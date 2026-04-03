# Triton GPU Kernel Cheat Sheet — Group B

## What You're Building
Vector Addition, Mean, GeMM, ReLU, Fused Softmax, Batch Norm, Max Pooling, Global Average Pooling — all as Triton GPU kernels.

---

## 1. The Mental Model: SPMD

Triton uses **Single Program, Multiple Data**. You write one kernel function, but the GPU launches hundreds of identical copies (called **programs**) simultaneously. Each program processes a different *tile* (chunk) of the data.

```
Input array:  [  0..1023  |  1024..2047  |  2048..3071  | ... ]
                  pid=0         pid=1          pid=2
```

Each `pid` handles its block in parallel — no waiting for the others.

**Why this is fast:** A CPU executes one loop iteration at a time. The GPU runs all programs simultaneously on thousands of cores.

---

## 2. The Three Core Builtins

Everything in Triton flows from three primitives:

```python
pid     = tl.program_id(axis=0)                        # which tile am I? (0-indexed program index)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global indices this program owns
mask    = offsets < n_elements                          # True for valid indices, False for out-of-bounds
```

Then load, compute, store:

```python
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)  # fetch this tile's elements from GPU memory; pad OOB slots with 0.0
y = x + 1                                            # any computation — runs in registers (fast)
tl.store(out_ptr + offsets, y, mask=mask)            # write results back to GPU memory; mask prevents OOB writes
```

`BLOCK_SIZE` must be declared `tl.constexpr` so the compiler can optimize it at compile time.

---

## 3. Kernel Structure + Launch

```python
import triton
import triton.language as tl

@triton.jit  # compiles this function to GPU machine code; boundary between CPU and GPU
def my_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)                               # which program am I (determines which tile I own)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global indices for this tile
    mask = offs < n                                      # don't touch memory beyond the array
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)     # load this tile from GPU memory into registers
    tl.store(out_ptr + offs, x * 2, mask=mask)          # multiply each element by 2, write back to GPU memory

# Launch: grid tells Triton how many programs to spawn
grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)  # one program per tile; cdiv ensures last partial tile is covered
my_kernel[grid](x, out, n, BLOCK_SIZE=1024)                # pass tensors + params; Triton extracts raw pointers automatically
```

`triton.cdiv(n, BLOCK_SIZE)` = ceiling division — ensures the last partial block is still covered.

---

## 4. The Kernels

---

### Vector Addition
Fully parallel — N independent adds with no data dependencies.

```python
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)                               # which tile this program owns
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global indices for this tile
    mask = offs < n                                      # guard against out-of-bounds on last tile
    tl.store(out_ptr + offs,                             # write result to output tensor
             tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask),  # load both tiles and add elementwise
             mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda, "inputs must be CUDA tensors"
    assert x.shape == y.shape
    out = torch.empty_like(x)            # allocate output on GPU, same shape/dtype as input
    n = x.numel()                        # total number of elements
    BLOCK_SIZE = 1024                    # elements per program; power of 2, standard default
    grid = (triton.cdiv(n, BLOCK_SIZE),) # number of programs to launch
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


# --- usage ---
n = 2 ** 20
x = torch.rand(n, device='cuda', dtype=torch.float32)
y = torch.rand(n, device='cuda', dtype=torch.float32)
out = vector_add(x, y)
assert torch.allclose(out, x + y)
```

---

### ReLU
Same tile pattern as vector addition. `tl.where` maps to a single GPU select instruction.

```python
import torch
import triton
import triton.language as tl


@triton.jit
def relu_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)                               # which tile this program owns
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global indices for this tile
    mask = offs < n                                      # guard against out-of-bounds
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)     # load tile; OOB slots get 0.0 (safe for relu)
    tl.store(out_ptr + offs, tl.where(x >= 0, x, 0.0), mask=mask)  # keep positives, zero negatives


def relu(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    relu_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


# --- usage ---
x = torch.randn(2 ** 20, device='cuda', dtype=torch.float32)
out = relu(x)
assert torch.allclose(out, x.clamp(min=0))
```

---

### Mean (Reduction)
Two-phase: each program sums its block in parallel, then a final CPU-side sum combines partial results.

```python
import torch
import triton
import triton.language as tl


@triton.jit
def partial_sum_kernel(x_ptr, partial_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)                               # which tile this program owns
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global indices for this tile
    mask = offs < n                                      # guard against out-of-bounds
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)     # load tile; OOB slots get 0.0 so they don't affect the sum
    tl.store(partial_ptr + pid, tl.sum(x, axis=0))      # reduce [BLOCK_SIZE] vector to scalar, store at pid's slot


def mean(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    n = x.numel()
    BLOCK_SIZE = 1024
    n_programs = triton.cdiv(n, BLOCK_SIZE)
    partial = torch.empty(n_programs, device='cuda', dtype=torch.float32)  # one slot per program for partial sums
    grid = (n_programs,)
    partial_sum_kernel[grid](x, partial, n, BLOCK_SIZE=BLOCK_SIZE)
    return partial.sum() / n  # combine partial sums on CPU and divide for global mean


# --- usage ---
x = torch.randn(2 ** 20, device='cuda', dtype=torch.float32)
out = mean(x)
assert torch.allclose(out, x.mean(), atol=1e-4)
```

`tl.sum(x, axis=0)` reduces a `[BLOCK_SIZE]` vector to a scalar within one program. O(n) → O(n/BLOCK_SIZE) parallel steps.

---

### GeMM (General Matrix Multiply)
Tiles both matrices into blocks and accumulates dot products in registers — never returning to DRAM mid-computation.

**Why it's fast:** `tl.dot` maps to GPU tensor cores — specialized hardware that does a full `[BM×BK] × [BK×BN]` multiply in ~1 cycle.

```python
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)  # which row-tile of C this program computes
    pid_n = tl.program_id(1)  # which col-tile of C this program computes

    offs_m = pid_m * BM + tl.arange(0, BM)  # row indices for this tile in A and C
    offs_n = pid_n * BN + tl.arange(0, BN)  # col indices for this tile in B and C
    offs_k = tl.arange(0, BK)               # indices along the shared K dimension (advances each loop iteration)

    # build 2D pointer blocks via broadcasting: [:, None] makes a column vector, [None, :] makes a row vector
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak  # [BM, BK] pointer block into A
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn  # [BK, BN] pointer block into B

    acc = tl.zeros((BM, BN), dtype=tl.float32)  # accumulator for output tile, starts at zero
    for k in range(tl.cdiv(K, BK)):             # step through K dimension one BK-wide slice at a time
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k*BK), other=0.0)  # load A tile; mask boundary
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k*BK) & (offs_n[None, :] < N), other=0.0)  # load B tile; mask boundary
        acc = tl.dot(a, b, acc)   # fused multiply-accumulate: acc += a @ b, uses tensor cores
        a_ptrs += BK * stride_ak  # advance A pointer to next K slice
        b_ptrs += BK * stride_bk  # advance B pointer to next K slice

    offs_cm = pid_m * BM + tl.arange(0, BM)  # row indices for writing output tile to C
    offs_cn = pid_n * BN + tl.arange(0, BN)  # col indices for writing output tile to C
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn  # 2D pointer block into C
    tl.store(c_ptrs, acc.to(tl.float16), mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))  # write; cast to fp16; mask boundary


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    K, N = b.shape
    a = a.to(torch.float16)                                   # tensor cores require fp16 input
    b = b.to(torch.float16)
    c = torch.empty((M, N), device='cuda', dtype=torch.float16)
    BM, BN, BK = 64, 64, 32                                   # tile sizes; tune for your GPU
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))           # one program per output tile
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),  # strides tell the kernel how many elements to skip per row/col step
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BM=BM, BN=BN, BK=BK
    )
    return c


# --- usage ---
M, K, N = 512, 512, 512
a = torch.randn(M, K, device='cuda')
b = torch.randn(K, N, device='cuda')
c = matmul(a, b)
ref = (a.float() @ b.float()).half()
assert torch.allclose(c, ref, atol=1e-1)  # fp16 accumulation has larger tolerance
```

Key pattern: `offs_m[:, None]` and `offs_k[None, :]` broadcast 1D vectors into a 2D pointer matrix.

---

### Fused Softmax
Naive softmax needs 3 DRAM passes (max scan → exp+sum → divide). Fused softmax keeps the entire row in registers and does all three in one pass.

**Why it's fast:** DRAM bandwidth is the bottleneck. 1 read + 1 write vs 3 reads + 2 writes = ~3× speedup for large rows.

```python
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(out_ptr, x_ptr, row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)                 # one program per row
    offs = tl.arange(0, BLOCK_SIZE)        # column indices (BLOCK_SIZE must fit the entire row)
    mask = offs < n_cols                   # guard against padding columns beyond actual row length

    x = tl.load(x_ptr + row * row_stride + offs, mask=mask, other=-float('inf'))  # load entire row; OOB slots get -inf so they don't affect max/sum
    x = x - tl.max(x, axis=0)             # subtract row max for numerical stability (prevents exp overflow)
    x = tl.exp(x)                          # exponentiate shifted values
    x = x / tl.sum(x, axis=0)             # normalize by sum to get probabilities

    tl.store(out_ptr + row * row_stride + offs, x, mask=mask)  # write softmax output back to GPU memory


def softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.ndim == 2
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)  # must be power of 2 and large enough to hold the full row
    out = torch.empty_like(x)
    grid = (n_rows,)                             # one program per row
    softmax_kernel[grid](out, x, x.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out


# --- usage ---
x = torch.randn(128, 512, device='cuda', dtype=torch.float32)
out = softmax(x)
assert torch.allclose(out, torch.softmax(x, dim=1), atol=1e-4)
```

Subtract max before `exp` to avoid overflow (`e^700 = inf` in float32) — the max cancels out mathematically.

---

### Batch Norm
One program per feature/channel. Each independently normalizes its column across the batch.

**Why it's fast:** C features = C independent programs run in parallel. The serial CPU bottleneck (iterate over N samples per feature) becomes simultaneous.

```python
import torch
import triton
import triton.language as tl


@triton.jit
def batchnorm_kernel(X, Y, W, B, N, C, eps, BLOCK_SIZE: tl.constexpr):
    feat = tl.program_id(0)  # one program per feature (column)

    # Pass 1: mean
    _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)  # accumulator for partial sums across rows
    for off in range(0, N, BLOCK_SIZE):              # iterate over rows in BLOCK_SIZE chunks
        rows = off + tl.arange(0, BLOCK_SIZE)        # row indices for this chunk
        x = tl.load(X + rows * C + feat, mask=rows < N, other=0.0).to(tl.float32)  # load this feature's values for these rows
        _sum += x                                    # accumulate partial sum
    mean = tl.sum(_sum, axis=0) / N                  # reduce partial sums to scalar mean

    # Pass 2: variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)  # accumulator for partial variance
    for off in range(0, N, BLOCK_SIZE):
        rows = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + rows * C + feat, mask=rows < N, other=0.0).to(tl.float32)  # reload feature values
        d = tl.where(rows < N, x - mean, 0.0)       # deviation from mean; zero out OOB slots
        _var += d * d                                # accumulate squared deviations
    rstd = 1.0 / tl.sqrt(tl.sum(_var, axis=0) / N + eps)  # reciprocal std dev; eps prevents division by zero

    # Pass 3: normalize
    w = tl.load(W + feat)      # learnable scale parameter for this feature
    b_val = tl.load(B + feat)  # learnable bias parameter for this feature
    for off in range(0, N, BLOCK_SIZE):
        rows = off + tl.arange(0, BLOCK_SIZE)
        mask = rows < N
        x = tl.load(X + rows * C + feat, mask=mask, other=0.0).to(tl.float32)       # reload feature values
        tl.store(Y + rows * C + feat, (x - mean) * rstd * w + b_val, mask=mask)     # normalize, scale, shift, write output


def batch_norm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    assert x.is_cuda and x.ndim == 2  # x shape: [N, C]
    N, C = x.shape
    out = torch.empty_like(x)
    grid = (C,)                       # one program per feature column
    batchnorm_kernel[grid](x, out, w, b, N, C, eps, BLOCK_SIZE=min(1024, triton.next_power_of_2(N)))
    return out


# --- usage ---
N, C = 512, 64
x = torch.randn(N, C, device='cuda', dtype=torch.float32)
w = torch.ones(C, device='cuda', dtype=torch.float32)   # scale initialized to 1
b = torch.zeros(C, device='cuda', dtype=torch.float32)  # bias initialized to 0
out = batch_norm(x, w, b)
ref = torch.nn.functional.batch_norm(x, None, None, w, b, training=True)
assert torch.allclose(out, ref, atol=1e-4)
```

---

### Max Pooling
One program per output element. Each loads its kernel window and reduces with `tl.max`.

**Why it's fast:** All output elements are independent — all computed simultaneously.

```python
import torch
import triton
import triton.language as tl


@triton.jit
def max_pool1d_kernel(X_ptr, Out_ptr, L, kernel_size, stride, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)                                     # which output element this program computes
    in_start = pid * stride                                    # where this output's receptive field starts in the input
    offs = in_start + tl.arange(0, BLOCK_SIZE)                 # indices of input elements in the kernel window
    mask = (offs < L) & (offs < in_start + kernel_size)        # stay within input bounds AND within kernel window
    x = tl.load(X_ptr + offs, mask=mask, other=-float('inf'))  # load window; OOB slots get -inf so they don't win the max
    tl.store(Out_ptr + pid, tl.max(x, axis=0))                 # reduce window to scalar max, write to output


def max_pool1d(x: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    assert x.is_cuda and x.ndim == 1
    L = x.numel()
    L_out = (L - kernel_size) // stride + 1  # number of output elements
    out = torch.empty(L_out, device='cuda', dtype=x.dtype)
    BLOCK_SIZE = triton.next_power_of_2(kernel_size)  # must be power of 2 and >= kernel_size
    grid = (L_out,)                                   # one program per output element
    max_pool1d_kernel[grid](x, out, L, kernel_size, stride, BLOCK_SIZE=BLOCK_SIZE)
    return out


# --- usage ---
x = torch.randn(1024, device='cuda', dtype=torch.float32)
out = max_pool1d(x, kernel_size=4, stride=2)
ref = torch.nn.functional.max_pool1d(x.unsqueeze(0).unsqueeze(0), 4, stride=2).squeeze()
assert torch.allclose(out, ref)
```

For 2D pooling, use the `[:, None]` / `[None, :]` broadcasting pattern from GeMM to build a `[kH, kW]` tile, then reduce.

---

### Global Average Pooling
One program per `(batch, channel)` pair. Sums spatial dimensions `H×W` and divides.

```python
import torch
import triton
import triton.language as tl


@triton.jit
def gap_kernel(X_ptr, Out_ptr, spatial, BLOCK_SIZE: tl.constexpr):
    bc = tl.program_id(0)   # flattened (batch, channel) index — one program per (batch, channel) pair
    base = bc * spatial     # offset to the start of this (batch, channel)'s spatial data in the flat array
    _acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)  # accumulator for partial sums over spatial elements
    for off in range(0, spatial, BLOCK_SIZE):         # iterate over spatial elements in chunks
        offs = off + tl.arange(0, BLOCK_SIZE)         # indices within the spatial chunk
        _acc += tl.load(X_ptr + base + offs, mask=offs < spatial, other=0.0).to(tl.float32)  # accumulate
    tl.store(Out_ptr + bc, tl.sum(_acc, axis=0) / spatial)  # reduce to scalar mean, write to output


def global_average_pool(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.ndim == 4  # x shape: [B, C, H, W]
    B, C, H, W = x.shape
    spatial = H * W
    x_flat = x.reshape(B * C, spatial)              # flatten to [B*C, H*W] for 1D indexing
    out = torch.empty(B * C, device='cuda', dtype=torch.float32)
    BLOCK_SIZE = min(1024, triton.next_power_of_2(spatial))
    grid = (B * C,)                                 # one program per (batch, channel) pair
    gap_kernel[grid](x_flat, out, spatial, BLOCK_SIZE=BLOCK_SIZE)
    return out.reshape(B, C)                        # return shape [B, C]


# --- usage ---
x = torch.randn(8, 16, 14, 14, device='cuda', dtype=torch.float32)  # batch=8, channels=16, 14x14 spatial
out = global_average_pool(x)                                          # shape: [8, 16]
ref = x.mean(dim=[2, 3])
assert torch.allclose(out, ref, atol=1e-4)
```

---

## 5. Quick Reference

| Pattern | Code |
|---|---|
| Tile indices | `offs = pid * BS + tl.arange(0, BS)` |
| Boundary mask | `mask = offs < n` |
| Safe load | `tl.load(ptr + offs, mask=mask, other=0.0)` |
| Row reduction | `tl.sum(x, axis=0)` / `tl.max(x, axis=0)` |
| 2D pointer block | `ptr + rows[:, None] * stride_r + cols[None, :] * stride_c` |
| Matrix multiply | `acc = tl.dot(a_tile, b_tile, acc)` |
| Conditional | `tl.where(cond, x, y)` |
| Grid size (1D) | `grid = (triton.cdiv(n, BLOCK_SIZE),)` |
| Grid size (2D) | `grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))` |
| Power of 2 | `triton.next_power_of_2(n)` |
| Input conversion | `torch.from_numpy(arr).cuda()` / `tensor.cuda()` |

**Docs:** https://triton-lang.org/main/index.html  
**Tutorials:** vector-add → fused-softmax → matrix-multiplication → layer-norm (tutorials 01–05)
