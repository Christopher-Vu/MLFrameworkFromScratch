import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    """
    x_ptr: pointer (integer that points to the memory address of the start) for the first arr
    y_ptr: pointer for 2nd arr
    out_ptr: pointer for where to store the result
    # n is the number of elements in the array
    # BLOCK_SIZE: number of elements each program processes (likely don't have to edit)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    tl.store(out_ptr + offs,
             tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask),
             mask=mask)

def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda, "inputs must be CUDA tensors"
    if x.shape != y.shape:
        y = y.expand_as(x)
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def relu_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, tl.where(x >= 0, x, 0.0), mask=mask)

def relu(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    relu_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def partial_sum_kernel(x_ptr, partial_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(partial_ptr + pid, tl.sum(x, axis=0))

def mean(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    n = x.numel()
    BLOCK_SIZE = 1024
    n_programs = triton.cdiv(n, BLOCK_SIZE)
    partial = torch.empty(n_programs, device='cuda', dtype=torch.float32)
    grid = (n_programs,)
    partial_sum_kernel[grid](x, partial, n, BLOCK_SIZE=BLOCK_SIZE)
    return partial.sum() / n


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(tl.cdiv(K, BK)):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k*BK), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k*BK) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_bk

    offs_cm = pid_m * BM + tl.arange(0, BM)
    offs_cn = pid_n * BN + tl.arange(0, BN)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    K, N = b.shape
    a = a.to(torch.float16)
    b = b.to(torch.float16)
    c = torch.empty((M, N), device='cuda', dtype=torch.float16)
    BM, BN, BK = 64, 64, 32
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BM=BM, BN=BN, BK=BK
    )
    return c


@triton.jit
def softmax_kernel(out_ptr, x_ptr, row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    x = tl.load(x_ptr + row * row_stride + offs, mask=mask, other=-float('inf'))
    x = x - tl.max(x, axis=0)
    x = tl.exp(x)
    x = x / tl.sum(x, axis=0)

    tl.store(out_ptr + row * row_stride + offs, x, mask=mask)

def softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.ndim == 2
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    out = torch.empty_like(x)
    grid = (n_rows,)
    softmax_kernel[grid](out, x, x.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def batchnorm_kernel(X, Y, W, B, N, C, eps, BLOCK_SIZE: tl.constexpr):
    feat = tl.program_id(0)

    _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        rows = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + rows * C + feat, mask=rows < N, other=0.0).to(tl.float32)
        _sum += x
    mean = tl.sum(_sum, axis=0) / N

    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        rows = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + rows * C + feat, mask=rows < N, other=0.0).to(tl.float32)
        d = tl.where(rows < N, x - mean, 0.0)
        _var += d * d
    rstd = 1.0 / tl.sqrt(tl.sum(_var, axis=0) / N + eps)

    w = tl.load(W + feat)
    b_val = tl.load(B + feat)
    for off in range(0, N, BLOCK_SIZE):
        rows = off + tl.arange(0, BLOCK_SIZE)
        mask = rows < N
        x = tl.load(X + rows * C + feat, mask=mask, other=0.0).to(tl.float32)
        tl.store(Y + rows * C + feat, (x - mean) * rstd * w + b_val, mask=mask)

def batch_norm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    assert x.is_cuda and x.ndim == 2
    N, C = x.shape
    out = torch.empty_like(x)
    grid = (C,)
    batchnorm_kernel[grid](x, out, w, b, N, C, eps, BLOCK_SIZE=min(1024, triton.next_power_of_2(N)))
    return out


@triton.jit
def max_pool1d_kernel(X_ptr, Out_ptr, L, kernel_size, stride, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    in_start = pid * stride
    offs = in_start + tl.arange(0, BLOCK_SIZE)
    mask = (offs < L) & (offs < in_start + kernel_size)
    x = tl.load(X_ptr + offs, mask=mask, other=-float('inf'))
    tl.store(Out_ptr + pid, tl.max(x, axis=0))

def max_pool1d(x: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    assert x.is_cuda and x.ndim == 1
    L = x.numel()
    L_out = (L - kernel_size) // stride + 1
    out = torch.empty(L_out, device='cuda', dtype=x.dtype)
    BLOCK_SIZE = triton.next_power_of_2(kernel_size)
    grid = (L_out,)
    max_pool1d_kernel[grid](x, out, L, kernel_size, stride, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def gap_kernel(X_ptr, Out_ptr, spatial, BLOCK_SIZE: tl.constexpr):
    bc = tl.program_id(0)
    base = bc * spatial
    _acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, spatial, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        _acc += tl.load(X_ptr + base + offs, mask=offs < spatial, other=0.0).to(tl.float32)
    tl.store(Out_ptr + bc, tl.sum(_acc, axis=0) / spatial)

def global_average_pool(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.ndim == 4
    B, C, H, W = x.shape
    spatial = H * W
    x_flat = x.reshape(B * C, spatial)
    out = torch.empty(B * C, device='cuda', dtype=torch.float32)
    BLOCK_SIZE = min(1024, triton.next_power_of_2(spatial))
    grid = (B * C,)
    gap_kernel[grid](x_flat, out, spatial, BLOCK_SIZE=BLOCK_SIZE)
    return out.reshape(B, C)
