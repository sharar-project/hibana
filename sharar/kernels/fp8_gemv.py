"""Small-M FP8 GEMM as a weight-streaming GEMV (Triton).

For decode (M <= 4) cuBLAS's FP8 GEMM on SM120 reaches ~60% of DRAM bandwidth
(measured 36.9 ms for ~6 GB of FP8 weights per step). Here each program owns a
block of output columns, streams its W rows exactly once in K chunks, and keeps
the M activation rows in registers. y = (x_q @ W_q^T) * (in_scale * w_scale).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_gemv_kernel(
    x_ptr, w_ptr, y_ptr, alpha_ptr,
    M, N, K,
    stride_xm, stride_wn, stride_ym,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, M_PAD: tl.constexpr,
):
    pid = tl.program_id(0)
    rows_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = rows_n < N
    rows_m = tl.arange(0, M_PAD)
    mask_m = rows_m < M
    acc = tl.zeros((M_PAD, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        w = tl.load(w_ptr + rows_n[:, None] * stride_wn + ks[None, :], mask=mask_n[:, None], other=0.0)  # [BN, BK] fp8
        x = tl.load(x_ptr + rows_m[:, None] * stride_xm + ks[None, :], mask=mask_m[:, None], other=0.0)  # [MP, BK] fp8
        acc += tl.dot(x.to(tl.bfloat16), tl.trans(w.to(tl.bfloat16)), out_dtype=tl.float32)
    alpha = tl.load(alpha_ptr)
    y = acc * alpha
    tl.store(y_ptr + rows_m[:, None] * stride_ym + rows_n[None, :], y.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


def _config(n: int, k: int) -> tuple[int, int, int]:
    # From tools/bench_fp8_gemv.py on GB10 (M=1): wide-N projections stream
    # best with small row blocks; the 5120x6144 output projections with wider.
    if n >= 8192:
        return 32, 256, 4
    return 64, 512, 8


def fp8_gemv(x_q: torch.Tensor, w_q: torch.Tensor, alpha: torch.Tensor, block_n: int | None = None, block_k: int | None = None, num_warps: int | None = None) -> torch.Tensor:
    """x_q: [M, K] fp8 (M <= 16), w_q: [N, K] fp8 row-major, alpha: [] fp32. Returns [M, N] bf16."""
    m, k = x_q.shape
    n = w_q.shape[0]
    bn, bk, nw = _config(n, k)
    block_n = block_n or bn
    block_k = block_k or bk
    num_warps = num_warps or nw
    assert w_q.stride(1) == 1 and x_q.stride(1) == 1 and k % block_k == 0
    y = torch.empty((m, n), dtype=torch.bfloat16, device=x_q.device)
    m_pad = 16
    grid = (triton.cdiv(n, block_n),)
    _fp8_gemv_kernel[grid](
        x_q, w_q, y, alpha, m, n, k, x_q.stride(0), w_q.stride(0), y.stride(0),
        BLOCK_N=block_n, BLOCK_K=block_k, M_PAD=m_pad, num_warps=num_warps,
    )
    return y


@triton.jit
def _fp8_gemv_fma_kernel(
    x_ptr, w_ptr, y_ptr, alpha_ptr,
    M, N, K,
    stride_xm, stride_wn, stride_ym,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, M_ROWS: tl.constexpr,
):
    """FMA variant: no tensor cores, each program streams BLOCK_N rows of W once
    and dots them with up to M_ROWS activation rows kept in registers."""
    pid = tl.program_id(0)
    rows_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = rows_n < N
    acc = tl.zeros((M_ROWS, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        w = tl.load(w_ptr + rows_n[:, None] * stride_wn + ks[None, :], mask=mask_n[:, None], other=0.0).to(tl.float32)  # [BN, BK]
        for m in tl.static_range(M_ROWS):
            xm = tl.load(x_ptr + m * stride_xm + ks, mask=(m < M) & (ks < K), other=0.0).to(tl.float32)  # [BK]
            part = tl.sum(w * xm[None, :], axis=1)  # [BN]
            acc = tl.where(tl.arange(0, M_ROWS)[:, None] == m, acc + part[None, :], acc)
    alpha = tl.load(alpha_ptr)
    rows_m = tl.arange(0, M_ROWS)
    tl.store(y_ptr + rows_m[:, None] * stride_ym + rows_n[None, :], (acc * alpha).to(tl.bfloat16), mask=(rows_m[:, None] < M) & mask_n[None, :])


def fp8_gemv_fma(x_q: torch.Tensor, w_q: torch.Tensor, alpha: torch.Tensor, block_n: int = 32, block_k: int = 512, num_warps: int = 4, num_stages: int = 3) -> torch.Tensor:
    m, k = x_q.shape
    n = w_q.shape[0]
    y = torch.empty((m, n), dtype=torch.bfloat16, device=x_q.device)
    m_rows = 4 if m > 1 else 1
    _fp8_gemv_fma_kernel[(triton.cdiv(n, block_n),)](
        x_q, w_q, y, alpha, m, n, k, x_q.stride(0), w_q.stride(0), y.stride(0),
        BLOCK_N=block_n, BLOCK_K=block_k, M_ROWS=m_rows, num_warps=num_warps, num_stages=num_stages,
    )
    return y


@triton.jit
def _fp8_gemv_splitk_kernel(
    x_ptr, w_ptr, part_ptr,
    M, N, K, K_PER_SPLIT,
    stride_xm, stride_wn, stride_ps, stride_pm,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, M_PAD: tl.constexpr, EVICT: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    rows_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = rows_n < N
    rows_m = tl.arange(0, M_PAD)
    mask_m = rows_m < M
    acc = tl.zeros((M_PAD, BLOCK_N), dtype=tl.float32)
    k_start = pid_s * K_PER_SPLIT
    for k0 in range(k_start, k_start + K_PER_SPLIT, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        if EVICT:
            w = tl.load(w_ptr + rows_n[:, None] * stride_wn + ks[None, :], mask=mask_n[:, None], other=0.0, eviction_policy="evict_first")
        else:
            w = tl.load(w_ptr + rows_n[:, None] * stride_wn + ks[None, :], mask=mask_n[:, None], other=0.0)
        x = tl.load(x_ptr + rows_m[:, None] * stride_xm + ks[None, :], mask=mask_m[:, None], other=0.0)
        acc += tl.dot(x.to(tl.bfloat16), tl.trans(w.to(tl.bfloat16)), out_dtype=tl.float32)
    tl.store(part_ptr + pid_s * stride_ps + rows_m[:, None] * stride_pm + rows_n[None, :], acc, mask=mask_m[:, None] & mask_n[None, :])


def fp8_gemv_splitk(x_q: torch.Tensor, w_q: torch.Tensor, alpha: torch.Tensor, block_n: int = 64, block_k: int = 256, num_warps: int = 4, splits: int = 4, evict: bool = True) -> torch.Tensor:
    m, k = x_q.shape
    n = w_q.shape[0]
    assert k % (splits * block_k) == 0
    part = torch.empty((splits, m, n), dtype=torch.float32, device=x_q.device)
    _fp8_gemv_splitk_kernel[(triton.cdiv(n, block_n), splits)](
        x_q, w_q, part, m, n, k, k // splits, x_q.stride(0), w_q.stride(0), part.stride(0), part.stride(1),
        BLOCK_N=block_n, BLOCK_K=block_k, M_PAD=16, EVICT=evict, num_warps=num_warps,
    )
    return (part.sum(0) * alpha).to(torch.bfloat16)
