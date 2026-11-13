"""M-invariant BF16 projection for the GDN b/a gates (Triton).

``ba_project(x [M, K], w [N, K]) -> [M, N]`` in BF16 with FP32 accumulation.
Every output row is computed by a fixed tile sequence: K is cut into SPLIT_K
equal ranges, each accumulated over [BLOCK_M, BLOCK_K] x [BLOCK_K, BLOCK_N]
tiles in order, and the SPLIT_K partial sums are added in a fixed order. So
the arithmetic of a row does not depend on M, on which block the row lands in
or on how many blocks run. cuBLAS's BF16 GEMM changes algorithm with M and
flips BF16 roundings of these gates in most layers
(target-numerics study, tag archive/pre-cleanup-2026-09-26). The split keeps small-M calls off
a single SM: 48 layers of a 16-row prefill cost 0.7 ms instead of 3.8 ms.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK_M, BLOCK_K, SPLIT_K = 16, 64, 16


@triton.jit
def _ba_partial_kernel(x_ptr, w_ptr, part_ptr, M, stride_xm, stride_wn, K: tl.constexpr, N: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
    row0 = tl.program_id(0) * BLOCK_M
    split = tl.program_id(1)
    rows = row0 + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_begin = split * (K // SPLIT_K)
    for k0 in tl.static_range(0, K // SPLIT_K, BLOCK_K):
        ks = k_begin + k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + rows[:, None] * stride_xm + ks[None, :], mask=rows[:, None] < M, other=0.0)
        w = tl.load(w_ptr + cols[None, :] * stride_wn + ks[:, None], mask=cols[None, :] < N, other=0.0)  # [BLOCK_K, BLOCK_N]
        acc = tl.dot(x, w, acc)
    out = part_ptr + split * M * N + rows[:, None] * N + cols[None, :]
    tl.store(out, acc, mask=(rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def _ba_reduce_kernel(part_ptr, out_ptr, M, stride_om, N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                      SPLIT_K: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for split in tl.static_range(0, SPLIT_K):  # fixed order
        acc += tl.load(part_ptr + split * M * N + rows[:, None] * N + cols[None, :], mask=mask, other=0.0)
    tl.store(out_ptr + rows[:, None] * stride_om + cols[None, :], acc.to(tl.bfloat16), mask=mask)


def ba_project(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16 or x.shape[-1] != w.shape[1]:
        raise ValueError("ba_project expects BF16 x [M, K] and w [N, K]")
    n, k = w.shape
    if k % (BLOCK_K * SPLIT_K) or n > 256:
        raise ValueError("K must be a multiple of the split block and N at most 256")
    x2 = x.reshape(-1, k)
    if x2.stride(1) != 1:
        x2 = x2.contiguous()
    m = x2.shape[0]
    block_n = max(16, triton.next_power_of_2(n))
    partial = torch.empty((SPLIT_K, m, n), dtype=torch.float32, device=x.device)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    blocks = triton.cdiv(m, BLOCK_M)
    _ba_partial_kernel[(blocks, SPLIT_K)](x2, w, partial, m, x2.stride(0), w.stride(0), K=k, N=n, BLOCK_M=BLOCK_M,
                                         BLOCK_N=block_n, BLOCK_K=BLOCK_K, SPLIT_K=SPLIT_K, num_warps=4)
    _ba_reduce_kernel[(blocks,)](partial, out, m, out.stride(0), N=n, BLOCK_M=BLOCK_M, BLOCK_N=block_n, SPLIT_K=SPLIT_K, num_warps=4)
    return out.view(*x.shape[:-1], n)
