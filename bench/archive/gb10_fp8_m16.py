"""M=16 FP8 weight-streaming GEMM for the 176 target verification projections.

`torch._scaled_mm` at M=16 reaches only 156-214 GB/s on the five serving shapes
while this GB10's streaming-read ceiling is 239 GB/s (2026-09-10 microbenchmark).
One program owns 16 output columns, streams its weight rows once in K=256 chunks
with evict-first loads, and keeps the 16 activation rows in registers, which
measured 214-235 GB/s on the same shapes.

The FP32 accumulation order differs from cuBLAS, so outputs differ in the last
BF16 bit; this is a numerics-changing candidate judged by the 21-task protocol
(`bench/gb10_family_eval.py`), not an exact-output kernel.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK_N, BLOCK_K, NUM_WARPS, NUM_STAGES = 16, 256, 4, 4
SHAPES = ((10240, 5120), (6144, 5120), (5120, 6144), (14336, 5120), (4096, 5120))


@triton.jit
def _fp8_m16_kernel(x_ptr, w_ptr, y_ptr, alpha_ptr, N, K, stride_xm, stride_wn, stride_ym,
                    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, M_PAD: tl.constexpr):
    pid = tl.program_id(0)
    rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = tl.arange(0, M_PAD)
    rk = tl.arange(0, BLOCK_K)
    mask_n = rn < N
    acc = tl.zeros((M_PAD, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + rk
        w = tl.load(w_ptr + rn[:, None] * stride_wn + ks[None, :], mask=mask_n[:, None], other=0.0,
                    eviction_policy='evict_first')
        x = tl.load(x_ptr + rm[:, None] * stride_xm + ks[None, :])
        acc = tl.dot(x, tl.trans(w), acc)
    alpha = tl.load(alpha_ptr)
    tl.store(y_ptr + rm[:, None] * stride_ym + rn[None, :], (acc * alpha).to(tl.bfloat16), mask=mask_n[None, :])


def fp8_gemm_m16(x_q: torch.Tensor, weight: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """x_q: [16, K] float8_e4m3fn, weight: [N, K] float8_e4m3fn row-major, alpha: [] float32."""
    m, k = x_q.shape
    n = weight.shape[0]
    if m != 16 or k % BLOCK_K or n % BLOCK_N or x_q.stride(1) != 1 or weight.stride(1) != 1:
        raise ValueError('unsupported shape for the M=16 FP8 kernel')
    y = torch.empty((m, n), dtype=torch.bfloat16, device=x_q.device)
    _fp8_m16_kernel[(triton.cdiv(n, BLOCK_N),)](
        x_q, weight, y, alpha, n, k, x_q.stride(0), weight.stride(0), y.stride(0),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, M_PAD=16, num_warps=NUM_WARPS, num_stages=NUM_STAGES)
    return y


def supported(x_q: torch.Tensor, lin) -> bool:
    n, k = lin.weight.shape
    return (x_q.shape[0] == 16 and x_q.shape[1] == k and k % BLOCK_K == 0 and n % BLOCK_N == 0
            and x_q.stride(1) == 1 and lin.weight.stride(1) == 1)


def validate(device='cuda', seed=0, samples=3) -> dict:
    """Compare against torch._scaled_mm on random operands of the serving shapes."""
    torch.manual_seed(seed)
    worst = {'nrmse': 0.0, 'max_abs': 0.0, 'argmax_equal': True, 'exact': True, 'shapes': []}
    a = torch.tensor(0.023, device=device)
    b = torch.tensor(0.0031, device=device)
    alpha = (a * b).float()
    for n, k in SHAPES:
        for _ in range(samples):
            x = (torch.randn(16, k, device=device) * 40).to(torch.float8_e4m3fn)
            w = (torch.randn(n, k, device=device) * 40).to(torch.float8_e4m3fn)
            ref = torch._scaled_mm(x, w.t(), scale_a=a.reshape(1, 1), scale_b=b.reshape(1, 1), out_dtype=torch.bfloat16)
            got = fp8_gemm_m16(x, w, alpha)
            delta = (got.float() - ref.float())
            worst['nrmse'] = max(worst['nrmse'], float(delta.norm() / ref.float().norm().clamp_min(1e-30)))
            worst['max_abs'] = max(worst['max_abs'], float(delta.abs().max()))
            worst['argmax_equal'] &= torch.equal(got.argmax(-1), ref.argmax(-1))
            worst['exact'] &= torch.equal(got, ref)
        worst['shapes'].append([n, k])
    return worst
