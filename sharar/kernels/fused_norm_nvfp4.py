"""Gemma RMSNorm (+ residual add) with the normalized row written straight
as an NVFP4 activation (packed e2m1 + swizzled e4m3 block scales).

The MLP input is only ever consumed by the NVFP4 gate/up GEMM, so the bf16
normalized tensor that `fused_norm.gemma_rmsnorm` writes and
`nvfp4_quant.nvfp4_quantize` reads back is pure traffic: 2 x M x 5120 x 2
bytes per layer, about 2.7 GB per 2K-token prefill. This kernel does both in
one pass over the row. The quantization math is a copy of
`nvfp4_quant._nvfp4_quant_kernel` applied to the bf16-rounded normalized
values, so its output is bit-identical to the two-kernel path
(`tests/unit/test_fused_kernels.py` checks that).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sharar.kernels.nvfp4_quant import _scale_buffer, rcp_approx_ftz


@triton.jit
def _gemma_rmsnorm_nvfp4_kernel(
    x_ptr, res_ptr, w_ptr, gs_ptr, q_ptr, sf_ptr,
    stride_x, stride_res,
    N, K_PAD_GROUPS, eps,
    HAS_RES: tl.constexpr, FP32_RESIDUAL_SUM: tl.constexpr, BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_RES:
        r = tl.load(res_ptr + row * stride_res + cols, mask=mask, other=0.0).to(tl.float32)
        # Use the same arithmetic choice as the separate norm + quant path.
        x = x + r
        tl.store(res_ptr + row * stride_res + cols, x.to(tl.bfloat16), mask=mask)
        if not FP32_RESIDUAL_SUM:
            x = x.to(tl.bfloat16).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x * rstd * (1.0 + w)).to(tl.bfloat16).to(tl.float32)  # what the bf16 tensor would have held
    y = tl.where(mask, y, 0.0)
    # ---- NVFP4 quantization of the row (same math as nvfp4_quant) ----
    NB: tl.constexpr = BLOCK_N // 16
    xb = tl.reshape(y, (NB, 16))
    amax = tl.max(tl.abs(xb), axis=1)
    gs = tl.load(gs_ptr)
    sf = (gs * (amax * rcp_approx_ftz(tl.full((), 6.0, tl.float32)))).to(tl.float8e4nv)
    sf_f = sf.to(tl.float32)
    inv = tl.where(amax != 0.0, rcp_approx_ftz(sf_f * rcp_approx_ftz(gs)), 0.0)
    v = xb * inv[:, None]
    a = tl.minimum(tl.abs(v), 6.0)
    q = tl.where(a <= 0.25, 0.0,
        tl.where(a < 0.75, 0.5,
        tl.where(a <= 1.25, 1.0,
        tl.where(a < 1.75, 1.5,
        tl.where(a <= 2.5, 2.0,
        tl.where(a < 3.5, 3.0,
        tl.where(a <= 5.0, 4.0, 6.0)))))))
    code = tl.where(q == 0.0, 0, tl.where(q == 0.5, 1, tl.where(q == 1.0, 2, tl.where(q == 1.5, 3,
           tl.where(q == 2.0, 4, tl.where(q == 3.0, 5, tl.where(q == 4.0, 6, 7))))))).to(tl.int32)
    code = tl.where(v.to(tl.int32, bitcast=True) < 0, code | 8, code)
    code = tl.reshape(code, (BLOCK_N,))
    lo = tl.reshape(code, (BLOCK_N // 2, 2))
    packed = (tl.sum(lo * tl.reshape(tl.where(tl.arange(0, BLOCK_N) % 2 == 0, 1, 16), (BLOCK_N // 2, 2)), axis=1)).to(tl.uint8)
    pcols = tl.arange(0, BLOCK_N // 2)
    tl.store(q_ptr + row * (N // 2) + pcols, packed, mask=pcols < N // 2)
    c = tl.arange(0, NB)
    tile_r = row // 128
    r_in = row % 128
    off = tile_r * (128 * K_PAD_GROUPS) + (r_in % 32) * 16 + ((r_in // 32) % 4) * 4 + (c // 4) * 512 + (c % 4)
    tl.store(sf_ptr + off, sf, mask=c < (N // 16))


def gemma_rmsnorm_nvfp4(
    x: torch.Tensor, weight: torch.Tensor, eps: float, global_scale: torch.Tensor, residual: torch.Tensor | None = None,
    *, fp32_residual_sum: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize rows of x (adding `residual` in place first when given) and
    return the NVFP4 activation (packed [M, N/2] uint8, swizzled scales)."""
    assert x.dim() == 2 and x.stride(-1) == 1
    m, n = x.shape
    assert n % 16 == 0
    if residual is not None:
        assert residual.shape == x.shape and residual.stride(-1) == 1
    q = torch.empty((m, n // 2), dtype=torch.uint8, device=x.device)
    sf = _scale_buffer(m, n, x.device)
    block_n = triton.next_power_of_2(n)
    num_warps = min(max(block_n // 512, 1), 16)
    _gemma_rmsnorm_nvfp4_kernel[(m,)](
        x, residual if residual is not None else x, weight, global_scale, q, sf,
        x.stride(0), residual.stride(0) if residual is not None else 0,
        n, sf.shape[1], eps,
        HAS_RES=residual is not None, FP32_RESIDUAL_SUM=fp32_residual_sum, BLOCK_N=block_n, num_warps=num_warps,
    )
    return q, sf
