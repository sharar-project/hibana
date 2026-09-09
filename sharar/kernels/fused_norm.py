"""Experimental fused Gemma RMSNorm kernels (Triton).

Serving uses the native CUDA norm in model.norm to preserve its reduction
order. These kernels retain both residual modes for component experiments.

gemma_rmsnorm(x, weight, eps, residual=None, fp8_scale=None):
  - if residual is given: store x + residual as bf16 in residual and normalize
    that rounded sum by default; fp32_residual_sum opts into the unrounded sum
    for reference arithmetic experiments;
  - out_bf16 = x_norm * (1 + weight) in bf16;
  - if fp8_scale is given, also out_fp8 = clamp(out / scale) as float8_e4m3fn,
    the static per-tensor quantization every FP8 linear needs as input.
One read of x (+residual), one pass, no intermediate tensors.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

FP8_MAX = tl.constexpr(448.0)


@triton.jit
def _gemma_rmsnorm_kernel(
    x_ptr, res_ptr, w_ptr, out_ptr, fp8_ptr, scale_ptr,
    stride_x, stride_res, stride_out, stride_fp8,
    N, eps,
    HAS_RES: tl.constexpr, HAS_FP8: tl.constexpr,
    FP32_RESIDUAL_SUM: tl.constexpr, BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_RES:
        r = tl.load(res_ptr + row * stride_res + cols, mask=mask, other=0.0).to(tl.float32)
        # Explicit legacy mode rounds before normalization. The native target
        # norm uses the unrounded sum and its own CUDA reduction order.
        x = x + r
        tl.store(res_ptr + row * stride_res + cols, x.to(tl.bfloat16), mask=mask)
        if not FP32_RESIDUAL_SUM:
            x = x.to(tl.bfloat16).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * (1.0 + w)
    y_bf16 = y.to(tl.bfloat16)
    tl.store(out_ptr + row * stride_out + cols, y_bf16, mask=mask)
    if HAS_FP8:
        scale = tl.load(scale_ptr)
        q = y_bf16.to(tl.float32) / scale
        q = tl.minimum(tl.maximum(q, -FP8_MAX), FP8_MAX)
        tl.store(fp8_ptr + row * stride_fp8 + cols, q.to(tl.float8e4nv), mask=mask)


def gemma_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
    fp8_scale: torch.Tensor | None = None,
    *,
    fp32_residual_sum: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    assert x.dim() == 2 and x.stride(-1) == 1
    m, n = x.shape
    out = torch.empty_like(x)
    fp8 = torch.empty((m, n), dtype=torch.float8_e4m3fn, device=x.device) if fp8_scale is not None else None
    if residual is not None:
        assert residual.shape == x.shape and residual.stride(-1) == 1
    block_n = triton.next_power_of_2(n)
    num_warps = min(max(block_n // 512, 1), 16)
    _gemma_rmsnorm_kernel[(m,)](
        x, residual if residual is not None else x, weight, out,
        fp8 if fp8 is not None else out, fp8_scale if fp8_scale is not None else weight,
        x.stride(0), residual.stride(0) if residual is not None else 0, out.stride(0), fp8.stride(0) if fp8 is not None else 0,
        n, eps,
        HAS_RES=residual is not None, HAS_FP8=fp8 is not None,
        FP32_RESIDUAL_SUM=fp32_residual_sum, BLOCK_N=block_n, num_warps=num_warps,
    )
    return out, fp8
