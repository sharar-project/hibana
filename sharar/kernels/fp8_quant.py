"""Static per-tensor FP8 (e4m3) activation quantization (Triton).

quant_fp8(x, scale) -> x_q with x_q = clamp(x / scale, -448, 448).
gated_quant_fp8(x, gate, scale) -> quant(x * sigmoid(gate)), the attention
output gate fused with the o_proj input quantization. Row strides are free, so
strided views (e.g. a [T, H, D] gate slice) need no copy.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

FP8_MAX = tl.constexpr(448.0)


@triton.jit
def _quant_kernel(
    x_ptr, g_ptr, out_ptr, scale_ptr,
    stride_x, stride_g, stride_out, N,
    HAS_GATE: tl.constexpr, BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cb = tl.program_id(1)
    cols = cb * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_GATE:
        g = tl.load(g_ptr + row * stride_g + cols, mask=mask, other=0.0).to(tl.float32)
        x = (x * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
    scale = tl.load(scale_ptr)
    q = x / scale
    q = tl.minimum(tl.maximum(q, -FP8_MAX), FP8_MAX)
    tl.store(out_ptr + row * stride_out + cols, q.to(tl.float8e4nv), mask=mask)


def _launch(x2: torch.Tensor, gate2: torch.Tensor | None, scale: torch.Tensor) -> torch.Tensor:
    m, n = x2.shape
    out = torch.empty((m, n), dtype=torch.float8_e4m3fn, device=x2.device)
    block_n = min(4096, triton.next_power_of_2(n))
    grid = (m, triton.cdiv(n, block_n))
    _quant_kernel[grid](
        x2, gate2 if gate2 is not None else x2, out, scale,
        x2.stride(0), gate2.stride(0) if gate2 is not None else 0, out.stride(0), n,
        HAS_GATE=gate2 is not None, BLOCK_N=block_n, num_warps=8,
    )
    return out


def quant_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    x2 = x.reshape(-1, x.shape[-1]) if x.dim() != 2 else x
    assert x2.stride(-1) == 1
    return _launch(x2, None, scale)


@triton.jit
def _gated_quant_kernel(
    x_ptr, g_ptr, out_ptr, scale_ptr,
    stride_xt, stride_xh, stride_gt, stride_gh, stride_ot, stride_oh, D,
    BLOCK_D: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    x = tl.load(x_ptr + t * stride_xt + h * stride_xh + cols, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + t * stride_gt + h * stride_gh + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
    scale = tl.load(scale_ptr)
    q = y / scale
    q = tl.minimum(tl.maximum(q, -FP8_MAX), FP8_MAX)
    tl.store(out_ptr + t * stride_ot + h * stride_oh + cols, q.to(tl.float8e4nv), mask=mask)


def gated_quant_fp8(x: torch.Tensor, gate: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """x: [T, H, D] and gate: [T, H, D], both with unit last stride and any
    (token, head) strides, e.g. the gate half of a [T, H, 2D] view of the qkv
    projection. Returns [T, H*D] fp8."""
    t, h, d = x.shape
    assert gate.shape == x.shape and x.stride(2) == 1 and gate.stride(2) == 1
    out = torch.empty((t, h * d), dtype=torch.float8_e4m3fn, device=x.device)
    _gated_quant_kernel[(t, h)](
        x, gate, out, scale,
        x.stride(0), x.stride(1), gate.stride(0), gate.stride(1), out.stride(0), d, d,
        BLOCK_D=triton.next_power_of_2(d), num_warps=4,
    )
    return out
