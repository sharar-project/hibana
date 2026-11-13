"""Fused per-head Gemma RMSNorm + partial NeoX RoPE (Triton).

One launch per tensor (q or k): reads heads straight out of the qkv projection
output (any token/head strides) and writes a contiguous [T, H, D] bf16 tensor.
Replaces reshape copies, the norm, the rope and the cats of the torch path.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _norm_rope_kernel(
    src_ptr, cos_sin_ptr, pos_ptr, w_ptr, out_ptr,
    stride_st, stride_sh, stride_ot, stride_oh,
    eps,
    D: tl.constexpr, R: tl.constexpr, HALF: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    base = src_ptr + t * stride_st + h * stride_sh
    cols = tl.arange(0, D)
    x = tl.load(base + cols).to(tl.float32)
    w = tl.load(w_ptr + cols).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / D + eps)
    y = (x * rstd * (1.0 + w)).to(tl.bfloat16).to(tl.float32)
    out = out_ptr + t * stride_ot + h * stride_oh
    # dims >= R pass through
    tl.store(out + cols, y.to(tl.bfloat16), mask=cols >= R)
    # rope on dims < R, NeoX pairing (i, i + R/2); cache and rotary products remain FP32 until the output store
    lane = tl.arange(0, HALF)
    pos = tl.load(pos_ptr + t)
    cos = tl.load(cos_sin_ptr + pos * R + lane).to(tl.float32)
    sin = tl.load(cos_sin_ptr + pos * R + HALF + lane).to(tl.float32)
    x1 = tl.load(base + lane).to(tl.float32)
    x2 = tl.load(base + HALF + lane).to(tl.float32)
    w1 = tl.load(w_ptr + lane).to(tl.float32)
    w2 = tl.load(w_ptr + HALF + lane).to(tl.float32)
    y1 = (x1 * rstd * (1.0 + w1)).to(tl.bfloat16).to(tl.float32)
    y2 = (x2 * rstd * (1.0 + w2)).to(tl.bfloat16).to(tl.float32)
    o1 = y1 * cos - y2 * sin
    o2 = y2 * cos + y1 * sin
    tl.store(out + lane, o1.to(tl.bfloat16))
    tl.store(out + HALF + lane, o2.to(tl.bfloat16))


def norm_rope(src: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor, weight: torch.Tensor, rotary_dim: int, eps: float) -> torch.Tensor:
    """src: [T, H, D] view with unit last stride. Returns contiguous [T, H, D] bf16."""
    t, h, d = src.shape
    assert src.stride(2) == 1
    out = torch.empty((t, h, d), dtype=torch.bfloat16, device=src.device)
    _norm_rope_kernel[(t, h)](
        src, cos_sin, positions, weight, out,
        src.stride(0), src.stride(1), out.stride(0), out.stride(1), eps,
        D=d, R=rotary_dim, HALF=rotary_dim // 2, num_warps=4,
    )
    return out


def qk_norm_rope(qkv: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor, num_heads: int, num_kv_heads: int, head_dim: int, rotary_dim: int, eps: float):
    t = qkv.shape[0]
    q_gate = qkv[:, : num_heads * 2 * head_dim].view(t, num_heads, 2 * head_dim)
    q_src = q_gate[..., :head_dim]
    k_src = qkv[:, num_heads * 2 * head_dim : num_heads * 2 * head_dim + num_kv_heads * head_dim].view(t, num_kv_heads, head_dim)
    q = norm_rope(q_src, positions, cos_sin, q_weight, rotary_dim, eps)
    k = norm_rope(k_src, positions, cos_sin, k_weight, rotary_dim, eps)
    return q, k
