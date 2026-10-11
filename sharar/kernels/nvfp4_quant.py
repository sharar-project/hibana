"""NVFP4 activation quantization in FlashInfer's swizzled block-scale layout.

E2M1 ties round to even and preserve signed zero. Full quantizer agreement
also depends on the reciprocal evaluation order and the BF16 SiLU/product
boundary. Preserve both to match the CUDA quantizer used by this checkpoint.

Per 16-value block: sf = e4m3(global_scale * amax / 6); the block is scaled by
reciprocal(sf * reciprocal(global_scale)) and rounded to e2m1 (round-to-nearest-even, saturating at 6).
Values are packed two per byte (element 2i in the low nibble). Block scales are
written in the 128x4 swizzled layout the CUTLASS SM120 GEMM reads:
  rows in tiles of 128 (4 groups of 32), columns in groups of 4:
  offset = tile_r*(128*Kp) + (r%32)*16 + (r//32 % 4)*4 + tile_c*512 + (c%4)
FlashInfer's plain kernel streams at ~43 GB/s on GB10; this one is memory-bound.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def rcp_approx_ftz(x):
    return tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", constraints="=f,f", args=[x], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _nvfp4_quant_kernel(
    x_ptr, gs_ptr, q_ptr, sf_ptr,
    M, K, K_PAD_GROUPS,
    stride_xm,
    BLOCK_K: tl.constexpr, SILU_MUL: tl.constexpr,
):
    # Widen BEFORE multiplying by strides. B32x2K's dense gate/up tensor
    # crosses 2^31 elements at row 61681; widening the product is too late.
    # Keep packed-output and swizzled-scale row offsets wide as well.
    row = tl.program_id(0).to(tl.int64)
    kb = tl.program_id(1)
    NB: tl.constexpr = BLOCK_K // 16
    cols = kb * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = cols < K
    if SILU_MUL:
        # x holds [gate | up] with K values each; quantize silu(gate) * up.
        g = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(x_ptr + row * stride_xm + K + cols, mask=mask, other=0.0).to(tl.float32)
        # The fused path preserves the BF16 activation boundary of the MLP.
        x = (tl.div_rn(g, 1.0 + libdevice.exp(-g)) * u).to(tl.bfloat16).to(tl.float32)
    else:
        x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    xb = tl.reshape(x, (NB, 16))
    amax = tl.max(tl.abs(xb), axis=1)  # [NB]
    gs = tl.load(gs_ptr)
    sf = (gs * (amax * rcp_approx_ftz(tl.full((), 6.0, tl.float32)))).to(tl.float8e4nv)
    sf_f = sf.to(tl.float32)
    # Algebraically gs/sf, but its floating-point rounding differs at FP4 ties.
    inv = tl.where(amax != 0.0, rcp_approx_ftz(sf_f * rcp_approx_ftz(gs)), 0.0)
    v = xb * inv[:, None]
    # e2m1 round-to-nearest-even over magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6}, saturate at 6
    a = tl.minimum(tl.abs(v), 6.0)
    # candidates by range; ties go to even mantissa
    q = tl.where(a <= 0.25, 0.0,
        tl.where(a < 0.75, 0.5,
        tl.where(a <= 1.25, 1.0,  # 1.25 tie -> 1.0 (even)
        tl.where(a < 1.75, 1.5,
        tl.where(a <= 2.5, 2.0,   # 2.5 tie -> 2.0 (even)
        tl.where(a < 3.5, 3.0,
        tl.where(a <= 5.0, 4.0, 6.0)))))))  # 5.0 tie -> 4.0 (even)
    # The inclusive first boundary rounds 0.25 to even zero. At 0.75 the
    # strict second comparison falls through to even 1.0.
    code = tl.where(q == 0.0, 0, tl.where(q == 0.5, 1, tl.where(q == 1.0, 2, tl.where(q == 1.5, 3,
           tl.where(q == 2.0, 4, tl.where(q == 3.0, 5, tl.where(q == 4.0, 6, 7))))))).to(tl.int32)
    # E2M1 preserves the sign of zero, including inputs that underflow to zero.
    code = tl.where(v.to(tl.int32, bitcast=True) < 0, code | 8, code)
    code = tl.reshape(code, (BLOCK_K,))
    lo = tl.reshape(code, (BLOCK_K // 2, 2))
    packed = (tl.sum(lo * tl.reshape(tl.where(tl.arange(0, BLOCK_K) % 2 == 0, 1, 16), (BLOCK_K // 2, 2)), axis=1)).to(tl.uint8)
    pcols = kb * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)
    tl.store(q_ptr + row * (K // 2) + pcols, packed, mask=pcols < K // 2)
    # swizzled scale factors
    c = kb * NB + tl.arange(0, NB)
    tile_r = row // 128
    r_in = row % 128
    off = tile_r * (128 * K_PAD_GROUPS) + (r_in % 32) * 16 + ((r_in // 32) % 4) * 4 + (c // 4) * 512 + (c % 4)
    tl.store(sf_ptr + off, sf, mask=c < (K // 16))


def _scale_buffer(m: int, k: int, device: torch.device) -> torch.Tensor:
    """Swizzled block-scale buffer [M_pad128, (K/16)_pad4]. The kernel writes
    every scale of the M real rows; the padding rows and columns only feed
    the GEMM's discarded tiles, so they are zeroed once and the buffer is
    reused per shape instead of being zero-filled on every call."""
    m_pad = (m + 127) // 128 * 128
    k_pad_groups = (k // 16 + 3) // 4 * 4
    key = (m_pad, k_pad_groups, device)
    buf = _SCALE_BUFFERS.get(key)
    if buf is None:
        # Never evicted: CUDA graphs capture these addresses, and the key space
        # is bounded (M rounded to 128 up to the prefill budget, two K values).
        buf = torch.zeros((m_pad, k_pad_groups), dtype=torch.float8_e4m3fn, device=device)
        _SCALE_BUFFERS[key] = buf
    return buf


_SCALE_BUFFERS: dict = {}


def nvfp4_quantize(x: torch.Tensor, global_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x: [M, K] bf16 (K % 16 == 0). Returns (packed [M, K/2] uint8, sf [M_pad128, (K/16)_pad4] fp8 swizzled).
    The scale buffer is shared per shape: consume it before the next quantize of the same shape."""
    m, k = x.shape
    assert k % 16 == 0 and x.stride(1) == 1
    groups = k // 16
    k_pad_groups = (groups + 3) // 4 * 4
    q = torch.empty((m, k // 2), dtype=torch.uint8, device=x.device)
    sf = _scale_buffer(m, k, x.device)
    block_k = min(1024, triton.next_power_of_2(k))
    _nvfp4_quant_kernel[(m, triton.cdiv(k, block_k))](
        x, global_scale, q, sf, m, k, k_pad_groups, x.stride(0), BLOCK_K=block_k, SILU_MUL=False, num_warps=4,
    )
    return q, sf


def silu_mul_nvfp4_quantize(gate_up: torch.Tensor, global_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """gate_up: [M, 2*I] bf16 -> quantized silu(gate) * up as ([M, I/2] uint8, swizzled sf)."""
    m, two_i = gate_up.shape
    k = two_i // 2
    assert k % 16 == 0 and gate_up.stride(1) == 1
    groups = k // 16
    k_pad_groups = (groups + 3) // 4 * 4
    q = torch.empty((m, k // 2), dtype=torch.uint8, device=gate_up.device)
    sf = _scale_buffer(m, k, gate_up.device)
    block_k = min(1024, triton.next_power_of_2(k))
    _nvfp4_quant_kernel[(m, triton.cdiv(k, block_k))](
        gate_up, global_scale, q, sf, m, k, k_pad_groups, gate_up.stride(0), BLOCK_K=block_k, SILU_MUL=True, num_warps=4,
    )
    return q, sf
