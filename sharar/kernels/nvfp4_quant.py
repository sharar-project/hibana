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
FlashInfer's plain kernel streams at ~43 GB/s on GB10; this one is memory-bound. The E2M1 rounding is the hardware
conversion (cvt.rn.satfinite.e2m1x2), with the saturation and NaN rule of the comparison chain it replaced: equal
for every FP32 input (tests/gpu/test_nvfp4_quant.py checks all 2^32). Prefill-sized calls (from PREFILL_ROWS rows) use
wide column blocks, consecutive blocks along a row and streaming loads.
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
def e2m1_pairs(v, N: tl.constexpr):
    """The E2M1 codes of v [N] packed two per byte (element 2i in the low nibble): round to nearest even,
    saturate at 6, NaN to 6, the sign kept (also of zero and of what underflows to zero)."""
    a = tl.minimum(tl.abs(v), 6.0)
    a = tl.where(a == a, a, 6.0)
    signed = (a.to(tl.int32, bitcast=True) | (v.to(tl.int32, bitcast=True) & -2147483648)).to(tl.float32, bitcast=True)
    lo, hi = tl.split(tl.reshape(signed, (N // 2, 2)))
    return tl.inline_asm_elementwise(
        "{ .reg .b8 e; cvt.rn.satfinite.e2m1x2.f32 e, $2, $1; cvt.u32.u8 $0, e; }",
        constraints="=r,f,f", args=[lo, hi], dtype=tl.int32, is_pure=True, pack=1).to(tl.uint8)


@triton.jit
def _nvfp4_quant_kernel(
    x_ptr, gs_ptr, q_ptr, sf_ptr,
    M, K, K_PAD_GROUPS,
    stride_xm,
    BLOCK_K: tl.constexpr, SILU_MUL: tl.constexpr, COL_BLOCKS: tl.constexpr, STREAM: tl.constexpr,
):
    # One program per (row, column block), consecutive programs along a row.
    # Widen BEFORE multiplying by strides. B32x2K's dense gate/up tensor
    # crosses 2^31 elements at row 61681; widening the product is too late.
    # Keep packed-output and swizzled-scale row offsets wide as well.
    pid = tl.program_id(0)
    row = (pid // COL_BLOCKS).to(tl.int64)
    kb = pid % COL_BLOCKS
    NB: tl.constexpr = BLOCK_K // 16
    cols = kb * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = cols < K
    # STREAM: prefill reads each activation once; do not keep it in L2.
    policy: tl.constexpr = "evict_first" if STREAM else ""
    if SILU_MUL:
        # x holds [gate | up] with K values each; quantize silu(gate) * up.
        g = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=0.0, eviction_policy=policy).to(tl.float32)
        u = tl.load(x_ptr + row * stride_xm + K + cols, mask=mask, other=0.0, eviction_policy=policy).to(tl.float32)
        # The fused path preserves the BF16 activation boundary of the MLP.
        x = (tl.div_rn(g, 1.0 + libdevice.exp(-g)) * u).to(tl.bfloat16).to(tl.float32)
    else:
        x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=0.0, eviction_policy=policy).to(tl.float32)
    xb = tl.reshape(x, (NB, 16))
    amax = tl.max(tl.abs(xb), axis=1)  # [NB]
    gs = tl.load(gs_ptr)
    sf = (gs * (amax * rcp_approx_ftz(tl.full((), 6.0, tl.float32)))).to(tl.float8e4nv)
    sf_f = sf.to(tl.float32)
    # Algebraically gs/sf, but its floating-point rounding differs at FP4 ties.
    inv = tl.where(amax != 0.0, rcp_approx_ftz(sf_f * rcp_approx_ftz(gs)), 0.0)
    v = xb * inv[:, None]
    packed = e2m1_pairs(tl.reshape(v, (BLOCK_K,)), BLOCK_K)
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

# From this many rows a call is a prefill chunk: wide column blocks and streaming loads (measured on the 27B's
# [2048, 5120] MLP input and [2048, 2 x 17408] gate/up output; decode and verification keep the narrow blocks).
PREFILL_ROWS = 256


def _launch(x, global_scale, q, sf, m, k, k_pad_groups, *, silu_mul):
    prefill = m >= PREFILL_ROWS
    block_k = min(4096 if prefill else 1024, triton.next_power_of_2(k))
    col_blocks = triton.cdiv(k, block_k)
    _nvfp4_quant_kernel[(m * col_blocks,)](
        x, global_scale, q, sf, m, k, k_pad_groups, x.stride(0), BLOCK_K=block_k, SILU_MUL=silu_mul,
        COL_BLOCKS=col_blocks, STREAM=prefill, num_warps=8 if prefill else 4,
    )


def nvfp4_quantize(x: torch.Tensor, global_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x: [M, K] bf16 (K % 16 == 0). Returns (packed [M, K/2] uint8, sf [M_pad128, (K/16)_pad4] fp8 swizzled).
    The scale buffer is shared per shape: consume it before the next quantize of the same shape."""
    m, k = x.shape
    assert k % 16 == 0 and x.stride(1) == 1
    groups = k // 16
    k_pad_groups = (groups + 3) // 4 * 4
    q = torch.empty((m, k // 2), dtype=torch.uint8, device=x.device)
    sf = _scale_buffer(m, k, x.device)
    _launch(x, global_scale, q, sf, m, k, k_pad_groups, silu_mul=False)
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
    _launch(gate_up, global_scale, q, sf, m, k, k_pad_groups, silu_mul=True)
    return q, sf
