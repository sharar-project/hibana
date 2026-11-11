# SPDX-License-Identifier: Apache-2.0
"""Hyper-connections for prefill-sized calls: the [T, hc*H] streams read and written as few times as possible.

A hyper-connection site (`hibana/model/flashnext/hyper.py`) mixes the four streams into one block input and adds
the block's output back. The decode kernels (`flashnext_fused`) write the normalized streams in BF16, quantize them
per tensor for the down projection, write the up projection's gate and read both back for the mean, and run the
inject projection as its own BF16 GEMM: about 190 KB of DRAM traffic per row and site, more than the row's GEMMs
take at prefill row counts. Here a site is three launches and about 90 KB per row:

  norm   `combine_norm`: one program per row adds the block output to each stream (hyper'), takes the next site's
         grouped Gemma RMS norm in registers, writes it as E4M3 with one scale per row (the down projection's
         input), the per-stream 1/rms (so the mean can recompute the norm bit for bit) and the next site's inject
         logits in FP32 (a dot over the normalized row; the decode path rounds them to BF16).
  down   `torch._scaled_mm` with the row scales: FP8 tensor cores, BF16 out.
  up     `up_gate_mean`: the up projection on BF16 activations against the FP8 weights (exact in BF16), with the
         gate, the recomputed norm and the mean over the streams in its epilogue; the gate never leaves registers. It
         also writes each tile's maximum, so the block's projections share one FP8 copy of its input (`fp8_input`).

Every step keeps or raises the decode path's precision (per-row instead of per-tensor activation scales, unquantized
up-projection input, FP32 inject logits); the results are not bitwise the decode path's.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

ROWS = 1024  # from this many rows a site takes this path (prefill chunks; decode and verify stay below)


@triton.jit
def _sigmoid(x):
    return 1.0 / (1.0 + tl.exp(-x))


@triton.jit
def _stream(I, Y, W, Rstd, t, s: tl.constexpr, x, b, stride_y, H: tl.constexpr, eps, HAS_B: tl.constexpr, BLOCK: tl.constexpr):
    """Stream s of row t, x its loaded values (FP32): hyper' (stored when HAS_B), its norm (returned as the BF16 value
    in FP32), 1/rms stored."""
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    off = s * H + cols
    if HAS_B:
        g = 2.0 * _sigmoid(tl.load(I + t * 4 + s))
        hb = (x + b * g).to(tl.bfloat16)
        tl.store(Y + t * stride_y + off, hb, mask=mask)
        x = hb.to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, 0) / H + eps)
    tl.store(Rstd + t * 4 + s, rstd)
    w = tl.load(W + off, mask=mask, other=0.0).to(tl.float32)
    return ((x * rstd) * (1.0 + w)).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _inject_dot(WI, j: tl.constexpr, n0, n1, n2, n3, H: tl.constexpr, BLOCK: tl.constexpr):
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    base = WI + j * 4 * H
    acc = n0 * tl.load(base + cols, mask=mask, other=0.0).to(tl.float32)
    acc += n1 * tl.load(base + H + cols, mask=mask, other=0.0).to(tl.float32)
    acc += n2 * tl.load(base + 2 * H + cols, mask=mask, other=0.0).to(tl.float32)
    acc += n3 * tl.load(base + 3 * H + cols, mask=mask, other=0.0).to(tl.float32)
    return tl.sum(acc, 0)


@triton.jit
def _quant_store(NQ, n, t, s: tl.constexpr, inv, stride_q, H: tl.constexpr, BLOCK: tl.constexpr):
    cols = tl.arange(0, BLOCK)
    q = tl.minimum(tl.maximum(n * inv, -448.0), 448.0)
    tl.store(NQ + t * stride_q + s * H + cols, q.to(tl.float8e4nv), mask=cols < H)


@triton.jit
def _combine_norm_kernel(R, B, I, Y, W, WI, NQ, XS, Rstd, Inj, stride_r, stride_b, stride_y, stride_q, H: tl.constexpr, eps,
                         HAS_B: tl.constexpr, HAS_INJ: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    # the row's four streams and the block output are loaded before any store (a store would order the loads after it)
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    x0 = tl.load(R + t * stride_r + cols, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(R + t * stride_r + H + cols, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(R + t * stride_r + 2 * H + cols, mask=mask, other=0.0).to(tl.float32)
    x3 = tl.load(R + t * stride_r + 3 * H + cols, mask=mask, other=0.0).to(tl.float32)
    b = x0
    if HAS_B:
        b = tl.load(B + t * stride_b + cols, mask=mask, other=0.0).to(tl.float32)
    n0 = _stream(I, Y, W, Rstd, t, 0, x0, b, stride_y, H, eps, HAS_B, BLOCK)
    n1 = _stream(I, Y, W, Rstd, t, 1, x1, b, stride_y, H, eps, HAS_B, BLOCK)
    n2 = _stream(I, Y, W, Rstd, t, 2, x2, b, stride_y, H, eps, HAS_B, BLOCK)
    n3 = _stream(I, Y, W, Rstd, t, 3, x3, b, stride_y, H, eps, HAS_B, BLOCK)
    if HAS_INJ:
        tl.store(Inj + t * 4 + 0, _inject_dot(WI, 0, n0, n1, n2, n3, H, BLOCK))
        tl.store(Inj + t * 4 + 1, _inject_dot(WI, 1, n0, n1, n2, n3, H, BLOCK))
        tl.store(Inj + t * 4 + 2, _inject_dot(WI, 2, n0, n1, n2, n3, H, BLOCK))
        tl.store(Inj + t * 4 + 3, _inject_dot(WI, 3, n0, n1, n2, n3, H, BLOCK))
    amax = tl.maximum(tl.maximum(tl.max(tl.abs(n0), 0), tl.max(tl.abs(n1), 0)), tl.maximum(tl.max(tl.abs(n2), 0), tl.max(tl.abs(n3), 0)))
    scale = tl.maximum(amax * (1.0 / 448.0), 1.0e-12)
    tl.store(XS + t, scale)
    inv = 1.0 / scale
    _quant_store(NQ, n0, t, 0, inv, stride_q, H, BLOCK)
    _quant_store(NQ, n1, t, 1, inv, stride_q, H, BLOCK)
    _quant_store(NQ, n2, t, 2, inv, stride_q, H, BLOCK)
    _quant_store(NQ, n3, t, 3, inv, stride_q, H, BLOCK)


def combine_norm(hyper: torch.Tensor, block: torch.Tensor | None, inj: torch.Tensor | None, norm: torch.Tensor, inject_w: torch.Tensor | None,
                 hidden: int, eps: float):
    """hyper [T, 4H] bf16; with `block` [T, H] and `inj` [T, 4] fp32: hyper' = hyper + block * 2 sigmoid(inj) per stream.
    Then the grouped Gemma norm of hyper' under `norm` [4H]. -> (hyper' (hyper itself without a block), normed E4M3
    [T, 4H], row scales [T, 1] fp32, 1/rms [T, 4] fp32, inject logits [T, 4] fp32 under `inject_w` [4, 4H] or None)."""
    t, n = hyper.shape
    assert n == 4 * hidden and hyper.stride(1) == 1
    dev = hyper.device
    y = torch.empty_like(hyper) if block is not None else hyper
    nq = torch.empty((t, n), dtype=torch.float8_e4m3fn, device=dev)
    xs = torch.empty((t, 1), dtype=torch.float32, device=dev)
    rstd = torch.empty((t, 4), dtype=torch.float32, device=dev)
    out_inj = torch.empty((t, 4), dtype=torch.float32, device=dev) if inject_w is not None else None
    _combine_norm_kernel[(t,)](hyper, block if block is not None else hyper, inj if inj is not None else xs, y, norm,
                               inject_w if inject_w is not None else norm, nq, xs, rstd, out_inj if out_inj is not None else xs,
                               hyper.stride(0), block.stride(0) if block is not None else 0, y.stride(0), nq.stride(0), hidden, eps,
                               HAS_B=block is not None, HAS_INJ=inject_w is not None, BLOCK=triton.next_power_of_2(hidden), num_warps=8)
    return y, nq, xs, rstd, out_inj


@triton.jit
def _up_gate_mean_kernel(A, Wq, WS, Hp, Rstd, Nw, Out, Amax, T, stride_a, stride_w, stride_h, stride_o, H: tl.constexpr, K: tl.constexpr,
                         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    mm = rm < T
    mn = rn < H
    ws = tl.load(WS)
    mix = tl.zeros((BM, BN), dtype=tl.float32)
    for s in tl.static_range(4):
        # BK columns of K at a time (a pipelined loop): the products accumulate in the same order as one dot over K
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in tl.range(0, K, BK):
            kk = k0 + tl.arange(0, BK)
            a = tl.load(A + rm[:, None] * stride_a + kk[None, :], mask=mm[:, None] & (kk[None, :] < K), other=0.0)
            w = tl.load(Wq + (s * H + rn)[None, :] * stride_w + kk[:, None], mask=mn[None, :] & (kk[:, None] < K), other=0.0)
            acc = tl.dot(a, w.to(tl.bfloat16), acc)
        u = (acc * ws).to(tl.bfloat16).to(tl.float32)
        x = tl.load(Hp + rm[:, None] * stride_h + (s * H + rn)[None, :], mask=mm[:, None] & mn[None, :], other=0.0).to(tl.float32)
        r = tl.load(Rstd + rm * 4 + s, mask=mm, other=0.0)
        nw = tl.load(Nw + s * H + rn, mask=mn, other=0.0).to(tl.float32)
        n = ((x * r[:, None]) * (1.0 + nw[None, :])).to(tl.bfloat16).to(tl.float32)
        mix += _sigmoid(u) * n
    y = (mix / 4).to(tl.bfloat16)
    tl.store(Out + rm[:, None] * stride_o + rn[None, :], y, mask=mm[:, None] & mn[None, :])
    # this tile's max |y| as stored: the per-tensor FP8 quantization of y then needs no pass of its own (`fp8_input`)
    peak = tl.max(tl.max(tl.where(mm[:, None] & mn[None, :], tl.abs(y.to(tl.float32)), 0.0), 1), 0)
    tl.store(Amax + pm * tl.num_programs(1) + pn, peak)


def up_gate_mean(a: torch.Tensor, up_w: torch.Tensor, up_scale: torch.Tensor, hyper: torch.Tensor, rstd: torch.Tensor, norm: torch.Tensor,
                 hidden: int) -> tuple[torch.Tensor, torch.Tensor]:
    """mixed [T, H] = mean_s sigmoid(bf16(a @ W_s^T)) * norm_s(hyper'), a [T, K] bf16 the SiLU of the down projection,
    up_w [4H, K] E4M3 with its scalar scale; norm_s recomputed from hyper' and `rstd` as `combine_norm` computed it.
    -> (mixed, the max |mixed| of each output tile, fp32)."""
    t, k = a.shape
    assert up_w.shape == (4 * hidden, k) and a.stride(1) == 1 and up_w.stride(1) == 1
    # 32 x 64 tiles, K 64 at a time in three stages (8,192 rows: 1.68 -> 1.30 ms against hoisting all of K;
    # larger tiles spill or lose occupancy)
    bm, bn = 32, 64
    grid = (triton.cdiv(t, bm), triton.cdiv(hidden, bn))
    out = torch.empty((t, hidden), dtype=torch.bfloat16, device=a.device)
    parts = torch.empty((grid[0] * grid[1],), dtype=torch.float32, device=a.device)
    _up_gate_mean_kernel[grid](a, up_w, up_scale, hyper, rstd, norm, out, parts, t, a.stride(0), up_w.stride(0), hyper.stride(0), out.stride(0),
                               hidden, k, BM=bm, BN=bn, BK=64, num_warps=4, num_stages=3)
    return out, parts


def mix(nq: torch.Tensor, xs: torch.Tensor, rstd: torch.Tensor, hyper: torch.Tensor, down, up, norm: torch.Tensor, hidden: int) -> torch.Tensor:
    """The block input from `combine_norm`'s outputs: down (FP8, row scales), SiLU, then `up_gate_mean`.
    `down` / `up` are the site's FP8 `TrunkLinear`s (the 1/hc of silu(down(x)/hc) folded into `down`)."""
    ws = down.scale.reshape(1, 1).expand(1, down.out_features).contiguous()
    d = torch._scaled_mm(nq, down.weight.T, scale_a=xs, scale_b=ws, out_dtype=torch.bfloat16)
    mixed, parts = up_gate_mean(torch.nn.functional.silu(d), up.weight, up.scale, hyper, rstd, norm, hidden)
    mixed._sharar_amax_parts = parts  # for `fp8_input`
    return mixed


def fp8_input(x: torch.Tensor):
    """The per-tensor dynamic FP8 of a block input `mix` produced, or None for any other tensor: quantized once from the
    tile maxima `up_gate_mean` wrote and kept on the tensor, so every projection of the block (GDN in_qkv and in_z, the
    attention's qkv, the shared expert) reads the same E4M3 copy. Bitwise the quantization each would do itself
    (`draft_fp8.quant_dynamic`: the same maximum, scale and rounding). -> (fp8 like x, scale [1, 1]) or None."""
    cached = getattr(x, "_sharar_fp8", None)
    if cached is not None:
        return cached
    parts = getattr(x, "_sharar_amax_parts", None)
    if parts is None:
        return None
    from hibana.kernels.flashnext_fused import quant_fp8_parts
    x._sharar_fp8 = quant_fp8_parts(x, parts)
    return x._sharar_fp8
