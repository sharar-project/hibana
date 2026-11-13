# SPDX-License-Identifier: Apache-2.0
"""The QSA indexer's graph-safe key maintenance and query prep (`FlashNextIndexer.maintain` / `select`) in two
Triton launches instead of ~40 torch ops (~70 with the selection's query prep).

`prep`: each row's raw index key into its page (aux row 0) and, for a selecting forward, its four index queries
Gemma-normed and NeoX-rotated, plus each row's base offset in the flat page table. `compress`: each row's group
key (FP32 mean of its four members' raw keys, Gemma norm, rotation at the group's first position) into its write
page (aux row 1). `compress` reads raw keys `prep` wrote, so the two are separate launches.

The arithmetic is `reference.gemma_rms` / `rope_neox` exactly: FP32 inside, the sum of squares four elements a lane
then a descending shuffle tree (one warp per 128-wide head, torch's order for this reduction), BF16 roundings where
torch rounds, no FMA contraction (a product feeding a sum is scalar PTX with explicit rounding), and cos/sin read from
the rotary table (`PartialRotary.cos_sin`, the same FP32 values `rope_neox` computes per call). Bitwise checks:
tests/gpu/test_qsa.py.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _mul(a, b):
    return tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _add(a, b):
    return tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _sub(a, b):
    return tl.inline_asm_elementwise("sub.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _rstd(x, eps, D: tl.constexpr):
    """1 / sqrt(mean(x^2) + eps) of one head as torch computes it. The arithmetic is scalar PTX with explicit rounding:
    `enable_fp_fusion=False` alone leaves Triton's packed f32x2 ops, which the CUDA 13.0 ptxas (the one Triton takes
    once `flashinfer.triton` is imported) contracts into FMAs."""
    ssq = tl.reduce(_mul(x, x), 0, _add)
    var = _mul(ssq, tl.full(ssq.shape, 1.0 / D, tl.float32))  # torch's mean: the sum times 1/D (exact: D is a power of 2)
    return tl.rsqrt(_add(var, tl.full(var.shape, eps, tl.float32)))


@triton.jit
def _scale(x, rstd, w):
    """(x * rstd) * (1 + w), rounded to BF16, as FP32."""
    return (x * rstd * (1.0 + w)).to(tl.bfloat16).to(tl.float32)  # products only: nothing to contract


@triton.jit
def _norm_rope_store(x, x1, x2, w_ptr, cos_sin, pos, out, eps, D: tl.constexpr, R: tl.constexpr, HALF: tl.constexpr):
    """Normalize the head `x` (its rotary halves `x1`, `x2` loaded again: the same values) and rotate its first R
    dims at `pos`, storing BF16 at `out`."""
    d = tl.arange(0, D)
    lane = tl.arange(0, HALF)
    rstd = _rstd(x, eps, D)
    y = _scale(x, rstd, tl.load(w_ptr + d).to(tl.float32))
    tl.store(out + d, y.to(tl.bfloat16), mask=d >= R)
    y1 = _scale(x1, rstd, tl.load(w_ptr + lane).to(tl.float32))
    y2 = _scale(x2, rstd, tl.load(w_ptr + HALF + lane).to(tl.float32))
    cos = tl.load(cos_sin + pos * R + lane)
    sin = tl.load(cos_sin + pos * R + HALF + lane)
    tl.store(out + lane, _sub(_mul(y1, cos), _mul(y2, sin)).to(tl.bfloat16))
    tl.store(out + HALF + lane, _add(_mul(y2, cos), _mul(y1, sin)).to(tl.bfloat16))


@triton.jit
def _prep_kernel(QK, Aux, Pages, Pos, W, CosSin, Q, IndPtr, RowSeq, Base, stride_qk, stride_page, stride_qr, eps,
                 HEADS: tl.constexpr, D: tl.constexpr, R: tl.constexpr, HALF: tl.constexpr, QUERY: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, D)
    row = QK + t * stride_qk
    if h == 0:
        page = tl.load(Pages + t).to(tl.int64)
        tl.store(Aux + page * stride_page + d, tl.load(row + HEADS * D + d))
        if QUERY:
            tl.store(Base + t, tl.load(IndPtr + tl.load(RowSeq + t)))
    if QUERY:
        lane = tl.arange(0, HALF)
        x = tl.load(row + h * D + d).to(tl.float32)
        x1 = tl.load(row + h * D + lane).to(tl.float32)
        x2 = tl.load(row + h * D + HALF + lane).to(tl.float32)
        pos = tl.load(Pos + t).to(tl.int64)
        _norm_rope_store(x, x1, x2, W, CosSin, pos, Q + t * stride_qr + h * D, eps, D, R, HALF)


@triton.jit
def _mean4(Aux, m0, m1, m2, m3, cols):
    """FP32 mean of the members' raw keys at `cols`, as torch's `.float().mean(dim=1)` adds them (from 0, in order,
    times 1/4), rounded to BF16."""
    acc = tl.zeros(cols.shape, tl.float32) + tl.load(Aux + m0 + cols).to(tl.float32)
    acc += tl.load(Aux + m1 + cols).to(tl.float32)
    acc += tl.load(Aux + m2 + cols).to(tl.float32)
    acc += tl.load(Aux + m3 + cols).to(tl.float32)
    return (acc * 0.25).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _compress_kernel(Aux, Members, Writes, Pos, W, CosSin, stride_page, eps, RATIO: tl.constexpr, D: tl.constexpr,
                     R: tl.constexpr, HALF: tl.constexpr):
    t = tl.program_id(0)
    d = tl.arange(0, D)
    lane = tl.arange(0, HALF)
    m0 = tl.load(Members + t * RATIO + 0).to(tl.int64) * stride_page
    m1 = tl.load(Members + t * RATIO + 1).to(tl.int64) * stride_page
    m2 = tl.load(Members + t * RATIO + 2).to(tl.int64) * stride_page
    m3 = tl.load(Members + t * RATIO + 3).to(tl.int64) * stride_page
    x = _mean4(Aux, m0, m1, m2, m3, d)
    x1 = _mean4(Aux, m0, m1, m2, m3, lane)
    x2 = _mean4(Aux, m0, m1, m2, m3, HALF + lane)
    first = tl.maximum(tl.load(Pos + t).to(tl.int64) - (RATIO - 1), 0)
    out = Aux + tl.load(Writes + t).to(tl.int64) * stride_page + D
    _norm_rope_store(x, x1, x2, W, CosSin, first, out, eps, D, R, HALF)


def supported(indexer, cos_sin) -> bool:
    return (cos_sin is not None and cos_sin.is_cuda and cos_sin.dtype == torch.float32 and cos_sin.is_contiguous()
            and cos_sin.shape[-1] == indexer.rotary_dim and indexer.ratio == 4 and indexer.dim == 128
            and indexer.rotary_dim % 2 == 0 and indexer.rotary_dim <= indexer.dim
            and indexer.q_norm.is_contiguous() and indexer.k_norm.is_contiguous())


def prep(qk: torch.Tensor, aux: torch.Tensor, pages: torch.Tensor, positions: torch.Tensor, weight: torch.Tensor,
         cos_sin: torch.Tensor, heads: int, eps: float, query: bool, indptr=None, row_seq=None):
    """Raw keys `qk[:, heads * D:]` into `aux[pages, 0]`; with `query`, also returns (q [T, heads, D] normed and
    rotated, base [T] = indptr[row_seq])."""
    t, dim = qk.shape[0], aux.shape[-1]
    assert qk.stride(1) == 1 and aux.is_contiguous() and aux.shape[1] == 2
    r = cos_sin.shape[-1]
    if query:
        q = torch.empty((t, heads, dim), dtype=qk.dtype, device=qk.device)
        base = torch.empty((t,), dtype=indptr.dtype, device=qk.device)
    else:
        q = base = qk
        indptr = row_seq = pages
    if t:
        _prep_kernel[(t, heads if query else 1)](qk, aux, pages, positions, weight, cos_sin, q, indptr, row_seq, base,
                                                  qk.stride(0), aux.stride(0), q.stride(0) if query else 0, eps,
                                                  HEADS=heads, D=dim, R=r, HALF=r // 2, QUERY=query, num_warps=1,
                                                  enable_fp_fusion=False)
    return (q, base) if query else None


def compress(aux: torch.Tensor, members: torch.Tensor, writes: torch.Tensor, positions: torch.Tensor, weight: torch.Tensor,
             cos_sin: torch.Tensor, eps: float) -> None:
    """Each row's group key from its members [T, 4] into `aux[writes, 1]`, rotated at max(position - 3, 0)."""
    t, ratio = members.shape
    assert members.is_contiguous() and aux.is_contiguous()
    r = cos_sin.shape[-1]
    if t:
        _compress_kernel[(t,)](aux, members, writes, positions, weight, cos_sin, aux.stride(0), eps, RATIO=ratio,
                               D=aux.shape[-1], R=r, HALF=r // 2, num_warps=1, enable_fp_fusion=False)
