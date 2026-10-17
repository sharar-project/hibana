# SPDX-License-Identifier: Apache-2.0
"""Graph-safe indexer selection (the `original` QSA backend) with fixed shapes (CUDA-graph safe).

Every row scores the compressed keys of the groups it sees in one Triton kernel:
the key of group g lives in page aux row 1 of the page holding token
g * ratio + ratio - 1, found through a flat page table at the row's base offset.
The score is relu(q . k) summed over the index heads, times `scale`; groups the
row cannot see score -inf. The top `budget // ratio` groups of a fixed-width
buffer, their tokens and the row's pending tail form the row's selection, valid
entries first, as `FlashNextIndexer.select`
(sharar/model/flashnext/indexer.py) builds it.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _score_kernel(Q, Aux, Table, Base, Pos, Out, stride_qr, stride_qh, stride_page, key_offset, stride_or, gcap, scale,
                  RATIO: tl.constexpr, HEADS: tl.constexpr, DIM: tl.constexpr, BLOCK_G: tl.constexpr):
    row = tl.program_id(0)
    g = tl.program_id(1) * BLOCK_G + tl.arange(0, BLOCK_G)
    visible = (tl.load(Pos + row).to(tl.int32) + 1) // RATIO
    live = g < visible
    out = tl.full((BLOCK_G,), float("-inf"), tl.float32)
    if tl.program_id(1) * BLOCK_G < visible:
        base = tl.load(Base + row).to(tl.int64)
        page = tl.load(Table + base + (g * RATIO + RATIO - 1).to(tl.int64), mask=live, other=0).to(tl.int64)
        d = tl.arange(0, DIM)
        k = tl.load(Aux + page[:, None] * stride_page + key_offset + d[None, :], mask=live[:, None], other=0.0).to(tl.float32)
        acc = tl.zeros((BLOCK_G,), tl.float32)
        for h in tl.static_range(HEADS):
            q = tl.load(Q + row * stride_qr + h * stride_qh + d).to(tl.float32)
            acc += tl.maximum(tl.sum(k * q[None, :], axis=1), 0.0)
        out = tl.where(live, acc * scale, out)
    tl.store(Out + row * stride_or + g, out, mask=g < gcap)


def score_groups(q: torch.Tensor, aux: torch.Tensor, table: torch.Tensor, base: torch.Tensor, positions: torch.Tensor,
                 ratio: int, gcap: int, scale: float) -> torch.Tensor:
    """q [R, H, D] (normed, rotated); aux [pages, 2, D] (row 1: compressed keys); table flat int32
    page ids; base [R] offsets of each row's sequence in `table`; positions [R]. Returns [R, gcap] fp32."""
    rows, heads, dim = q.shape
    out = torch.empty((rows, gcap), dtype=torch.float32, device=q.device)
    if rows == 0 or gcap == 0:
        return out
    block = 64
    _score_kernel[(rows, triton.cdiv(gcap, block))](
        q, aux, table, base, positions, out, q.stride(0), q.stride(1), aux.stride(0), aux.stride(1), out.stride(0), gcap, scale,
        RATIO=ratio, HEADS=heads, DIM=dim, BLOCK_G=block, num_warps=4)
    return out


@triton.jit
def _assemble_kernel(Values, Idx, Pos, Base, Table, Pages, Counts, stride_v, stride_p, top, width,
                     RATIO: tl.constexpr, TOPP: tl.constexpr, WIDTHP: tl.constexpr):
    """One row's selection from its top-k: the chosen groups' tokens (finite scores, in top-k order, each group's
    tokens ascending), then its pending tail (the tokens of the incomplete group through itself), as pages through the
    table, valid entries first and -1 after them; and their count."""
    row = tl.program_id(0)
    pos = tl.load(Pos + row).to(tl.int64)
    base = tl.load(Base + row).to(tl.int64)
    j = tl.arange(0, TOPP)
    live = j < top
    v = tl.load(Values + row * stride_v + j, mask=live, other=float("-inf"))
    g = tl.load(Idx + row * stride_v + j, mask=live, other=0)
    chosen = live & (v > float("-inf")) & (v < float("inf"))  # torch.isfinite (scores are never NaN)
    m = tl.arange(0, RATIO)
    tok = g[:, None] * RATIO + m[None, :]  # [TOPP, RATIO]
    ok = chosen[:, None] & (m[None, :] >= 0)
    flat_ok = tl.reshape(ok, (TOPP * RATIO,))
    flat_tok = tl.reshape(tok, (TOPP * RATIO,))
    slot = tl.cumsum(flat_ok.to(tl.int32), 0) - 1
    n_members = tl.sum(flat_ok.to(tl.int32), 0)
    page = tl.load(Table + base + flat_tok, mask=flat_ok, other=0)
    tl.store(Pages + row * stride_p + slot, page, mask=flat_ok)
    i = tl.arange(0, RATIO)
    tail = ((pos + 1) // RATIO) * RATIO + i
    tail_ok = (i < RATIO - 1) & (tail <= pos)
    tail_slot = n_members + tl.cumsum(tail_ok.to(tl.int32), 0) - 1
    tl.store(Pages + row * stride_p + tail_slot, tl.load(Table + base + tail, mask=tail_ok, other=0), mask=tail_ok)
    count = n_members + tl.sum(tail_ok.to(tl.int32), 0)
    k = tl.arange(0, WIDTHP)
    tl.store(Pages + row * stride_p + k, tl.full((WIDTHP,), -1, tl.int32), mask=(k >= count) & (k < width))
    tl.store(Counts + row, count)


def select_pages(q: torch.Tensor, aux: torch.Tensor, table: torch.Tensor, base: torch.Tensor, positions: torch.Tensor,
                 ratio: int, budget: int, gcap: int, scale: float, chunk: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """(pages [R, budget + ratio - 1] int32, valid first then -1; counts [R] int32). No host syncs:
    every shape depends on R and the static `gcap` (groups the widest row may see). The top-k in torch, the rest in one
kernel (`_assemble_kernel`). `chunk`: rows scored and ranked per
    top-k call (bounding the score buffer; the result is that of calling this per `chunk` rows, since only the top-k's
    tie order depends on the call's shape and everything after it is exact selection arithmetic)."""
    rows, device = q.shape[0], q.device
    width = budget + ratio - 1
    top = min(budget // ratio, gcap)
    positions = positions.long()
    if top:
        step = rows if chunk is None else chunk
        if step >= rows:
            values, idx = score_groups(q, aux, table, base, positions, ratio, gcap, scale).topk(top, dim=-1)
        else:
            values = torch.empty((rows, top), dtype=torch.float32, device=device)
            idx = torch.empty((rows, top), dtype=torch.int64, device=device)
            for a in range(0, rows, step):
                b = min(rows, a + step)
                torch.topk(score_groups(q[a:b], aux, table, base[a:b], positions[a:b], ratio, gcap, scale), top, dim=-1,
                           out=(values[a:b], idx[a:b]))
    else:
        values = torch.empty((rows, 0), dtype=torch.float32, device=device)
        idx = torch.empty((rows, 0), dtype=torch.int64, device=device)
    pages = torch.empty((rows, width), dtype=torch.int32, device=device)
    counts = torch.empty((rows,), dtype=torch.int32, device=device)
    if rows:
        tp = max(1, triton.next_power_of_2(top))
        _assemble_kernel[(rows,)](values, idx, positions, base, table, pages, counts, values.stride(0), pages.stride(0), top, width,
                                  RATIO=ratio, TOPP=tp, WIDTHP=triton.next_power_of_2(width), num_warps=4)
    return pages, counts
