# SPDX-License-Identifier: Apache-2.0
"""Original Flash-Next indexer selection with fixed shapes (CUDA-graph safe).

Every row scores the compressed keys of the groups it sees in one Triton kernel:
the key of group g lives in page aux row 1 of the page holding token
g * ratio + ratio - 1, found through a flat page table at the row's base offset.
The score is relu(q . k) summed over the index heads, times `scale`; groups the
row cannot see score -inf. The top `budget // ratio` groups of a fixed-width
buffer, their tokens and the row's pending tail form the row's selection, valid
entries first, as `FlashNextIndexer._select` builds it.
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


def select_pages(q: torch.Tensor, aux: torch.Tensor, table: torch.Tensor, base: torch.Tensor, positions: torch.Tensor,
                 ratio: int, budget: int, gcap: int, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(pages [R, budget + ratio - 1] int32, valid first then -1; counts [R] int32). No host syncs:
    every shape depends on R and the static `gcap` (groups the widest row may see)."""
    rows, device = q.shape[0], q.device
    width = budget + ratio - 1
    top = min(budget // ratio, gcap)
    positions = positions.long()
    sel = torch.full((rows, width), -1, dtype=torch.int64, device=device)
    if top:
        values, idx = score_groups(q, aux, table, base, positions, ratio, gcap, scale).topk(top, dim=-1)
        chosen = torch.where(torch.isfinite(values), idx, torch.full_like(idx, -1))
        members = chosen.unsqueeze(-1) * ratio + torch.arange(ratio, device=device)
        sel[:, : top * ratio] = torch.where(chosen.unsqueeze(-1) >= 0, members, torch.full_like(members, -1)).reshape(rows, top * ratio)
    visible = (positions + 1) // ratio
    tail = (visible * ratio).unsqueeze(1) + torch.arange(ratio - 1, device=device)  # the pending group, itself included
    sel[:, width - (ratio - 1):] = torch.where(tail <= positions.unsqueeze(1), tail, torch.full_like(tail, -1))
    order = torch.argsort((sel < 0).to(torch.int8), dim=1, stable=True)  # valid entries first, in order
    sel = sel.gather(1, order)
    pages = torch.where(sel >= 0, table[(base.long().unsqueeze(1) + sel.clamp_min(0))].long(), torch.full_like(sel, -1)).to(torch.int32)
    return pages, (pages >= 0).sum(1, dtype=torch.int32)
