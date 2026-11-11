"""Immutable GDN metadata shared by every layer of one prefill forward.

Sequence lengths stay on the CPU; planning never reads a CUDA tensor. This
plan is independent of ``ForwardContext.batched`` and therefore cannot change
the cold-ragged versus cached-paged attention path. A context owns one plan:
create a new context when its sequence, slot or checkpoint metadata changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate
from math import gcd
from typing import TYPE_CHECKING

import torch

from hibana.engine.upload import upload

if TYPE_CHECKING:
    from hibana.engine.context import ForwardContext


@dataclass(frozen=True)
class GdnPrefillPlan:
    query_start_loc: torch.Tensor
    cache_indices: torch.Tensor
    has_initial_state: torch.Tensor
    seq_lens: tuple[int, ...]
    chunk_indices: torch.Tensor
    chunk_offsets: torch.Tensor
    # (destination slot, end of pre-conv inputs, entering-chunk state index)
    checkpoints: tuple[tuple[int, int, int], ...]
    cute_cu_seqlens: torch.Tensor
    checkpoint_interval: int
    checkpoint_cu_starts: torch.Tensor
    checkpoint_indices: torch.Tensor
    checkpoint_destinations: torch.Tensor
    checkpoint_count: int


def prepare_gdn_prefill(ctx: ForwardContext | None, tokens: int, device: torch.device | str) -> GdnPrefillPlan:
    if ctx is not None and ctx.gdn_plan is not None:
        return ctx.gdn_plan
    if ctx is not None and ctx.batched:
        lengths = tuple(ctx.seq_lens)
        prefixes = tuple(ctx.seq_prefix_lens)
        cu, slots = ctx.cu_seqlens, ctx.seq_slots
        checkpoint_seqs = ctx.checkpoint_seqs
        if slots is None or slots.numel() != len(lengths) or cu.numel() != len(lengths) + 1:
            raise ValueError("GDN prefill sequence metadata has inconsistent shapes")
    else:
        lengths = (tokens,)
        prefixes = (ctx.prefix_len if ctx is not None else 0,)
        slot = ctx.slot if ctx is not None and ctx.state_pool is not None else 0
        cu = upload([0, tokens], torch.int32, device)
        slots = upload([slot], torch.int32, device)
        checkpoint_seqs = [0] * (len(ctx.checkpoint_positions) if ctx is not None else 0)
    if not lengths or any(n <= 0 for n in lengths) or sum(lengths) != tokens or len(prefixes) != len(lengths):
        raise ValueError("GDN prefill needs positive CPU sequence lengths covering all tokens")
    starts = (0, *accumulate(lengths))
    offsets = (0, *accumulate((n + 63) // 64 for n in lengths))
    indices = [(i, c) for i, n in enumerate(lengths) for c in range((n + 63) // 64)]
    checkpoints = []
    checkpoint_local = []
    if ctx is not None:
        if len(ctx.checkpoint_positions) != len(ctx.checkpoint_slots) or len(checkpoint_seqs) != len(ctx.checkpoint_slots):
            raise ValueError("GDN prefill checkpoint metadata has inconsistent lengths")
        for pos, slot, i in zip(ctx.checkpoint_positions, ctx.checkpoint_slots, checkpoint_seqs):
            local = pos - prefixes[i]
            if local % 64 or not 0 < local < lengths[i]:
                raise ValueError("GDN checkpoint must be an interior 64-token boundary")
            checkpoints.append((slot, starts[i] + local, offsets[i] + local // 64))
            checkpoint_local.append((i, local))
    # CuTe snapshots the state AFTER each interval, unlike FLA's entering-chunk
    # history. Use the largest interval covering every requested boundary;
    # ordinary serving checkpoints therefore do not need a full 64-token history.
    interval = 0
    for _, local in checkpoint_local:
        interval = gcd(interval, local)
    checkpoint_starts = (0, *accumulate(n // interval for n in lengths)) if interval else (0,) * (len(lengths) + 1)
    unique_checkpoints = sorted(set(checkpoint_local))
    packed_indices = {boundary: j for j, boundary in enumerate(unique_checkpoints)}
    checkpoint_indices = [packed_indices[boundary] for boundary in checkpoint_local]
    lookup = [-1] * checkpoint_starts[-1]
    for (i, local), j in packed_indices.items():
        lookup[checkpoint_starts[i] + local // interval - 1] = j
    plan = GdnPrefillPlan(
        cu, slots, upload([p > 0 for p in prefixes], torch.bool, device), lengths,
        upload(indices, cu.dtype, device),
        upload(offsets, torch.int64, device), tuple(checkpoints),
        upload(starts, torch.int64, device), interval,
        # Header: per-sequence lookup offsets. Body: compact snapshot index or-1.
        upload((*checkpoint_starts, *lookup), torch.int64, device),
        upload(checkpoint_indices, torch.int64, device),
        upload([slot for slot, _, _ in checkpoints], torch.int64, device),
        len(unique_checkpoints),
    )
    if ctx is not None:
        ctx.gdn_plan = plan
    return plan
