"""Greedy acceptance, stop detection and budget clamp as static-shaped tensor ops.

This is the device half of `SpecDecoder._accept`: the same rule the host
applies after its download (`y == z` or the relaxed rule, cumulative prefix,
`(y[:a] + [z[a]])[:budget]`, cut at the first stop token), written so it can
be captured in a CUDA graph and drive reconstruction and draft feedback
before the host learns the counts. `tests/unit/test_device_accept.py` checks
it against the host rule.
"""

from __future__ import annotations

import torch

MAX_STOPS = 8  # stop token ids a request may carry on the device path (more: eager path)


def accept_block(y: torch.Tensor, z: torch.Tensor, logits: torch.Tensor | None, relax: torch.Tensor,
                 budget: torch.Tensor, stops: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (accepted [b], counts [b], flags [b, k]) for one verify block.

    y [b, k] drafted tokens; z [b, k+1] the target's argmax rows; logits
    [b, k, V] the target's logits for the drafted positions (None: exact only);
    relax [2, b] per-row (top-k, margin), top-k 0 disables the relaxed rule;
    budget [b] output tokens the request may still emit (0 for padded rows);
    stops [b, MAX_STOPS] stop token ids padded with -1.

    counts is the number of emitted tokens: accepted + 1, cut at the budget
    and at the first stop token; flags marks drafted positions in the
    accepted prefix that the relaxed rule accepted rather than the argmax.
    """
    b, k = y.shape
    s = k + 1
    exact = y == z[:, :k]
    if logits is not None:
        ly = logits.gather(-1, y.unsqueeze(-1)).squeeze(-1).float()
        lmax = logits.max(-1).values.float()
        rank = (logits > ly.unsqueeze(-1)).sum(-1).float()
        relaxed = (rank < relax[0].unsqueeze(1)) & ((lmax - ly) <= relax[1].unsqueeze(1))
        match = exact | relaxed
    else:
        match = exact
    prefix = torch.cumprod(match.to(torch.int64), dim=1)
    accepted = prefix.sum(dim=1)
    flags = prefix.bool() & ~exact
    # Emitted candidates: y[j] for j < a, then the correction z[a].
    offsets = torch.arange(s, device=y.device)
    correction = z.gather(1, accepted.unsqueeze(1))
    candidates = torch.where(offsets.unsqueeze(0) < accepted.unsqueeze(1),
                             torch.cat([y, correction], dim=1), correction)
    within = offsets.unsqueeze(0) <= accepted.unsqueeze(1)
    is_stop = (candidates.unsqueeze(-1) == stops.unsqueeze(1)).any(-1) & within
    first_stop = is_stop.to(torch.int64).argmax(dim=1)
    limit = torch.where(is_stop.any(dim=1), first_stop + 1, torch.full_like(first_stop, s))
    counts = torch.minimum(torch.minimum(accepted + 1, budget), limit).clamp(min=0)
    return accepted, counts, flags


def host_emitted(y: list[int], z: list[int], accepted: int, budget: int, stops, ignore_eos: bool) -> list[int]:
    """The host rule `SpecDecoder._accept` applies; the reference for accept_block."""
    emitted = (y[:accepted] + [z[accepted]])[:budget]
    if not ignore_eos:
        for j, tok in enumerate(emitted):
            if tok in stops:
                emitted = emitted[:j + 1]
                break
    return emitted
