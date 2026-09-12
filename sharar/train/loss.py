"""DFlash2 block objective, matching SpecForge's ``OnlineDFlashModel``.

Reference: SpecForge ``specforge/algorithms/common/dflash_family_model.py``
(``_dflash_objective_chunk_terms``, ``_selector_chunk_terms``, label and weight
construction) and the naive formulas in ``tests/test_utils/test_dflash_losses.py``.
Every function here is plain torch so the CPU tests can check it against those
formulas exactly.

A block starts at an anchor token ``a`` and position ``k`` predicts token
``a + k``. Position 0 is the clean anchor and never contributes. The label
weight is ``exp(-max(k - 1, 0) / gamma)`` masked to labels inside the sequence
and inside the supervised (response) region. The selector is trained as a
classifier over the target head's strict top-k candidates with the true
previous token as predecessor, only where the label is a candidate. Both terms
share one denominator, the sum of the label weights over the whole batch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def sample_anchors(prompt_len: int, length: int, count: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Random anchors whose own token and successor are supervised, ascending.

    SpecForge samples anchors where ``loss_mask[t]`` and ``loss_mask[t + 1]``
    hold; here the supervised region is every response position.
    """
    if not 0 <= prompt_len < length:
        raise ValueError("prompt must be shorter than the sequence")
    low, high = prompt_len, length - 1  # anchors in [prompt_len, length - 2]
    if high <= low:
        return torch.empty(0, dtype=torch.int64)
    population = high - low
    if count >= population:
        return torch.arange(low, high, dtype=torch.int64)
    picked = torch.randperm(population, generator=generator)[:count]
    return (picked + low).sort().values


def block_targets(ids: torch.Tensor, anchors: torch.Tensor, block: int, prompt_len: int, gamma: float | None):
    """Return (target_ids, predecessor_ids, weight_mask, loss_weights) as [A, block]."""
    length = ids.shape[0]
    offsets = torch.arange(block, device=ids.device)
    indices = anchors[:, None] + offsets[None, :]
    safe = indices.clamp(max=length - 1)
    target_ids = ids[safe]
    predecessor_ids = torch.cat([target_ids[:, :1], target_ids[:, :-1]], dim=1)
    weight_mask = ((indices < length) & (offsets[None, :] > 0) & (safe >= prompt_len)).to(torch.float32)
    loss_weights = weight_mask
    if gamma is not None and gamma > 0:
        decay = torch.exp(-(offsets - 1).clamp(min=0).to(torch.float32) / gamma)
        loss_weights = weight_mask * decay[None, :]
    return target_ids, predecessor_ids, weight_mask, loss_weights


@dataclass
class Terms:
    """Additive numerators; divide by the shared ``loss_den`` at the end."""
    ce_num: torch.Tensor
    loss_den: torch.Tensor
    correct_num: torch.Tensor
    label_count: torch.Tensor
    selector_ce_num: torch.Tensor
    selector_correct_num: torch.Tensor
    selector_weight_den: torch.Tensor
    selector_covered_num: torch.Tensor

    def loss(self, selector_alpha: float) -> torch.Tensor:
        return (self.ce_num + selector_alpha * self.selector_ce_num) / self.loss_den.clamp_min(torch.finfo(torch.float32).tiny)

    def metrics(self) -> dict:
        den = self.label_count.clamp_min(1)
        return {
            "ce": (self.ce_num / self.loss_den.clamp_min(1e-30)).item(),
            "accuracy": (self.correct_num / den).item(),
            "selector_ce": (self.selector_ce_num / self.selector_weight_den.clamp_min(1e-30)).item(),
            "selector_accuracy": (self.selector_correct_num / self.selector_weight_den.clamp_min(1e-30)).item(),
            "selector_coverage": (self.selector_covered_num / den).item(),
        }

    def __add__(self, other: "Terms") -> "Terms":
        return Terms(*(a + b for a, b in zip(self.__dict__.values(), other.__dict__.values())))


def selector_scores(unary: torch.Tensor, candidates: torch.Tensor, hidden: torch.Tensor, predecessor_ids: torch.Tensor,
                    hidden_projection: torch.Tensor, predecessor_codebook: torch.Tensor,
                    successor_codebook: torch.Tensor) -> torch.Tensor:
    """Unary logits plus the low-rank predecessor transition, as SpecForge's
    ``CandidateSelector.score_candidates`` and Sharar's serving selector."""
    gate = hidden @ hidden_projection.T
    context = predecessor_codebook[predecessor_ids] * gate
    return unary + torch.einsum("...r,...kr->...k", context, successor_codebook[candidates])


def block_terms(logits: torch.Tensor, hidden: torch.Tensor, target_ids: torch.Tensor, predecessor_ids: torch.Tensor,
                weight_mask: torch.Tensor, loss_weights: torch.Tensor, *, top_k: int,
                hidden_projection: torch.Tensor | None, predecessor_codebook: torch.Tensor | None,
                successor_codebook: torch.Tensor | None) -> Terms:
    """logits [A, B, V] (any float dtype), hidden [A, B, H]; selector terms are
    zero when the selector weights are None."""
    vocab = logits.shape[-1]
    flat = logits.reshape(-1, vocab)
    if flat.dtype not in (torch.float32, torch.float64):
        flat = flat.float()
    neg_log_q = F.cross_entropy(flat, target_ids.reshape(-1), reduction="none").reshape_as(target_ids.to(torch.float32))
    ce_num = (neg_log_q * loss_weights).sum()
    loss_den = loss_weights.sum()
    with torch.no_grad():
        correct_num = ((flat.argmax(-1).reshape_as(target_ids) == target_ids) & (weight_mask > 0.5)).sum().to(torch.float32)
        label_count = weight_mask.sum()
    zero = ce_num.new_zeros(())
    if hidden_projection is None:
        return Terms(ce_num, loss_den, correct_num, label_count, zero, zero, zero, zero)
    unary, candidates = logits.float().topk(top_k, dim=-1)
    matches = candidates.eq(target_ids.unsqueeze(-1))
    is_candidate = matches.any(dim=-1)
    target_index = matches.long().argmax(dim=-1)
    scores = selector_scores(unary, candidates, hidden.float(), predecessor_ids, hidden_projection.float(),
                             predecessor_codebook.float(), successor_codebook.float())
    selector_ce = F.cross_entropy(scores.reshape(-1, top_k), target_index.reshape(-1), reduction="none").reshape_as(target_ids.to(torch.float32))
    selector_weights = loss_weights * is_candidate.to(torch.float32)
    selector_ce_num = (selector_ce * selector_weights).sum()
    with torch.no_grad():
        selected = candidates.gather(-1, scores.argmax(dim=-1, keepdim=True)).squeeze(-1)
        selector_correct_num = ((selected == target_ids).to(torch.float32) * selector_weights).sum()
        selector_weight_den = selector_weights.sum()
        selector_covered_num = (weight_mask * is_candidate.to(torch.float32)).sum()
    return Terms(ce_num, loss_den, correct_num, label_count, selector_ce_num, selector_correct_num,
                 selector_weight_den, selector_covered_num)


def greedy_walk(candidates: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Serving's greedy lattice traversal in plain torch.

    candidates [A, S, W], scores [A, S, W, W] where scores[a, j, p, c] scores
    candidate c at step j given predecessor rank p at step j - 1 (rank 0 at
    step 0 is the anchor). Returns [A, S] token ids. Matches
    ``sharar.kernels.draft_selection.greedy_lattice`` for finite scores.
    """
    rows, steps, _ = candidates.shape
    previous = torch.zeros(rows, dtype=torch.int64, device=candidates.device)
    out = []
    for j in range(steps):
        row_scores = scores[torch.arange(rows), j, previous]  # [A, W]
        previous = row_scores.argmax(dim=-1)
        out.append(candidates[torch.arange(rows), j, previous])
    return torch.stack(out, dim=1)
