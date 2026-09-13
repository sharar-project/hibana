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


def block_teacher(anchors: torch.Tensor, block: int, ids: torch.Tensor, logits: torch.Tensor, valid: torch.Tensor):
    """Teacher rows for each block position: the label at ``a + k`` is described by row ``a + k - 1``."""
    length = ids.shape[0]
    offsets = torch.arange(block, device=anchors.device)
    rows = (anchors[:, None] + offsets[None, :] - 1).clamp(min=0, max=length - 1)
    keep = valid[rows] & (offsets[None, :] > 0) & (anchors[:, None] + offsets[None, :] < length)
    return ids[rows], logits[rows], keep


def on_policy_selector(anchor_tokens: torch.Tensor, proposals: torch.Tensor, target_rows: torch.Tensor, block: int, gamma: float | None):
    """Selector predecessors, labels and weights from verification rounds.

    anchor_tokens [A], proposals [A, k], target_rows [A, k + 1] with k = block - 1.
    Block position j (1..k) has predecessor block token j - 1 (the anchor token
    for j = 1, else proposal j - 1) and label target_rows[j - 1], the target's
    argmax after that prefix. Position 0 carries no label.
    """
    rows = anchor_tokens.shape[0]
    tokens = torch.cat([anchor_tokens[:, None], proposals], dim=1)  # [A, block]
    predecessor = torch.cat([tokens[:, :1], tokens[:, :-1]], dim=1)
    labels = torch.cat([tokens[:, :1], target_rows[:, :block - 1]], dim=1)
    offsets = torch.arange(block, device=anchor_tokens.device)
    weights = (offsets[None, :] > 0).to(torch.float32).expand(rows, block).clone()
    if gamma is not None and gamma > 0:
        weights = weights * torch.exp(-(offsets - 1).clamp(min=0).to(torch.float32) / gamma)[None, :]
    return predecessor, labels, weights


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
    kl_num: torch.Tensor | None = None  # KL(target top-k || draft over the same ids), weighted like the CE
    kl_den: torch.Tensor | None = None

    def loss(self, selector_alpha: float, kl_alpha: float = 0.0) -> torch.Tensor:
        total = self.ce_num + selector_alpha * self.selector_ce_num
        if kl_alpha and self.kl_num is not None:
            total = total + kl_alpha * self.kl_num
        return total / self.loss_den.clamp_min(torch.finfo(torch.float32).tiny)

    def metrics(self) -> dict:
        den = self.label_count.clamp_min(1)
        out = {
            "ce": (self.ce_num / self.loss_den.clamp_min(1e-30)).item(),
            "accuracy": (self.correct_num / den).item(),
            "selector_ce": (self.selector_ce_num / self.selector_weight_den.clamp_min(1e-30)).item(),
            "selector_accuracy": (self.selector_correct_num / self.selector_weight_den.clamp_min(1e-30)).item(),
            "selector_coverage": (self.selector_covered_num / den).item(),
        }
        if self.kl_num is not None:
            out["kl"] = (self.kl_num / self.kl_den.clamp_min(1e-30)).item()
        return out

    def __add__(self, other: "Terms") -> "Terms":
        return Terms(*(None if a is None or b is None else a + b for a, b in zip(self.__dict__.values(), other.__dict__.values())))

    def detached(self) -> "Terms":
        return Terms(*(None if t is None else t.detach() for t in self.__dict__.values()))


def teacher_kl(logits: torch.Tensor, teacher_ids: torch.Tensor, teacher_logits: torch.Tensor, weights: torch.Tensor):
    """KL from the target's distribution over its top-k ids to the draft's distribution over the same ids.

    logits [A, B, V] student, teacher_ids [A, B, k] (int64), teacher_logits [A, B, k],
    weights [A, B] (zero where no teacher row exists). Returns (numerator, denominator).
    """
    student = torch.gather(logits.float(), -1, teacher_ids)
    log_q = torch.log_softmax(student, dim=-1)
    log_p = torch.log_softmax(teacher_logits.float(), dim=-1)
    kl = (log_p.exp() * (log_p - log_q)).sum(-1)
    return (kl * weights).sum(), weights.sum()

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
                successor_codebook: torch.Tensor | None, teacher=None,
                selector_predecessor_ids: torch.Tensor | None = None, selector_target_ids: torch.Tensor | None = None,
                selector_weights_override: torch.Tensor | None = None) -> Terms:
    """logits [A, B, V] (any float dtype), hidden [A, B, H]; selector terms are
    zero when the selector weights are None. ``teacher`` is an optional
    (ids [A, B, k], logits [A, B, k], valid [A, B]) for the KL term. The
    selector's predecessors, labels and weights default to the teacher-forced
    ones and can be replaced by on-policy rows (the drafter's own proposals as
    predecessors, the target's answers to them as labels)."""
    kl_num = kl_den = None
    if teacher is not None:
        teacher_ids, teacher_logits, teacher_valid = teacher
        kl_num, kl_den = teacher_kl(logits, teacher_ids, teacher_logits, loss_weights * teacher_valid.to(torch.float32))
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
        return Terms(ce_num, loss_den, correct_num, label_count, zero, zero, zero, zero, kl_num, kl_den)
    sel_pred = predecessor_ids if selector_predecessor_ids is None else selector_predecessor_ids
    sel_target = target_ids if selector_target_ids is None else selector_target_ids
    sel_base = loss_weights if selector_weights_override is None else selector_weights_override
    unary, candidates = logits.float().topk(top_k, dim=-1)
    matches = candidates.eq(sel_target.unsqueeze(-1))
    is_candidate = matches.any(dim=-1)
    target_index = matches.long().argmax(dim=-1)
    scores = selector_scores(unary, candidates, hidden.float(), sel_pred, hidden_projection.float(),
                             predecessor_codebook.float(), successor_codebook.float())
    selector_ce = F.cross_entropy(scores.reshape(-1, top_k), target_index.reshape(-1), reduction="none").reshape_as(sel_target.to(torch.float32))
    selector_weights = sel_base * is_candidate.to(torch.float32)
    selector_ce_num = (selector_ce * selector_weights).sum()
    with torch.no_grad():
        selected = candidates.gather(-1, scores.argmax(dim=-1, keepdim=True)).squeeze(-1)
        selector_correct_num = ((selected == sel_target).to(torch.float32) * selector_weights).sum()
        selector_weight_den = selector_weights.sum()
        selector_covered_num = ((sel_base > 0).to(torch.float32) * is_candidate.to(torch.float32)).sum()
    return Terms(ce_num, loss_den, correct_num, label_count, selector_ce_num, selector_correct_num,
                 selector_weight_den, selector_covered_num, kl_num, kl_den)


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
