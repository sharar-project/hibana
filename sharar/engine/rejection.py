"""Exact speculative rejection, independent of draft architecture and kernels.

q is the *realized conditional proposal law*, after selector corrections
and draft sampling transforms. It is never reconstructed from uncorrected logits.
None denotes deterministic proposals (point masses at the proposed tokens).
"""
from dataclasses import dataclass

import torch

from sharar.engine.sampler import SamplingParams


class DraftColumns:
    """Host copies of a sequential draft's columns, each readable once the step that drafted it ends.

    A consumer that needs the drafted tokens on the host (Flash-Next's PLE rows) reads column j while
    step j + 1 still runs, instead of waiting for the whole block (#36). `host` is pinned int64
    [k, max_batch], written by a non-blocking copy behind each step, then `events[j]` is recorded."""

    def __init__(self, host: torch.Tensor, events: list, batch: int):
        self.host, self.events, self.batch = host, events, batch

    def column(self, j: int) -> list[int]:
        self.events[j].synchronize()
        return self.host[j, :self.batch].tolist()


@dataclass
class Proposal:
    tokens: torch.Tensor  # int64 [B, k]
    probabilities: torch.Tensor | None = None  # FP32 [B, k, V] or sparse support
    support: torch.Tensor | None = None  # int64 [B, k, C], unique per row
    confidence: torch.Tensor | None = None  # FP32 [B, k, 2]: the draft's top-1 probability and top-1 minus top-2 (SHARAR_TRACE_CONFIDENCE only)
    columns: DraftColumns | None = None  # sequential drafts: host copies of tokens' columns as they complete


NUCLEUS_CANDIDATES = 1024  # top-p without top-k: candidates ranked before falling back to a full sort


def distribution(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
    """The same target/draft temperature, top-k (ties included), top-p law."""
    row = logits.float()
    if params.greedy:
        return torch.zeros_like(row).scatter_(-1, row.argmax(-1, keepdim=True), 1)
    row = row / params.temperature
    top_k = params.top_k if 0 < params.top_k < row.shape[-1] else 0
    if top_k:
        kth = row.topk(top_k, dim=-1).values[..., -1:]
        row = row.masked_fill(row < kth, float('-inf'))
    probs = row.softmax(-1)
    if params.top_p < 1:
        probs = _nucleus(probs, params.top_p, top_k)
    return probs


def _nucleus(probs: torch.Tensor, top_p: float, top_k: int) -> torch.Tensor:
    """Keep the smallest descending-probability prefix reaching top_p.

    A full sort of the vocabulary costs about 6 ms per 16-row verification on
    the GB10. Rank only the leading candidates (top-k plus room for ties, or
    NUCLEUS_CANDIDATES): when no row keeps its last candidate, every token
    outside them comes later in the full order and is dropped too, so the
    result equals the full sort's (up to the order of exactly equal
    probabilities, which a sort leaves unspecified as well). Otherwise, one
    host check away, the full sort decides."""
    candidates = min(probs.shape[-1], top_k + 64 if top_k else NUCLEUS_CANDIDATES)
    if candidates < probs.shape[-1]:
        ordered, indices = probs.topk(candidates, dim=-1)
        keep = ordered.cumsum(-1) - ordered < top_p
        if not bool(keep[..., -1].any()):
            ordered = ordered.masked_fill(~keep, 0)
            out = torch.zeros_like(probs).scatter(-1, indices, ordered)
            return out / out.sum(-1, keepdim=True)
    ordered, indices = probs.sort(dim=-1, descending=True)
    keep = ordered.cumsum(-1) - ordered < top_p
    ordered = ordered.masked_fill(~keep, 0)
    probs = torch.zeros_like(probs).scatter(-1, indices, ordered)
    return probs / probs.sum(-1, keepdim=True)


def corrected_distribution(p, q):
    """Law at a rejection: max(p-q, 0), normalized. No epsilon mass added."""
    residual = (p - q).clamp_min(0)
    return residual / residual.sum(-1, keepdim=True)


def verify_row(tokens, p, q, support, generator):
    """Return accepted draft count and one correction/bonus token.

    Only the accepted prefix consumes target random draws. Python control and
    scalar transfers make this an explicit reference path for sampled requests.
    Greedy serving retains the existing vectorized acceptance fast path.
    """
    for j in range(len(tokens)):
        token = int(tokens[j])
        if q is None:
            # Drawing from p and matching a deterministic draft is equivalent
            # to acceptance p[y], followed by p with y removed on rejection.
            sampled = int(torch.multinomial(p[j], 1, generator=generator))
            if sampled != token:
                return j, sampled
            continue
        row = q[j]
        if support is not None:
            row = torch.zeros_like(p[j]).scatter(0, support[j], row)
        q_y = row[token]
        if not torch.isfinite(q_y) or q_y <= 0:
            raise ValueError('proposed token must have positive finite conditional probability')
        threshold = (p[j, token] / q_y).clamp_max(1)
        if torch.rand((), device=p.device, generator=generator) >= threshold:
            residual = corrected_distribution(p[j], row)
            return j, int(torch.multinomial(residual, 1, generator=generator))
    return len(tokens), int(torch.multinomial(p[-1], 1, generator=generator))


def plausible(tokens, p, top_k: int, margin: float):
    """[k] bool: drafted token j among p_j's top `top_k` and within `margin` of
    its most likely token in log-probability, with positive probability."""
    k = tokens.shape[0]
    p_y = p[torch.arange(k, device=p.device), tokens]
    rank = (p[:k] > p_y.unsqueeze(-1)).sum(-1)
    gap = p[:k].max(-1).values.log() - p_y.log()
    return (p_y > 0) & (rank < top_k) & (gap <= margin)


def verify_rows(tokens, p, q, support, generator, relaxed=None):
    """Vectorized rejection sampling for one request without host syncs.

    Same acceptance rule as `verify_row` (accept y_j with probability
    min(1, p_j[y_j] / q_j[y_j]), then sample the residual max(p-q, 0) at the
    first rejection or the bonus from p_k after k acceptances) but the k
    uniform draws are taken at once and the correction/bonus comes from one
    categorical draw over the selected law, so the request's generator is
    consumed identically whatever the outcome. Returns device scalars
    (accepted count, correction token) for the caller to download in one
    batch. Unlike `verify_row` this path is not bit-identical to the
    sequential reference for a fixed seed; it is equal in distribution.

    `relaxed` ([k] bool, `plausible`) also accepts those drafted tokens
    whatever the draw; the correction at the first position failing both
    tests is the residual as before. The third result marks accepted
    positions that only the relaxed rule accepted ([k] bool)."""
    k = tokens.shape[0]
    idx = torch.arange(k, device=p.device)
    if q is None:  # deterministic draft: acceptance p[y], residual p without y
        q_rows = torch.zeros_like(p[:k]).scatter(1, tokens[:, None], 1.0)
    else:
        q_rows = q
        if support is not None:
            q_rows = torch.zeros_like(p[:k]).scatter(1, support, q)
    q_y = q_rows[idx, tokens]
    p_y = p[idx, tokens]
    threshold = (p_y / q_y.clamp_min(1e-30)).clamp_max(1)
    threshold = torch.where(q_y > 0, threshold, torch.zeros_like(threshold))
    u = torch.rand(k, device=p.device, generator=generator)
    exact = u < threshold
    ok = exact if relaxed is None else exact | relaxed
    prefix = torch.cumprod(ok.to(torch.int64), dim=0)
    accepted = prefix.sum()
    residual = corrected_distribution(p[:k], q_rows)
    residual = torch.where(residual.isfinite(), residual, torch.zeros_like(residual))
    laws = torch.cat([residual, p[k:k + 1]], dim=0)  # law at rejection position a, or the bonus law
    law = laws.index_select(0, accepted.reshape(1))
    correction = torch.multinomial(law, 1, generator=generator).reshape(())
    return accepted, correction, prefix.bool() & ~exact
