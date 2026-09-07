"""Exact speculative rejection, independent of draft architecture and kernels.

q is the *realized conditional proposal law*, after selector/Markov corrections
and draft sampling transforms. It is never reconstructed from uncorrected logits.
None denotes deterministic proposals (point masses at the proposed tokens).
"""
from dataclasses import dataclass

import torch

from sharar.engine.sampler import SamplingParams


@dataclass
class Proposal:
    tokens: torch.Tensor  # int64 [B, k]
    probabilities: torch.Tensor | None = None  # FP32 [B, k, V] or sparse support
    support: torch.Tensor | None = None  # int64 [B, k, C], unique per row


def distribution(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
    """The same target/draft temperature, top-k (ties included), top-p law."""
    row = logits.float()
    if params.greedy:
        return torch.zeros_like(row).scatter_(-1, row.argmax(-1, keepdim=True), 1)
    row = row / params.temperature
    if 0 < params.top_k < row.shape[-1]:
        kth = row.topk(params.top_k, dim=-1).values[..., -1:]
        row = row.masked_fill(row < kth, float('-inf'))
    probs = row.softmax(-1)
    if params.top_p < 1:
        ordered, indices = probs.sort(dim=-1, descending=True)
        keep = ordered.cumsum(-1) - ordered < params.top_p
        ordered = ordered.masked_fill(~keep, 0)
        probs = torch.zeros_like(probs).scatter(-1, indices, ordered)
        probs = probs / probs.sum(-1, keepdim=True)
    return probs


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


def verify_rows(tokens, p, q, support, generator):
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
    """
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
    accepted = torch.cumprod((u < threshold).to(torch.int64), dim=0).sum()
    residual = corrected_distribution(p[:k], q_rows)
    residual = torch.where(residual.isfinite(), residual, torch.zeros_like(residual))
    laws = torch.cat([residual, p[k:k + 1]], dim=0)  # law at rejection position a, or the bonus law
    law = laws.index_select(0, accepted.reshape(1))
    correction = torch.multinomial(law, 1, generator=generator).reshape(())
    return accepted, correction
