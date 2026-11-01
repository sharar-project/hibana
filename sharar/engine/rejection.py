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
    step j + 1 still runs, instead of waiting for the whole block. `host` is pinned int64
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
    # FP32 [B, k]: position j's probability of acceptance once positions < j were accepted (DFlash2's lattice).
    # It reads only the drafts before position j, never y_j, so a depth chosen from it keeps rejection sampling exact.
    acceptance: torch.Tensor | None = None

    def head(self, n: int) -> "Proposal":
        """The first n drafts of every request."""
        cut = lambda t: t[:, :n] if t is not None else None  # noqa: E731
        return Proposal(self.tokens[:, :n], cut(self.probabilities), cut(self.support), cut(self.confidence),
                        self.columns, cut(self.acceptance))


NUCLEUS_CANDIDATES = 1024  # top-p without top-k: candidates ranked before falling back to a full sort


_ROWS: dict = {}  # (kind, values, device) -> device tensor: a batch's per-row parameters, uploaded once


def _rows(kind: str, values: list, device) -> torch.Tensor:
    """A device tensor of per-row parameters, reused while the batch's requests stay the same. An upload from
    pageable memory waits for the work already queued on the stream (the draft or verify graph just replayed), so
    building it each step or round would stall the host behind the GPU."""
    key = (kind, tuple(map(tuple, values)), str(device))
    t = _ROWS.get(key)
    if t is None:
        if len(_ROWS) > 512:
            _ROWS.clear()
        t = _ROWS[key] = torch.tensor(values, device=device)
    return t


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
    Greedy serving uses the vectorized path.
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


def verify_rows(tokens, p, q, support, generator, soft_scale: float = 1.0):
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

    With `soft_scale` = e^-s < 1 (SamplingParams.accept_soft s > 0) a drafted
    token is also accepted when u * e^-s * q_j[y_j] < p_j[y_j]; the correction
    at the first position failing both tests is the residual as before. The
    third result marks accepted positions that only the relaxed rule accepted
    ([k] bool)."""
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
    ok = exact | ((q_y > 0) & (u * soft_scale * q_y < p_y)) if soft_scale < 1 else exact  # a draft its law never gives: rejected
    prefix = torch.cumprod(ok.to(torch.int64), dim=0)
    accepted = prefix.sum()
    residual = corrected_distribution(p[:k], q_rows)
    residual = torch.where(residual.isfinite(), residual, torch.zeros_like(residual))
    laws = torch.cat([residual, p[k:k + 1]], dim=0)  # law at rejection position a, or the bonus law
    law = laws.index_select(0, accepted.reshape(1))
    correction = torch.multinomial(law, 1, generator=generator).reshape(())
    return accepted, correction, prefix.bool() & ~exact


def candidate_laws(logits, params):
    """Each row's `distribution` law on its candidates: (law [B, T, C], token ids [B, T, C]), from one top-k over
    the vocabulary of logits [B, T, V] (top_k plus room for ties, or NUCLEUS_CANDIDATES without top_k).
    None when a sampled row's law does not fit its candidates (ties of the k-th value past them, a nucleus
    wider than them, or no truncation at all). Greedy rows are not checked; their law is not used."""
    b, t, v = logits.shape
    dev = logits.device
    top_k = [p.top_k if 0 < p.top_k < v else 0 for p in params]
    sampled = [not p.greedy for p in params]
    if any(s and not tk and p.top_p >= 1 for s, p, tk in zip(sampled, params, top_k)) and v > NUCLEUS_CANDIDATES:
        return None  # an untruncated law has the whole vocabulary as support
    c = min(v, max(tk + 64 if tk else NUCLEUS_CANDIDATES for tk in top_k))
    values, index = logits.topk(c, dim=-1)  # descending
    # one upload: temperature, top_k - 1, top_p and whether the row's law must fit its candidates
    rows = _rows("laws", [[max(p.temperature, 1e-6), max(tk - 1, 0), p.top_p, float(s)] for p, tk, s in zip(params, top_k, sampled)],
                 dev).view(b, 1, 4)
    temp, top_p, checked = rows[..., 0:1], rows[..., 2:3], rows[..., 3:4] > 0
    cand = values.float() / temp  # the tempered logits `distribution` computes, at the candidates
    every_k = all(tk for tk in top_k)
    if every_k:  # the usual case: every row truncates to its top_k (ties included, as distribution's mask)
        kth = cand.gather(-1, rows[..., 1:2].long().expand(b, t, 1))
        keep = cand >= kth
        probs = cand.masked_fill(~keep, float('-inf')).softmax(-1)
        nucleus = probs.cumsum(-1) - probs < top_p
        bad = (cand[..., -1:] >= kth) & checked if c < v else None  # a tie of the k-th value past the candidates
    else:
        has_k = _rows("has_k", [[float(tk > 0)] for tk in top_k], dev).view(b, 1, 1) > 0
        kth = cand.gather(-1, rows[..., 1:2].long().expand(b, t, 1))
        keep = ~has_k | (cand >= kth)
        full = torch.logsumexp(logits.float() / temp, dim=-1, keepdim=True)  # no top_k: normalize over the vocabulary
        probs = torch.where(has_k, cand.masked_fill(~keep, float('-inf')).softmax(-1), (cand - full).exp())
        nucleus = probs.cumsum(-1) - probs < top_p
        bad = ((has_k & (cand[..., -1:] >= kth)) | (~has_k & nucleus[..., -1:])) & checked if c < v else None
    nucleus |= top_p >= 1
    if bad is not None and bool(bad.any()):
        return None
    law = probs * (keep & nucleus)
    return law / law.sum(-1, keepdim=True), index


def verify_batch(tokens, logits, params, generators, q=None, support=None, rng=None):
    """Rejection sampling for a sampled batch on each row's candidates.

    tokens [B, k] drafts, logits [B, k + 1, V] (the verify output), one SamplingParams and generator per row, and
    the drafts' law: none (deterministic drafts, MTP: point masses) or q [B, k, W] on support [B, k, W] (DFlash2).
    The same law as `distribution` (`candidate_laws`) and the same rule as `verify_rows` (accept y_j with
    probability min(1, p_j[y_j] / q_j[y_j]); at the first rejection draw from max(p_j - q_j, 0); after k
    acceptances draw the bonus from p_k; relaxed rows also accept y_j when u * e^-accept_soft * q_j[y_j] < p_j[y_j];
    the residual is positive only where p_j is, so it lives on p_j's candidates), for the whole batch at once and
    on a few dozen candidates instead of the 248K-token vocabulary. Each row draws k + 1 uniforms from its own
    generator (the k acceptance tests, then the correction by inverse CDF), so a row's result does not depend
    on the batch; equal in distribution to `verify_rows`, not seed-for-seed. Greedy rows match the target's
    argmax as the greedy path does.

    With `rng` ([seed, offset] per row, engine/sampling_rng.py) on CUDA, rows that set top_k run in one fused
    kernel (kernels/sampled_verify.py) with Philox uniforms instead of the generators.

    Returns device tensors (accepted [B], correction [B], relaxed-only flags [B, k], argmax [B, k + 1] or
    None), or None when `candidate_laws` does: the caller then takes the per-row path."""
    b, k = tokens.shape
    dev = logits.device
    v = logits.shape[-1]
    if rng is not None and logits.is_cuda and all(p.greedy or 0 < p.top_k < v for p in params):
        fused = _verify_fused(tokens, logits, params, rng, q, support)
        if fused is not None:
            return fused
    laws = candidate_laws(logits, params)
    if laws is None:
        return None
    law, index = laws
    c = law.shape[-1]
    any_greedy = any(p.greedy for p in params)
    argmax = logits.argmax(-1) if any_greedy else None  # greedy rows: the same argmax as the greedy path
    p_y = (law[:, :k] * (index[:, :k] == tokens.unsqueeze(-1))).sum(-1)
    if q is None:  # point masses: the threshold is p_j[y_j], and q_j at p's candidates is 1 at y_j only
        threshold = p_y
        q_at = None
    else:
        q_y = (q * (support == tokens.unsqueeze(-1))).sum(-1)
        threshold = torch.where(q_y > 0, (p_y / q_y.clamp_min(1e-30)).clamp_max(1), torch.zeros_like(p_y))
        q_at = (q.unsqueeze(-2) * (index[:, :k].unsqueeze(-1) == support.unsqueeze(-2))).sum(-1)  # q_j at p's candidates [B, k, C]
    u = torch.stack([torch.rand(k + 1, device=dev, generator=g) for g in generators])  # [B, k + 1]
    exact = u[:, :k] < threshold
    ok = exact
    relaxed_rows = [p.relaxed and not p.greedy for p in params]
    if any(relaxed_rows):
        rule = _rows("rule", [[float(r), p.soft_scale] for p, r in zip(params, relaxed_rows)], dev)
        boosted = u[:, :k] * rule[:, 1:] * (1.0 if q is None else q_y) < p_y  # accept_soft: u q < e^s p
        if q is not None:
            boosted = boosted & (q_y > 0)  # a draft its law never gives: rejected, as by the exact test
        ok = exact | ((rule[:, :1] > 0) & boosted)  # rows not relaxed: exact
    prefix = torch.cumprod(ok.to(torch.int64), dim=1)
    accepted = prefix.sum(1)
    # the law at the first rejection (without its drafted token), or the bonus law after k acceptances
    at = accepted.view(b, 1, 1).expand(b, 1, c)
    sel, sel_index = law.gather(1, at).squeeze(1), index.gather(1, at).squeeze(1)  # [B, C]
    rejected = (accepted < k).view(b, 1)
    if q_at is None:  # max(p - q, 0) for a point mass: p without the drafted token
        drafted = tokens.gather(1, accepted.clamp_max(k - 1).view(b, 1))
        sel = torch.where(rejected & (sel_index == drafted), torch.zeros_like(sel), sel)
    else:
        q_sel = q_at.gather(1, accepted.clamp_max(k - 1).view(b, 1, 1).expand(b, 1, c)).squeeze(1)
        sel = torch.where(rejected, (sel - q_sel).clamp_min(0), sel)
    cdf = sel.cumsum(-1)
    draw = torch.searchsorted(cdf, (u[:, k:] * cdf[:, -1:]).contiguous(), right=True).clamp_max(c - 1)
    correction = sel_index.gather(1, draw).squeeze(1)
    flags = prefix.bool() & ~exact
    if argmax is not None:  # greedy rows: the exact-match prefix and the argmax after it
        greedy = _rows("greedy", [[float(p.greedy)] for p in params], dev).view(b) > 0
        g_acc = torch.cumprod((tokens == argmax[:, :k]).to(torch.int64), dim=1).sum(1)
        accepted = torch.where(greedy, g_acc, accepted)
        correction = torch.where(greedy, argmax.gather(1, g_acc.view(b, 1)).squeeze(1), correction)
        flags = flags & ~greedy.view(b, 1)
    return accepted, correction, flags, argmax


def small_laws(scores, params):
    """`distribution` for each row of a few scores at once: scores [B, W] (W draft candidates), one SamplingParams per
    row (greedy rows get the point mass at their argmax)."""
    b, w = scores.shape
    rows = _rows("small", [[max(p.temperature, 1e-6), p.top_k if 0 < p.top_k < w else w, p.top_p, float(p.greedy)] for p in params],
                 scores.device)
    x = scores.float() / rows[:, :1]
    ordered, order = x.sort(dim=-1, descending=True)
    kth = ordered.gather(1, (rows[:, 1:2].long() - 1))
    probs = x.masked_fill(x < kth, float('-inf')).softmax(-1)  # top-k with ties, then the softmax
    sorted_p = probs.gather(1, order)
    keep = (sorted_p.cumsum(-1) - sorted_p < rows[:, 2:3]) | (rows[:, 2:3] >= 1)  # the nucleus in descending order
    law = torch.zeros_like(probs).scatter(1, order, sorted_p * keep)
    law = law / law.sum(-1, keepdim=True)
    point = torch.zeros_like(law).scatter_(1, x.argmax(-1, keepdim=True), 1)
    return torch.where(rows[:, 3:4] > 0, point, law)


def sample_lattice(candidates, scores, params, generators):
    """Sampled drafts from DFlash2's candidate lattice for a whole batch: candidates [B, k, W] token ids, scores
    [B, k, W, W] (the score of each candidate given the previous step's choice). Step j draws from `distribution`
    over the W scores that follow the previous choice, as the per-row walk did, with each row's k uniforms drawn
    at once from its own generator (inverse CDF; equal in distribution to one multinomial per step). Returns
    (tokens [B, k], laws [B, k, W]): the drafts and the realized conditional law q of each."""
    b, k, w = candidates.shape
    dev = candidates.device
    u = torch.stack([torch.rand(k, device=dev, generator=g) for g in generators])  # [B, k]
    local = torch.zeros(b, 1, dtype=torch.int64, device=dev)
    tokens, laws = [], []
    for j in range(k):
        row = scores[:, j].gather(1, local.view(b, 1, 1).expand(b, 1, w)).squeeze(1)  # [B, W] after the previous choice
        law = small_laws(row, params)
        cdf = law.cumsum(-1)
        local = torch.searchsorted(cdf, (u[:, j:j + 1] * cdf[:, -1:]).contiguous(), right=True).clamp_max(w - 1)
        tokens.append(candidates[:, j].gather(1, local).squeeze(1))
        laws.append(law)
    return torch.stack(tokens, 1), torch.stack(laws, 1)


def _verify_fused(tokens, logits, params, rng, q, support):
    """verify_batch through kernels/sampled_verify.py: one vocabulary top-k, then one program per row. None when a
    row's top-k ties run past its candidates (the caller takes the candidate-law path)."""
    from sharar.engine import sampling_rng
    from sharar.kernels.sampled_verify import verify_candidates
    b, k = tokens.shape
    dev = logits.device
    c = min(logits.shape[-1], max((p.top_k for p in params if not p.greedy), default=1) + 64)
    values, index = logits.topk(c, dim=-1)
    row_params = sampling_rng.rows("verify-params", [[1.0, 1, 1.0, 0.0, 0.0, 0, 0, 0] if p.greedy else
                                                     [max(p.temperature, 1e-6), p.top_k, p.top_p, float(p.relaxed), 0.0,
                                                      p.soft_scale, 0, 0] for p in params], torch.float32, dev)
    rng_t = sampling_rng.rows("verify-rng", rng, torch.int64, dev)
    accepted, correction, flags, fit = verify_candidates(values, index, tokens, row_params, rng_t, q, support)
    greedy = [p.greedy for p in params]
    if not bool(fit.bool().logical_or(torch.tensor(greedy, device=dev) if any(greedy) else torch.zeros_like(fit, dtype=torch.bool)).all()):
        return None
    argmax = None
    if any(greedy):  # greedy rows: the exact-match prefix and the argmax after it, as the greedy path
        argmax = logits.argmax(-1)
        g = _rows("greedy", [[float(x)] for x in greedy], dev).view(b) > 0
        g_acc = torch.cumprod((tokens == argmax[:, :k]).to(torch.int64), dim=1).sum(1)
        accepted = torch.where(g, g_acc, accepted)
        correction = torch.where(g, argmax.gather(1, g_acc.view(b, 1)).squeeze(1), correction)
        flags = flags * (~g).view(b, 1)
    return accepted, correction, flags.bool(), argmax
