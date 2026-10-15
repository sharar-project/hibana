# SPDX-License-Identifier: Apache-2.0
"""Speculative rejection sampling for a sampled batch in one kernel, on each row's candidates.

After one top-k over the vocabulary (the candidates, sorted), one program per request computes `distribution`'s law
at every verified position (temperature, top-k with ties, the nucleus in candidate order), accepts drafted token j
with probability min(1, p_j[y_j] / q_j[y_j]) (q a point mass for MTP's drafts, a law on its support for DFlash2's;
`relaxed` rows also accept what `rejection.plausible` passes), and draws the correction from max(p_a - q_a, 0) at
the first rejection a, or the bonus from p_k, by inverse CDF. Uniforms are Philox draws at the request's (seed,
offset + j) (engine/sampling_rng.py). A row whose top-k ties run past its candidates is flagged unfit.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _verify(values, index, drafts, q, support, row_params, rng, accepted_out, correction_out, flags_out, fit_out,
            VS0, VS1, IS0, IS1, QS0, QS1, SS0, SS1, K, C, W,
            BLOCK_C: tl.constexpr, BLOCK_W: tl.constexpr, HAS_Q: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, BLOCK_C)
    in_c = c < C
    temperature = tl.load(row_params + row * 8)
    top_k = tl.load(row_params + row * 8 + 1).to(tl.int32)
    top_p = tl.load(row_params + row * 8 + 2)
    relax_top = tl.load(row_params + row * 8 + 3)
    relax_margin = tl.load(row_params + row * 8 + 4)
    seed = tl.load(rng + row * 2)
    offset = tl.load(rng + row * 2 + 1).to(tl.int32)
    u_last = tl.rand(seed, offset + K + c * 0)  # the correction / bonus uniform (broadcast)
    alive = tl.full((), 1, tl.int32)
    accepted = tl.full((), 0, tl.int64)
    correction = tl.full((), 0, tl.int64)
    fit = tl.full((), 1, tl.int32)
    for j in range(K + 1):
        x = tl.load(values + row * VS0 + j * VS1 + c, in_c, float('-inf')).to(tl.float32) / temperature
        tokens = tl.load(index + row * IS0 + j * IS1 + c, in_c, -1)
        kth = tl.sum(tl.where(c == top_k - 1, x, 0.0), 0)
        last = tl.sum(tl.where(c == C - 1, x, 0.0), 0)
        fit = fit & ((last < kth) | (kth == float('-inf'))).to(tl.int32)  # no finite tie of the k-th value past the candidates
        keep = in_c & (x >= kth)
        m = tl.max(tl.where(keep, x, float('-inf')), 0)
        e = tl.where(keep, tl.exp(x - m), 0.0)
        p = e / tl.sum(e, 0)
        nucleus = (tl.cumsum(p, 0) - p < top_p) | (top_p >= 1)  # candidate order is the descending order
        law = tl.where(nucleus, p, 0.0)
        law = law / tl.sum(law, 0)
        if j < K:
            y = tl.load(drafts + row * K + j)
            p_y = tl.sum(tl.where(tokens == y, law, 0.0), 0)
            if HAS_Q:
                w = tl.arange(0, BLOCK_W)
                in_w = w < W
                qw = tl.load(q + row * QS0 + j * QS1 + w, in_w, 0.0)
                sw = tl.load(support + row * SS0 + j * SS1 + w, in_w, -2)
                q_y = tl.sum(tl.where(sw == y, qw, 0.0), 0)
                threshold = tl.where(q_y > 0, tl.minimum(p_y / tl.maximum(q_y, 1e-30), 1.0), 0.0)
                q_at = tl.sum(tl.where(tokens[:, None] == sw[None, :], qw[None, :], 0.0), 1)  # q_j at the candidates
            else:
                threshold = p_y
                q_at = tl.where(tokens == y, 1.0, 0.0)
            u = tl.sum(tl.where(c == 0, tl.rand(seed, offset + j + c * 0), 0.0), 0)
            exact = u < threshold
            rank = tl.sum(tl.where(law > p_y, 1, 0), 0)
            gap = tl.log(tl.max(law, 0)) - tl.log(p_y)
            plausible = (relax_top > 0) & (p_y > 0) & (rank.to(tl.float32) < relax_top) & (gap <= relax_margin)
            ok = exact | plausible
            tl.store(flags_out + row * K + j, tl.where((alive > 0) & ok & (exact == 0), 1, 0).to(tl.int64))
            reject = (alive > 0) & (ok == 0)
            residual = tl.maximum(law - q_at, 0.0)
            cdf = tl.cumsum(residual, 0)
            pick = tl.minimum(tl.sum(tl.where(in_c & (cdf <= u_last * tl.sum(residual, 0)), 1, 0), 0), C - 1)
            token = tl.sum(tl.where(c == pick, tokens, 0), 0)
            correction = tl.where(reject, token, correction)
            accepted = tl.where((alive > 0) & ok, accepted + 1, accepted)
            alive = tl.where(reject, 0, alive)
        else:  # the bonus after K acceptances
            cdf = tl.cumsum(law, 0)
            pick = tl.minimum(tl.sum(tl.where(in_c & (cdf <= u_last * tl.sum(law, 0)), 1, 0), 0), C - 1)
            token = tl.sum(tl.where(c == pick, tokens, 0), 0)
            correction = tl.where(alive > 0, token, correction)
    tl.store(accepted_out + row, accepted)
    tl.store(correction_out + row, correction)
    tl.store(fit_out + row, fit)


def verify_candidates(values, index, drafts, row_params, rng, q=None, support=None):
    """values [B, K + 1, C] (the vocabulary top-k of the verify logits, descending), index [B, K + 1, C] their token
    ids, drafts int64 [B, K], row_params float32 [B, 8] (temperature, top_k, top_p, relaxed top_k or 0, relaxed
    margin), rng int64 [B, 2] (seed, offset), and the drafts' law q [B, K, W] on support [B, K, W] or None (point
    masses). Every row must have 0 < top_k <= C. Returns (accepted [B], correction [B], relaxed-only flags [B, K],
    fit [B] int32: 0 where ties of the k-th value run past the candidates)."""
    b, s, c = values.shape
    k = s - 1
    dev = values.device
    accepted = torch.empty(b, dtype=torch.int64, device=dev)
    correction = torch.empty(b, dtype=torch.int64, device=dev)
    flags = torch.empty((b, k), dtype=torch.int64, device=dev)
    fit = torch.empty(b, dtype=torch.int32, device=dev)
    has_q = q is not None
    w = q.shape[-1] if has_q else 1
    qq = q if has_q else values
    ss = support if has_q else index
    _verify[(b,)](values, index, drafts.contiguous(), qq, ss, row_params, rng, accepted, correction, flags, fit,
                  values.stride(0), values.stride(1), index.stride(0), index.stride(1),
                  qq.stride(0), qq.stride(1), ss.stride(0), ss.stride(1),
                  k, c, w, triton.next_power_of_2(c), triton.next_power_of_2(w), has_q, num_warps=4)
    return accepted, correction, flags, fit
