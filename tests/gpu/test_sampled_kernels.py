"""The fused sampled-speculation kernels: DFlash2's sampled lattice walk (kernels/draft_selection.sampled_lattice)
and sampled verification (kernels/sampled_verify.verify_candidates) give `distribution`'s law and are distributed as
the per-row reference walk and `rejection.verify_rows`, with Philox draws."""
from collections import Counter

import pytest
import torch

from sharar.engine.rejection import distribution, plausible, small_laws, verify_rows
from sharar.engine.sampler import SamplingParams

DEV = torch.device("cuda")


def close(ref: Counter, got: Counter) -> list:
    """Outcomes whose counts differ by more than 4.5 sigma (plus a small floor)."""
    return [(key, ref[key], got[key]) for key in set(ref) | set(got)
            if abs(ref[key] - got[key]) > 4.5 * (max(ref[key], got[key]) + 1) ** 0.5 + 5]


def test_sampled_lattice_law_and_draws():
    from sharar.kernels.draft_selection import sampled_lattice
    torch.manual_seed(1)
    b, k, w = 4, 15, 32
    params = [SamplingParams(temperature=1.0, top_k=20, top_p=0.95), SamplingParams(temperature=0.6, top_k=5, top_p=0.9),
              SamplingParams(temperature=1.0, top_k=-1, top_p=1.0), SamplingParams(temperature=0.0)]
    cand = torch.randint(0, 248320, (b, k, w), device=DEV)
    scores = torch.randn(b, k, w, w, device=DEV) * 2
    rows = torch.tensor([[max(p.temperature, 1e-6), p.top_k if 0 < p.top_k < w else 0, p.top_p, float(p.greedy)] for p in params], device=DEV)
    tokens, laws = sampled_lattice(cand, scores, rows, torch.tensor([[11, 0], [22, 0], [33, 0], [44, 0]], device=DEV))
    previous = torch.zeros(b, dtype=torch.long, device=DEV)
    for j in range(k):  # the law along the walked path is small_laws' (distribution's)
        row = scores[:, j].gather(1, previous.view(b, 1, 1).expand(b, 1, w)).squeeze(1)
        torch.testing.assert_close(laws[:, j], small_laws(row, params), rtol=1e-5, atol=1e-6)
        previous = (cand[:, j] == tokens[:, j:j + 1]).float().argmax(-1)
    # the drafts are distributed as the per-row walk: one launch of n seeds against n reference walks
    k, w, n = 3, 5, 8000
    scores = torch.randn(1, k, w, w, device=DEV) * 1.5
    cand = torch.arange(k * w, device=DEV).view(1, k, w)
    p = SamplingParams(temperature=1.0, top_k=4, top_p=0.9)
    rng = torch.stack([torch.arange(n, device=DEV) * 7919 + 3, torch.zeros(n, dtype=torch.int64, device=DEV)], 1)
    got = Counter(map(tuple, sampled_lattice(cand.expand(n, k, w).contiguous(), scores.expand(n, k, w, w).contiguous(),
                                             torch.tensor([[1.0, 4, 0.9, 0.0]], device=DEV).expand(n, 4).contiguous(), rng)[0].tolist()))
    ref = Counter()
    for seed in range(n):
        g, local, out = torch.Generator(device=DEV).manual_seed(seed), 0, []
        for j in range(k):
            local = int(torch.multinomial(distribution(scores[0, j, local], p), 1, generator=g))
            out.append(int(cand[0, j, local]))
        ref[tuple(out)] += 1
    assert not close(ref, got)


@pytest.mark.parametrize("relaxed,has_q", [(False, False), (True, False), (False, True)])
def test_verify_candidates_is_distributed_as_verify_rows(relaxed, has_q):
    from sharar.kernels.sampled_verify import verify_candidates
    torch.manual_seed(2)
    k, v, w, n = 3, 300, 6, 8000
    params = SamplingParams(temperature=1.0, top_k=8, top_p=0.9, **({"accept_top_k": 3, "accept_margin": 1.0} if relaxed else {}))
    logits = (torch.randn(1, k + 1, v, device=DEV) * 2).to(torch.bfloat16)
    p = distribution(logits[0], params)
    q = support = None
    if has_q:
        support = torch.stack([torch.randperm(v, device=DEV)[:w] for _ in range(k)]).view(1, k, w)
        support[0, :, 0] = p[:k].argmax(-1)
        q = torch.softmax(torch.randn(1, k, w, device=DEV), -1)
        y = support.gather(2, torch.multinomial(q[0], 1).view(1, k, 1)).view(1, k)
    else:
        y = torch.multinomial(p[:k], 1).view(1, k)
    rel = plausible(y[0], p, params.accept_top_k, params.accept_margin) if params.relaxed else None
    ref = Counter()
    for seed in range(n):
        a, c, _ = verify_rows(y[0], p, None if q is None else q[0], None if q is None else support[0],
                              torch.Generator(device=DEV).manual_seed(seed), relaxed=rel)
        ref[(int(a), int(c))] += 1
    values, index = logits.topk(params.top_k + 64, dim=-1)
    rows = torch.tensor([[1.0, params.top_k, params.top_p, float(params.accept_top_k if relaxed else 0), float(params.accept_margin), 0, 0, 0]], device=DEV)
    rng = torch.stack([torch.arange(n, device=DEV) * 104729 + 17, torch.full((n,), 32, device=DEV)], 1)
    rep = lambda t: None if t is None else t.expand(n, *t.shape[1:]).contiguous()  # noqa: E731
    a, c, _, fit = verify_candidates(rep(values), rep(index), rep(y), rep(rows), rng, rep(q), rep(support))
    assert int(fit.min()) == 1
    assert not close(ref, Counter(zip(a.tolist(), c.tolist())))


def test_verify_candidates_flags_ties_past_the_candidates():
    from sharar.kernels.sampled_verify import verify_candidates
    values, index = torch.zeros(1, 4, 300, device=DEV, dtype=torch.bfloat16).topk(30, dim=-1)
    rows = torch.tensor([[1.0, 5, 0.95, 0, 0, 0, 0, 0]], device=DEV)
    *_, fit = verify_candidates(values, index, torch.zeros(1, 3, dtype=torch.int64, device=DEV), rows, torch.tensor([[1, 32]], device=DEV))
    assert int(fit[0]) == 0


@pytest.mark.parametrize("draft", ["mtp-plausible", "mtp-implausible", "dflash2"])
def test_verify_candidates_soft_law(draft):
    """accept_soft s (row_params[5] = e^-s): the first token's law is min(q, c p) + (1 - alpha) (p - q)+ / Z with
    c = e^s on the plausible set (tests/unit/test_accept_soft.py); s = inf (0 in the slot) is the forced accept."""
    import math
    from sharar.kernels.sampled_verify import verify_candidates
    p = torch.tensor([0.60, 0.30, 0.07, 0.03], device=DEV)  # top-2 within 1.0 plausible
    q = torch.tensor([0.90, 0.06, 0.02, 0.02], device=DEV)
    plaus = torch.tensor([True, True, False, False], device=DEV)
    n = 400_000
    torch.manual_seed(5)
    values = torch.cat([p.log(), torch.tensor([-30.0], device=DEV)]).expand(n, 2, 5).contiguous()  # top_k 4 drops the fifth
    index = torch.arange(5, device=DEV).expand(n, 2, 5).contiguous()
    rng = torch.stack([torch.arange(n, device=DEV) * 7919 + 1, torch.zeros(n, dtype=torch.int64, device=DEV)], 1)
    for s in (0.0, 0.1, 0.25, 0.5, math.inf):
        rows = torch.tensor([[1.0, 4, 1.0, 2, 1.0, math.exp(-s), 0, 0]], device=DEV).expand(n, 8).contiguous()
        if draft == "dflash2":
            drafts = torch.multinomial(q, n, replacement=True).view(n, 1)
            qq, support, point = q.expand(n, 1, 4).contiguous(), torch.arange(4, device=DEV).expand(n, 1, 4).contiguous(), q
        else:
            y = 0 if draft == "mtp-plausible" else 2
            drafts, qq, support = torch.full((n, 1), y, device=DEV), None, None
            point = torch.nn.functional.one_hot(torch.tensor(y, device=DEV), 4).float()
        a, c, flags, fit = verify_candidates(values, index, drafts, rows, rng, qq, support)
        assert bool(fit.all())
        got = torch.bincount(torch.where(a > 0, drafts[:, 0], c), minlength=4).float() / n
        cap = torch.where(plaus, torch.tensor(math.exp(s), device=DEV), torch.tensor(1.0, device=DEV)) * p
        acc = torch.minimum(point, cap)
        left = (p - point).clamp(min=0)
        law = acc + (1 - acc.sum()).clamp(min=0) * left / left.sum()
        assert (got - law).abs().max() < 0.004, (draft, s, got, law)
        assert abs(a.float().mean().item() - acc.sum().item()) < 0.004, (draft, s)
        assert not (flags[:, 0].bool() & (a == 0)).any()  # flags mark accepts only
        assert s > 0 or not flags.any()  # s = 0 boosts nothing
