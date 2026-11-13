"""Batched sampled verification on candidates (rejection.verify_batch): the law equals `distribution`, acceptance and
correction are distributed as `verify_rows`, a row's result does not depend on its batch, and laws that do not fit
the candidates fall back to the per-row path."""
from collections import Counter

import torch

from hibana.engine.rejection import NUCLEUS_CANDIDATES, candidate_laws, distribution, verify_batch, verify_rows
from hibana.engine.sampler import SamplingParams


def dense(law, index, v):
    return torch.zeros(*law.shape[:-1], v, dtype=law.dtype).scatter(-1, index, law)


def test_candidate_law_equals_distribution():
    g = torch.Generator().manual_seed(1)
    v = 3000  # above NUCLEUS_CANDIDATES, so rows without top_k use candidates too
    params = [SamplingParams(temperature=1.0, top_k=20, top_p=0.95), SamplingParams(temperature=0.6, top_k=20, top_p=0.95),
              SamplingParams(temperature=0.7, top_k=0, top_p=0.9), SamplingParams(temperature=1.3, top_k=5, top_p=1.0),
              SamplingParams(temperature=1.0, top_k=-1, top_p=0.5)]
    logits = torch.randn(len(params), 5, v, generator=g) * 6
    logits[3, 0, :10] = logits[3, 0].max() + 1  # a tie of ten tokens at the top inside top_k = 5: all ten kept
    law, index = candidate_laws(logits, params)
    for i, p in enumerate(params):
        torch.testing.assert_close(dense(law[i], index[i], v), distribution(logits[i], p), rtol=1e-5, atol=1e-7)


def test_laws_that_do_not_fit_the_candidates_fall_back():
    v = 3000
    flat = torch.zeros(1, 5, v)
    assert candidate_laws(flat, [SamplingParams(temperature=1.0, top_k=5, top_p=0.95)]) is None  # ties past the candidates
    assert candidate_laws(flat, [SamplingParams(temperature=1.0, top_k=0, top_p=0.99)]) is None  # nucleus past them
    peaked = torch.randn(1, 5, v) * 8
    assert candidate_laws(peaked, [SamplingParams(temperature=1.0, top_k=-1, top_p=1.0)]) is None  # no truncation
    assert NUCLEUS_CANDIDATES < v
    small = torch.randn(1, 5, 50)  # a vocabulary inside the candidates: every law fits
    assert candidate_laws(small, [SamplingParams(temperature=1.0, top_k=-1, top_p=1.0)]) is not None


def _counts(fn, n):
    out = Counter()
    for seed in range(n):
        a, c = fn(seed)
        out[(a, c)] += 1
    return out


def test_acceptance_and_correction_are_distributed_as_verify_rows():
    torch.manual_seed(0)
    k, v, n = 3, 40, 6000
    for params in (SamplingParams(temperature=1.0, top_k=6, top_p=0.9),
                   SamplingParams(temperature=1.0, top_k=6, top_p=0.9, accept_soft=0.5)):
        logits = torch.randn(1, k + 1, v) * 2
        p = distribution(logits[0], params)
        tokens = torch.multinomial(p[:k], 1).view(1, k)  # drafts the target can sample

        def ref(seed):
            a, c, _ = verify_rows(tokens[0], p, None, None, torch.Generator().manual_seed(seed), soft_scale=params.soft_scale)
            return int(a), int(c)

        def new(seed):
            a, c, _, _ = verify_batch(tokens, logits, [params], [torch.Generator().manual_seed(50_000 + seed)])
            return int(a[0]), int(c[0])

        r, b = _counts(ref, n), _counts(new, n)
        for key in set(r) | set(b):
            # 4.5 sigma of a binomial count at n = 6000 (p <= 1/2): about 170
            assert abs(r[key] - b[key]) <= 170, (params, key, r[key], b[key])


def test_a_rows_result_does_not_depend_on_its_batch():
    torch.manual_seed(3)
    k, v = 4, 500
    params = [SamplingParams(temperature=1.0, top_k=20, top_p=0.95, seed=s) for s in range(4)]
    logits = torch.randn(4, k + 1, v) * 3
    tokens = logits[:, :k].argmax(-1)
    alone = [verify_batch(tokens[i:i + 1], logits[i:i + 1], [params[i]], [torch.Generator().manual_seed(i)]) for i in range(4)]
    batch = verify_batch(tokens, logits, params, [torch.Generator().manual_seed(i) for i in range(4)])
    for i in range(4):
        assert int(batch[0][i]) == int(alone[i][0][0]) and int(batch[1][i]) == int(alone[i][1][0])
        assert torch.equal(batch[2][i], alone[i][2][0])


def test_greedy_rows_match_the_argmax():
    torch.manual_seed(4)
    k, v = 4, 300
    logits = torch.randn(2, k + 1, v)
    argmax = logits.argmax(-1)
    tokens = argmax[:, :k].clone()
    tokens[0, 2] = (tokens[0, 2] + 1) % v  # row 0 diverges at its third draft
    params = [SamplingParams(temperature=0.0), SamplingParams(temperature=1.0, top_k=20, top_p=0.95)]
    a, c, flags, am = verify_batch(tokens, logits, params, [torch.Generator().manual_seed(0), torch.Generator().manual_seed(1)])
    assert int(a[0]) == 2 and int(c[0]) == int(argmax[0, 2]) and not flags[0].any()
    assert torch.equal(am, argmax)


def test_small_laws_equal_distribution():
    from hibana.engine.rejection import small_laws
    g = torch.Generator().manual_seed(7)
    params = [SamplingParams(temperature=1.0, top_k=20, top_p=0.95), SamplingParams(temperature=0.6, top_k=5, top_p=0.9),
              SamplingParams(temperature=1.0, top_k=-1, top_p=1.0), SamplingParams(temperature=0.0), SamplingParams(temperature=2.0, top_k=3, top_p=0.5)]
    scores = torch.randn(len(params), 32, generator=g) * 3
    scores[1, :6] = 9.0  # a tie of six at the top inside top_k = 5: all six kept
    laws = small_laws(scores, params)
    for i, p in enumerate(params):
        torch.testing.assert_close(laws[i], distribution(scores[i], p), rtol=1e-5, atol=1e-7)


def test_sample_lattice_is_distributed_as_the_per_row_walk():
    from hibana.engine.rejection import sample_lattice
    torch.manual_seed(8)
    b, k, w, n = 1, 3, 5, 6000
    candidates = torch.arange(b * k * w).view(b, k, w)
    scores = torch.randn(b, k, w, w) * 1.5
    params = SamplingParams(temperature=1.0, top_k=4, top_p=0.9)

    def walk(gen):  # the per-row walk sample_lattice replaces
        local, out = 0, []
        for j in range(k):
            law = distribution(scores[0, j, local], params)
            local = int(torch.multinomial(law, 1, generator=gen))
            out.append(int(candidates[0, j, local]))
        return tuple(out)

    ref = Counter(walk(torch.Generator().manual_seed(s)) for s in range(n))
    new = Counter(tuple(sample_lattice(candidates, scores, [params], [torch.Generator().manual_seed(90_000 + s)])[0][0].tolist())
                  for s in range(n))
    for key in set(ref) | set(new):
        assert abs(ref[key] - new[key]) <= 170, (key, ref[key], new[key])
    tokens, laws = sample_lattice(candidates, scores, [params], [torch.Generator().manual_seed(1)])
    assert laws.shape == (b, k, w) and torch.allclose(laws.sum(-1), torch.ones(b, k))


def test_sparse_draft_laws_verify_as_verify_rows():
    torch.manual_seed(9)
    k, v, w, n = 3, 40, 6, 6000
    params = SamplingParams(temperature=1.0, top_k=8, top_p=0.95)
    logits = torch.randn(1, k + 1, v) * 2
    p = distribution(logits[0], params)
    support = torch.stack([torch.randperm(v)[:w] for _ in range(k)]).view(1, k, w)
    support[0, :, 0] = p[:k].argmax(-1)  # the draft's support overlaps the target's mass
    q = torch.softmax(torch.randn(1, k, w), -1)
    tokens = support.gather(2, torch.multinomial(q[0], 1).view(1, k, 1)).view(1, k)

    def ref(seed):
        a, c, _ = verify_rows(tokens[0], p, q[0], support[0], torch.Generator().manual_seed(seed))
        return int(a), int(c)

    def new(seed):
        a, c, _, _ = verify_batch(tokens, logits, [params], [torch.Generator().manual_seed(70_000 + seed)], q, support)
        return int(a[0]), int(c[0])

    r, b = _counts(ref, n), _counts(new, n)
    for key in set(r) | set(b):
        assert abs(r[key] - b[key]) <= 170, (key, r[key], b[key])
