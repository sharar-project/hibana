import torch

from hibana.engine.rejection import corrected_distribution, distribution, verify_row
from hibana.engine.sampler import Sampler, SamplingParams
from hibana.engine.contracts import ProposerContext, ProposerRequirements
from hibana.engine.rejection import Proposal
from tests.ngram_proposer import NGramProposer
from tests.unit.test_runtime_safety import tiny_executor


def test_unseeded_requests_also_own_independent_rng_state():
    sampler = Sampler(torch.device('cpu'))
    a, b = sampler.generator(None), sampler.generator(None)
    state = b.get_state().clone()
    torch.rand(100, generator=a)
    assert a is not b and torch.equal(b.get_state(), state)


def test_rejection_mixture_is_exact_with_zero_support_and_markov_rows():
    # Each row represents a different realized predecessor/selector outcome.
    p = torch.tensor([[.1, .3, .6], [.8, .2, 0.], [0., 0., 1.]], dtype=torch.float64)
    q = torch.tensor([[.7, .2, .1], [.1, .4, .5], [1., 0., 0.]], dtype=torch.float64)
    accepted = torch.minimum(p, q)
    rejected_mass = 1 - accepted.sum(-1, keepdim=True)
    restored = accepted + rejected_mass * corrected_distribution(p, q)
    torch.testing.assert_close(restored, p, rtol=0, atol=2e-16)


def test_transforms_preserve_existing_sampler_distribution_and_seed():
    params = SamplingParams(temperature=.7, top_k=3, top_p=.8, seed=12)
    logits = torch.tensor([[1., 2., 2., 4., -1.]])
    sampler = Sampler(torch.device('cpu'))
    a, b = sampler.generator(12), sampler.generator(12)
    for _ in range(50):
        expected = sampler.sample(logits, [params], [a])
        actual = torch.multinomial(distribution(logits, params), 1, generator=b).flatten()
        assert torch.equal(actual, expected)


def test_sparse_conditional_law_matches_dense_and_does_not_draw_past_rejection():
    tokens = torch.tensor([2, 0])
    p = torch.tensor([[1., 0., 0.], [0., 1., 0.], [.2, .3, .5]])
    sparse = torch.tensor([[.25, .75], [.8, .2]])
    support = torch.tensor([[1, 2], [0, 2]])
    dense = torch.zeros((2, 3)).scatter(1, support, sparse)
    a, b = torch.Generator().manual_seed(123), torch.Generator().manual_seed(123)
    assert verify_row(tokens, p, sparse, support, a) == verify_row(tokens, p, dense, None, b) == (0, 0)
    assert torch.equal(a.get_state(), b.get_state())


def test_equal_distributions_accept_without_zero_residual_sampling():
    q = torch.tensor([[0., 1.], [1., 0.]])
    p = torch.cat([q, torch.tensor([[0., 1.]])])
    assert verify_row(torch.tensor([1, 0]), p, q, None, torch.Generator().manual_seed(1)) == (2, 1)


def test_sampled_proposals_use_request_rng_and_shared_stop_aware_commit(monkeypatch):
    ex = tiny_executor(monkeypatch)
    class Toy(NGramProposer):
        requirements = ProposerRequirements('toy', greedy_only=False)
        def ready(self, req):
            return True
        def propose(self, requests, pages, inputs, *, graphs):
            laws = torch.full((len(requests), self.k, 128), 1 / 128)
            tokens = torch.stack([torch.multinomial(laws[i], 1, generator=r.generator).flatten() for i, r in enumerate(requests)])
            return Proposal(tokens, laws)
    ex.set_proposer(Toy(ProposerContext(ex.device, 4, 512)))
    params = SamplingParams(max_new_tokens=11, temperature=1, top_k=-1, top_p=1, seed=113)
    single = ex.generate([1, 2, 3], params).output_ids
    batch = ex.generate_batch([([1, 2, 3], params) for _ in range(4)])
    assert all(r.output_ids == single and r.spec_rounds > 0 and r.committed_len == 13 for r in batch)
    assert ex.kv_pool.free_count == 1024 and ex.state_pool.free_count == 15


def test_deterministic_proposals_serve_sampled_requests(monkeypatch):
    """A proposer without a draft law (MTP's argmax drafts) speculates for sampled requests:
    point-mass rejection, the request's own RNG, identical output alone and in a batch."""
    ex = tiny_executor(monkeypatch)
    class Toy(NGramProposer):
        requirements = ProposerRequirements('toy', greedy_only=False)
        def ready(self, req):
            return True
        def propose(self, requests, pages, inputs, *, graphs):
            return torch.stack([torch.tensor([7, 7, 7][:self.k]) for _ in requests])
    ex.set_proposer(Toy(ProposerContext(ex.device, 4, 512)))
    params = SamplingParams(max_new_tokens=11, temperature=1, top_k=-1, top_p=1, seed=7)
    single = ex.generate([1, 2, 3], params).output_ids
    batch = ex.generate_batch([([1, 2, 3], params) for _ in range(4)])
    assert all(r.output_ids == single and r.spec_rounds > 0 and r.committed_len == 13 for r in batch)


def test_mtp_proposer_serves_sampled_requests():
    from types import SimpleNamespace
    from hibana.proposers.mtp import MtpProposer
    proposer = MtpProposer(SimpleNamespace(bytes=0), ProposerContext(torch.device('cpu'), 4, 512))
    assert proposer.requirements.greedy_only is False


def test_vectorized_verify_matches_sequential_reference_in_distribution():
    """verify_rows draws differently from verify_row (all k uniforms at once) but
    must agree in distribution: acceptance counts and correction tokens."""
    from collections import Counter

    from hibana.engine.rejection import verify_rows

    torch.manual_seed(0)
    k, v = 3, 6
    p = torch.softmax(torch.randn(k + 1, v) * 2, -1)
    q = torch.softmax(torch.randn(k, v) * 2, -1)
    tokens = torch.tensor([1, 4, 2])
    n = 4000
    ref, vec = Counter(), Counter()
    for seed in range(n):
        a, c = verify_row(tokens, p, q, None, torch.Generator().manual_seed(seed))
        ref[(a, c)] += 1
        a_t, c_t, _ = verify_rows(tokens, p, q, None, torch.Generator().manual_seed(10_000 + seed))
        vec[(int(a_t), int(c_t))] += 1
    for key in set(ref) | set(vec):
        # binomial tolerance: 4 sigma of n*p*(1-p) with p up to 1/2 is ~130 counts
        assert abs(ref[key] - vec[key]) <= 130, (key, ref[key], vec[key])
    assert sum(vec.values()) == n


def test_vectorized_verify_deterministic_draft_and_sparse_support():
    from hibana.engine.rejection import verify_rows

    p = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]])
    tokens = torch.tensor([0, 1])
    a, c, _ = verify_rows(tokens, p, None, None, torch.Generator().manual_seed(3))
    assert 0 <= int(a) <= 2 and 0 <= int(c) < 3
    support = torch.tensor([[0, 2], [1, 2]])
    q = torch.tensor([[0.9, 0.1], [0.6, 0.4]])
    a, c, _ = verify_rows(tokens, p, q, support, torch.Generator().manual_seed(3))
    assert 0 <= int(a) <= 2 and 0 <= int(c) < 3


def test_candidate_nucleus_equals_the_full_sort():
    from hibana.engine.rejection import _nucleus
    g = torch.Generator().manual_seed(5)
    for scale in (0.05, 1.0, 8.0):  # flat rows exercise the full-sort fallback
        logits = torch.randn(8, 3000, generator=g) * scale
        for top_p, top_k in ((0.95, 20), (0.9, 0), (0.99, 0), (0.5, 7)):
            probs = logits.softmax(-1)
            if top_k:
                kth = logits.topk(top_k, dim=-1).values[..., -1:]
                probs = logits.masked_fill(logits < kth, float('-inf')).softmax(-1)
            ordered, indices = probs.sort(dim=-1, descending=True)
            keep = ordered.cumsum(-1) - ordered < top_p
            ref = torch.zeros_like(probs).scatter(-1, indices, ordered.masked_fill(~keep, 0))
            ref = ref / ref.sum(-1, keepdim=True)
            assert torch.equal(_nucleus(probs, top_p, top_k), ref)


def test_relaxed_rows_accept_what_rejection_would_not():
    from hibana.engine.rejection import verify_rows
    k, V = 4, 6
    p = torch.full((k + 1, V), 0.02)
    p[:, 0] = 0.9  # the target strongly prefers token 0 everywhere
    tokens = torch.tensor([1, 1, 1, 1])  # the draft proposes token 1 (p = 0.02) deterministically
    exact_accepts = relaxed_accepts = 0
    for seed in range(200):
        a, c, rx = verify_rows(tokens, p, None, None, torch.Generator().manual_seed(seed))
        exact_accepts += int(a)
        assert not rx.any()
        # accept_soft s = log 40: u * q < 40 p accepts token 1 (p = 0.02) at u < 0.8
        a2, c, rx = verify_rows(tokens, p, None, None, torch.Generator().manual_seed(seed), soft_scale=1 / 40)
        relaxed_accepts += int(a2)
        assert int(a2) >= int(a)  # the same uniforms: an exact accept stays an accept
        assert not rx[:int(a)].any() and not rx[int(a2):].any()  # flags: relaxed-only accepts inside the prefix
    assert exact_accepts < 20 and relaxed_accepts >= 400
