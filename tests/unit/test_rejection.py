import torch

from sharar.engine.rejection import corrected_distribution, distribution, verify_row
from sharar.engine.sampler import Sampler, SamplingParams
from sharar.engine.contracts import ProposerContext, ProposerRequirements
from sharar.engine.rejection import Proposal
from sharar.proposers.ngram import NGramProposer
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
