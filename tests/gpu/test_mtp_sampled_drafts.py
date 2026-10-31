"""Sampled MTP drafts on the device (SHARAR_MTP_SAMPLE): the Philox draft draw (proposers/mtp.draft_laws through
kernels/draft_selection.sampled_lattice) and exact verification of those drafts by the fused kernel
(rejection.verify_batch -> kernels/sampled_verify.verify_candidates, the one the device acceptance graph replays)."""
import pytest
import torch

from sharar.engine.rejection import distribution, small_laws, verify_batch
from sharar.engine.sampler import SamplingParams
from sharar.engine import sampling_rng
from sharar.proposers.mtp import draft_laws, draft_rows


@pytest.fixture(autouse=True)
def _inference_mode():
    """As the engine runs: sampling_rng's pinned buffers are process-wide, and one an engine test created under
    inference mode cannot be written outside it."""
    with torch.inference_mode():
        yield


class _Req:
    def __init__(self, params, seed, rounds=0):
        self.params = params
        self.generator = torch.Generator(device="cuda").manual_seed(seed)
        self.rng_seed, self.spec_rounds = seed, rounds


def _candidates(b, r, seed=0):
    torch.manual_seed(seed)
    ids = torch.stack([torch.randperm(5000, device="cuda")[:r] for _ in range(b)])
    logits = (torch.randn(b, r, device="cuda") * 3).sort(-1, descending=True).values
    return ids, logits


def test_device_draft_law_is_the_request_law():
    b, r = 5, 16
    ids, logits = _candidates(b, r)
    params = [SamplingParams(temperature=1.0, top_k=20, top_p=0.95), SamplingParams(temperature=0.6, top_k=4),
              SamplingParams(temperature=1.0, top_k=0, top_p=1.0), SamplingParams(temperature=1.3, top_k=8, top_p=0.5),
              SamplingParams(temperature=0.0)]
    reqs = [_Req(p, 50 + i) for i, p in enumerate(params)]
    rows = draft_rows(reqs, 3, r, ids.device)
    for j in range(3):
        tokens, laws = draft_laws(ids, logits, reqs, rows, j)
        assert torch.allclose(laws, small_laws(logits, params), atol=1e-5)
        assert all(int(t) in row for t, row in zip(tokens.tolist(), ids.tolist()))
        assert int(tokens[4]) == int(ids[4, 0])  # greedy row: the best candidate


def test_device_draws_are_seeded_by_request_round_and_step():
    b, r = 3, 16
    ids, logits = _candidates(b, r, seed=1)
    params = SamplingParams(temperature=1.0, top_k=0, top_p=1.0)

    def draw(seed, rounds, j):
        reqs = [_Req(params, seed + i, rounds) for i in range(b)]
        return draft_laws(ids, logits, reqs, draft_rows(reqs, 4, r, ids.device), j)[0]

    assert torch.equal(draw(7, 3, 2), draw(7, 3, 2))  # reproducible: (seed, round, step) fix the draw
    many = torch.stack([draw(7, rd, j) for rd in range(40) for j in range(4)])
    assert len({tuple(t) for t in many.tolist()}) > 20  # and they move with the round and the step


def test_device_draws_follow_their_law():
    n, r = 100000, 16
    ids, logits = _candidates(1, r, seed=2)
    params = SamplingParams(temperature=1.0, top_k=20, top_p=0.95)
    reqs = [_Req(params, 10_000 + i) for i in range(n)]
    tokens, laws = draft_laws(ids.expand(n, r).contiguous(), logits.expand(n, r).contiguous(), reqs,
                              draft_rows(reqs, 1, r, ids.device), 0)
    pick = (tokens.unsqueeze(1) == ids).int().argmax(1)
    freq = torch.bincount(pick, minlength=r).float() / n
    assert (freq - laws[0]).abs().max() < 0.006


@pytest.mark.parametrize("r", [16, 6])
def test_fused_verify_of_sampled_drafts_emits_the_target_law(r):
    """k = 2 sampled drafts per row from a head that disagrees with the target, verified by the fused kernel with
    Philox draws at the verify offsets: the first emitted token follows the target's law exactly (also when the 6
    candidates miss some of its support), and acceptance at the first slot is sum(min(p, q)), above p(argmax)."""
    n, v, k = 200000, 1000, 2
    torch.manual_seed(4)
    target = torch.randn(v, device="cuda") * 2.5
    head = target + torch.randn(v, device="cuda") * 0.7
    hv, hi = head.topk(r)
    params = SamplingParams(temperature=1.0, top_k=20, top_p=0.95)
    reqs = [_Req(params, 20_000 + i) for i in range(n)]
    rows = draft_rows(reqs, k, r, hv.device)
    ids, logits = hi.expand(n, r).contiguous(), hv.expand(n, r).contiguous()
    drafted = [draft_laws(ids, logits, reqs, rows, j) for j in range(k)]
    tokens = torch.stack([t for t, _ in drafted], 1)
    q = torch.stack([law for _, law in drafted], 1)
    support = ids.unsqueeze(1).expand(n, k, r).contiguous()
    verify_logits = target.view(1, 1, v).expand(n, k + 1, v).contiguous()
    rng = [[q_.rng_seed, o] for q_, o in zip(reqs, sampling_rng.offsets(reqs, sampling_rng.VERIFY))]
    from sharar.engine.rejection import _verify_fused
    assert _verify_fused(tokens[:4], verify_logits[:4], [params] * 4, rng[:4], q[:4], support[:4]) is not None  # the fused kernel decides
    acc, corr, flags, _ = verify_batch(tokens, verify_logits, [params] * n, [q_.generator for q_ in reqs], q, support, rng=rng)
    first = torch.where(acc > 0, tokens[:, 0], corr)
    p = distribution(target, params)
    freq = torch.bincount(first, minlength=v).float() / n
    assert (freq - p).abs().max() < 0.005, (freq - p).abs().max()
    q0 = torch.zeros(v, device="cuda").scatter(0, hi, q[0, 0])
    alpha = torch.minimum(p, q0).sum().item()
    assert abs((acc > 0).float().mean().item() - alpha) < 0.005 and alpha > p.max().item()
    assert not flags.any()  # exact: nothing accepted by a relaxed rule
