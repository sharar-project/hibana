"""Sampled MTP drafts (HIBANA_MTP_SAMPLE): the draft head's candidates, the drafts' law, exact verification against it,
the settings and the profiles that turn it on."""
import math

import pytest
import torch

from hibana.engine.config import EngineConfig
from hibana.engine.rejection import distribution, small_laws, verify_batch
from hibana.engine.sampler import SamplingParams
from hibana.launch import settings as launch
from hibana.proposers.draft_head import ShortlistHead, TargetHead
from hibana.proposers.mtp import draft_laws


class _Req:
    def __init__(self, params, seed):
        self.params = params
        self.generator = torch.Generator().manual_seed(seed)
        self.rng_seed, self.spec_rounds = seed, 0


def test_settings_parse_and_validate():
    assert not EngineConfig.from_env({}).mtp_sample
    assert EngineConfig.from_env({"HIBANA_MTP_SAMPLE": "1"}).mtp_sample


@pytest.mark.parametrize("profile", ["fast", "throughput"])
def test_mtp_profiles_sample_their_drafts(profile):
    for p in (launch.profile_defaults(profile, "flashnext"), launch.profile_defaults(profile, "flashnext", proposer="mtp"),
              launch.profile_defaults(profile, "qwen38", proposer="mtp"), launch.profile_defaults(profile, "nemotron")):
        assert p["HIBANA_MTP_SAMPLE"] == "1", p
        assert p["HIBANA_ACCEPT_DEFAULT"] == "3:2.0"  # greedy requests keep the relaxed rule
    # DFlash2 drafts are sampled already
    for p in (launch.profile_defaults(profile, "flashnext", proposer="dflash2"), launch.profile_defaults(profile, "qwen38")):
        assert "HIBANA_MTP_SAMPLE" not in p


def test_exact_profiles_keep_argmax_drafts():
    for target in ("flashnext", "qwen38", "nemotron"):
        p = launch.profile_defaults("exact", target)
        assert "HIBANA_MTP_SAMPLE" not in p and p["HIBANA_ACCEPT_SOFT"] == "0"


def test_target_head_candidates_are_its_best_logits():
    torch.manual_seed(0)
    w = torch.randn(300, 32)
    head = TargetHead(lambda x: x @ w.T)
    x = torch.randn(5, 32)
    ids, values = head.candidates(x, 16)
    want_v, want_i = (x @ w.T).topk(16, dim=-1)
    assert torch.equal(ids, want_i) and torch.allclose(values, want_v) and values.dtype == torch.float32
    assert torch.equal(ids[:, 0], head.argmax(x))


def _shortlist(rows: int, real: int | None, rescore: int, scale: float):
    """A ShortlistHead without its low-bit kernel: the "low-bit" logits are the FP8 rows' plus noise, so the rescore
    has something to fix. `real` rows of `rows` hold ids (the rest pad with the last one)."""
    torch.manual_seed(1)
    weight = torch.randn(rows, 24)
    head = object.__new__(ShortlistHead)
    head.real_rows = real
    head.ids = None if real is None else torch.cat([torch.arange(real), torch.full((rows - real,), real - 1)])
    head.rescore = rescore
    head.source = type("Fp8", (), {"weight": weight, "scale": torch.tensor(scale)})()
    noise = torch.randn(rows) * 0.3
    head._logits = lambda flat: flat @ weight.T + noise
    return head, weight


@pytest.mark.parametrize("real", [None, 90])
def test_shortlist_candidates_rescored_scaled_and_unique(real):
    head, weight = _shortlist(128, real, rescore=16, scale=0.5)
    x = torch.randn(7, 24)
    ids, values = head.candidates(x, 16)
    assert ids.shape == values.shape == (7, 16) and values.dtype == torch.float32
    exact = 0.5 * (x @ weight.T)
    assert torch.allclose(values, exact.gather(1, ids), atol=1e-5)  # the FP8 rows' logits, weight scale applied
    assert (values[:, :-1] >= values[:, 1:]).all()  # descending
    assert all(len(set(r)) == 16 for r in ids.tolist())  # padded rows never repeat an id
    if real is not None:
        assert int(ids.max()) < real
    assert torch.equal(ids[:, 0], head._argmax(x)[0])  # the shortlist holds the argmax pick, first


def test_draft_laws_cpu_match_the_request_law_and_draw_from_it():
    torch.manual_seed(2)
    b, r = 4, 16
    ids = torch.stack([torch.randperm(1000)[:r] for _ in range(b)])
    logits = torch.randn(b, r).sort(-1, descending=True).values * 3
    params = [SamplingParams(temperature=1.0, top_k=20, top_p=0.95), SamplingParams(temperature=0.7, top_k=5),
              SamplingParams(temperature=1.0, top_k=0, top_p=1.0), SamplingParams(temperature=0.0)]
    reqs = [_Req(p, 10 + i) for i, p in enumerate(params)]
    tokens, laws = draft_laws(ids, logits, reqs)
    assert torch.allclose(laws, small_laws(logits, params))
    assert torch.allclose(laws.sum(-1), torch.ones(b))
    assert (laws[1] > 0).sum() <= 5 and laws[3].max() == 1 and laws[3, 0] == 1  # top-k; greedy: the best candidate
    assert all(int(t) in row for t, row in zip(tokens.tolist(), ids.tolist()))
    # one row many times: the draws follow its law
    n = 20000
    rows = [_Req(params[0], 1000 + i) for i in range(n)]
    tokens, laws = draft_laws(ids[:1].expand(n, r), logits[:1].expand(n, r), rows)
    freq = torch.bincount((ids[0].unsqueeze(0) == tokens.unsqueeze(1)).int().argmax(1), minlength=r).float() / n
    assert (freq - laws[0]).abs().max() < 0.015


def test_sampled_drafts_verify_to_the_target_law():
    """The whole rule on the host path: drafts drawn by draft_laws from a head law that differs from the target's,
    on 8 candidates that miss some of the target's support, verified by verify_batch against that law: the first
    emitted token follows the target's law exactly, and acceptance is sum(min(p, q)) (> p(argmax), the argmax draft's)."""
    torch.manual_seed(3)
    v, r, n = 40, 8, 30000
    target = torch.randn(v) * 2
    head = target + torch.randn(v) * 0.8  # the draft head disagrees a little
    hv, hi = head.topk(r)
    params = SamplingParams(temperature=1.0, top_k=20, top_p=0.95)
    reqs = [_Req(params, i) for i in range(n)]
    tokens, laws = draft_laws(hi.expand(n, r), hv.expand(n, r), reqs)
    logits = target.view(1, 1, v).expand(n, 2, v)  # k = 1 draft and the bonus row
    acc, corr, _, _ = verify_batch(tokens.view(n, 1), logits, [params] * n, [q.generator for q in reqs],
                                   laws.view(n, 1, r), hi.view(1, 1, r).expand(n, 1, r))
    first = torch.where(acc > 0, tokens, corr)
    p = distribution(target, params)
    freq = torch.bincount(first, minlength=v).float() / n
    assert (freq - p).abs().max() < 0.012, (freq - p).abs().max()
    q = torch.zeros(v).scatter(0, hi, laws[0])
    alpha = torch.minimum(p, q).sum().item()
    assert abs(acc.float().mean().item() - alpha) < 0.012 and alpha > p.max().item()
