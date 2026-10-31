"""Relaxed acceptance for sampled requests (SamplingParams.accept_soft s): a draft keeps its uniform, scaled by e^-s,
and is accepted when u * q(y) < e^s * p(y). At one verify slot the first emitted token then has the law
    P(x) = min(q(x), e^s p(x)) + (1 - alpha) (p(x) - q(x))+ / Z,
with alpha = sum_y min(q(y), e^s p(y)), so P(x) <= e^s p(x); s = 0 is exact sampling. Checked by Monte Carlo through
verify_rows and verify_batch for MTP's point-mass drafts and DFlash2's draft laws (the fused kernel:
tests/gpu/test_sampled_kernels.py)."""
import math

import pytest
import torch

from sharar.api import validation
from sharar.api.validation import _sampling_from
from sharar.engine.rejection import verify_batch, verify_rows
from sharar.engine.sampler import InvalidRequest, SamplingParams

P = torch.tensor([0.60, 0.30, 0.07, 0.03])  # the target's law
Q = torch.tensor([0.90, 0.06, 0.02, 0.02])  # a drafter sharper than its target
SOFTS = (0.0, 0.25, 0.5)


def params(s, **kw):
    return SamplingParams(temperature=1.0, top_k=-1, top_p=1.0, accept_soft=s, **kw)


def soft_law(s, y=None):
    """The first token's law: DFlash2's draft law Q, or (y given) MTP's point mass at y."""
    q = Q if y is None else torch.nn.functional.one_hot(torch.tensor(y), 4).float()
    a = torch.minimum(q, math.exp(s) * P)
    left = (P - q).clamp(min=0)
    return a + (1 - a.sum()).clamp(min=0) * left / left.sum()


def first_tokens(tokens, accepted, correction):
    return torch.where(accepted > 0, tokens[:, 0], correction)


def dflash2_rows(n, seed):
    tokens = torch.multinomial(Q, n, replacement=True, generator=torch.Generator().manual_seed(seed)).view(n, 1)
    return tokens, P.log().expand(n, 2, 4).contiguous(), Q.expand(n, 1, 4).contiguous(), torch.arange(4).expand(n, 1, 4).contiguous()


def generators(n):
    return [torch.Generator().manual_seed(i) for i in range(n)]


@pytest.mark.parametrize("draft", ["mtp-top", "mtp-tail", "dflash2"])
def test_verify_batch_law(draft):
    n = 40_000
    torch.manual_seed(1)
    logits = P.log().expand(n, 2, 4).contiguous()
    for s in SOFTS:
        if draft == "dflash2":
            tokens = torch.multinomial(Q, n, replacement=True).view(n, 1)
            q, support, law = Q.expand(n, 1, 4).contiguous(), torch.arange(4).expand(n, 1, 4).contiguous(), soft_law(s)
        else:
            y = 0 if draft == "mtp-top" else 2
            tokens, q, support, law = torch.full((n, 1), y), None, None, soft_law(s, y)
        gens = [torch.Generator().manual_seed(10 * n + i) for i in range(n)]
        accepted, correction, flags, _ = verify_batch(tokens, logits, [params(s)] * n, gens, q, support)
        got = torch.bincount(first_tokens(tokens, accepted, correction), minlength=4).float() / n
        assert (got - law).abs().max() < 0.012, (draft, s, got, law)
        assert (got <= math.exp(s) * P + 0.012).all()


@pytest.mark.parametrize("draft", ["mtp", "dflash2"])
def test_verify_rows_law(draft):
    n = 12_000
    torch.manual_seed(2)
    p = P.expand(2, 4).contiguous()
    for s in (0.0, 0.25, 1.0):
        counts = torch.zeros(4)
        for i in range(n):
            g = torch.Generator().manual_seed(i)
            if draft == "dflash2":
                tokens, q, support = torch.multinomial(Q, 1, generator=g), Q.view(1, 4), torch.arange(4).view(1, 4)
            else:
                tokens, q, support = torch.tensor([1]), None, None
            a, c, _ = verify_rows(tokens, p, q, support, g, soft_scale=params(s).soft_scale)
            counts[int(tokens[0]) if int(a) > 0 else int(c)] += 1
        law = soft_law(s) if draft == "dflash2" else soft_law(s, 1)
        assert (counts / n - law).abs().max() < 0.02, (draft, s, counts / n, law)


def test_soft_zero_is_exact():
    assert torch.allclose(soft_law(0.0), P) and torch.allclose(soft_law(0.0, 1), P)
    assert params(0.0).soft_scale == 1.0 and SamplingParams().accept_soft == 0.0
    assert not params(0.0).relaxed and params(0.5).relaxed


def test_exact_rows_are_unchanged_beside_relaxed_ones():
    n = 2_000
    tokens, logits, q, support = dflash2_rows(n, 4)
    alone = verify_batch(tokens, logits, [params(0.0)] * n, generators(n), q, support)
    mixed = verify_batch(tokens, logits, [params(0.0 if i % 2 else 0.5) for i in range(n)], generators(n), q, support)
    for a, b in zip(alone[:3], mixed[:3]):
        assert torch.equal(a[1::2], b[1::2])


def test_flags_mark_only_the_boosted_accepts():
    n = 20_000
    tokens, logits, q, support = dflash2_rows(n, 3)
    exact = verify_batch(tokens, logits, [params(0.0)] * n, generators(n), q, support)
    soft = verify_batch(tokens, logits, [params(0.25)] * n, generators(n), q, support)
    assert not exact[2].any()  # s = 0 boosts nothing
    # the same uniforms: an accept at s = 0 stays an accept, and the flags are the accepts s = 0.25 adds
    assert torch.equal(soft[2][:, 0], (soft[0] > 0) & (exact[0] == 0))
    assert soft[2].any()


def test_sampled_rows_ignore_top_k_and_margin():
    n = 2_000
    tokens, logits, q, support = dflash2_rows(n, 5)
    for s in (0.0, 0.5):
        plain = verify_batch(tokens, logits, [params(s)] * n, generators(n), q, support)
        km = verify_batch(tokens, logits, [params(s, accept_top_k=2, accept_margin=1.0)] * n, generators(n), q, support)
        assert all(torch.equal(a, b) for a, b in zip(plain[:3], km[:3]))
    assert not params(0.0, accept_top_k=3, accept_margin=2.0).relaxed


def test_greedy_rows_ignore_accept_soft():
    tokens = torch.tensor([[1]])
    logits = P.log().view(1, 1, 4).expand(1, 2, 4).contiguous()
    greedy = SamplingParams(temperature=0.0, accept_soft=0.5)
    assert not greedy.relaxed
    a, c, flags, _ = verify_batch(tokens, logits, [greedy], [torch.Generator().manual_seed(0)])
    assert int(a[0]) == 0 and int(c[0]) == 0 and not flags.any()


def test_validation_and_server_default():
    SamplingParams(accept_soft=0.0).validate()
    SamplingParams(accept_soft=2).validate()
    for bad in (math.inf, -0.1, float("nan"), "1", True):
        with pytest.raises(InvalidRequest, match="finite number >= 0"):
            SamplingParams(accept_soft=bad).validate()
    with pytest.raises(InvalidRequest):
        _sampling_from({"temperature": 1.0, "accept_soft": math.inf})
    with pytest.raises(InvalidRequest):
        validation.set_accept_default(1, 0.0, soft=math.inf)
    assert _sampling_from({"temperature": 1.0}).accept_soft == 0.0
    assert _sampling_from({"temperature": 1.0, "accept_soft": 0.25}).accept_soft == 0.25
    assert _sampling_from({"sampling_params": {"temperature": 1.0, "accept_soft": 0.5}}).accept_soft == 0.5
    try:
        validation.set_accept_default(1, 0.0, soft=0.3)
        assert _sampling_from({"temperature": 1.0}).relaxed and _sampling_from({"temperature": 1.0}).accept_soft == 0.3
        assert not _sampling_from({"temperature": 1.0, "accept_soft": 0}).relaxed
    finally:
        validation.set_accept_default(1, 0.0)
    assert validation.ACCEPT_SOFT_DEFAULT == 0.0
