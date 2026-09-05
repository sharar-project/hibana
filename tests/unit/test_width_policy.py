from types import SimpleNamespace

import pytest

from sharar import config
from sharar.engine.sampler import SamplingParams
from sharar.engine.width_policy import WidthPolicy
from sharar.proposers.parallel import PINS
from sharar.proposers.ngram import NGramProposer
from sharar.engine.contracts import ProposerContext
from tests.unit.test_runtime_safety import tiny_executor


def test_calibration_does_not_label_unverified_tail_as_rejections():
    from bench.calibrate_width import fit
    result = fit(dict(name='dspark', passed=True, target=config.MODEL_REVISION,
                      draft_pin=PINS['dspark'], width_costs=[dict(batch=1, depth=3, context=177, round_ms=120)],
                      acceptance_trace=[dict(sampled=False, accepted=0, verified_depth=3, confidence=[.5] * 3)]))
    assert result['observations']['greedy'][5] == [0, 1]
    assert 'sampled' not in result['calibration']


def profile():
    return dict(hardware='GB10', target_revision=config.MODEL_REVISION,
                draft_revision=PINS['dspark'][1], state_strategy='reconstruct',
                round_ms={'1': {'1': 100., '3': 120., '7': 160.}},
                calibration={'greedy': [.1, .95], 'sampled': [.05, .5]})


def test_width_uses_measured_cost_confidence_and_remaining_budget():
    policy = WidthPolicy(profile(), 'dspark', 'reconstruct')
    req = SimpleNamespace(params=SamplingParams(max_new_tokens=100), output_ids=[1],
                          proposer_state=SimpleNamespace(confidence=None))
    assert policy.choose([req], (1, 3, 7)) == 7
    req.proposer_state.confidence = [.1] * 7
    assert policy.choose([req], (1, 3, 7)) == 1
    req.proposer_state.confidence = [.9] * 7
    assert policy.choose([req], (1, 3, 7)) == 7
    req.output_ids = [1] * 99
    assert policy.choose([req], (1, 3, 7)) == 1


@pytest.mark.parametrize('field,value', [('hardware', 'other'), ('target_revision', 'old'),
                                        ('draft_revision', 'old'), ('state_strategy', 'snapshots'),
                                        ('round_ms', {'1': {'1': float('nan')}})])
def test_width_rejects_foreign_or_invalid_profiles(field, value):
    p = profile()
    p[field] = value
    with pytest.raises(ValueError):
        WidthPolicy(p, 'dspark', 'reconstruct')


@pytest.mark.parametrize('depth', [1, 3])
@pytest.mark.parametrize('budget', [2, 4, 8])
def test_short_verify_releases_entire_unverified_reservation(monkeypatch, depth, budget):
    ex = tiny_executor(monkeypatch)
    expected = ex.generate([7, 7, 7], SamplingParams(max_new_tokens=budget)).output_ids
    ex.set_proposer(NGramProposer(ProposerContext(ex.device, 4, 512), k=3))
    ex.spec.verify_depths = (1, 3)
    ex.spec.verify_depth = depth
    requests = ex.generate_batch([([7, 7, 7], SamplingParams(max_new_tokens=budget)) for _ in range(4)])
    assert all(r.output_ids == expected and r.committed_len == r.context_len - 1 for r in requests)
    assert ex.kv_pool.free_count == 1024 and ex.state_pool.free_count == 15
