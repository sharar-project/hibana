"""Sampled MTP drafts on a served target (SHARAR_MTP_SAMPLE): sampled batches propose the draft head's law on its
candidates, the device acceptance graph decides on that law as the host path does, and greedy batches keep argmax
drafts."""
import dataclasses
import gc

import pytest
import torch

from sharar.engine import config as engine_config
from sharar.engine.sampler import SamplingParams
from tests.gpu.test_mtp_device_accept import run_rounds

SAMPLED = dict(max_new_tokens=40, ignore_eos=True, temperature=1.0, top_k=20, top_p=0.95)


@pytest.fixture(scope='module')
def runtime(qwen38):
    from sharar import config
    from sharar.engine.executor import Executor
    from sharar.engine.prompts import PARAGRAPH
    from sharar.tokenizer import SnapshotTokenizer
    gc.collect()
    torch.cuda.empty_cache()
    sampled_cfg = dataclasses.replace(engine_config.current(), mtp_sample=True)
    with engine_config.using(sampled_cfg), torch.inference_mode():
        ex = Executor(qwen38, num_pages=35000, num_slots=12)
        ex.enable_spec(3, proposer='mtp', state_strategy='reconstruct')
        ex.capture_graphs(log=print)
        ex.warmup(log=print)
        tok = SnapshotTokenizer(config.snapshot_dir())
        prompts = [tok.encode(PARAGRAPH * n)[:length] for n, length in ((8, 300), (6, 220), (4, 150), (3, 90))]
        yield ex, prompts
        ex.flush_cache()
        del ex
        gc.collect()
        torch.cuda.empty_cache()


def _propose(ex, prompts, params):
    """The first round's proposal (recorded from the proposer while the round runs as usual)."""
    seen = []
    propose = ex.spec.proposer.propose

    def record(*args, **kwargs):
        seen.append(propose(*args, **kwargs))
        return seen[-1]
    ex.spec.proposer.propose = record
    try:
        run_rounds(ex, prompts, params, 1, device_accept=False)
    finally:
        ex.spec.proposer.propose = propose
    return seen[0]


def test_sampled_batches_propose_a_law_and_greedy_batches_argmax(runtime):
    ex, prompts = runtime
    assert ex.spec.proposer.sample and ex.spec.verify_graphs.accept.q_width == 16
    sampled = _propose(ex, prompts[:3], [SamplingParams(**SAMPLED, seed=1 + i) for i in range(3)])
    b, k = sampled.tokens.shape
    assert sampled.probabilities.shape == sampled.support.shape == (b, k, 16)
    assert torch.allclose(sampled.probabilities.sum(-1), torch.ones(b, k, device=sampled.tokens.device), atol=1e-4)
    q_y = (sampled.probabilities * (sampled.support == sampled.tokens.unsqueeze(-1))).sum(-1)
    assert (q_y > 0).all()  # every draft is on its candidates with positive probability
    greedy = _propose(ex, prompts[:3], [SamplingParams(max_new_tokens=40, ignore_eos=True) for _ in range(3)])
    assert greedy.probabilities is None and greedy.support is None


@pytest.mark.parametrize('batch', [1, 3])
@pytest.mark.parametrize('rounds', [1, 8])
def test_sampled_draft_device_round_matches_host_round(runtime, batch, rounds):
    """As test_mtp_device_accept's sampled test, with sampled drafts: both paths draw the same drafts (Philox at the
    draft offsets) and decide on the same law with the same kernel at the verify offsets, so the first round agrees
    exactly and later rounds up to the first whose counts differ."""
    ex, prompts = runtime
    params = [SamplingParams(**SAMPLED, seed=100 + i) for i in range(batch)]
    host = run_rounds(ex, prompts[:batch], params, rounds, device_accept=False)
    device = run_rounds(ex, prompts[:batch], params, rounds, device_accept=True)
    agree = next((j for j, (a, b) in enumerate(zip(device[3], host[3])) if a != b), min(len(device[3]), len(host[3])))
    assert agree >= 1, (device[3], host[3])
    for i, (out_d, out_h) in enumerate(zip(device[0], host[0])):
        n = 1 + sum(c[i] for c in host[3][:agree])
        assert out_d[:n] == out_h[:n], (batch, rounds, i, agree)
    if rounds == 1:
        for (ssm_d, conv_d), (ssm_h, conv_h) in zip(device[1], host[1]):
            assert torch.equal(ssm_d, ssm_h) and torch.equal(conv_d, conv_h)


def test_a_seeded_sampled_request_is_reproducible(runtime):
    ex, prompts = runtime
    params = [SamplingParams(**SAMPLED, seed=42)]
    first = run_rounds(ex, prompts[:1], params, 6, device_accept=True)
    again = run_rounds(ex, prompts[:1], params, 6, device_accept=True)
    assert first[0] == again[0] and first[3] == again[3]
