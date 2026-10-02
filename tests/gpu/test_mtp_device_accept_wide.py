"""Device-resident acceptance with the MTP proposer in a wide bucket at a depth below k (#59 with #36): the
throughput profile's buckets 8/16 verify fewer drafts; their accept graphs must commit like the host round."""
import gc

import pytest
import torch

from sharar.engine.sampler import SamplingParams
from tests.gpu.test_mtp_device_accept import run_rounds


@pytest.fixture(scope='module')
def runtime():
    from sharar import config
    from sharar.engine.config import current, using
    from sharar.engine.executor import Executor
    from sharar.model.qwen38 import Qwen38
    from sharar.engine.prompts import PARAGRAPH
    from sharar.tokenizer import SnapshotTokenizer
    gc.collect()
    torch.cuda.empty_cache()
    with torch.inference_mode(), using(current().replace(spec_depths="8:2")):
        model = Qwen38(config.snapshot_dir(), log=print)
        ex = Executor(model, num_pages=35000, num_slots=24, max_batch=8)
        ex.enable_spec(3, proposer='mtp', state_strategy='reconstruct')
        ex.capture_graphs(log=print)
        ex.warmup(log=print)
        tok = SnapshotTokenizer(config.snapshot_dir())
        prompts = [tok.encode(PARAGRAPH * 8)[i * 17:i * 17 + length] for i, length in enumerate((300, 260, 220, 190, 150, 120, 90, 70))]
        yield ex, prompts
        ex.flush_cache()
        del ex, model
        gc.collect()
        torch.cuda.empty_cache()


def test_wide_bucket_captures_its_own_depth(runtime):
    ex, _ = runtime
    assert ex.spec.verify_depths == (2, 3)
    assert ex.spec.width_graphs[3].sizes == (1, 2, 4) and ex.spec.width_graphs[2].sizes == (8,)
    assert all(hasattr(g, 'accept') for g in ex.spec.width_graphs.values()), 'MTP device accept graphs were not captured'


@pytest.mark.parametrize('batch', [5, 8])
@pytest.mark.parametrize('rounds', [1, 6])
def test_wide_bucket_device_round_matches_host_round(runtime, batch, rounds):
    """As test_mtp_device_accept: the first round agrees exactly (tokens and committed state); later rounds
    until the first round whose counts differ."""
    ex, prompts = runtime
    params = [SamplingParams(max_new_tokens=40, ignore_eos=True) for _ in range(batch)]
    host = run_rounds(ex, prompts[:batch], params, rounds, device_accept=False)
    device = run_rounds(ex, prompts[:batch], params, rounds, device_accept=True)
    assert all(max(c) <= 3 for c in host[3]), 'a round at bucket 8 verifies two drafts: at most three tokens'
    agree = next((j for j, (a, b) in enumerate(zip(device[3], host[3])) if a != b), min(len(device[3]), len(host[3])))
    assert agree >= 1, (device[3], host[3])
    for i, (out_d, out_h) in enumerate(zip(device[0], host[0])):
        n = 1 + sum(c[i] for c in host[3][:agree])
        assert out_d[:n] == out_h[:n], (batch, rounds, i, agree)
    if device[3] == host[3]:
        assert device[0] == host[0] and device[4] == host[4]
        for (ssm_d, conv_d), (ssm_h, conv_h) in zip(device[1], host[1]):
            assert torch.equal(ssm_d, ssm_h) and torch.equal(conv_d, conv_h)


def test_wide_bucket_pools_are_clean_after_device_rounds(runtime):
    ex, prompts = runtime
    run_rounds(ex, prompts, [SamplingParams(max_new_tokens=20, ignore_eos=True)] * 8, rounds=6, device_accept=True)
    assert ex.kv_pool.free_count == 35000 - 1 and ex.state_pool.free_count == 24 - 1
