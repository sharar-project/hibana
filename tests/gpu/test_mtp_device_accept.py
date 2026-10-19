"""Device-resident acceptance with the MTP proposer: the captured pass commits the target state
bitwise like the host round and the draft KV to rounding of the whole-bucket projection."""
import gc

import pytest
import torch

from sharar.engine.sampler import SamplingParams


@pytest.fixture(scope='module')
def runtime(qwen38):
    from sharar import config
    from sharar.engine.executor import Executor
    from sharar.engine.prompts import PARAGRAPH
    from sharar.tokenizer import SnapshotTokenizer
    gc.collect()  # the previous module's executor
    torch.cuda.empty_cache()
    with torch.inference_mode():
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


def run_rounds(ex, prompts, params, rounds, device_accept):
    ex.spec.device_accept = device_accept
    ex.flush_cache()
    reqs = [ex.admit(ids, p) for ids, p in zip(prompts, params)]
    for req in reqs:
        ex.prefill(req)
    active = [r for r in reqs if not r.finished]
    chunks = []
    for _ in range(rounds):
        if not active:
            break
        before = [len(r.output_ids) for r in reqs]
        ex.decode_step(active)
        chunks.append([len(r.output_ids) - b for r, b in zip(reqs, before)])
        active = [r for r in active if not r.finished]
    torch.cuda.synchronize()
    outputs = [r.output_ids[:] for r in reqs]
    states = [(ex.state_pool.ssm_all[:, r.slot].clone(), ex.state_pool.conv_all[:, r.slot].clone()) for r in reqs]
    finished = [(r.finished, r.finish_reason, r.committed_len) for r in reqs]
    pending = [(r.proposer_state.draft_pos, r.proposer_state.pending_token) for r in reqs if r.proposer_state is not None]
    for r in reqs:
        r.cacheable = False
        ex.release(r)
    return outputs, states, finished, chunks, pending


@pytest.mark.parametrize('batch', [1, 3, 4])
@pytest.mark.parametrize('rounds', [1, 8])
def test_mtp_device_round_matches_host_round(runtime, batch, rounds):
    """The first round drafts from the prefill's draft KV on both paths, so it must agree exactly: same
    tokens, bitwise the same committed recurrent state. Later rounds draft from the fed-back KV, which the
    device pass projects over every row of the bucket (the host path per request), so proposals and
    round boundaries may move; outputs are compared up to the first round whose counts differ."""
    ex, prompts = runtime
    assert hasattr(ex.spec.verify_graphs, 'accept'), 'MTP device accept graphs were not captured'
    params = [SamplingParams(max_new_tokens=40, ignore_eos=True) for _ in range(batch)]
    host = run_rounds(ex, prompts[:batch], params, rounds, device_accept=False)
    device = run_rounds(ex, prompts[:batch], params, rounds, device_accept=True)
    agree = next((j for j, (a, b) in enumerate(zip(device[3], host[3])) if a != b), min(len(device[3]), len(host[3])))
    assert agree >= 1, (device[3], host[3])
    for i, (out_d, out_h) in enumerate(zip(device[0], host[0])):
        n = 1 + sum(c[i] for c in host[3][:agree])
        assert out_d[:n] == out_h[:n], (batch, rounds, i, agree)
    if device[3] == host[3]:
        assert device[0] == host[0] and device[4] == host[4]
        for (ssm_d, conv_d), (ssm_h, conv_h) in zip(device[1], host[1]):
            assert torch.equal(ssm_d, ssm_h) and torch.equal(conv_d, conv_h)


def test_mtp_stops_and_budget_cut_identically(runtime):
    ex, prompts = runtime
    probe = run_rounds(ex, prompts[:2], [SamplingParams(max_new_tokens=40, ignore_eos=True)] * 2, rounds=1, device_accept=False)
    stop_a = probe[0][0][1]  # the first token the first round emits (the round both paths draft identically)
    params = [SamplingParams(max_new_tokens=40, stop_token_ids=(stop_a,)), SamplingParams(max_new_tokens=2, ignore_eos=True)]
    host = run_rounds(ex, prompts[:2], params, rounds=1, device_accept=False)
    device = run_rounds(ex, prompts[:2], params, rounds=1, device_accept=True)
    assert host[2][0][:2] == (True, 'stop') and host[2][1][:2] == (True, 'length')
    assert device[0] == host[0] and device[2] == host[2]


def test_mtp_pools_are_clean_after_device_rounds(runtime):
    ex, prompts = runtime
    run_rounds(ex, prompts[:4], [SamplingParams(max_new_tokens=20, ignore_eos=True)] * 4, rounds=6, device_accept=True)
    assert ex.kv_pool.free_count == 35000 - 1 and ex.state_pool.free_count == 12 - 1


@pytest.mark.parametrize('batch', [1, 3])
@pytest.mark.parametrize('rounds', [1, 8])
def test_mtp_sampled_device_round_matches_host_round(runtime, batch, rounds):
    """Sampled requests: the device path (decision and commit graphs) and the host path decide with the same
    kernel at the same Philox offsets, so the first round agrees exactly (tokens and committed state) and later
    rounds up to the first whose counts differ (the device pass's draft feedback rounds differently, as greedy)."""
    ex, prompts = runtime
    params = [SamplingParams(max_new_tokens=40, ignore_eos=True, temperature=1.0, top_k=20, top_p=0.95, seed=100 + i) for i in range(batch)]
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


def test_mtp_sampled_rows_wider_than_the_candidates_take_the_host_path(runtime):
    """top_k above SAMPLED_TOP_K is not eligible for the device path, and the round still completes on the host."""
    ex, prompts = runtime
    params = [SamplingParams(max_new_tokens=12, ignore_eos=True, temperature=1.0, top_k=200, top_p=0.95, seed=7)]
    out = run_rounds(ex, prompts[:1], params, rounds=3, device_accept=True)
    assert len(out[0][0]) > 1
