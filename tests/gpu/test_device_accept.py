"""Device-resident acceptance reproduces the host round: same tokens, same committed state, same stops."""
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
        ex.enable_spec(15, proposer='dflash2', state_strategy='reconstruct', draft_precision='fp8')
        ex.capture_graphs(log=print)
        ex.warmup(log=print)
        tok = SnapshotTokenizer(config.snapshot_dir())
        prompts = [tok.encode(PARAGRAPH * n)[:length] for n, length in ((8, 300), (6, 220), (4, 150), (3, 90))]
        yield ex, prompts
        ex.flush_cache()


def run_rounds(ex, prompts, params, rounds, device_accept):
    """Prefill fresh requests and run `rounds` decode steps; returns outputs and committed recurrent state."""
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
    finished = [(r.finished, r.finish_reason, r.committed_len, r.spec_emitted, chunks) for r in reqs]
    p = ex.spec.proposer
    rings = []
    for r in reqs:  # only the rows the draft wrote: the rest of a ring is uninitialized
        st = r.proposer_state
        pos = torch.arange(max(0, st.end - p.window), st.end, device=p.features.device)
        rings.append(p.features[st.slot, pos % p.window].clone())
    for r in reqs:
        r.cacheable = False
        ex.release(r)
    return outputs, states, finished, rings


@pytest.mark.parametrize('batch', [1, 2, 3, 4])
@pytest.mark.parametrize('relaxed', [False, True])
@pytest.mark.parametrize('rounds', [2, 12])
def test_device_round_matches_host_round(runtime, batch, relaxed, rounds):
    """Same outputs while the rounds agree. The device path projects every
    verify row of the bucket in one GEMM while the host path projects exactly
    the accepted rows, and cuBLAS rounds bf16 differently per row count, so the
    drafter's rings agree to bf16 precision and its proposals, hence the
    per-round counts, can drift after a few rounds. While the counts agree, the
    outputs are identical and the committed recurrent state is bitwise the
    host's; once round boundaries move, the target's own shape sensitivity
    (docs/operations.md#numerical-limits) can change later tokens (seen at 12 rounds,
    batch 3), so outputs are compared up to the first round whose counts differ.
    The first round depends only on the shared prefill and always agrees; the drift
    can start from the second (batch 3 since the CuTe GDN prefill: one
    request accepts 15 drafts on the device, 14 on the host). A relaxed request can also accept a different draft
    with the same count once the drafts drift (the target ranks both within its top-k: seen at 12 rounds, batch 3,
    round 3), so its outputs are compared through its first round only."""
    ex, prompts = runtime
    assert hasattr(ex.spec.verify_graphs, 'accept'), 'device accept graphs were not captured'
    params = [SamplingParams(max_new_tokens=48, ignore_eos=True, accept_top_k=2 if relaxed and i % 2 == 0 else 1,
                             accept_margin=1.0 if relaxed and i % 2 == 0 else 0.0) for i in range(batch)]
    host = run_rounds(ex, prompts[:batch], params, rounds=rounds, device_accept=False)
    device = run_rounds(ex, prompts[:batch], params, rounds=rounds, device_accept=True)
    chunks_d, chunks_h = device[2][0][4], host[2][0][4]
    same_chunks = chunks_d == chunks_h
    agree = next((j for j, (a, b) in enumerate(zip(chunks_d, chunks_h)) if a != b), min(len(chunks_d), len(chunks_h)))
    for i, (out_d, out_h) in enumerate(zip(device[0], host[0])):
        # the prefill token plus the rounds before the first moved boundary; a relaxed request's first round
        n = len(out_h) if same_chunks else 1 + sum(c[i] for c in chunks_h[:agree])
        if params[i].accept_top_k > 1:
            n = min(n, 1 + chunks_h[0][i])
        assert out_d[:n] == out_h[:n], (batch, relaxed, rounds, i, agree)
    assert chunks_d[0] == chunks_h[0], (chunks_d, chunks_h)
    if same_chunks:
        assert [f[:3] for f in device[2]] == [f[:3] for f in host[2]]
        for (ssm_d, conv_d), (ssm_h, conv_h) in zip(device[1], host[1]):
            assert torch.equal(ssm_d, ssm_h) and torch.equal(conv_d, conv_h)
        for ring_d, ring_h in zip(device[3], host[3]):  # 2 bf16 ulps: cuBLAS splits K with bf16 partials at some row counts
            assert torch.allclose(ring_d.float(), ring_h.float(), rtol=2e-2, atol=1e-2)


def test_feedback_device_matches_host_feedback_rows(runtime):
    """Same projection (one GEMM over the bucket's rows), then exactly the accepted rows land in each ring."""
    ex, prompts = runtime
    p = ex.spec.proposer
    s = ex.spec.steps
    torch.manual_seed(3)
    features = {lid: torch.randn((4 * s, 5120), device='cuda', dtype=torch.bfloat16) for lid in p.model.features}
    ring_before = p.features.clone()
    caches_before = [(k.clone(), v.clone()) for k, v in p.caches]
    starts, slots = [100, 2049, 7, 4000], [1, 2, 3, 4]
    hidden_all = p.model.project_features({lid: rows[:4 * s] for lid, rows in features.items()})
    for counts in ([s] * 4, [1, 5, 16, 0], [3, 0, 0, 2]):
        p.features.copy_(ring_before)
        for (k, v), (k0, v0) in zip(p.caches, caches_before):
            k.copy_(k0); v.copy_(v0)
        p.feedback_device(features, torch.tensor(starts, device='cuda'), torch.tensor(slots, device='cuda'),
                          torch.tensor(counts, device='cuda'), s)
        torch.cuda.synchronize()
        device_ring, device_caches = p.features.clone(), [(k.clone(), v.clone()) for k, v in p.caches]
        p.features.copy_(ring_before)
        for (k, v), (k0, v0) in zip(p.caches, caches_before):
            k.copy_(k0); v.copy_(v0)
        # Same 4*s-row projections as the device pass, into scratch pages of the
        # dummy slot 0; then only the accepted rows are copied where they belong.
        positions_all = (torch.tensor(starts, device='cuda')[:, None] + torch.arange(s, device='cuda')[None, :]).reshape(-1)
        scratch = [(k.clone(), v.clone()) for k, v in p.caches]
        p.model.context_kv(hidden_all, positions_all, scratch, torch.arange(4 * s, device='cuda'))
        for row, (slot, start, count) in enumerate(zip(slots, starts, counts)):
            if not count:
                continue
            ring = torch.arange(start, start + count, device='cuda') % p.window
            p.features[slot, ring] = hidden_all[row * s:row * s + count]
            for (k, v), (ks, vs) in zip(p.caches, scratch):
                k[slot * p.pitch + ring] = ks[row * s:row * s + count]
                v[slot * p.pitch + ring] = vs[row * s:row * s + count]
        torch.cuda.synchronize()
        bits = lambda t: t.contiguous().view(torch.int16)  # noqa: E731  bitwise: untouched rows may hold NaN garbage
        for slot in slots:  # private rings and their KV: bit-identical rows, untouched rows untouched
            assert torch.equal(bits(device_ring[slot]), bits(p.features[slot])), (counts, slot)
            lo, hi = slot * p.pitch, (slot + 1) * p.pitch
            for (kd, vd), (kh, vh) in zip(device_caches, p.caches):
                assert torch.equal(bits(kd[lo:hi]), bits(kh[lo:hi])) and torch.equal(bits(vd[lo:hi]), bits(vh[lo:hi])), (counts, slot)
    p.features.copy_(ring_before)
    for (k, v), (k0, v0) in zip(p.caches, caches_before):
        k.copy_(k0); v.copy_(v0)


def test_stop_tokens_and_budget_cut_identically(runtime):
    ex, prompts = runtime
    probe = run_rounds(ex, prompts[:2], [SamplingParams(max_new_tokens=40, ignore_eos=True)] * 2, rounds=10, device_accept=False)
    # Stop on a token each request actually emits, and give the other a tiny budget.
    stop_a = probe[0][0][7]
    params = [SamplingParams(max_new_tokens=40, stop_token_ids=(stop_a,)), SamplingParams(max_new_tokens=5, ignore_eos=True)]
    host = run_rounds(ex, prompts[:2], params, rounds=10, device_accept=False)
    device = run_rounds(ex, prompts[:2], params, rounds=10, device_accept=True)
    assert host[0][0][-1] == stop_a and host[2][0][:2] == (True, 'stop')
    assert host[2][1][:2] == (True, 'length') and len(host[0][1]) == 5
    assert device[0] == host[0] and device[2] == host[2]
    for (ssm_d, conv_d), (ssm_h, conv_h) in zip(device[1], host[1]):
        assert torch.equal(ssm_d, ssm_h) and torch.equal(conv_d, conv_h)


def test_pools_are_clean_after_device_rounds(runtime):
    ex, prompts = runtime
    run_rounds(ex, prompts[:4], [SamplingParams(max_new_tokens=20, ignore_eos=True)] * 4, rounds=6, device_accept=True)
    # the prompts' chunk-end snapshots stay cached (unlocked) after release; nothing else may hold pages or slots
    assert ex.cache.stats()["locks"]["total"] == 0
    ex.flush_cache()
    assert ex.kv_pool.free_count == 35000 - 1 and ex.state_pool.free_count == 12 - 1
    assert ex.spec.stats()['rounds'] > 0
