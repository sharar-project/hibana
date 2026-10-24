"""Real DFlash2 integration: conditional selection, cache reuse and mixed laws."""

import pytest
import torch



@pytest.fixture(scope='module', params=['bf16', 'fp8'])
def draft_runtime(request, qwen38):
    import gc
    from sharar import config
    from sharar.engine.executor import Executor
    from sharar.engine.prompts import PARAGRAPH
    from sharar.tokenizer import SnapshotTokenizer
    gc.collect()  # the other precision's executor
    torch.cuda.empty_cache()
    with torch.inference_mode():
        ex = Executor(qwen38, num_pages=35000, num_slots=12)
        ex.enable_spec(7, proposer='dflash2', state_strategy='reconstruct', draft_precision=request.param)
        ex.spec.verify_depths = (1, 3, 7)
        ex.capture_graphs(log=print)
        ids = SnapshotTokenizer(config.snapshot_dir()).encode(PARAGRAPH * 16)[:512]
        yield ex, ids
        ex.flush_cache()


@pytest.mark.parametrize('graphs', [False, True])
@pytest.mark.parametrize('depth,batch,mixed', [(1, 1, False), (3, 2, False), (7, 3, False), (7, 4, False), (3, 3, True)])
@torch.inference_mode()
def test_selection_preserves_real_outputs_and_cache(draft_runtime, monkeypatch, graphs, depth, batch, mixed):
    from sharar.engine.sampler import SamplingParams
    from sharar.kernels import draft_selection
    ex, ids = draft_runtime
    ex.use_graphs, ex.spec.verify_depth = graphs, depth
    native = draft_selection.greedy_lattice
    calls = []
    def reference(candidates, scores):
        choices, values = candidates.cpu(), scores.float().cpu()
        out = torch.empty(choices.shape[:2], dtype=torch.int64)
        for row in range(len(choices)):
            previous = 0
            for step in range(choices.shape[1]):
                previous = int(values[row, step, previous].argmax())
                out[row, step] = choices[row, step, previous]
        return out.to(candidates.device)
    def checked(candidates, scores):
        result = native(candidates, scores)
        assert torch.equal(result, reference(candidates, scores))
        calls.append(len(candidates))
        return result
    outcomes = []
    try:
        for select in (reference, checked):
            monkeypatch.setattr(draft_selection, 'greedy_lattice', select)
            ex.flush_cache()
            runs = []
            for repeat in range(2):
                reqs = ex.generate_batch([(ids, SamplingParams(max_new_tokens=4 + i, seed=31,
                                          temperature=.7 if mixed and i == 1 else 0., ignore_eos=True))
                                         for i in range(batch)])
                assert all(r.released and len(r.output_ids) == 4 + i for i, r in enumerate(reqs))
                if repeat:
                    assert all(r.prefix_len > 0 for r in reqs)
                runs.append([r.output_ids for r in reqs])
            outcomes.append(runs)
        assert outcomes[0] == outcomes[1]
        if not mixed:
            assert calls
        ex.flush_cache()
        assert ex.kv_pool.free_count == ex.kv_pool.num_pages - 1
        assert ex.state_pool.free_count == ex.state_pool.num_slots - 1
        assert ex.spec.proposer.cache_fallbacks == 0
    finally:
        ex.flush_cache()
        ex.use_graphs = True
        ex.spec.verify_depth = ex.spec.k


@torch.inference_mode()
def test_long_batched_prefill_retains_each_rows_endpoint(draft_runtime):
    from sharar.engine.sampler import SamplingParams
    ex, tape = draft_runtime
    ex.flush_cache()
    reqs = [ex.admit([100 + i] + (tape * 16)[:8189], SamplingParams(max_new_tokens=8, ignore_eos=True))
            for i in range(4)]
    ex.prefill_batch([(r, len(r.prompt_ids)) for r in reqs])
    # each row's last chunk end (8128: the last 64-token grid point before the end), locked in the tree
    assert all(r.lock.node.depth == 8128 for r in reqs)
    endpoints = [r.lock.node.snapshot.slot for r in reqs]
    assert len(set(endpoints)) == 4 and not set(endpoints) & {r.slot for r in reqs}
    while active := [r for r in reqs if not r.finished]:
        ex.decode_step(active)
    for req in reqs:
        assert req.committed_len == 8197 and len(req.output_ids) == 8 and req.spec_rounds > 0
        ex.release(req)
    ex.flush_cache()
    assert ex.kv_pool.free_count == 34999 and ex.state_pool.free_count == 11
    assert len(ex.spec.proposer.free_slots) == 4


@torch.inference_mode()
def test_branch_from_prompt_end_keeps_draft_after_longer_completion(draft_runtime):
    from sharar import config
    from sharar.engine.sampler import SamplingParams
    ex, ids = draft_runtime
    ex.flush_cache()
    p = ex.spec.proposer
    before = p.cache_fallbacks
    seed = ex.generate(ids, SamplingParams(max_new_tokens=32, ignore_eos=True))
    # Diverge immediately after the original prompt, so the completion's newer
    # endpoint cannot conceal a missing feature window at the prompt boundary.
    branch_ids = ids + [(seed.output_ids[0] + 1) % config.VOCAB_SIZE, 42]
    match = ex.cache.match(branch_ids[:-1], touch=False)
    last = (len(ids) - 1) // 64 * 64  # the prompt's last chunk end (updated from main: no prompt-end copy)
    assert match.length == last
    handle = match.snapshot.auxiliary[p.cache_identity]
    assert p.feature_cache.slots[match.snapshot_slot] == handle
    assert p.feature_cache.get(handle)[1] == last
    results = [ex.generate(branch_ids, SamplingParams(max_new_tokens=16, ignore_eos=True)) for _ in range(2)]
    assert all(r.spec_rounds > 0 for r in results)
    assert results[0].output_ids == results[1].output_ids
    assert p.cache_fallbacks == before
    ex.flush_cache()
    assert p.feature_cache.bytes == 0
