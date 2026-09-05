"""Real DFlash2 integration: conditional selection, cache reuse and mixed laws."""
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get('SHARAR_RUN_GPU') != '1', reason='opt-in GB10 checkpoint tests')


@pytest.fixture(scope='module')
def draft_runtime():
    import gc
    from sharar import config
    from sharar.engine.executor import Executor
    from sharar.model.qwen38 import Qwen38
    from sharar.parity.prompts import PARAGRAPH
    from sharar.tokenizer import SnapshotTokenizer
    gc.collect()
    torch.cuda.empty_cache()
    with torch.inference_mode():
        model = Qwen38(config.snapshot_dir(), log=print)
        torch.cuda.empty_cache()
        ex = Executor(model, num_pages=35000, num_slots=12)
        ex.enable_spec(7, proposer='dflash2', state_strategy='reconstruct')
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
