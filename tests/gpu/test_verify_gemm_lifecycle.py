"""Compare complete k=15 rounds and committed state against the old kernels."""
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get('SHARAR_RUN_GPU') != '1', reason='opt-in GB10 tests')


@pytest.fixture(scope='module')
def runtime():
    from sharar import config
    from sharar.engine.executor import Executor
    from sharar.engine.spec_graphs import VerifyGraphs
    from sharar.kernels import ops
    from sharar.model.qwen38 import Qwen38
    from sharar.tokenizer import SnapshotTokenizer
    with torch.inference_mode():
        original = ops.VERIFY_GEMM_ENABLED
        try:
            ops.VERIFY_GEMM_ENABLED = False
            ex = Executor(Qwen38(config.snapshot_dir()), num_pages=35000, num_slots=12)
            ex.enable_spec(15, proposer='dflash2', state_strategy='reconstruct', draft_precision='fp8')
            ex.capture_graphs()
            graphs = {False: ex.spec.verify_graphs}
            ops.VERIFY_GEMM_ENABLED = True
            graphs[True] = VerifyGraphs(ex.spec)
            graphs[True].capture()
            ex.warmup()
            tok = SnapshotTokenizer(config.snapshot_dir())
            ids = tok.encode('Implement a Python ring buffer with append and pop methods. Explain the invariants.\n')
            yield ex, graphs, ids
            ex.flush_cache()
        finally:
            ops.VERIFY_GEMM_ENABLED = original


@pytest.mark.parametrize('batch,stop', [(1, 'length'), (1, 'eos'), (3, 'length'), (4, 'length')])
@torch.inference_mode()
def test_outputs_and_committed_state_are_exact(runtime, monkeypatch, batch, stop):
    from sharar.engine.sampler import SamplingParams
    from sharar.kernels import ops
    ex, graphs, ids = runtime
    reference = []
    for enabled in (False, True):
        monkeypatch.setattr(ops, 'VERIFY_GEMM_ENABLED', enabled)
        ex.spec.verify_graphs = graphs[enabled]
        ex.spec.width_graphs[15] = graphs[enabled]
        ex.flush_cache()
        reqs = [ex.admit(ids + [100 + i], SamplingParams(max_new_tokens=17 + i, ignore_eos=stop == 'length'))
                for i in range(batch)]
        original_accept = ex.spec._accept
        def accept(rs, y, z, *args, **kwargs):
            if stop == 'eos':
                # z[:,0] is either the first accepted proposal or the correction.
                for i, r in enumerate(rs):
                    r.params.stop_token_ids = (int(z[i, 0]),)
            return original_accept(rs, y, z, *args, **kwargs)
        monkeypatch.setattr(ex.spec, '_accept', accept)
        try:
            for req in reqs:
                ex.prefill(req)
            step = 0
            while active := [r for r in reqs if not r.finished]:
                ex.decode_step(active)
                state = []
                for r in active:
                    assert r.committed_len == r.context_len - 1 == len(r.pages)
                    state.append((r.output_ids[:], r.finish_reason,
                                  ex.state_pool.ssm_all[:, r.slot].clone(),
                                  ex.state_pool.conv_all[:, r.slot].clone()))
                if not enabled:
                    reference.append(state)
                else:
                    assert step < len(reference) and len(state) == len(reference[step])
                    for got, wanted in zip(state, reference[step]):
                        assert got[:2] == wanted[:2]
                        assert torch.equal(got[2], wanted[2])
                        assert torch.equal(got[3], wanted[3])
                step += 1
            assert step == len(reference)
            assert all(r.finish_reason == ('stop' if stop == 'eos' else 'length') for r in reqs)
        finally:
            monkeypatch.setattr(ex.spec, '_accept', original_accept)
            for req in reqs:
                req.cacheable = False
                ex.release(req)
            ex.flush_cache()
        assert ex.kv_pool.free_count == ex.kv_pool.num_pages - 1
        assert ex.state_pool.free_count == ex.state_pool.num_slots - 1
        assert len(ex.spec.proposer.free_slots) == ex.max_batch
