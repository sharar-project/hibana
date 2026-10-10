"""Pinned external drafts: FP32 projection oracle and real commit/cache checks."""
import gc
import json
from types import SimpleNamespace

import pytest
import torch

from sharar.engine.contracts import VerifiedFeatures
from sharar.engine.sampler import SamplingParams
from sharar.proposers.parallel import DraftState



@pytest.fixture(scope='module', params=['dflash2', 'dspark'])
def runtime(request):
    from sharar import config
    from sharar.proposers.factory import draft_snapshot
    from sharar.proposers.parallel import PINS
    if not (draft_snapshot(request.param) / 'model.safetensors').is_file():
        name, revision = PINS[request.param]
        pytest.skip(f'{request.param} draft weights not in the model cache: {name}@{revision[:7]}')
    from sharar.engine.executor import Executor
    from sharar.model.qwen38 import Qwen38
    from sharar.engine.prompts import PARAGRAPH
    from sharar.tokenizer import SnapshotTokenizer
    gc.collect()
    torch.cuda.empty_cache()
    with torch.inference_mode():
        model = Qwen38(config.snapshot_dir(), log=print)
        torch.cuda.empty_cache()
        ex = Executor(model, num_pages=35000, num_slots=12)
        ex.enable_spec(7, proposer=request.param, state_strategy='reconstruct')
        ex.capture_graphs(log=print)
        ids = SnapshotTokenizer(config.snapshot_dir()).encode(PARAGRAPH * 16)[:512]
        captured, original = {}, model.forward
        def forward(*args):
            out = original(*args)
            captured.update({lid: value[-32:].clone().view(4, 8, -1) for lid, value in out['features'].items()})
            return out
        model.forward = forward
        req = ex.admit(ids, SamplingParams(max_new_tokens=1))
        ex.prefill(req)
        model.forward = original
        ex.release(req)
        ex.flush_cache()
        yield ex, ids, captured
        ex.flush_cache()


def errors(actual, expected):
    actual, expected = actual.float(), expected.float()
    delta = actual - expected
    result = {'relative_rms': (delta.square().mean() / expected.square().mean().clamp_min(1e-30)).sqrt().item(),
              'max_abs': delta.abs().max().item(),
              'cosine': torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item()}
    # Frozen before the experiment: a backend numerical gate, not model quality.
    assert result['relative_rms'] < .01 and result['cosine'] > .9999, result
    return result


@pytest.fixture
def checkpoint_slots(runtime):
    pool = runtime[0].state_pool
    slots = [pool.alloc() for _ in range(4)]
    try:
        yield slots
    finally:
        for slot in slots:
            pool.free(slot)


@pytest.mark.parametrize('batch', [1, 2, 3, 4])
@torch.inference_mode()
def test_real_projections_against_fp32_oracle_and_private_ring_boundaries(runtime, batch, tmp_path, checkpoint_slots):
    from sharar.model.parallel_draft import rms, rotate
    ex, _, source = runtime
    p, model = ex.spec.proposer, ex.spec.proposer.model
    counts, starts, slots = [1, 3, 5, 8][:batch], [2047, 2046, 2045, 2044][:batch], [4, 1, 3, 2][:batch]
    layers = {lid: value[:batch].clone() for lid, value in source.items()}
    for value in layers.values():
        for row, count in enumerate(counts):
            value[row, count:] = float('nan')  # verify padding must never enter KV
    joined = torch.cat([torch.cat([layers[lid][row, :count] for row, count in enumerate(counts)]) for lid in model.features], -1)
    positions = torch.tensor([pos for start, count in zip(starts, counts) for pos in range(start, start + count)], device='cuda')
    pages = torch.tensor([slot * p.pitch + pos % p.window for slot, start, count in zip(slots, starts, counts)
                          for pos in range(start, start + count)], device='cuda')
    feature_rows = torch.tensor([slot * p.window + pos % p.window for slot, start, count in zip(slots, starts, counts)
                                 for pos in range(start, start + count)], device='cuda')
    guards = torch.tensor([slot * p.pitch + pos % p.window for slot, start, count in zip(slots, starts, counts)
                           for pos in (start - 1, start + count)], device='cuda')
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        expected_hidden = rms((joined.float() @ model.weights['fc.weight'].float().T).bfloat16(), model.weights['hidden_norm.weight'])
        expected_kv = []
        for layer in range(5):
            prefix = f'layers.{layer}.self_attn.'
            k, v = (expected_hidden.float() @ model.weights[prefix + 'kv'].float().T).bfloat16().chunk(2, -1)
            k = rms(k.view(-1, 8, 128), model.weights[prefix + 'k_norm.weight'])
            expected_kv.append((rotate(k, positions, model.cos, model.sin), v.view(-1, 8, 128)))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    report = {'draft': p.name, 'batch': batch, 'counts': counts, 'fp32_oracle': {}, 'control_vs_batch': {}}
    results = []
    for batched in (False, True):
        reqs = []
        for row, (slot, start, count) in enumerate(zip(slots, starts, counts)):
            p.features[slot, :min(p.window, start + count)].fill_(.25)
            reqs.append(SimpleNamespace(proposer_state=DraftState(slot, end=start, pending=77),
                        committed_len=start + count, finished=row % 2 == 0, cache_enabled=True,
                        checkpoints={start + count: checkpoint_slots[row]}, cache_auxiliary={}))
        for keys, values in p.caches:
            keys[guards] = -37
            values[guards] = 37
        emitted = [[100 + row] * count for row, count in enumerate(counts)]
        features = VerifiedFeatures(layers=layers)
        if batched:
            p.accepted_batch(reqs, emitted, features)
        else:
            for row, (req, tokens) in enumerate(zip(reqs, emitted)):
                p._accepted(req, tokens, features, row)
        hidden = p.features.view(-1, p.features.shape[-1])[feature_rows].clone()
        kv = [(keys[pages, 0].clone(), values[pages, 0].clone()) for keys, values in p.caches]
        report['fp32_oracle']['batch' if batched else 'control'] = {
            'features': errors(hidden, expected_hidden),
            'kv': [errors(value, expected) for pair, oracle in zip(kv, expected_kv) for value, expected in zip(pair, oracle)]}
        for keys, values in p.caches:
            assert (keys[guards] == -37).all() and (values[guards] == 37).all()
        for req, tokens in zip(reqs, emitted):
            assert req.proposer_state.end == req.committed_len
            assert req.proposer_state.pending == (None if req.finished else tokens[-1])
            handle = req.cache_auxiliary[req.committed_len][p.cache_identity]
            start, end, retained = p.feature_cache.get(handle)
            ring = torch.arange(start, end, device='cuda') % p.window
            assert torch.equal(retained, p.features[req.proposer_state.slot, ring])
        results.append((hidden, kv))
    report['control_vs_batch'] = {'features': errors(results[1][0], results[0][0]),
        'kv': [errors(value, expected) for pair, oracle in zip(results[1][1], results[0][1]) for value, expected in zip(pair, oracle)]}
    if batch == 1:
        assert torch.equal(results[1][0], results[0][0])
        assert all(torch.equal(x, y) for pair, oracle in zip(results[1][1], results[0][1]) for x, y in zip(pair, oracle))
    (tmp_path / f'{p.name}-b{batch}-numerics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


@pytest.mark.parametrize('graphs', [False, True])
@pytest.mark.parametrize('mixed', [False, True])
@torch.inference_mode()
def test_real_feedback_stops_cache_and_mixed_sampling(runtime, monkeypatch, graphs, mixed):
    ex, ids, _ = runtime
    p = ex.spec.proposer
    ex.use_graphs = graphs
    original = p.accepted_batch
    batches = []
    def checked(reqs, emitted, features):
        assert all(r.reservation is None and r.committed_len == r.context_len - 1 for r in reqs)
        original(reqs, emitted, features)
        assert all(r.proposer_state.end == r.committed_len for r in reqs)
        assert all(r.proposer_state.pending == (None if r.finished else tokens[-1]) for r, tokens in zip(reqs, emitted))
        batches.append(len(reqs))
    monkeypatch.setattr(p, 'accepted_batch', checked)
    if not mixed:
        accept = ex.spec._accept
        def stop_inside(reqs, y, z, *args, **kwargs):
            for row, req in enumerate(reqs):
                draft, target = y[row].tolist(), z[row].tolist()
                count = next((j for j, token in enumerate(draft) if token != target[j]), len(draft))
                choices = draft[:count] + [target[count]]
                req.params.stop_token_ids = (choices[min(row, len(choices) - 1)],)
            return accept(reqs, y, z, *args, **kwargs)
        monkeypatch.setattr(ex.spec, '_accept', stop_inside)
    try:
        ex.flush_cache()
        for repeat in range(2):
            reqs = ex.generate_batch([(ids, SamplingParams(max_new_tokens=5 + row, seed=31,
                                        temperature=.7 if mixed and row == 1 else 0., ignore_eos=mixed)) for row in range(4)])
            assert all(r.released and r.finish_reason == ('length' if mixed else 'stop') for r in reqs)
            if repeat:
                assert all(r.prefix_len > 0 for r in reqs)
        assert 4 in batches and p.cache_fallbacks == 0
        ex.flush_cache()
        assert ex.kv_pool.free_count == ex.kv_pool.num_pages - 1
        assert ex.state_pool.free_count == ex.state_pool.num_slots - 1
    finally:
        ex.flush_cache()
        ex.use_graphs = True
