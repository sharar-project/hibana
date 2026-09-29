"""Feature ownership follows real target-slot and radix lifecycle operations."""
from types import SimpleNamespace

import pytest
import torch

from sharar.engine.sampler import SamplingParams
from sharar.proposers.parallel import CheckpointFeatureCache, DraftState, ParallelProposer
from tests.unit.test_runtime_safety import tiny_executor


def setup(monkeypatch):
    ex = tiny_executor(monkeypatch, cache=True)
    features = CheckpointFeatureCache(ex.state_pool, 32)
    p = ParallelProposer.__new__(ParallelProposer)
    p.context = SimpleNamespace(device=ex.device)
    p.window, p.cache_identity = 4, 'test-draft'
    p.features = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
    p.feature_cache = features
    return ex, p, features


def test_retention_is_bounded_by_owned_slots_without_independent_eviction(monkeypatch):
    ex, p, features = setup(monkeypatch)
    handles = []
    while ex.state_pool.free_count:
        slot = ex.state_pool.alloc()
        handles.append((slot, features.put(0, 4, p.features[0], slot=slot)))
    assert features.bytes == features.capacity == 15 * 32
    assert all(features.get(handle) is not None for _, handle in handles)
    slot, old = handles[0]
    ex.state_pool.free(slot)
    assert features.get(old) is None and features.bytes == 14 * 32
    assert ex.state_pool.alloc() == slot
    new = features.put(4, 8, p.features[0] + 10, slot=slot)
    assert new != old and features.get(old) is None
    assert features.bytes == features.capacity
    for slot, _ in handles:
        ex.state_pool.free(slot)
    assert features.bytes == 0 and not features.entries


def test_reject_unowned_slot_and_oversize_before_replacing_features(monkeypatch):
    ex, p, features = setup(monkeypatch)
    slot = ex.state_pool.alloc()
    handle = features.put(0, 4, p.features[0], slot=slot)
    for invalid in (None, 0, 2, 16, True):
        with pytest.raises(ValueError, match='owned recurrent slot'):
            features.put(0, 4, p.features[0], slot=invalid)
    with pytest.raises(ValueError, match='reserved window'):
        features.put(0, 5, torch.ones(5, 2), slot=slot)
    assert features.get(handle) is not None and features.bytes == 32


def seed(ex, p, ids):
    req = ex.admit(ids, SamplingParams(max_new_tokens=1))
    ex.prefill(req)
    req.proposer_state = DraftState(0, end=req.committed_len)
    p._save(req, req.committed_len)
    return req


def test_release_moves_endpoint_features_to_committed_slot_and_discards_duplicate(monkeypatch):
    ex, p, features = setup(monkeypatch)
    ids = [1, 2, 3, 4]
    req = seed(ex, p, ids)
    duplicate_slot = req.checkpoints[4]
    old = req.cache_auxiliary[4][p.cache_identity]
    assert features.slots[duplicate_slot] == old
    p.prepare_release(req)
    endpoint = req.cache_auxiliary[4][p.cache_identity]
    assert features.slots[req.slot] == endpoint and endpoint != old
    ex.state.release(req)
    assert features.get(old) is None
    match = ex.cache.match(ids, touch=False)
    assert match.node.auxiliary[p.cache_identity] == endpoint
    assert torch.equal(features.get(endpoint)[2], p.features[0])
    # A duplicate radix insertion frees the incoming slot and its features,
    # preserving the existing snapshot and its own matching payload.
    req = seed(ex, p, ids)
    p.prepare_release(req)
    incoming = req.cache_auxiliary[4][p.cache_identity]
    ex.state.release(req)
    assert features.get(incoming) is None and features.get(endpoint) is not None
    ex.flush_cache()
    assert features.bytes == 0 and not features.entries


@pytest.mark.parametrize('action', ['cancel', 'retract', 'failed'])
def test_unpublished_request_cleanup_releases_features(monkeypatch, action):
    ex, p, features = setup(monkeypatch)
    req = seed(ex, p, [1, 2, 3, 4])
    assert features.bytes
    if action == 'retract':
        ex.state.retract(req)
    else:
        req.cacheable = False
        if action == 'cancel':
            req.cancel()
        ex.state.release(req)
    assert features.bytes == 0 and not features.entries


def test_locked_prefix_keeps_features_through_flush_until_reader_releases(monkeypatch):
    ex, p, features = setup(monkeypatch)
    req = seed(ex, p, [1, 2, 3, 4])
    p.prepare_release(req)
    ex.state.release(req)
    reader = ex.admit([1, 2, 3, 4, 5], SamplingParams(max_new_tokens=1))
    handle = reader.match.node.auxiliary[p.cache_identity]
    ex.flush_cache()
    assert ex.cache.flushing and features.get(handle) is not None
    reader.cacheable = False
    ex.state.release(reader)
    assert not ex.cache.flushing and features.get(handle) is None
    assert features.bytes == 0


def test_interior_snapshot_eviction_drops_its_features_only(monkeypatch):
    ex, p, features = setup(monkeypatch)
    first, last = ex.state_pool.alloc(), ex.state_pool.alloc()
    a = features.put(0, 2, p.features[0, :2], slot=first)
    b = features.put(0, 4, p.features[0], slot=last)
    ex.cache.insert([1, 2, 3, 4], ex.kv_pool.alloc(4), {2: first, 4: last},
                    {2: {p.cache_identity: a}, 4: {p.cache_identity: b}})
    ex.cache.evict(need_slots=14)
    assert features.get(a) is None and features.get(b) is not None
    assert ex.cache.cached_tokens == 4 and features.bytes == 32
    ex.flush_cache()
    assert features.bytes == 0


def test_prospective_budget_accounts_for_each_recurrent_feature_window(monkeypatch, tmp_path):
    import json
    from sharar import config
    from sharar.cache import spec_budget
    from sharar.proposers.parallel import PINS
    snapshot = tmp_path / 'models--target' / 'snapshots' / 'revision'
    name, revision = PINS['dflash2']
    draft = tmp_path / ('models--' + name.replace('/', '--')) / 'snapshots' / revision
    draft.mkdir(parents=True)
    (draft / 'config.json').write_text(json.dumps({'sliding_window': 2048}))
    # monkeypatch.setattr(config, 'snapshot_dir', lambda: snapshot)
    monkeypatch.setattr(type(config.DEFAULT_TARGET), 'snapshot_dir', lambda self, cache_dir=None: snapshot)
    monkeypatch.setattr(spec_budget, 'tensor_bytes', lambda *args: (128, 64))
    spec = spec_budget.SpecConfig('dflash2', 15, 'reconstruct', 'fp8')
    small, _ = spec_budget.budget(spec, 4, 32768, cache_slots=5)
    large, _ = spec_budget.budget(spec, 4, 32768, cache_slots=24)
    assert large - small == 19 * 2048 * 5120 * 2


def test_prompt_endpoint_features_survive_subsequent_decode_and_branch(monkeypatch):
    from sharar.engine.contracts import FeatureRequest
    ex, p, features = setup(monkeypatch)
    req = ex.admit([1, 2, 3, 4], SamplingParams(max_new_tokens=2))
    req.proposer_state = DraftState(0)
    p.requirements = SimpleNamespace(features=FeatureRequest())
    def ingest(req, ids, start, result):
        req.proposer_state.end = start + len(ids)
    ex.spec = SimpleNamespace(proposer=p, ingest_prefill=ingest, start=p.start, k=1)
    ex.prefill(req)
    endpoint = req.cache_auxiliary[4][p.cache_identity]
    assert features.slots[req.checkpoints[4]] == endpoint
    saved = features.get(endpoint)[2].clone()
    ex.state.reserve([req], 1)
    ex.state.commit(req, 1)
    req.push(77)
    req.proposer_state.end = req.committed_len
    p.features.add_(10)
    p.prepare_release(req)
    ex.state.release(req)
    match = ex.cache.match([1, 2, 3, 4], touch=False)
    assert match.node.auxiliary[p.cache_identity] == endpoint
    assert features.slots[match.snapshot_slot] == endpoint
    assert torch.equal(features.get(endpoint)[2], saved)
    ex.flush_cache()
    assert features.bytes == 0
