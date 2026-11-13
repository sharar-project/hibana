"""Feature ownership follows real target-slot and radix lifecycle operations.

Updated from main (no prompt-end copy; drafter metadata in `Snapshot.auxiliary`): a request's windows are bound to its
private snapshots inside a prefill step, and to its own slot, the reply end, at release (`prepare_release`)."""
from types import SimpleNamespace

import pytest
import torch

from hibana.cache.snapshot import Snapshot
from hibana.engine.sampler import SamplingParams
from tests.unit.test_radix import snaps
from hibana.proposers.parallel import CheckpointFeatureCache, DraftState, ParallelProposer
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
    """A prefilled one-token request (four tokens place no snapshot of their own) whose draft ring reached its end."""
    req = ex.admit(ids, SamplingParams(max_new_tokens=1))
    ex.prefill(req)
    req.proposer_state = DraftState(0, end=req.committed_len)
    return req


def release(ex, p, req):
    """`Executor.release`: the drafter attaches its window to the reply end; returns that handle."""
    handles = []

    def attach(r, snapshot):
        p.prepare_release(r, snapshot)
        handles.append(snapshot.auxiliary.get(p.cache_identity))
    ex.state.release(req, attach=attach)
    return handles[0] if handles else None


def test_release_binds_the_reply_end_window_to_its_slot_and_a_duplicate_frees_its_own(monkeypatch):
    ex, p, features = setup(monkeypatch)
    ids = [1, 2, 3, 4]
    req = seed(ex, p, ids)
    slot = req.slot
    endpoint = release(ex, p, req)
    match = ex.cache.match(ids, touch=False)
    assert match.length == 4 and match.snapshot_slot == slot and features.slots[slot] == endpoint
    assert match.snapshot.auxiliary[p.cache_identity] == endpoint
    assert torch.equal(features.get(endpoint)[2], p.features[0])
    # A duplicate radix insertion frees the incoming slot and its features,
    # preserving the existing snapshot and its own matching payload.
    req = seed(ex, p, ids)
    incoming = release(ex, p, req)
    assert incoming != endpoint and features.get(incoming) is None and features.get(endpoint) is not None
    ex.flush_cache()
    assert features.bytes == 0 and not features.entries


@pytest.mark.parametrize('action', ['cancel', 'retract', 'failed'])
def test_unpublished_request_cleanup_releases_features(monkeypatch, action):
    """A window bound to one of a request's private snapshots (inside a prefill step) goes with it when the request
    goes without being cached."""
    ex, p, features = setup(monkeypatch)
    req = ex.admit([1, 2, 3, 4], SamplingParams(max_new_tokens=1))
    slot = ex.state_pool.alloc()
    req.private[3] = Snapshot(slot, 3)
    req.private[3].auxiliary[p.cache_identity] = features.put(0, 3, p.features[0, :3], slot=slot)
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
    release(ex, p, req)
    reader = ex.admit([1, 2, 3, 4, 5], SamplingParams(max_new_tokens=1))
    handle = reader.match.snapshot.auxiliary[p.cache_identity]
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
    ex.cache.insert([1, 2, 3, 4], ex.kv_pool.alloc(4), snaps({2: first, 4: last},
                    {2: {p.cache_identity: a}, 4: {p.cache_identity: b}}))
    ex.cache.evict(need_slots=14)
    assert features.get(a) is None and features.get(b) is not None
    assert ex.cache.cached_tokens == 4 and features.bytes == 32
    ex.flush_cache()
    assert features.bytes == 0


def test_prospective_budget_accounts_for_each_recurrent_feature_window(monkeypatch, tmp_path):
    import json
    from hibana.cache import spec_budget
    from hibana.target import DEFAULT_TARGET, Target
    name, revision = DEFAULT_TARGET.draft_pin('dflash2')
    draft = tmp_path / ('models--' + name.replace('/', '--')) / 'snapshots' / revision
    draft.mkdir(parents=True)
    (draft / 'config.json').write_text(json.dumps({'sliding_window': 2048, 'hidden_size': 5120, 'num_hidden_layers': 5,
                                                  'num_key_value_heads': 8, 'head_dim': 128}))
    monkeypatch.setattr(Target, 'cache_root', lambda self, cache_dir=None: tmp_path)  # the draft alone, no target
    monkeypatch.setattr(spec_budget, 'tensor_bytes', lambda *args: (128, 64))
    spec = spec_budget.SpecConfig('dflash2', 15, 'reconstruct', 'fp8')
    small, _ = spec_budget.budget(spec, 4, cache_slots=5)
    large, _ = spec_budget.budget(spec, 4, cache_slots=24)
    assert large - small == 19 * 2048 * 5120 * 2


def test_a_last_chunk_end_window_survives_decode_and_release(monkeypatch):
    """Updated from main (the prompt-end copy is gone): the window bound to the prompt's last chunk end (64, placed by
    its forward and saved at ingest) stays with that snapshot through decoding and the reply end's own window."""
    from hibana.engine.contracts import FeatureRequest
    ex, p, features = setup(monkeypatch)
    prompt = list(range(1, 71))
    req = ex.admit(prompt, SamplingParams(max_new_tokens=2))
    req.proposer_state = DraftState(0)
    p.requirements = SimpleNamespace(features=FeatureRequest())

    def ingest(req, ids, start, result):  # what ParallelProposer.ingest_prefill does at each of the forward's snapshots
        for position in sorted(req.private):
            req.proposer_state.end = position
            p._save(req, req.private[position])
        req.proposer_state.end = start + len(ids)
    ex.spec = SimpleNamespace(proposer=p, ingest_prefill=ingest, start=p.start, k=1)
    ex.prefill(req)
    node = req.lock.node
    assert node.depth == 64 and not req.private
    endpoint = node.snapshot.auxiliary[p.cache_identity]
    assert features.slots[node.snapshot.slot] == endpoint
    saved = features.get(endpoint)[2].clone()
    ex.state.reserve([req], 1)
    ex.state.commit(req, 1)
    req.push(77)
    req.proposer_state.end = req.committed_len
    p.features.add_(10)
    reply = release(ex, p, req)
    match = ex.cache.match(prompt, touch=False)
    assert match.length == 64 and match.snapshot.auxiliary[p.cache_identity] == endpoint
    assert features.slots[match.snapshot_slot] == endpoint
    assert torch.equal(features.get(endpoint)[2], saved)
    assert reply != endpoint and features.get(reply) is not None
    ex.flush_cache()
    assert features.bytes == 0
