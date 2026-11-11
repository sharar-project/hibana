"""Retained resume windows must not be displaced by duplicate checkpoint saves."""
from types import SimpleNamespace

import torch

from hibana.cache.snapshot import Snapshot
from hibana.proposers.parallel import DraftState, FeatureCache, ParallelProposer


def setup(capacity=96):
    p = object.__new__(ParallelProposer)
    p.context = SimpleNamespace(device=torch.device('cpu'))
    p.cache_identity = 'test-draft'
    p.window = 4
    p.features = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
    p.feature_cache = FeatureCache(capacity)
    req = SimpleNamespace(proposer_state=DraftState(0, end=8), cache_enabled=True,
                          private={4: Snapshot(1, 4), 8: Snapshot(2, 8)})
    return p, req


def handle(req, p, position):
    return req.private[position].auxiliary[p.cache_identity]


def save(p, req, position):
    """`_save` where prefill calls it: once the draft ring has reached the snapshot's position."""
    req.proposer_state.end = position
    p._save(req, req.private[position])


def test_repeated_endpoint_save_preserves_interior_resume_window():
    p, req = setup()
    save(p, req, 4)
    interior = handle(req, p, 4)
    p.features.add_(10)
    save(p, req, 8)
    endpoint = handle(req, p, 8)
    # Prompt completion and release can save the same endpoint twice.
    save(p, req, 8)
    p.feature_cache.put(20, 24, torch.full((4, 2), 99.))
    assert p.feature_cache.get(interior) is not None
    assert handle(req, p, 8) == endpoint
    assert torch.equal(p.feature_cache.get(interior)[2], torch.arange(8).reshape(4, 2))


def test_evicted_endpoint_is_saved_again():
    p, req = setup(capacity=32)
    save(p, req, 8)
    old = handle(req, p, 8)
    p.feature_cache.put(20, 24, torch.full((4, 2), 99.))
    assert p.feature_cache.get(old) is None
    save(p, req, 8)
    restored = p.feature_cache.get(handle(req, p, 8))
    assert restored[:2] == (4, 8)
    assert torch.equal(restored[2], p.features[0])


def test_different_requests_at_same_position_have_distinct_features():
    p, req = setup()
    save(p, req, 8)
    first = handle(req, p, 8)
    other = SimpleNamespace(proposer_state=DraftState(0, end=8), cache_enabled=True,
                            private={8: Snapshot(3, 8)})
    p.features.add_(20)
    save(p, other, 8)
    second = handle(other, p, 8)
    assert first != second
    assert not torch.equal(p.feature_cache.get(first)[2], p.feature_cache.get(second)[2])
