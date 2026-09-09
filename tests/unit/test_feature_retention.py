"""Retained resume windows must not be displaced by duplicate checkpoint saves."""
from types import SimpleNamespace

import torch

from sharar.proposers.parallel import DraftState, FeatureCache, ParallelProposer


def setup(capacity=96):
    p = object.__new__(ParallelProposer)
    p.context = SimpleNamespace(device=torch.device('cpu'))
    p.cache_identity = 'test-draft'
    p.window = 4
    p.features = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
    p.feature_cache = FeatureCache(capacity)
    req = SimpleNamespace(proposer_state=DraftState(0, end=8), cache_enabled=True,
                          cache_auxiliary={}, checkpoints={4: 1, 8: 2})
    return p, req


def test_repeated_endpoint_save_preserves_interior_resume_window():
    p, req = setup()
    p._save(req, 4)
    interior = req.cache_auxiliary[4][p.cache_identity]
    p.features.add_(10)
    p._save(req, 8)
    endpoint = req.cache_auxiliary[8][p.cache_identity]
    # Prompt completion and release can save the same endpoint twice.
    p._save(req, 8)
    p.feature_cache.put(20, 24, torch.full((4, 2), 99.))
    assert p.feature_cache.get(interior) is not None
    assert req.cache_auxiliary[8][p.cache_identity] == endpoint
    assert torch.equal(p.feature_cache.get(interior)[2], torch.arange(8).reshape(4, 2))


def test_evicted_endpoint_is_saved_again():
    p, req = setup(capacity=32)
    p._save(req, 8)
    old = req.cache_auxiliary[8][p.cache_identity]
    p.feature_cache.put(20, 24, torch.full((4, 2), 99.))
    assert p.feature_cache.get(old) is None
    p._save(req, 8)
    restored = p.feature_cache.get(req.cache_auxiliary[8][p.cache_identity])
    assert restored[:2] == (4, 8)
    assert torch.equal(restored[2], p.features[0])


def test_different_requests_at_same_position_have_distinct_features():
    p, req = setup()
    p._save(req, 8)
    first = req.cache_auxiliary[8][p.cache_identity]
    other = SimpleNamespace(proposer_state=DraftState(0, end=8), cache_enabled=True,
                            cache_auxiliary={}, checkpoints={8: 3})
    p.features.add_(20)
    p._save(other, 8)
    second = other.cache_auxiliary[8][p.cache_identity]
    assert first != second
    assert not torch.equal(p.feature_cache.get(first)[2], p.feature_cache.get(second)[2])
