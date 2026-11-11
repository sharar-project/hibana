"""Exact row routing with poisoned padding, ring wrap and endpoint snapshots."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from hibana.cache.snapshot import Snapshot
from hibana.engine.contracts import ProposerContext, VerifiedFeatures
from hibana.proposers.parallel import DraftState, FeatureCache, ParallelProposer


class MarkedModel:
    features = (5, 19)

    def __init__(self):
        self.projections = self.kv_writes = 0

    def project_features(self, features):
        self.projections += 1
        return features[5] + features[19] * 10

    def context_kv(self, hidden, positions, caches, pages):
        self.kv_writes += 1
        caches[0][0][pages] = hidden + positions[:, None]
        caches[0][1][pages] = hidden - positions[:, None]


def setup(batch, checkpoint=False):
    p = ParallelProposer.__new__(ParallelProposer)
    p.feedback_graphs, p._feedback_enabled = None, False
    p.model, p.k, p.window, p.pitch = MarkedModel(), 7, 8, 16
    p.context = ProposerContext(torch.device('cpu'), 4, 128)
    p.features = torch.arange(5 * 8 * 3).reshape(5, 8, 3).float()
    p.caches = [(torch.full((5 * 16, 3), -999.), torch.full((5 * 16, 3), -999.))]
    p.cache_identity = ('test',)
    p.feature_cache = FeatureCache(8192)
    p.cache_hits = p.cache_fallbacks = 0
    starts, counts = [7, 15, 24, 31][:batch], [1, 5, 3, 8][:batch]
    emitted = [[100 + i] * count for i, count in enumerate(counts)]
    requests = [SimpleNamespace(proposer_state=DraftState(slot, end=start, pending=17),
                committed_len=start + count, finished=i % 2 == 0, cache_enabled=True,
                private={start + count: Snapshot(1, start + count),
                         **({start + 2: Snapshot(2, start + 2)} if checkpoint and count > 2 else {})})
                for i, (slot, start, count) in enumerate(zip([4, 1, 3, 2], starts, counts))]
    layers = {lid: torch.full((batch, 8, 3), float('nan')) for lid in p.model.features}
    for lid, values in layers.items():
        for row, (start, count) in enumerate(zip(starts, counts)):
            values[row, :count] = torch.arange(start, start + count)[:, None] + row * 100 + lid
    return p, requests, emitted, VerifiedFeatures(layers=layers)


@pytest.mark.parametrize('batch', [1, 2, 3, 4])
@pytest.mark.parametrize('checkpoint', [False, True])
def test_batched_rows_match_sequential_oracle_and_preserve_endpoints(batch, checkpoint):
    p, requests, emitted, features = setup(batch, checkpoint)
    control, control_reqs = deepcopy(p), deepcopy(requests)
    for row, (req, tokens) in enumerate(zip(control_reqs, emitted)):
        control._accepted(req, tokens, features, row)
    p.accepted_batch(requests, emitted, features)
    assert torch.equal(p.features, control.features)
    for pair, expected in zip(p.caches, control.caches):
        assert all(torch.equal(x, y) for x, y in zip(pair, expected))
    if batch > 1 and not checkpoint:
        assert p.model.projections == p.model.kv_writes == 1
        assert control.model.projections == batch
    for req, expected in zip(requests, control_reqs):
        assert req.proposer_state == expected.proposer_state
        assert req.proposer_state.pending == (None if req.finished else emitted[requests.index(req)][-1])
        saved = {q: s.auxiliary for q, s in req.private.items() if s.auxiliary}
        oracles = {q: s.auxiliary for q, s in expected.private.items() if s.auxiliary}
        assert saved.keys() == oracles.keys()
        for position, handles in saved.items():
            cached = p.feature_cache.get(handles[p.cache_identity])
            oracle = control.feature_cache.get(oracles[position][p.cache_identity])
            assert cached[:2] == oracle[:2]
            assert torch.equal(cached[2], oracle[2])
    # Graph/output buffers may be overwritten after feedback; rings/cache own data.
    saved = p.features.clone()
    for values in features.layers.values():
        values.zero_()
    assert torch.equal(p.features, saved)


@pytest.mark.parametrize('failure', ['empty', 'duplicate_slot', 'prefix', 'invalid', 'missing_features', 'short_features'])
def test_invalid_accepted_batch_fails_before_private_writes(failure):
    p, reqs, emitted, features = setup(3)
    if failure == 'empty':
        emitted[1] = []
    elif failure == 'duplicate_slot':
        reqs[1].proposer_state.slot = reqs[0].proposer_state.slot
    elif failure == 'prefix':
        reqs[1].proposer_state.end -= 1
    elif failure == 'invalid':
        reqs[1].proposer_state.valid = False
    elif failure == 'missing_features':
        features.layers = None
    else:
        features.layers = {lid: value[:, :1] for lid, value in features.layers.items()}
    before = p.features.clone()
    with pytest.raises(ValueError):
        p.accepted_batch(reqs, emitted, features)
    assert torch.equal(p.features, before)
    assert p.model.projections == p.model.kv_writes == 0


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('interior', [False, True])
def test_feedback_graph_dispatch_preserves_interior_checkpoint_fallback(enabled, interior):
    p, reqs, emitted, features = setup(2, checkpoint=interior)
    calls = []
    p._feedback_enabled = enabled
    p.feedback_graphs = SimpleNamespace(graphs={5: object()}, run=lambda *args: calls.append(args))
    p._accepted(reqs[1], emitted[1], features, 1)
    if enabled and not interior:
        assert calls == [(reqs[1], emitted[1], features, 1)]
        assert p.model.projections == 0
    else:
        assert not calls
        assert p.model.projections == 1
        assert reqs[1].proposer_state.end == reqs[1].committed_len
        assert reqs[1].private[reqs[1].committed_len].auxiliary
