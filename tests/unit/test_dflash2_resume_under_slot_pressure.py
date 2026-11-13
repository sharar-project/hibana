"""DFlash2 resumes a cached prefix exactly when slot pressure strikes between admission and the first prefill chunk.

A request resuming from a cached snapshot reads two things from the matched radix node: the recurrent state, copied at
admission, and DFlash2's feature handle, read by `ParallelProposer.ingest_prefill` at the first prefill chunk. The
handle lives exactly as long as the snapshot's state slot (`CheckpointFeatureCache`), and slot pressure takes every
snapshot no lock sits on, those on a running request's own path included. The request's one lock stays on the matched
node until the drafter ingested the first chunk (`StateManager.checkpoint` moves it after that), so pressure
in between (the request's own prefill or chunk checkpoints, another request's, any eviction) cannot take the matched
snapshot: the drafter would fall back, never draft for the request, and cache its snapshots without features, so the
conversation's next turns would miss too.
"""
from types import SimpleNamespace

import pytest
import torch

from hibana.engine.contracts import Capabilities, FeatureRequest, ProposerContext, ProposerRequirements
from hibana.engine.sampler import SamplingParams
from hibana.proposers.parallel import CheckpointFeatureCache, FeatureCache, ParallelProposer
from tests.unit.test_runtime_safety import tiny_executor

VOCAB = 128  # tiny_executor's logits
TAP = 3  # the target layer the draft reads


class _Draft:
    """A stand-in DFlash2 model: the projected feature of a position is the target's tap row there; drafts are valid
    tokens on a flat lattice (the real `propose`, `sample_lattice`, `accepted_batch` and feature cache run)."""
    mask_token, window, hidden, width, bytes = 0, 8, 2, 4, 0
    features = (TAP,)

    def project_features(self, layers):
        return layers[TAP].to(torch.bfloat16)

    def context_kv(self, hidden, positions, caches, pages):
        pass

    def forward(self, tokens, positions, pages, caches, attention):
        return torch.zeros(*tokens.shape, self.hidden)

    def selector(self, hidden, anchor):
        b, steps = hidden.shape[:2]
        candidates = (anchor.view(b, 1, 1) + 1 + torch.arange(steps).view(1, -1, 1) + torch.arange(self.width).view(1, 1, -1)) % VOCAB
        return candidates, torch.zeros(b, steps, self.width, self.width)


def setup(monkeypatch):
    ex = tiny_executor(monkeypatch, cache=True)
    target = ex.model.forward

    def forward(ids, ctx):
        out = target(ids, ctx)
        if TAP in ctx.feature_request.layers:
            # (the target's state after the position, the position): exact in bf16, and a function of the whole prefix
            rows = torch.stack([out['norm'][:, 0], ctx.positions % 256], -1)
            out['features'] = {TAP: rows.to(torch.bfloat16)}
        return out
    ex.model.forward = forward
    p = ParallelProposer.__new__(ParallelProposer)
    p.model, p.k, p.name, p.window = _Draft(), 3, 'dflash2', _Draft.window
    p.pitch = p.window + p.k + 1
    p.context = ProposerContext(ex.device, ex.max_batch, ex.max_context)
    p.free_slots = list(range(ex.max_batch, 0, -1))
    p.cache_identity = ('test-dflash2', p.window)
    p.caches, p.attention, p.graphs, p.feedback_graphs, p._feedback_enabled = [], object(), None, None, False
    p.planner = SimpleNamespace(paged_prefill=lambda *a, **kw: None)
    p.workspace = torch.empty(0, dtype=torch.uint8)
    p.features = torch.zeros((ex.max_batch + 1, p.window, _Draft.hidden), dtype=torch.bfloat16)
    p.feature_cache = FeatureCache(0)
    p.requirements = ProposerRequirements('dflash2', FeatureRequest(layers=_Draft.features), greedy_only=False, max_depth=15,
                                          capabilities=Capabilities(device_feedback=True, checkpoint_cache=True,
                                                                    stored_payload=True, short_drafts=True))
    p.cache_hits = p.cache_fallbacks = 0
    ex.set_proposer(p)  # the cached executor binds the slot-bound feature cache, as in serving
    assert isinstance(p.feature_cache, CheckpointFeatureCache)
    writes = []
    write = p._write

    def recorded(state, start, hidden):
        writes.append((state.slot, start, hidden.clone()))
        write(state, start, hidden)
    p._write = recorded
    return ex, p, writes


def sampled(n, seed):
    return SamplingParams(max_new_tokens=n, temperature=1.0, seed=seed)  # the checkpoint's own sampling


# A 100-token first turn: its only snapshot below the reply end is its last chunk end, 64 (updated from main, whose
# 40-token turn resumed from its prompt-end copy, gone now).
FIRST = list(range(1, 101))
RESUME = 64


@pytest.mark.parametrize('pressure', ['prefill-checkpoint', 'chunk-checkpoint', 'batched', 'evict'])
def test_dflash2_resumes_its_window_when_slot_pressure_precedes_the_first_chunk(monkeypatch, pressure):
    ex, p, writes = setup(monkeypatch)
    first = ex.generate(FIRST, sampled(6, 1))
    assert first.spec_rounds > 0
    match = ex.cache.match(FIRST, touch=False)
    node = match.node
    # The last chunk end is an interior snapshot (the first reply continues below it) carrying the draft's window.
    assert match.length == RESUME and node.children and node.state_lock_ref == 0
    window = p.feature_cache.get(node.snapshot.auxiliary[p.cache_identity])
    assert window[:2] == (RESUME - p.window, RESUME)
    window = window[2].clone()
    # The second turn edits the first reply: it resumes at 64 and prefills 236 tokens, so its prefill places its own
    # last chunk end S = 256 (the forward's last 64-token grid point before its end, 300) and its lock moves there.
    diverge = (first.output_ids[0] + 1) % VOCAB
    second = ex.admit(FIRST + [(diverge + i) % VOCAB for i in range(200)], sampled(8, 2))
    assert second.prefix_len == RESUME and second.match.node is node and second.lock.node is node
    assert node.state_lock_ref == 1
    running = [second]
    if pressure == 'batched':
        running.append(ex.admit([120] * 30, sampled(8, 3)))  # unrelated, prefilled in the same forward
    hog, evicted = [], ex.cache.evictions + ex.cache.snapshot_evictions
    if pressure == 'evict':
        ex.cache.evict(need_slots=ex.state_pool.free_count + 1024)
    else:
        # Other requests own the rest of the pool: the second turn's own checkpoint needs a slot, and eviction finds one.
        hog = [ex.state_pool.alloc() for _ in range(ex.state_pool.free_count)]
    writes.clear()
    if pressure == 'batched':
        ex.prefill_batch([(r, r.prefill_len - r.prefill_pos) for r in running])
    elif pressure == 'chunk-checkpoint':
        # The chunk ends at 128, a (spare) chunk end the executor snapshots after the forward, before the drafter
        # ingests; the lock moves there only after that ingest.
        assert ex.prefill(second, max_tokens=64) is None
        assert second.lock.node.depth == 128 and node.state_lock_ref == 0
        ex.prefill(second)
    else:
        ex.prefill(second)
    assert second.prefill_done and len(second.output_ids) == 1
    assert ex.cache.evictions + ex.cache.snapshot_evictions > evicted  # the pressure was real
    state = second.proposer_state
    assert (p.cache_hits, p.cache_fallbacks, state.valid) == (1, 0, True), 'DFlash2 lost the window it resumes from'
    # The lock moved to S after the drafter read the window: the matched snapshot may go again, S may not.
    assert second.lock.node.depth == 256 and node.state_lock_ref == 0 and len(ex.cache.locks()) == 1
    assert second.lock.node.snapshot is not None and not second.private
    slot, start, rows = writes[0]  # the drafter resumed from the exact window the first turn saved
    assert (slot, start) == (state.slot, RESUME - p.window) and torch.equal(rows, window)
    assert state.end == second.committed_len == second.prefill_len and p.ready(second)
    while any(not r.finished for r in running):
        ex.decode_step([r for r in running if not r.finished])
    assert second.spec_rounds > 0 and len(second.output_ids) == 8
    for r in running:
        ex.release(r)
    for slot in hog:
        ex.state_pool.free(slot)
    # The second turn cached its reply end with a live window: the third turn resumes drafting there too.
    end = ex.cache.match(second.prompt_ids + second.output_ids[:-1], touch=False)
    assert end.length == second.committed_len
    assert p.feature_cache.get(end.snapshot.auxiliary[p.cache_identity])[1] == end.length
    assert not ex.cache.locks() and all(n.lock_ref == 0 for n in ex.cache._walk())
    ex.flush_cache()
    assert not ex.cache.flushing and ex.state_pool.free_count == 15 and p.feature_cache.bytes == 0
