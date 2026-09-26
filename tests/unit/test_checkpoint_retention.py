"""Bounded snapshot placement and ownership under real cache pressure."""
import pytest
import torch

from sharar.engine.executor import Executor
from sharar.engine.request import Request
from sharar.engine.sampler import SamplingParams
from sharar.cache.radix import RadixCache
from sharar.cache.request_state import StateManager
from tests.unit.test_radix import FakePool


class States(FakePool):
    def __init__(self, n):
        super().__init__(n)
        self.values = torch.zeros(n)

    def alloc(self):
        slot = self.alloc_slot()
        self.values[slot] = 0
        return slot

    def copy_slot(self, source, dest):
        self.values[dest] = self.values[source]


def setup(slots=12, batch=1, length=32766):
    kv, states = FakePool(batch * length + 20), States(slots)
    cache = RadixCache(kv, states, torch.device('cpu'))
    manager = StateManager(kv, states, cache)
    ex = Executor.__new__(Executor)
    ex.state = manager
    requests = [Request([i + 1] * length, SamplingParams(max_new_tokens=1)) for i in range(batch)]
    for req in requests:
        manager.admit(req, req.prompt_ids)
    return ex, manager, requests


@pytest.mark.parametrize('slots,batch,length', [(12, 1, 32766), (12, 4, 8190), (3, 1, 8190)])
def test_endpoint_survives_exhaustion_and_exact_end_copy(slots, batch, length):
    ex, state, reqs = setup(slots, batch, length)
    plans = state.prefill_checkpoints([(r, ex._checkpoint_positions(0, length, length)) for r in reqs])
    destinations = [s for _, _, s in plans]
    assert len(destinations) == len(set(destinations))
    for row, pos, slot in plans:
        state.recurrent.values[slot] = pos
    for req in reqs:
        state.commit_prefill(req, length)
        state.recurrent.values[req.slot] = length
        state.checkpoint(req, length, copy=True)
        endpoint = ((length - 1) // 64) * 64
        assert endpoint in req.checkpoints
        assert state.recurrent.values[req.checkpoints[endpoint]] == endpoint
        req.cache_auxiliary[endpoint] = {'draft': ('features', endpoint)}
        state.release(req)
    for req in reqs:
        match = state.cache.match(req.prompt_ids[:-1])
        assert 0 < length - match.length <= 64
        assert state.recurrent.values[match.snapshot_slot] == match.length
        assert match.node.auxiliary == {'draft': ('features', match.length)}
    state.flush()
    assert state.recurrent.free_count == slots - 1
    assert len(state.recurrent._free) == len(set(state.recurrent._free))
    assert state.kv.free_count == batch * length + 19


def test_recycle_completed_only_and_drop_stale_auxiliary():
    _, state, (req,) = setup(slots=4, length=1000)
    first = state.prefill_checkpoints([(req, [64, 128, 192])])
    assert {p for _, p, _ in first} == {64, 192}
    assert state.checkpoint(req, 256) is None  # both destinations still pending
    state.commit_prefill(req, 300)
    req.cache_auxiliary = {64: {'draft': 'old'}, 192: {'draft': 'keep'}}
    second = state.prefill_checkpoints([(req, [320, 384, 448])])
    assert {p for _, p, _ in second} == {320, 448}
    assert len({s for _, _, s in second}) == 2
    assert req.cache_auxiliary == {}
    assert state.checkpoint(req, 448) == req.checkpoints[448]
    assert state.checkpoint(req, 512) is None
    req.cacheable = False
    state.release(req)
    assert state.recurrent.free_count == 3


def test_snapshot_reservation_never_recycles_a_borrowed_prefix():
    _, state, (seed,) = setup(slots=4, length=1000)
    state.commit_prefill(seed, 1000)
    state.release(seed)
    state.kv._free.extend(range(1020, 2020))
    req = Request([1] * 1000 + [2] * 1000, SamplingParams())
    state.admit(req, req.prompt_ids)
    borrowed = req.match.snapshot_slot
    state.prefill_checkpoints([(req, [1064, 1128])])
    state.commit_prefill(req, 1200)
    state.prefill_checkpoints([(req, [1264, 1328])])
    assert borrowed not in state.recurrent.freed
    assert borrowed not in req.checkpoints.values()
    req.cacheable = False
    state.release(req)
    state.flush()
    assert state.recurrent.free_count == 3


def test_admission_budgets_suffix_without_evicting_or_counting_a_cache_hit():
    _, state, (seed,) = setup(slots=4, length=1000)
    state.commit_prefill(seed, 1000)
    state.release(seed)
    ids = [1] * 1000 + [2] * 10
    hits, misses = state.cache.hits, state.cache.misses
    assert state.can_admit(ids, reserve=9)
    assert (state.cache.hits, state.cache.misses) == (hits, misses)
    req = Request(ids, SamplingParams())
    state.admit(req, ids)
    assert req.prefix_len == 1000 and len(req.pages) == 1010
    req.cacheable = False
    state.release(req)
    state.flush()
    assert state.kv.free_count == 1019 and state.recurrent.free_count == 3


def test_full_snapshot_cache_can_admit_without_pinning_every_slot():
    _, state, (seed,) = setup(slots=4, length=1000)
    state.prefill_checkpoints([(seed, [64, 960])])
    state.commit_prefill(seed, 1000)
    state.release(seed)
    assert state.recurrent.free_count == 0
    ids = [1] * 1000 + [2]
    assert state.can_admit(ids)
    req = Request(ids, SamplingParams())
    state.admit(req, ids)
    assert req.prefix_len == 1000
    assert state.cache.snapshot_evictions == 1 and state.cache.evictions == 0
    req.cacheable = False
    state.release(req)
    state.flush()
    assert state.recurrent.free_count == 3 and state.kv.free_count == 1019


def chunked_prefill(ex, state, req, chunk=2048):
    """What serving does: 2,048-token forwards, then the prompt-end copy."""
    n = req.prefill_len
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        for _, pos, slot in state.prefill_checkpoints([(req, ex._checkpoint_positions(start, end, n))]):
            state.recurrent.values[slot] = pos
        state.recurrent.values[req.slot] = end
        state.commit_prefill(req, end)
        ex._chunk_checkpoint(req, start, end)
    state.checkpoint(req, n, copy=True)


def test_chunked_prefill_is_resumable_at_chunk_boundaries():
    ex, state, (req,) = setup(slots=24, length=10000)
    chunked_prefill(ex, state, req)
    assert {2048, 4096, 6144, 8192} <= set(req.checkpoints)
    assert all(state.recurrent.values[req.checkpoints[p]] == p for p in (2048, 4096, 6144, 8192))
    state.release(req)
    edited = req.prompt_ids[:5000] + [99] * 5000  # the prompt changes mid-way
    match = state.cache.match(edited[:-1])
    assert match.length == 4096 and state.recurrent.values[match.snapshot_slot] == 4096


def test_long_prompts_cap_chunk_checkpoints():
    from sharar.engine.executor import CHECKPOINT_TAIL, MAX_CHUNK_CHECKPOINTS
    ex, state, (req,) = setup(slots=64, length=262000)
    chunked_prefill(ex, state, req)
    chunk_ends = [p for p in req.checkpoints if p % 2048 == 0 and p < 262000]
    spaced = [p for p in chunk_ends if p < 262000 - CHECKPOINT_TAIL]
    assert 0 < len(spaced) <= MAX_CHUNK_CHECKPOINTS and all(p % 16384 == 0 for p in spaced)
    # The tail keeps every chunk end, so a changed final message falls back at most one chunk.
    assert [p for p in chunk_ends if p >= 262000 - CHECKPOINT_TAIL] == [253952, 256000, 258048, 260096]
