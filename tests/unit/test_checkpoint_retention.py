"""Bounded snapshot placement and ownership under real cache pressure.

Updated from main (cache policy): a forward's snapshots are private only inside its step and go into the tree at its
checkpoint, where the request's lock moves to the deepest; there is no prompt-end copy and no 2,048-token grid, and an
earlier spare chunk end goes at the next checkpoint (`StateManager.checkpoint`)."""
import pytest
import torch

from hibana.cache.radix import RadixCache
from hibana.cache.request_state import StateManager
from hibana.cache.snapshot import Snapshot
from hibana.engine.executor import Executor
from hibana.engine.request import Request
from hibana.engine.sampler import SamplingParams
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
    ex.state, ex.state_pool = manager, states
    requests = [Request([i + 1] * length, SamplingParams(max_new_tokens=1)) for i in range(batch)]
    for req in requests:
        manager.admit(req, req.prompt_ids)
    return ex, manager, requests


def forward(ex, state, req, start, end):
    """One prefill forward as the executor runs it (the fake state of a slot is its position): its planned snapshots
    written, committed, its chunk end copied from the request's slot, then the checkpoint."""
    for _, pos, slot in state.prefill_checkpoints([(req, ex._checkpoint_positions(req, start, end))]):
        state.recurrent.values[slot] = pos
    state.recurrent.values[req.slot] = end
    state.commit_prefill(req, end)
    ex._chunk_checkpoint(req, end)
    state.checkpoint(req, end)


@pytest.mark.parametrize('slots,batch,length', [(12, 1, 32766), (12, 4, 8190), (3, 1, 8190)])
def test_the_last_chunk_end_survives_exhaustion_and_the_same_prompt_resumes_there(slots, batch, length):
    """Updated from main (no prompt-end copy): one slot past each request's own is enough for its last chunk end, the
    last 64-token grid point before its end, where the same prompt sent again resumes."""
    ex, state, reqs = setup(slots, batch, length)
    endpoint = ((length - 1) // 64) * 64
    for req in reqs:
        forward(ex, state, req, 0, length)
        node = req.lock.node
        assert node.depth == endpoint and not node.snapshot.spare and not req.private
        assert state.recurrent.values[node.snapshot.slot] == endpoint
        node.snapshot.auxiliary = {'draft': ('features', endpoint)}
    for req in reqs:
        state.release(req)  # the reply end: the request's own slot
    for req in reqs:
        match = state.cache.match(req.prompt_ids[:-1])
        assert 0 < length - match.length <= 64
        assert state.recurrent.values[match.snapshot_slot] == match.length
        assert match.snapshot.auxiliary == {'draft': ('features', match.length)}
    state.flush()
    assert state.recurrent.free_count == slots - 1
    assert len(state.recurrent._free) == len(set(state.recurrent._free))
    assert state.kv.free_count == batch * length + 19


def test_a_spare_destination_takes_only_a_free_or_spare_slot_and_a_locked_snapshot_none():
    _, state, (req,) = setup(slots=6, length=1000)  # five usable: the request's own and four in the tree
    cache = state.cache
    other = [9] * 400
    state.kv._free.extend(range(1020, 1420))
    cache.insert(other, state.kv.alloc(400), [Snapshot(state.recurrent.alloc(), 100, spare=True),
                                              Snapshot(state.recurrent.alloc(), 200),
                                              Snapshot(state.recurrent.alloc(), 300, spare=True),
                                              Snapshot(state.recurrent.alloc(), 400)])
    lock = cache.lock(cache.node_at(other), owner="other")
    assert state.recurrent.free_count == 0
    assert state._private_snapshot(req, 64, True) is not None and cache.node_at(other[:100]).snapshot is None
    assert state._private_snapshot(req, 128, True) is not None and cache.node_at(other[:300]).snapshot is None
    assert state._private_snapshot(req, 192, True) is None  # only the not-spare 200 is unlocked
    assert state._private_snapshot(req, 256, False) is not None and cache.node_at(other[:200]).snapshot is None
    assert state._private_snapshot(req, 320, False) is None  # 400 is locked
    assert cache.node_at(other).snapshot is not None
    state.discard_private(req)
    cache.unlock(lock)
    req.cacheable = False
    state.release(req)
    state.flush()
    assert state.recurrent.free_count == 5


def test_a_locked_match_is_never_taken_for_a_destination():
    _, state, (seed,) = setup(slots=3, length=1000)
    state.commit_prefill(seed, 1000)
    state.release(seed)  # its reply end, 1000, is the tree's one snapshot
    state.kv._free.extend(range(1020, 2020))
    req = Request([1] * 1000 + [2] * 1000, SamplingParams())
    state.admit(req, req.prompt_ids)
    borrowed = req.match.snapshot_slot
    assert state.recurrent.free_count == 0
    assert state.prefill_checkpoints([(req, [(1064, False), (1128, True)])]) == []
    assert borrowed not in state.recurrent.freed and req.lock.node.snapshot.slot == borrowed and not req.private
    req.cacheable = False
    state.release(req)
    state.flush()
    assert state.recurrent.free_count == 2


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
    state.prefill_checkpoints([(seed, [(64, False), (960, False)])])
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


def test_a_chunked_prompt_keeps_one_spare_chunk_end_and_its_last_one():
    """Updated from main (main kept a chunk end every 2,048 tokens, at most 16 a prompt, and every one in the last
    8,192): each chunk end is spare, the lock moves to it, and the next checkpoint frees the one before; the prompt's
    last chunk end stays. An edit mid-prompt then has its KV but no snapshot below it, and its own prefill places a
    branching point there for the next request."""
    ex, state, (req,) = setup(slots=24, length=10000)
    for start in range(0, 10000, 2048):
        end = min(10000, start + 2048)
        forward(ex, state, req, start, end)
        spare = [n.depth for n in state.cache._walk() if n.snapshot is not None and n.snapshot.spare]
        assert spare == ([end] if end < 10000 else []) and req.lock.node.depth == (end if end < 10000 else 9984)
    assert [n.depth for n in state.cache._walk() if n.snapshot is not None] == [9984]
    state.release(req)
    edited = req.prompt_ids[:5000] + [99] * 5000  # the prompt changes mid-way
    match = state.cache.match(edited[:-1])
    assert match.length == 0 and match.raw_length == 5000 and match.branching_seqlen == 5000
