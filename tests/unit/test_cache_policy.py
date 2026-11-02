"""The prefix cache's policy (docs/cache_design.md, "Protection and eviction"): one way to protect (`radix.TreeLock`)
with two counters and the evictable totals they move, one owner per snapshot, eviction by `get_priority`, dead-KV
pruning, a pure admission check, placement inside prefill forwards (`Executor._checkpoint_positions`), and
`RadixCache.check`.

Tree-level tests use the fake pools of `test_radix`; engine-level ones the real Executor, StateManager, RadixCache,
Scheduler and MtpProposer around the stub models of `test_resume_under_slot_pressure` (an exact recurrence for the
target, an MTP head every draft of which is accepted, draft KV checked row by row)."""
import asyncio
import random
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
import torch

from sharar.cache.radix import RadixCache
from sharar.cache.request_state import StateManager
from sharar.cache.snapshot import Snapshot, get_priority
from sharar.engine.executor import Executor
from sharar.engine.request import Request
from sharar.engine.sampler import InvalidRequest, SamplingParams
from sharar.engine.scheduler import QueueFull, Scheduler, SchedulerConfig
from tests.unit.test_checkpoint_retention import States
from tests.unit.test_radix import FakePool, make, snaps
from tests.unit.test_resume_under_slot_pressure import depth, draw, harness, run_turn

CPU = torch.device("cpu")


def cached(cache, kv, st, ids, spare=False):
    """ids in the tree with one snapshot at its end."""
    cache.insert(ids, kv.alloc(len(ids)), [Snapshot(st.alloc_slot(), len(ids), spare=spare)])
    return cache.node_at(ids)


# ---- locks and the counters ----

def test_a_lock_counts_on_its_node_and_every_ancestor_and_check_sees_every_lock():
    cache, kv, st = make()
    a, b = [1, 2, 3, 4, 5, 6], [1, 2, 3, 9]
    cache.insert(a, kv.alloc(6), snaps({3: st.alloc_slot(), 6: st.alloc_slot()}))
    cache.insert(b, kv.alloc(4), snaps({4: st.alloc_slot()}))
    n3, n6, n4 = cache.node_at(a[:3]), cache.node_at(a), cache.node_at(b)
    owners = ["x", "y", "z"]
    l6, l4, l3 = cache.lock(n6, owner=owners[0]), cache.lock(n4, owner=owners[1]), cache.lock(n3, owner=owners[2])
    assert (n3.lock_ref, n3.state_lock_ref, n6.lock_ref, n6.state_lock_ref, n4.lock_ref, n4.state_lock_ref) == (3, 1, 1, 1, 1, 1)
    assert (cache.evictable_size(), cache.protected_size(), cache.state_evictable_size(), cache.state_protected_size()) == (0, 7, 0, 3)
    cache.check(owners)
    with pytest.raises(AssertionError, match="owner is gone"):
        cache.check(owners[:2])
    n3.lock_ref += 1  # a count written by hand breaks the identity
    with pytest.raises(AssertionError, match="lock_ref"):
        cache.check(owners)
    n3.lock_ref -= 1
    cache.unlock(l6)
    cache.unlock(l6)  # idempotent
    cache.unlock(None)
    assert (n6.lock_ref, n3.lock_ref, len(cache.locks())) == (0, 2, 2)
    assert (cache.evictable_size(), cache.state_evictable_size()) == (3, 1)
    with cache._lock_node(n6) as probe:  # a probe locks only for its block
        assert n6.state_lock_ref == 1 and probe in cache.locks() and cache.state_evictable_size() == 0
        with pytest.raises(AssertionError, match="probe"):
            cache.check(owners)
    assert n6.state_lock_ref == 0 and len(cache.locks()) == 2
    assert cache.lock(None, owner="r") is None and cache.lock(cache.root, owner="r") is None  # nothing matched
    cache.unlock(l4)
    cache.unlock(l3)
    cache.check()
    assert not cache.locks() and all(n.lock_ref == n.state_lock_ref == 0 for n in cache._walk())
    assert (cache.evictable_size(), cache.protected_size(), cache.state_evictable_size()) == (7, 0, 3)


def test_a_split_gives_the_head_the_path_count_and_no_lock_of_its_own():
    cache, kv, st = make()
    a = [1, 2, 3, 4, 5, 6]
    node = cached(cache, kv, st, a)
    lock = cache.lock(node, owner="r")
    cache.insert([1, 2, 3, 7], kv.alloc(4), snaps({4: st.alloc_slot()}))  # splits a's node at 3
    head = cache.node_at([1, 2, 3])
    assert head is not node and node.parent is head
    assert (head.lock_ref, head.state_lock_ref, head.snapshot) == (1, 0, None)
    assert (node.lock_ref, node.state_lock_ref, node.snapshot.position) == (1, 1, 6)
    assert cache.protected_size() == 6 and cache.evictable_size() == 1  # the head and a's rest stay protected
    cache.check(["r"])
    cache.unlock(lock)
    cache.check()


def test_check_finds_two_owners_no_owner_dead_kv_and_drifted_totals():
    cache, kv, st = make()
    node = cached(cache, kv, st, [1, 2, 3, 4])
    cache.check()
    with pytest.raises(AssertionError, match="two owners"):
        cache.check(owned_slots=[("request", node.snapshot.slot)])
    lost = st.alloc_slot()
    with pytest.raises(AssertionError, match="no owner"):
        cache.check()
    st.free(lost)
    cache._evictable_tokens += 1
    with pytest.raises(AssertionError, match="KV totals"):
        cache.check()
    cache._evictable_tokens -= 1
    cache.evictable_leaves.pop(node)
    with pytest.raises(AssertionError, match="evictable_leaves"):
        cache.check()
    cache._update_leaf(node)
    cache._take_snapshot(node)  # by hand: KV that nothing can resume from, and its slot leaked
    with pytest.raises(AssertionError, match="dead KV|no owner"):
        cache.check()


# ---- snapshots: one owner, priority, recency ----

def test_slot_pressure_takes_spare_snapshots_first_then_the_least_recently_used_and_never_a_locked_one():
    cache, kv, st = make()
    made = [cached(cache, kv, st, [100 + i, 1, 2], spare=spare) for i, spare in enumerate([False, True, False, True])]
    locked = cached(cache, kv, st, [200, 1, 2], spare=True)
    lock = cache.lock(locked, owner="r")
    order = []
    while (node := cache._lru_candidate(snapshot_only=True)) is not None:
        order.append(made.index(node))
        cache._evict_snapshot(node, "slots")
    assert order == [1, 3, 0, 2]  # spare ones oldest first, then the rest oldest first
    assert locked.snapshot is not None and cache.node_at([200, 1, 2]) is locked
    assert dict(cache.evicted) == {("slots", True): 2, ("slots", False): 2}
    cache.check(["r"])
    cache.unlock(lock)


def test_a_spare_destination_takes_only_a_spare_snapshots_slot():
    cache, kv, st = make()
    keep = cached(cache, kv, st, [1, 2, 3])
    spare = cached(cache, kv, st, [4, 5, 6], spare=True)
    st._free.clear()
    cache.evict(need_slots=1, spare_only=True)
    assert spare.snapshot is None and keep.snapshot is not None and st.free_count == 1
    st._free.clear()
    cache.evict(need_slots=1, spare_only=True)
    assert keep.snapshot is not None and st.free_count == 0
    cache.evict(need_slots=1)
    assert keep.snapshot is None


def test_walking_past_a_snapshot_does_not_make_it_recent_and_resuming_from_it_does():
    """`Snapshot.last_access` is set when the snapshot is made and when a request resumes from it, not when a request
    resumes deeper on the same path; resuming also clears `spare`."""
    cache, kv, st = make()
    a = list(range(1, 9))
    cache.insert(a, kv.alloc(8), snaps({4: st.alloc_slot(), 8: st.alloc_slot()}, spare=True))
    other = cached(cache, kv, st, [50, 51, 52, 53], spare=True)  # made after a's
    n4, n8 = cache.node_at(a[:4]), cache.node_at(a)
    for _ in range(3):
        assert cache.match(a + [99]).length == 8  # resumes at 8, walking past 4
    assert not n8.snapshot.spare and n4.snapshot.spare
    assert cache._lru_candidate(snapshot_only=True) is n4
    assert cache.match(a[:4] + [99]).length == 4  # a request resumes at 4
    assert cache._lru_candidate(snapshot_only=True) is other and not n4.snapshot.spare
    assert get_priority(other.snapshot) < get_priority(n8.snapshot) < get_priority(n4.snapshot)
    assert cache.match([50, 51, 52, 53, 99], touch=False).length == 4 and other.snapshot.spare  # a probe touches nothing


def test_a_duplicate_snapshot_keeps_the_trees_merges_spare_and_recency_and_frees_its_slot():
    cache, kv, st = make()
    ids = [1, 2, 3, 4]
    node = cached(cache, kv, st, ids, spare=True)
    mine = node.snapshot
    slot = st.alloc_slot()
    incoming = Snapshot(slot, 4, {"d": 1})
    pages = kv.alloc(4)
    kv.free(pages[:cache.insert(ids, pages, [incoming])])
    assert node.snapshot is mine and not mine.spare and mine.auxiliary == {}
    assert mine.last_access == incoming.last_access > 0
    assert slot in st._free and incoming.slot == -1 and incoming.auxiliary == {}  # the duplicate is dead
    cache.check()


def test_dead_kv_goes_up_the_path_once_nothing_can_resume_from_it():
    cache, kv, st = make()
    a = list(range(1, 9))
    cache.insert(a, kv.alloc(8), snaps({4: st.alloc_slot(), 8: st.alloc_slot()}))
    with cache._lock_node(cache.node_at(a)):
        cache.evict(need_slots=st.free_count + 1)  # 4's snapshot goes, its KV stays under the locked path
    assert cache.node_at(a[:4]).snapshot is None and cache.node_at(a[:4]).in_l1
    cache.check()
    cache.evict(need_pages=kv.free_count + 1)  # the leaf goes whole; the node above it can resume nothing now
    assert not cache.root.children and cache.pruned == 1 and kv.free_count == 999 and st.free_count == 19
    cache.check()


def test_kv_pressure_takes_unlocked_leaves_least_recently_used_first():
    cache, kv, st = make()
    a, b = cached(cache, kv, st, [1, 2, 3]), cached(cache, kv, st, [4, 5, 6])
    assert set(cache.evictable_leaves) == {a, b} and cache.evictable_size() == 6
    cache.match([1, 2, 3, 9])  # a is used: b is now the least recently used
    lock = cache.lock(b, owner="r")
    assert set(cache.evictable_leaves) == {a}
    cache.evict(need_pages=kv.free_count + 1)  # only a can go: b is locked
    assert cache.node_at([1, 2, 3]) is None and b.in_l1
    cache.unlock(lock)
    assert set(cache.evictable_leaves) == {b}
    cache.check()


# ---- admission ----

def seeded(slots=4, convs=(1, 2, 3), length=1000):
    """A StateManager whose tree holds each conversation's reply end, oldest first."""
    kv, st = FakePool(len(convs) * length + 4000), States(slots)
    kv.num_pages = len(kv._free) + 1
    cache = RadixCache(kv, st, CPU)
    state = StateManager(kv, st, cache)
    for t in convs:
        r = Request([t] * length, SamplingParams(max_new_tokens=1))
        state.admit(r, r.prompt_ids)
        state.commit_prefill(r, length)
        state.release(r)
    return kv, st, cache, state


def test_can_admit_is_pure_and_does_not_count_its_own_match_as_evictable():
    """The only slot to take is the candidate's own resume point: admitting would evict it (or fail), so the check says
    no, and changes nothing (no eviction, no hit counted, no lock left)."""
    kv, st, cache, state = seeded(slots=3, convs=(1,))
    other = Request([9] * 10, SamplingParams())
    state.admit(other, other.prompt_ids)  # a running request owns the last free slot
    assert st.free_count == 0 and cache.state_evictable_size() == 1
    ids = [1] * 1000 + [5] * 10
    before = (cache.hits, cache.misses, cache.evictions, cache.snapshot_evictions, kv.free_count)
    assert not state.can_admit(ids)
    assert (cache.hits, cache.misses, cache.evictions, cache.snapshot_evictions, kv.free_count) == before
    assert not cache.locks() and cache.state_evictable_size() == 1
    other.cacheable = False
    state.release(other)
    assert state.can_admit(ids)
    req = Request(ids, SamplingParams())
    state.admit(req, ids)
    assert req.prefix_len == 1000
    state.check()


def test_admission_locks_its_match_before_it_evicts_for_its_own_slot():
    """Every slot holds a reply end and the conversation that continues is the least recently used one: admission
    locks its match first, so slot pressure takes the next one instead."""
    kv, st, cache, state = seeded()
    assert st.free_count == 0
    ids = [1] * 1000 + [5] * 10
    assert state.can_admit(ids) and cache.node_at([1] * 1000) is not None and cache.node_at([2] * 1000) is not None
    req = Request(ids, SamplingParams())
    state.admit(req, ids)
    assert req.prefix_len == 1000 and req.lock.node is cache.node_at([1] * 1000)
    assert dict(cache.evicted) == {("slots", True): 1} and cache.node_at([2] * 1000) is None  # conversation 2's, whole
    state.check()
    req.cacheable = False
    state.release(req)
    state.check()


class Recording:
    """A scheduler-facing executor that records the admission calls: prefetch, then the pure check, then admit."""

    def __init__(self, prefetch=True):
        self.calls, self.max_batch, self.cache = [], 4, None
        if prefetch is not None:
            self.prefetch = lambda ids, reserve: (self.calls.append(("prefetch", len(ids), reserve)), prefetch)[1]

    def validate_request(self, ids, params, media=()):
        params.validate()

    def admission_reserve(self, req, limit):
        return min(limit, req.params.max_new_tokens - len(req.output_ids))

    def can_admit(self, ids, reserve_tokens=0):
        self.calls.append(("check", len(ids), reserve_tokens))
        return True

    def admit(self, ids, params, req):
        self.calls.append(("admit", len(ids)))
        req.slot = 1
        return req

    def cache_stats(self):
        return None


def test_the_scheduler_budgets_the_running_requests_outstanding_decode_reserve():
    ex = Recording(prefetch=None)
    sched = Scheduler(ex, SchedulerConfig(max_running=4, reserve_tokens=64))
    a = sched.submit([1] * 10, SamplingParams(max_new_tokens=100))
    b = sched.submit([2] * 10, SamplingParams(max_new_tokens=30))
    sched._admit()
    # a: its own 64; b: its own 30 and a's outstanding 64
    assert ex.calls == [("check", 10, 64), ("admit", 10), ("check", 10, 94), ("admit", 10)]
    a.output_ids = [7] * 90  # a has 10 tokens left to generate
    ex.calls.clear()
    sched.submit([3] * 10, SamplingParams(max_new_tokens=1))
    sched._admit()
    assert ex.calls[0] == ("check", 10, 1 + 10 + 30)
    assert b in sched.prefilling


def test_the_queue_heads_restore_comes_before_the_pure_check_and_holds_the_queue():
    ex = Recording(prefetch=False)
    sched = Scheduler(ex, SchedulerConfig(max_running=4, reserve_tokens=8))
    sched.submit([1] * 10, SamplingParams(max_new_tokens=4))
    sched.submit([2] * 10, SamplingParams(max_new_tokens=4))
    sched._admit()
    assert ex.calls == [("prefetch", 10, 4)] and len(sched.queue) == 2  # the head waits for its restore, FIFO
    ex.prefetch = lambda ids, reserve: True
    sched._admit()
    assert [c[0] for c in ex.calls[1:]] == ["check", "admit", "check", "admit"] and not sched.queue


# ---- placement inside prefill forwards ----

def placing():
    ex = Executor.__new__(Executor)
    ex.state = StateManager(FakePool(10), FakePool(10), RadixCache(FakePool(10), FakePool(10), CPU))
    return ex


def fake(n, boundary=0, branching=0, cached=True):
    return SimpleNamespace(prefill_len=n, branching_seqlen=branching, cache_enabled=cached, cacheable=True,
                           params=SamplingParams(message_boundary=boundary))


def test_the_last_forward_snapshots_its_last_grid_point_before_the_end():
    ex = placing()
    assert ex._checkpoint_positions(fake(1000), 0, 1000) == [(960, False)]
    assert ex._checkpoint_positions(fake(1000), 100, 1000) == [(996, False)]  # the forward's own grid
    assert ex._checkpoint_positions(fake(100), 0, 100) == [(64, False)]  # the same prompt sent again resumes there
    assert ex._checkpoint_positions(fake(65), 0, 65) == [(64, False)]
    assert ex._checkpoint_positions(fake(64), 0, 64) == []
    assert ex._checkpoint_positions(fake(1000), 960, 1000) == []  # its own grid has no point before the end
    assert ex._checkpoint_positions(fake(1000, cached=False), 0, 1000) == []
    ex.cache.flushing = True
    assert ex._checkpoint_positions(fake(1000), 0, 1000) == []


def test_an_earlier_chunk_end_is_spare_unless_it_is_the_last_or_the_next_forward_cannot_place_a_point():
    ex = placing()
    assert ex._checkpoint_positions(fake(5000), 0, 2048) == [(2048, True)]
    assert ex._checkpoint_positions(fake(2100), 0, 2048) == [(2048, False)]  # the last forward places none of its own
    assert ex._checkpoint_positions(fake(2200), 0, 2048) == [(2048, True)]
    # a message boundary or branching point in the next forward's first 64 tokens: this chunk end stands for it
    assert ex._checkpoint_positions(fake(5000, boundary=2100), 0, 2048) == [(2048, False)]
    assert ex._checkpoint_positions(fake(5000, branching=2048), 0, 2048) == [(2048, False)]
    assert ex._checkpoint_positions(fake(5000, boundary=2100), 2048, 4096) == [(4096, True)]


def test_the_message_boundary_goes_on_the_forward_grid_and_is_not_spare():
    ex = placing()
    assert ex._checkpoint_positions(fake(1000, boundary=700), 0, 1000) == [(960, False), (640, False)]
    assert ex._checkpoint_positions(fake(1000, boundary=980), 0, 1000) == [(960, False)]  # the same point
    assert ex._checkpoint_positions(fake(5000, boundary=3000), 2048, 4096) == [(2944, False), (4096, True)]
    assert ex._checkpoint_positions(fake(1000, boundary=1000), 0, 1000) == [(960, False)]  # not before the end
    for bad in (-1, 1.5, "3"):
        with pytest.raises(InvalidRequest):
            SamplingParams(message_boundary=bad).validate()


def test_the_branching_point_goes_in_the_chunk_that_contains_it():
    """SGLang places its branching point in whichever forward contains it (prototype 3 placed it only in the first
    forward, so a shared prefix longer than a chunk never got its end covered)."""
    ex = placing()
    req = fake(5800, branching=5000)
    assert ex._checkpoint_positions(req, 0, 2048) == [(2048, True)]
    assert ex._checkpoint_positions(req, 2048, 4096) == [(4096, True)]
    assert ex._checkpoint_positions(req, 4096, 5800) == [(5760, False), (4992, False)]
    assert ex._checkpoint_positions(fake(3000, branching=1500), 1000, 3000) == [(2984, False), (1448, False)]


def test_a_branching_point_lets_the_next_request_sharing_the_prefix_resume_there():
    h = harness(slots=24, max_batch=4)
    x = draw(random.Random("x"), 3000)
    run_turn(h, x)
    assert h.ex.cache.full_reprefills == 0
    y = run_turn(h, x[:1500] + draw(random.Random("y"), 1000))  # shares 1500 tokens, no snapshot among them
    assert y.resume == 0 and h.ex.cache.full_reprefills == 1  # the tree held its KV but nothing to resume from
    branch = h.ex.cache.node_at(x[:1472])
    assert branch is not None and not branch.snapshot.spare and branch.snapshot.auxiliary
    z = run_turn(h, x[:1500] + draw(random.Random("z"), 500))
    assert z.resume == 1472 and z.drafter == "hit" and h.ex.cache.full_reprefills == 1
    assert h.ex.cache.branching_resumes == 1 and h.ex.cache_stats()["branching_resumes"] == 1


def test_a_turn_that_drops_the_last_message_resumes_at_its_start():
    """An agent's transient last message (a turn counter, a reminder) is gone from its next prompt: the chat API's
    boundary (900) puts a snapshot at the last grid point before that message (896); the prompt's last chunk end
    (960) is past what the next turn keeps."""
    h = harness(slots=24, max_batch=4)
    prompt = draw(random.Random("chat"), 1000)
    first = h.sched.submit(prompt, SamplingParams(max_new_tokens=8, ignore_eos=True, message_boundary=900))
    h.sched.run_until_idle()
    assert first.finished
    nxt = run_turn(h, prompt[:901] + draw(random.Random("next"), 200))
    assert nxt.resume == 896 and nxt.drafter == "hit"


def chat_app(engine, render, tok):
    from sharar.api.server import create_app
    return create_app(engine, tok, SimpleNamespace(render=render), {})


@pytest.mark.parametrize("offsets", [False, True])
def test_the_chat_api_reports_where_the_last_message_starts(offsets):
    submitted, renders = [], []

    async def submit(ids, params, media=()):
        submitted.append((ids, params))
        queue = asyncio.Queue()
        queue.put_nowait(([ord("k")], True, "stop"))
        return SimpleNamespace(queue=queue, req=SimpleNamespace(prefix_len=0))

    def render(messages, add_generation_prompt=True, enable_thinking=False, tools=None, **kw):
        renders.append(len(messages))
        return "".join(f"<{m['role']}>{m['content']}</>" for m in messages) + ("<assistant>" if add_generation_prompt else "")

    def encode(text):
        return [ord(c) for c in text]
    tok = SimpleNamespace(encode=Mock(side_effect=encode), decode=lambda ids, **kw: "".join(map(chr, ids)))
    if offsets:
        tok.encode_with_offsets = lambda text: (encode(text), [(i, i + 1) for i in range(len(text))])
    engine = SimpleNamespace(submit=AsyncMock(side_effect=submit), cancel=Mock(), stats=lambda: {},
                             sched=SimpleNamespace(ex=SimpleNamespace(spec=None)))
    messages = [{"role": "user", "content": "hi " * 40}, {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "now this"}]

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(chat_app(engine, render, tok)), base_url="http://test") as c:
            return await c.post("/v1/chat/completions", json={"messages": messages, "max_tokens": 4})
    assert asyncio.run(go()).status_code == 200
    (ids, params), = submitted
    assert params.message_boundary == len(render(messages[:-1], add_generation_prompt=False)) > 0
    assert renders[:2] == [2, 3]  # the head, then the prompt: one more render
    assert ids == [ord(c) for c in render(messages)]


# ---- the step boundary: nothing private outlives it ----

def test_a_failed_batched_forward_frees_every_rows_unwritten_destinations():
    """A batched forward fails after its snapshots were planned (a MemoryError retracts only one row): every row's
    planned destinations were never written, and none outlives the step."""
    h = harness(slots=16, max_batch=4)
    a = h.ex.admit(draw(random.Random("a"), 1500), SamplingParams(max_new_tokens=4, ignore_eos=True))
    b = h.ex.admit(draw(random.Random("b"), 1500), SamplingParams(max_new_tokens=4, ignore_eos=True))
    free = h.ex.state_pool.free_count
    forward = h.ex.model.forward

    def fail(ids, ctx):
        assert a.private and b.private  # planned for both rows
        raise MemoryError("injected")
    h.ex.model.forward = fail
    with pytest.raises(MemoryError):
        h.ex.prefill_batch([(a, 1024), (b, 1024)])
    assert not a.private and not b.private and h.ex.state_pool.free_count == free
    h.ex.state.check()
    h.ex.model.forward = forward
    h.ex.prefill_batch([(a, 1024), (b, 1024)])
    assert depth(a) == depth(b) == 1024 and not a.private and not b.private
    for r in (a, b):
        r.cacheable = False
        h.ex.release(r)
    h.ex.state.check()


def test_no_snapshot_is_placed_or_inserted_while_the_cache_flushes():
    h = harness(slots=16, max_batch=4)
    prompt = draw(random.Random("flush"), 6000)
    req = h.ex.admit(prompt, SamplingParams(max_new_tokens=4, ignore_eos=True))
    h.ex.prefill(req, max_tokens=2048)
    assert depth(req) == 2048 and not req.private
    h.ex.flush_cache()
    assert h.ex.cache.flushing  # the request's lock keeps its path until it ends
    while not req.prefill_done:
        h.ex.prefill(req, max_tokens=2048)
        assert not req.private and depth(req) == 2048  # no chunk end, no last chunk end
        h.ex.state.check()
    while not req.finished:
        h.ex.decode_step([req])
    h.ex.release(req)
    assert not h.ex.cache.flushing and not h.ex.cache.root.children and h.ex.state_pool.free_count == 15


# ---- observability ----

def test_cache_stats_report_locks_snapshots_evictions_pruning_and_draft_resumes():
    h = harness(slots=8, max_batch=4)
    prompt = draw(random.Random("stats"), 700)
    run_turn(h, prompt)
    run_turn(h, prompt + draw(random.Random("more"), 300))
    req = h.ex.admit(prompt + draw(random.Random("locked"), 100), SamplingParams(max_new_tokens=4, ignore_eos=True))
    stats = h.ex.cache_stats()
    assert stats["locks"] == {"total": 1, "request": 1}
    assert set(stats["snapshots"]) == {"spare", "not_spare"} and stats["snapshots"]["spare"] == 2  # the reply ends
    assert stats["draft_resumes"] == {"hits": 1, "fallbacks": 0}
    assert {"evicted", "pruned", "full_reprefills", "branching_resumes"} <= set(stats)
    assert stats["evictable"]["snapshots_protected"] == 1 and stats["evictable"]["kv_tokens_protected"] == req.prefix_len
    h.ex.cache.evict(need_slots=h.ex.state_pool.num_slots)
    stats = h.ex.cache_stats()
    assert stats["evicted"] and all(set(by) <= {"spare", "not_spare"} for by in stats["evicted"].values())
    assert set(stats["evicted"]) <= {"slots", "kv", "flush", "superseded"}
    req.cacheable = False
    h.ex.release(req)
    assert h.ex.cache_stats()["locks"] == {"total": 0}


# ---- the whole engine, randomized ----

@pytest.mark.parametrize("seed, slots, pages, flush", [(1, 8, 40_000, 0.0), (2, 12, 7_000, 0.0), (3, 10, 9_000, 0.02),
                                                      (4, 6, 40_000, 0.01)])
def test_randomized_serving_keeps_every_invariant_after_every_step(seed, slots, pages, flush):
    """Shared prefixes, identical re-sends, chat continuations (with their message boundaries), agent re-renders,
    cancellations, KV-pressure retractions and flushes on small pools. After every scheduler step `StateManager.check`
    runs (tests/unit/conftest): locks, counters, evictable totals, slot and page ownership, no private snapshot; at the
    end nothing is locked, and no resume missed the drafter's row."""
    rng = random.Random(seed)
    h = harness(slots=slots, max_batch=4, pages=pages)
    sched = Scheduler(h.ex, SchedulerConfig(max_running=4, chunk_size=1024, prefill_budget=2048, solo_chunk_size=2048,
                                            max_queued=6, reserve_tokens=0))
    done, flushes = [], 0
    for _ in range(300):
        if len(sched.queue) < 3 and rng.random() < 0.6:
            kind, boundary = rng.random(), 0
            if not done or kind < 0.3:
                prompt = draw(rng, rng.randint(100, 3000))
            else:
                prev = rng.choice(done[-12:])
                if kind < 0.5:
                    prompt = list(prev.prompt_ids)
                elif kind < 0.75:
                    boundary = len(prev.prompt_ids) + len(prev.output_ids)
                    prompt = prev.prompt_ids + prev.output_ids + draw(rng, rng.randint(40, 1200))
                else:
                    prompt = prev.prompt_ids[:-30] + draw(rng, rng.randint(60, 1500))
            try:
                sched.submit(prompt, SamplingParams(max_new_tokens=rng.randint(1, 30), ignore_eos=True,
                                                    message_boundary=boundary))
            except (QueueFull, InvalidRequest):
                pass
        for r in list(sched.prefilling + sched.running):
            if rng.random() < 0.02:
                sched.cancel(r)
        if rng.random() < flush:
            h.ex.flush_cache()
            flushes += 1
        sched.step()
        done += [r for r in h.ex.admitted if r.released and r.output_ids and r.finish_reason != "cancelled" and r not in done]
    sched.run_until_idle()
    h.ex.state.check()
    assert not h.ex.cache.locks() and not h.ex.state.requests and len(done) > 20
    assert h.proposer.cache_hits > 0 and h.proposer.cache_fallbacks == 0
    assert (h.head.inexact, h.head.unwritten_reads, h.head.inexact_reads) == ([], 0, 0)
    assert all(t.state_exact for t in h.ex.turns.values())
    assert flushes > 0 or not flush


# ---- the GPU diagnostic built on the cache's structures ----

def test_cache_trace_runs_on_the_cache_policy(tmp_path, monkeypatch):
    """`engine/cache_trace.run` on the stub executor, its GPU trace hooks stubbed: every run but the last starts cold
    (prototype 3's version crashed, its second run matching the first's checkpoint), and the last resumes at the cold
    run's last chunk end with its exact state."""
    import contextlib
    from sharar.engine import cache_trace

    @contextlib.contextmanager
    def fake_trace(model, position):
        yield {0: {"input": torch.zeros(1), "output": torch.zeros(1), "stages": {}}}
    monkeypatch.setattr(cache_trace, "last_token_trace", fake_trace)
    monkeypatch.setattr(cache_trace, "first_projection_probe", lambda model, ids: {})
    h = harness(slots=16, max_batch=4)
    h.ex.spec = None
    h.ex.model.layers = [SimpleNamespace(kind="linear_attention")]
    report = cache_trace.run(h.ex, draw(random.Random("trace"), 1000), tmp_path / "trace.json")
    assert report["boundary"] == 960
    assert {name: run["cache_prefix"] for name, run in report["runs"].items()} == {
        "live_split": 0, "rounded_live_split": 0, "cold": 0, "manual_restore": 0, "radix_restore": 960}
    assert all(run["committed_len"] == 1000 for run in report["runs"].values())
    assert not h.ex.cache.locks() and not h.ex.state.requests
