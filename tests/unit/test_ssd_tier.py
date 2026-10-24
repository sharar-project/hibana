"""The SSD tier end to end on CPU pools: the tier backs up a released
sequence's blocks, evicting its leaf writes the tail and demotes it, and the
next request restores it byte-exact (KV, page aux, snapshot, pool aux, drafter
payload) into fresh pages; the same works after a restart from the stored
headers alone, a corrupt object is a miss, and the budget deletes least
recently used SSD-only leaves."""
import dataclasses
import json
import random
import threading

import pytest
import torch

from sharar.cache.kv_pool import KvPool
from sharar.cache.radix import RadixCache, owner_kind
from sharar.cache.request_state import StateManager
from sharar.cache.snapshot import Snapshot
from sharar.cache.ssd import SsdTier
from sharar.cache import ssd, ssd_store
from sharar.cache.ssd_store import BLOCK, SsdStore, memory_backed
from sharar.cache.state_pool import StatePool
from sharar.engine.request import Request
from sharar.engine.sampler import SamplingParams
from sharar.target import DEFAULT_TARGET
from tests.unit.test_radix import snaps

CPU = torch.device("cpu")
IDENTITY = {"target": "tiny", "revision": "r1"}
TINY = dataclasses.replace(
    DEFAULT_TARGET, num_layers=4, layer_types=("linear_attention", "full_attention") * 2, num_kv_heads=1, head_dim=8,
    linear_num_key_heads=1, linear_num_value_heads=2, linear_key_head_dim=4, linear_value_head_dim=4)
DRAFT = ("draft", "rev1")


class CpuKvPool(KvPool):
    def alloc_with_ids(self, n):  # the real pool pins its upload buffer, which needs a GPU
        if n > len(self._free):
            raise MemoryError("KV pool exhausted")
        pages = [self._free.pop() for _ in range(n)]
        return torch.tensor(pages, dtype=torch.int32), pages


def engine(tmp_path, pages=4096, slots=8, budget=2**30, block=BLOCK, chunk_bytes=ssd.CHUNK_BYTES):
    kv = CpuKvPool(pages, CPU, target=TINY)
    kv.register_page_aux("index_keys", (3,), torch.bfloat16)
    sp = StatePool(slots, CPU, target=TINY)
    sp.register_aux("ple_conv", (5,), torch.float32)
    cache = RadixCache(kv, sp, CPU)
    tier = SsdTier(SsdStore(tmp_path, IDENTITY, block=block), kv, sp, CPU, budget, min_tokens=block, min_hit=block, trace=True,
                   chunk_bytes=chunk_bytes)
    feats = {}
    tier.register_codec(DRAFT, lambda h: feats[h], lambda payload, *, slot: feats.setdefault(("imported", slot), payload))
    cache.ssd = tier
    return cache, tier, kv, sp, feats


def fill(kv, sp, pages, slot, seed):
    g = torch.Generator().manual_seed(seed)
    idx = pages.long()
    for lid in kv.layer_ids:
        for pool in (kv.k[lid], kv.v[lid]):
            pool.view(torch.uint8)[idx] = torch.randint(0, 120, (len(idx), *pool.shape[1:]), generator=g, dtype=torch.uint8)
    kv.page_aux["index_keys"][idx] = torch.randn(len(idx), 3, generator=g).to(torch.bfloat16)
    sp.conv_all[:, slot] = torch.randn(sp.conv_all[:, slot].shape, generator=g).to(torch.bfloat16)
    sp.ssm_all[:, slot] = torch.randn(sp.ssm_all[:, slot].shape, generator=g)
    sp.aux["ple_conv"][slot] = torch.randn(5, generator=g)


def snapshot(kv, sp, pages, slot):
    idx = pages.long()
    return ([kv.k[l][idx].view(torch.uint8).clone() for l in kv.layer_ids] + [kv.v[l][idx].view(torch.uint8).clone() for l in kv.layer_ids]
            + [kv.page_aux["index_keys"][idx].clone(), sp.conv_all[:, slot].clone(), sp.ssm_all[:, slot].clone(), sp.aux["ple_conv"][slot].clone()])


def release(cache, tier, kv, sp, feats, ids, seed):
    pages, page_ids = kv.alloc_with_ids(len(ids))
    slot = sp.alloc()
    fill(kv, sp, pages, slot, seed)
    before = snapshot(kv, sp, pages, slot)
    feats["h"] = (3, 9, torch.randn(4, 2, generator=torch.Generator().manual_seed(seed)))
    cache.insert(ids, pages, snaps({len(ids): slot}, {len(ids): {DRAFT: "h"}}, spare=True), page_ids=page_ids)
    tier.backup(cache, ids)
    return before


def drop_snapshot(cache, sp, node):
    """As slot pressure's snapshot-only eviction leaves a node (its pages kept): no slot to write a tail from."""
    sp.free(cache._take_snapshot(node).slot)


def settle(cache, tier):
    for _ in range(100):
        tier.step(cache)
        tier.store.wait()
        if not tier.queue and not tier.through and not tier.inflight:
            return
    raise AssertionError("backup did not settle")


def evict_all(cache, kv):
    cache.evict(need_pages=kv.num_pages, need_slots=cache.state_pool.num_slots)


def restored(cache, kv, sp, ids):
    m = cache.match(ids)
    assert m.length == len(ids)
    return snapshot(kv, sp, m.pages, m.snapshot_slot)


def same(a, b):
    return all(torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)) for x, y in zip(a, b))


IDS = list(range(1000, 1000 + 2 * BLOCK + 37))  # two full blocks and a 37-token tail


def test_back_up_evict_and_restore_byte_exact(tmp_path):
    cache, tier, kv, sp, feats = engine(tmp_path)
    before = release(cache, tier, kv, sp, feats, IDS, seed=1)
    settle(cache, tier)
    assert tier.backed_blocks == 2 and all(r.ready for n in cache.root.children.values() for r in n.blocks.values())
    free = kv.free_count
    evict_all(cache, kv)
    settle(cache, tier)
    assert kv.free_count == free + len(IDS) and cache.cached_tokens == 0 and tier.tails == 1  # demoted, pages back
    assert cache.match(IDS + [7]).length == 0
    got = tier.restore(cache, IDS + [7])
    assert got == len(IDS) and cache.promotions >= 1
    (row,) = _rows(tier)
    assert row["tokens"] == len(IDS) and row["objects"] == 3  # two blocks and the tail, traced when asked
    assert same(restored(cache, kv, sp, IDS), before)
    (scalars_and_views,) = [v for k, v in feats.items() if isinstance(k, tuple) and k[0] == "imported"]
    assert scalars_and_views[:2] == (3, 9) and torch.equal(scalars_and_views[2], feats["h"][2])


def test_a_block_backup_describes_its_payload_as_its_views_would(tmp_path):
    """Block backups take their header's segment spec from the cached payload layout, not from per-pool views."""
    cache, tier, kv, sp, feats = engine(tmp_path)
    release(cache, tier, kv, sp, feats, IDS, seed=3)
    (node,) = cache.root.children.values()
    (_, segs, host), = tier._gather_spans(node, [(BLOCK, 2 * BLOCK)])
    (_, none, packed), = tier._gather_spans(node, [(BLOCK, 2 * BLOCK)], views=False)
    assert none is None and torch.equal(host, packed)
    assert tier._payload(BLOCK)[3] == [[name, ssd._dtype_name(t), list(t.shape), t.numel() * t.element_size()] for name, t in segs]


def test_restore_after_a_restart_from_the_headers_alone(tmp_path):
    cache, tier, kv, sp, feats = engine(tmp_path)
    before = release(cache, tier, kv, sp, feats, IDS, seed=2)
    settle(cache, tier)
    evict_all(cache, kv)
    settle(cache, tier)
    cache2, tier2, kv2, sp2, _ = engine(tmp_path)
    assert tier2.rebuild(cache2) == 1 and cache2.ssd_nodes >= 1
    assert tier2.restore(cache2, IDS + [5, 6]) == len(IDS)
    assert same(restored(cache2, kv2, sp2, IDS), before)


def test_a_shared_prefix_is_written_once_and_a_corrupt_block_is_a_miss(tmp_path):
    cache, tier, kv, sp, feats = engine(tmp_path)
    release(cache, tier, kv, sp, feats, IDS, seed=3)
    settle(cache, tier)
    other = IDS[:2 * BLOCK] + [5] * 20  # same two blocks, a different tail
    release(cache, tier, kv, sp, feats, other, seed=4)
    settle(cache, tier)
    assert tier.backed_blocks == 2  # the shared blocks went out once
    evict_all(cache, kv)
    settle(cache, tier)
    first = next(r for n in cache.root.children.values() for r in n.blocks.values())
    path = tier.store.path("kv", first.key)
    data = bytearray(path.read_bytes())
    data[-5000] ^= 0xFF
    path.write_bytes(bytes(data))
    assert tier.restore(cache, IDS + [7]) == 0 and tier.restore_fails == 1 and not first.ready
    assert cache.match(IDS + [7]).length == 0


def test_the_budget_deletes_least_recently_used_ssd_only_leaves(tmp_path):
    cache, tier, kv, sp, feats = engine(tmp_path)
    a, b = list(range(0, 300)), list(range(5000, 5300))
    release(cache, tier, kv, sp, feats, a, seed=5)
    release(cache, tier, kv, sp, feats, b, seed=6)
    settle(cache, tier)
    evict_all(cache, kv)
    settle(cache, tier)
    per_prefix = tier.used // 2
    tier.budget = per_prefix + per_prefix // 2  # room for one
    tier.step(cache)
    assert tier.used <= tier.budget and tier.deleted >= 2 and len(cache.root.children) == 1
    tier.store.wait()
    assert sum(1 for _ in tier.store.scan()) == 2  # one prefix left: its block and its tail


def test_a_restore_right_after_eviction_waits_for_the_tail_instead_of_missing(tmp_path):
    cache, tier, kv, sp, feats = engine(tmp_path)
    before = release(cache, tier, kv, sp, feats, IDS, seed=7)
    settle(cache, tier)
    evict_all(cache, kv)  # its tail is in flight, not yet written
    assert tier.restore(cache, IDS + [7]) == len(IDS) and tier.waited == 1
    assert same(restored(cache, kv, sp, IDS), before)


def test_a_larger_block_size_restores_byte_exact(tmp_path):
    cache, tier, kv, sp, feats = engine(tmp_path, pages=8192, block=1024)
    ids = list(range(3000, 3000 + 3 * 1024 + 300))
    before = release(cache, tier, kv, sp, feats, ids, seed=9)
    settle(cache, tier)
    evict_all(cache, kv)
    settle(cache, tier)
    assert tier.backed_blocks == 3 and tier.restore(cache, ids + [7]) == len(ids)
    assert same(restored(cache, kv, sp, ids), before)


def test_a_budget_victim_still_being_written_leaves_nothing_behind(tmp_path):
    cache, tier, kv, sp, feats = engine(tmp_path)
    ids = list(range(5000, 5300))
    release(cache, tier, kv, sp, feats, ids, seed=6)
    evict_all(cache, kv)  # gathers the block and the tail: in flight, not yet ready
    tier.budget = -1  # the SSD is full: the just-demoted leaf is the victim
    tier.step(cache)
    tier.budget = 2**30
    settle(cache, tier)
    assert not cache.root.children and tier.used == 0 and list(tier.store.scan()) == []


def test_a_leaf_whose_tail_is_stored_stays_restorable_without_its_snapshot(tmp_path):
    cache, tier, kv, sp, feats = engine(tmp_path)
    before = release(cache, tier, kv, sp, feats, IDS, seed=1)
    settle(cache, tier)
    evict_all(cache, kv)
    settle(cache, tier)
    assert cache.ssd_nodes == 1 and tier.restore(cache, IDS + [7]) == len(IDS) and cache.ssd_nodes == 0
    node = cache.root.children[IDS[0]]
    drop_snapshot(cache, sp, node)
    used, prefixes = tier.used, tier.prefixes
    evict_all(cache, kv)
    settle(cache, tier)
    # the tail it was restored from is still its state at that end: the leaf stays on the SSD, and restores again
    assert cache.ssd_nodes == 1 and tier.used == used and tier.prefixes == prefixes
    assert tier.restore(cache, IDS + [7]) == len(IDS) and same(restored(cache, kv, sp, IDS), before)


def test_a_failed_tail_write_removes_the_unrestorable_leaf(tmp_path, monkeypatch):
    cache, tier, kv, sp, feats = engine(tmp_path)
    release(cache, tier, kv, sp, feats, IDS, seed=3)
    settle(cache, tier)
    real = tier.store._write

    def full_disk(header, tensors, buf):
        if header.kind == "tail":
            raise OSError(28, "No space left on device")
        return real(header, tensors, buf)
    monkeypatch.setattr(tier.store, "_write", full_disk)
    evict_all(cache, kv)
    settle(cache, tier)
    assert not cache.root.children and tier.store.write_errors == 1 and tier.restore(cache, IDS + [7]) == 0


LONG = list(range(5000, 5000 + 6 * BLOCK + 11))  # six full blocks and an 11-token tail


def test_a_block_queued_twice_is_written_once(tmp_path):
    """The disk budget's clearing of GPU-resident copies, then a re-queue, can leave one block in the queue twice."""
    cache, tier, kv, sp, feats = engine(tmp_path)
    release(cache, tier, kv, sp, feats, IDS, seed=9)
    tier.queue.extend(list(tier.queue))
    tier.step(cache)
    assert tier.backed_blocks == 2 and len(tier.inflight) == 2
    assert tier.staged == sum(n for _, n in tier.inflight.values())
    settle(cache, tier)
    assert tier.store.writes == 2 and all(r.ready for r in blocks_of(cache))


def test_a_prefill_step_backs_up_past_the_decode_step_budget_and_leaves_room_for_a_tail(tmp_path):
    """A decode step gathers `step_bytes`; a prefill step gathers what the staging limit takes, less one tail's room."""
    cache, tier, kv, sp, feats = engine(tmp_path)
    cost = tier._block_charge()
    tier.step_bytes = cost
    release(cache, tier, kv, sp, feats, LONG, seed=5)  # six blocks queued
    tier.step(cache)
    assert tier.backed_blocks == 1
    tier.store.wait()  # the next step applies that write first: nothing in flight
    tier.staging_bytes = tier._tail_room + 2 * cost
    tier.step(cache, prefilling=True)
    assert tier.backed_blocks == 3 and tier.staged == 2 * cost
    tier.staging_bytes = 2**30
    tier.step(cache, prefilling=True)
    assert tier.backed_blocks == 6 and not tier.queue


def test_a_restore_streams_through_a_staging_buffer_smaller_than_it(tmp_path):
    cache, tier, kv, sp, feats = engine(tmp_path, chunk_bytes=1)  # each half holds one object: seven chunks, alternating halves
    before = release(cache, tier, kv, sp, feats, LONG, seed=11)
    settle(cache, tier)
    evict_all(cache, kv)
    settle(cache, tier)
    size = tier._staging.nbytes
    assert tier.restore(cache, LONG + [7]) == len(LONG)
    assert same(restored(cache, kv, sp, LONG), before)
    assert tier._staging.nbytes == size  # sized at startup; a restore never grows it


def test_a_corrupt_block_mid_restore_returns_every_page_and_slot(tmp_path):
    cache, tier, kv, sp, feats = engine(tmp_path, chunk_bytes=1)
    release(cache, tier, kv, sp, feats, LONG, seed=12)
    settle(cache, tier)
    evict_all(cache, kv)
    settle(cache, tier)
    node, refs = cache.root, []
    while node.children:
        (node,) = node.children.values()
        refs += [r for _, r in sorted(node.blocks.items())]
    path = tier.store.path("kv", refs[3].key)  # the fourth chunk reads it, after three scatters
    data = bytearray(path.read_bytes())
    data[-5000] ^= 0xFF
    path.write_bytes(bytes(data))
    pages, slots = kv.free_count, sp.free_count
    assert tier.restore(cache, LONG + [7]) == 0 and tier.restore_fails == 1
    assert kv.free_count == pages and sp.free_count == slots and cache.match(LONG + [7]).length == 0


def test_a_block_deleted_behind_the_trees_back_fails_the_restore_cleanly(tmp_path):
    cache, tier, kv, sp, feats = engine(tmp_path, chunk_bytes=1)
    release(cache, tier, kv, sp, feats, LONG, seed=13)
    settle(cache, tier)
    evict_all(cache, kv)
    settle(cache, tier)
    node, refs = cache.root, []
    while node.children:
        (node,) = node.children.values()
        refs += [r for _, r in sorted(node.blocks.items())]
    tier.store.path("kv", refs[3].key).unlink()
    pages, slots = kv.free_count, sp.free_count
    assert tier.restore(cache, LONG + [7]) == 0 and tier.restore_fails == 1 and not refs[3].ready
    assert kv.free_count == pages and sp.free_count == slots and cache.match(LONG + [7]).length == 0


def test_the_plan_reserve_does_not_grow_with_the_context():
    assert SsdTier.memory_bound(DEFAULT_TARGET, 1024) < 1.2 * 2**30
    assert SsdTier.memory_bound(DEFAULT_TARGET, 256) == SsdTier.memory_bound(DEFAULT_TARGET, 1024)
    tail = 1024 * DEFAULT_TARGET.kv_bytes_per_token + DEFAULT_TARGET.state_bytes_per_slot
    assert SsdTier.memory_bound(DEFAULT_TARGET, 1024, staging_bytes=1, chunk_bytes=1) >= 3 * tail + DEFAULT_TARGET.state_bytes_per_slot


def test_a_store_kept_in_memory_is_refused(tmp_path, monkeypatch):
    mounts = "tmpfs /tmp tmpfs rw 0 0\n/dev/nvme0n1p2 / ext4 rw 0 0\n/dev/nvme1n1 /mnt/my\\040ssd xfs rw 0 0\n"
    assert memory_backed("/tmp/sharar-ssd", mounts) == "tmpfs" and memory_backed("/tmp", mounts) == "tmpfs"
    assert memory_backed("/tmpx/a", mounts) is None and memory_backed("/home/u", mounts) is None
    assert memory_backed("/mnt/my ssd/x", mounts) is None
    monkeypatch.setattr(ssd_store, "memory_backed", lambda path: "tmpfs")
    with pytest.raises(OSError, match="SHARAR_SSD_DIR"):
        SsdStore(tmp_path, {})


class Peer:
    """The other tensor-parallel rank, as this rank's agreements see it: how many of the writes (in submission order)
    it has completed, which of them failed there or are a 4 KiB page longer there, and which objects of a restore it
    could not read."""

    def __init__(self, landed=10**9, failed=(), longer=(), unreadable=(), restore_done=True):
        self.landed, self.failed, self.longer, self.unreadable = landed, set(failed), set(longer), set(unreadable)
        self.restore_done = restore_done  # whether its asynchronous restore's reads have landed
        self.applied = 0

    def min_over_ranks(self, question, values):
        if question == "restore":
            return [min(values[0], int(self.restore_done))]
        first = self.applied - len(values)  # the writes the last "landed" answer released
        if question == "landed":
            n = min(values[0], self.landed - self.applied)
            self.applied += n
            return [n]
        if question == "outcomes":  # written (0/1) for each released write, then their negated sizes
            n = len(values) // 2
            first = self.applied - n
            return ([0 if first + i in self.failed else v for i, v in enumerate(values[:n])]
                    + [v - 4096 if first + i in self.longer else v for i, v in enumerate(values[n:])])
        if question == "sizes":
            return [v - 4096 if first + i in self.longer else v for i, v in enumerate(values)]  # minus the negated size
        if question == "read":
            return [0 if i in self.unreadable else v for i, v in enumerate(values)]
        return values


def gated(tier):
    """Hold each of the tier's writes until the test opens its gate (by key), so writes finish when and in the order
    a test says. Returns the gate lookup."""
    gates, lock, real = {}, threading.Lock(), tier.store._write

    def gate(key):
        with lock:
            return gates.setdefault(key, threading.Event())

    def write(header, tensors, *rest):
        assert gate(header.key).wait(10), "a gated write was never let through"
        return real(header, tensors, *rest)

    tier.store._write = write
    return gate


def six_blocks_in_flight(tmp_path, seed):
    cache, tier, kv, sp, feats = engine(tmp_path)
    gate = gated(tier)
    tier.step_bytes = 2**30
    release(cache, tier, kv, sp, feats, LONG, seed=seed)
    tier.step(cache)
    cost = tier._block_charge()
    assert tier.staged == 6 * cost and len(tier.inflight) == 6
    tier.staging_bytes = 6 * cost
    return cache, tier, gate, cost


def test_a_wait_for_staging_room_waits_for_the_oldest_writes_that_free_it_and_no_others(tmp_path):
    cache, tier, gate, cost = six_blocks_in_flight(tmp_path, seed=7)
    for _, key in list(tier.inflight)[:2]:
        gate(key).set()
    assert tier._room(2 * cost, True)
    assert tier.room_waits == 1 and tier.room_wait_writes == 2 and len(tier.inflight) == 4 and tier.staged == 4 * cost
    for _, key in list(tier.inflight):
        gate(key).set()
    settle(cache, tier)


def test_tensor_parallel_a_wait_for_staging_room_applies_the_same_writes_on_every_rank(tmp_path):
    """Each rank waits for its own oldest writes, so one agreement applies them on every rank."""
    cache, tier, gate, cost = six_blocks_in_flight(tmp_path, seed=8)
    peer = Peer(landed=2)  # it waited for its own two oldest, and no more of its writes have finished
    asked = []

    def counted(question, values):
        asked.append(question)
        return peer.min_over_ranks(question, values)

    tier._min_over_ranks = counted
    for _, key in list(tier.inflight)[:2]:
        gate(key).set()
    assert tier._room(2 * cost, True)
    assert asked.count("landed") == 1 and peer.applied == 2 and tier.room_wait_writes == 2
    assert len(tier.inflight) == 4 and tier.staged + 2 * cost <= tier.staging_bytes
    peer.landed = 10**9
    for _, key in list(tier.inflight):
        gate(key).set()
    settle(cache, tier)


def blocks_of(cache):
    (node,) = cache.root.children.values()
    return [r for _, r in sorted(node.blocks.items())]


def test_tensor_parallel_a_write_counts_once_every_rank_completed_it(tmp_path):
    cache, tier, kv, sp, feats = engine(tmp_path)
    peer = Peer(landed=0)
    tier._min_over_ranks = peer.min_over_ranks
    release(cache, tier, kv, sp, feats, IDS, seed=1)
    tier.step(cache)  # gathers and writes both blocks
    tier.store.wait()
    tier.step(cache)
    assert len(tier.landed) == 2 and [r.ready for r in blocks_of(cache)] == [False, False] and tier.used == 0
    peer.landed = 1
    tier.step(cache)
    assert [r.ready for r in blocks_of(cache)] == [True, False] and len(tier.inflight) == 1  # the first, in order
    peer.landed = 2
    tier.step(cache)
    assert [r.ready for r in blocks_of(cache)] == [True, True] and not tier.inflight and not tier.landed


def test_tensor_parallel_a_write_that_failed_on_another_rank_fails_here_and_its_file_goes(tmp_path):
    cache, tier, kv, sp, feats = engine(tmp_path)
    tier._min_over_ranks = Peer(failed={1}).min_over_ranks
    release(cache, tier, kv, sp, feats, IDS, seed=2)
    settle(cache, tier)
    first, second = blocks_of(cache)
    assert first.ready and not second.ready and second.nbytes == -1
    tier.store.wait()
    assert tier.store.size("kv", first.key) is not None and tier.store.size("kv", second.key) is None
    assert tier.used == first.nbytes


def test_tensor_parallel_a_size_is_the_largest_any_rank_wrote_and_still_restores(tmp_path):
    cache, tier, kv, sp, feats = engine(tmp_path)
    tier._min_over_ranks = Peer(longer={0}).min_over_ranks  # the other rank's header crossed a 4 KiB page
    before = release(cache, tier, kv, sp, feats, IDS, seed=6)
    settle(cache, tier)
    first, second = blocks_of(cache)
    assert first.nbytes == tier.store.size("kv", first.key) + 4096 and second.nbytes == tier.store.size("kv", second.key)
    evict_all(cache, kv)
    settle(cache, tier)
    assert tier.restore(cache, IDS + [7]) == len(IDS) and same(restored(cache, kv, sp, IDS), before)


def test_tensor_parallel_a_restore_another_rank_could_not_read_fails_on_every_rank(tmp_path):
    cache, tier, kv, sp, feats = engine(tmp_path)
    release(cache, tier, kv, sp, feats, IDS, seed=3)
    settle(cache, tier)
    evict_all(cache, kv)
    settle(cache, tier)
    tier._min_over_ranks = Peer(unreadable={2}).min_over_ranks  # the tail: blocks 0 and 1, then the tail
    free = kv.free_count, sp.free_count
    assert tier.restore(cache, IDS + [7]) == 0 and tier.restore_fails == 1
    assert (kv.free_count, sp.free_count) == free and cache.match(IDS + [7]).length == 0
    assert not [k for k in feats if isinstance(k, tuple)]  # this rank imported no drafter features either
    assert tier.restore(cache, IDS + [7]) == 0  # the tail is lost on every rank: nothing left to resume from


def test_tensor_parallel_a_restart_lists_only_prefixes_every_rank_stored(tmp_path):
    cache, tier, kv, sp, feats = engine(tmp_path)
    other = list(range(5000, 5000 + 2 * BLOCK + 11))
    before = release(cache, tier, kv, sp, feats, IDS, seed=4)
    release(cache, tier, kv, sp, feats, other, seed=5)
    settle(cache, tier)
    evict_all(cache, kv)
    settle(cache, tier)
    cache2, tier2, kv2, sp2, _ = engine(tmp_path)
    tier2._held_by_all = lambda values: [v for v in values if v == ssd._digest(IDS)]  # the other rank lost `other`
    assert tier2.rebuild(cache2) == 1
    assert tier2.restore(cache2, other + [5]) == 0 and tier2.restore(cache2, IDS + [5]) == len(IDS)
    assert same(restored(cache2, kv2, sp2, IDS), before)
    tier2.store.wait()
    assert sum(1 for h, _ in tier2.store.scan() if h.kind == "tail") == 1  # rebuild deleted `other`'s tail and blocks


def test_a_tail_is_written_through_at_each_resume_point_it_is_given_and_the_spare_end_goes_first(tmp_path):
    """Updated from main (main wrote through only the end's tail, never the prompt end's). `backup` writes a tail
    through at each of its resume points: the caller gives a prompt's non-spare snapshots (its last chunk end here)
    and, at release, the reply end. Slot pressure then takes the spare reply end first (a leaf: whole, demoted with
    its tail), then the last chunk end's snapshot alone (its only child is on the SSD), which still restores from
    its tail."""
    cache, tier, kv, sp, feats = engine(tmp_path)
    prompt_end, n = 2 * BLOCK + 10, len(IDS)
    pages, page_ids = kv.alloc_with_ids(n)
    p_slot, n_slot = sp.alloc(), sp.alloc()
    fill(kv, sp, pages, n_slot, seed=21)
    g = torch.Generator().manual_seed(22)
    sp.conv_all[:, p_slot] = torch.randn(sp.conv_all[:, p_slot].shape, generator=g).to(torch.bfloat16)
    sp.ssm_all[:, p_slot] = torch.randn(sp.ssm_all[:, p_slot].shape, generator=g)
    sp.aux["ple_conv"][p_slot] = torch.randn(5, generator=g)
    before = snapshot(kv, sp, pages[:prompt_end], p_slot)
    feats["h"] = (3, 9, torch.randn(4, 2, generator=g))
    cache.insert(IDS, pages, two_snapshots(prompt_end, p_slot, n, n_slot), page_ids=page_ids)
    tier.backup(cache, IDS, (n, prompt_end))
    settle(cache, tier)
    (head,) = cache.root.children.values()
    (leaf,) = head.children.values()
    assert tier.tails_through == 2 and head.tail.ready and leaf.tail.ready and head.depth == prompt_end
    cache.evict(need_slots=sp.free_count + 1)  # slots short, pages ample: the spare reply end goes first, whole
    assert head.snapshot_slot is not None and head.in_l1 and not leaf.in_l1 and leaf.tail.ready
    assert cache.match(IDS[:prompt_end] + [7]).length == prompt_end  # the last chunk end still resumes from L1
    cache.evict(need_slots=sp.free_count + 1)  # then its snapshot alone: its only child is on the SSD
    assert head.snapshot_slot is None and head.in_l1 and cache.snapshot_evictions == 1
    assert cache.match(IDS[:prompt_end] + [7]).length == 0
    assert tier.restore(cache, IDS[:prompt_end] + [7]) == prompt_end  # its tail alone: its pages are still in L1
    assert same(restored(cache, kv, sp, IDS[:prompt_end]), before)
    assert tier.restore(cache, IDS + [7]) == 0  # the end is 27 tokens past it: under min_hit, prefilled instead
    evict_all(cache, kv)
    settle(cache, tier)
    assert tier.tails == 2  # each written once


def test_write_through_stays_within_the_staging_limit_and_skips_a_snapshot_dropped_meanwhile(tmp_path):
    """Writing a tail through copies a whole snapshot into pinned memory: requests finishing faster than the writer
    drains wait for staging room instead of each pinning a copy. The tier skips one whose snapshot the GPU cache drops
    while it waits, rather than write it from a slot it no longer owns."""
    cache, tier, kv, sp, feats = engine(tmp_path)
    tier.staging_bytes = 1  # one object in flight at a time
    prefixes = [[s] + list(range(1000, 1000 + 2 * BLOCK + 9)) for s in (1, 2, 3)]
    befores = []
    for i, ids in enumerate(prefixes):
        pages, page_ids = kv.alloc_with_ids(len(ids))
        slot = sp.alloc()
        fill(kv, sp, pages, slot, seed=30 + i)
        befores.append(snapshot(kv, sp, pages, slot))
        feats["h"] = (3, 9, torch.randn(4, 2, generator=torch.Generator().manual_seed(30 + i)))
        cache.insert(ids, pages, snaps({len(ids): slot}, {len(ids): {DRAFT: "h"}}, spare=True), page_ids=page_ids)
        tier.backup(cache, ids, (len(ids),))
    assert tier.staged == 0 and tier.tails_through == 0  # backup copies nothing as the requests finish
    tier.step(cache)
    assert tier.tails_through == 1 and len(tier.inflight) == 1
    third = cache.root.children[prefixes[2][0]]
    drop_snapshot(cache, sp, third)  # while its write waits for room
    for _ in range(100):
        tier.store.wait()
        tier.step(cache)
        assert len(tier.inflight) <= 1
        if not tier.queue and not tier.through and not tier.inflight:
            break
    assert tier.tails_through == 2 and tier.through_dropped == 1 and third.tail is None
    evict_all(cache, kv)
    settle(cache, tier)
    assert tier.tails == 2  # eviction found both written tails, and the third had nothing to write
    assert tier.restore(cache, prefixes[2] + [7]) == 0
    for ids, before in zip(prefixes[:2], befores):
        assert tier.restore(cache, ids + [7]) == len(ids)
        assert same(restored(cache, kv, sp, ids), before)


def test_a_flush_forgets_the_deletions_the_old_epoch_owed(tmp_path, monkeypatch):
    """The tier owes a budget victim's writes still in flight a deletion when they land; a flush forgets those with
    the old epoch. Written again in the new epoch, the same prefix has the same keys, which must stay."""
    cache, tier, kv, sp, feats = engine(tmp_path)
    release(cache, tier, kv, sp, feats, IDS, seed=7)
    evict_all(cache, kv)  # gathers its blocks and tail: in flight
    with monkeypatch.context() as m:
        m.setattr(tier.store, "completions", lambda: [])  # none has landed yet
        tier.budget = -1  # the SSD is full: the just-demoted leaf is the victim
        tier.step(cache)
    tier.budget = 2**30
    assert not cache.root.children and len(tier.doomed) == 3
    cache.flush()
    before = release(cache, tier, kv, sp, feats, IDS, seed=8)
    settle(cache, tier)
    evict_all(cache, kv)
    settle(cache, tier)
    assert tier.deleted == 0 and tier.restore(cache, IDS + [7]) == len(IDS)  # the old epoch's files went with it
    assert same(restored(cache, kv, sp, IDS), before)


@pytest.mark.parametrize("write_through", [False, True])
def test_a_prefix_reinserted_before_its_discarded_writes_finish_reuses_them(tmp_path, monkeypatch, write_through):
    cache, tier, kv, sp, feats = engine(tmp_path)
    before = release(cache, tier, kv, sp, feats, IDS, seed=41)
    evict_all(cache, kv)
    staged = tier.staged
    with monkeypatch.context() as m:
        m.setattr(tier.store, "completions", lambda: [])  # the old writes have not completed on all ranks yet
        tier.budget = -1
        tier.step(cache)
        assert not cache.root.children and len(tier.doomed) == 3
        tier.budget = 2**30
        release(cache, tier, kv, sp, feats, IDS, seed=41)
        if write_through:
            tier.backup(cache, IDS, (len(IDS),))
            tier.step(cache)
        else:
            evict_all(cache, kv)
        assert tier.staged == staged and len(tier.inflight) == 3 and not tier.doomed
    settle(cache, tier)
    assert tier.staged == 0 and not tier.inflight and tier.store.writes == 3
    assert len(list(tier.store.scan())) == 3
    evict_all(cache, kv)
    assert tier.restore(cache, IDS + [7]) == len(IDS)
    assert same(restored(cache, kv, sp, IDS), before)


@pytest.mark.parametrize("split", [False, True])
def test_the_disk_budget_drops_gpu_resident_copies_and_eviction_can_back_them_up_again(tmp_path, split):
    cache, tier, kv, sp, feats = engine(tmp_path, budget=16384)
    before = release(cache, tier, kv, sp, feats, IDS, seed=42)
    if split:
        cut = BLOCK + 10  # one block spans parent and child; another lies entirely in the parent
        cache.insert(IDS[:cut], cache.match(IDS).pages[:cut], snaps({cut: sp.alloc()}, spare=True))
    tier.backup(cache, IDS, (len(IDS),))
    free = kv.free_count, sp.free_count
    settle(cache, tier)
    assert tier.used <= tier.budget
    assert sum(n for _, n in tier.store.scan()) <= tier.budget
    assert (kv.free_count, sp.free_count) == free and cache.cached_tokens == len(IDS)
    assert same(restored(cache, kv, sp, IDS), before)
    tier.budget = 2**30
    evict_all(cache, kv)  # eviction re-queues the blocks the disk budget removed before it frees their pages
    settle(cache, tier)
    assert tier.restore(cache, IDS + [7]) == len(IDS)
    assert same(restored(cache, kv, sp, IDS), before)


@pytest.mark.parametrize("operation", ["step", "eviction", "drain", "restore"])
@pytest.mark.parametrize("capacity", [0, 1, 1.5])
def test_gathers_respect_the_staging_limit_including_when_forced(tmp_path, monkeypatch, operation, capacity):
    cache, tier, kv, sp, feats = engine(tmp_path)
    tier.staging_bytes = max(1, int(capacity * tier.block * tier._row))
    real_write, peaks = tier.store.write, []

    def write(header, tensors, ready=None):
        peaks.append(tier.staged)
        assert tier.staged <= tier.staging_bytes or (capacity == 0 and len(tier.inflight) == 1)
        real_write(header, tensors, ready)

    monkeypatch.setattr(tier.store, "write", write)
    before = release(cache, tier, kv, sp, feats, LONG, seed=44)
    if operation == "step":
        tier.step(cache)
        settle(cache, tier)
        evict_all(cache, kv)
    elif operation == "drain":
        tier.backup(cache, LONG, (len(LONG),))
        tier.drain(cache)
        assert not tier.queue and not tier.through and not tier.inflight and tier.staged == 0
        evict_all(cache, kv)
    else:
        tier.patient = True  # as at shutdown: serving eviction drops what does not fit (below)
        evict_all(cache, kv)
        if operation == "restore":
            other = [t + 20000 for t in LONG]
            release(cache, tier, kv, sp, feats, other, seed=45)
            tier.backup(cache, other, (len(other),))
            assert tier.restore(cache, LONG + [7]) == len(LONG) and tier.waited == 1
    settle(cache, tier)
    assert peaks and tier.staged == 0
    if operation != "restore":
        assert tier.restore(cache, LONG + [7]) == len(LONG)
    assert same(restored(cache, kv, sp, LONG), before)


def test_serving_eviction_drops_a_leaf_it_cannot_copy_instead_of_waiting_for_the_writers(tmp_path, monkeypatch):
    cache, tier, kv, sp, feats = engine(tmp_path)
    release(cache, tier, kv, sp, feats, LONG, seed=44)
    tier.staging_bytes = 1
    tier.staged = 1  # another copy holds the staging buffer
    monkeypatch.setattr(tier.store, "wait", lambda: pytest.fail("eviction waited for the writers"))
    monkeypatch.setattr(tier.store, "wait_first", lambda n: pytest.fail("eviction waited for the writers"))
    evict_all(cache, kv)
    assert tier.evict_dropped == 1 and not tier.inflight and cache.ssd_prefixes() == 0
    tier.staged = 0
    monkeypatch.undo()
    settle(cache, tier)
    assert tier.restore(cache, LONG + [7]) == 0 and tier.used == 0


def test_serving_eviction_waits_for_room_for_a_tail_instead_of_dropping_a_stored_prefix(tmp_path):
    cache, tier, kv, sp, feats = engine(tmp_path)
    before = release(cache, tier, kv, sp, feats, LONG, seed=44)
    settle(cache, tier)  # its blocks are stored; its tail is not
    other = [t + 20000 for t in LONG]
    release(cache, tier, kv, sp, feats, other, seed=45)
    cache.lock(cache.node_at(other), owner="test")
    tier.step(cache)  # another prefix's backups fill the staging buffer
    tier.staging_bytes = tier.staged
    evict_all(cache, kv)
    assert tier.evict_dropped == 0 and tier.tails == 1
    settle(cache, tier)
    assert tier.restore(cache, LONG + [7]) == len(LONG)
    assert same(restored(cache, kv, sp, LONG), before)


@pytest.mark.parametrize("entry", ["restore", "rebuild"])
@pytest.mark.parametrize("damage", ["state_name", "kv_name", "state_shape", "kv_shape", "state_dtype"])
def test_an_incompatible_tensor_layout_is_a_miss_before_scatter_or_import(tmp_path, monkeypatch, entry, damage):
    cache, tier, kv, sp, feats = engine(tmp_path)
    real_write = tier.store._write

    def write(header, tensors, *rest):
        if (header.kind == "tail") == damage.startswith("state"):
            name = "conv" if damage.startswith("state") else "kv.0.k"
            spec = next(s for s in header.segments if s[0] == name)
            if damage.endswith("name"):
                spec[0] += "_corrupt"
            elif damage.endswith("shape"):
                spec[2] = [1, *spec[2]]  # same byte count and dtype, wrong tensor geometry
            else:
                spec[1] = "int16"  # conv was bfloat16: same size, wrong interpretation
        return real_write(header, tensors, *rest)

    monkeypatch.setattr(tier.store, "_write", write)
    release(cache, tier, kv, sp, feats, IDS, seed=46)
    evict_all(cache, kv)
    settle(cache, tier)
    if entry == "rebuild":
        cache, tier, kv, sp, feats = engine(tmp_path)
        assert tier.rebuild(cache) == 0
    free = kv.free_count, sp.free_count
    assert tier.restore(cache, IDS + [7]) == 0
    assert tier.restore_fails == (entry == "restore")
    assert (kv.free_count, sp.free_count) == free and cache.match(IDS + [7]).length == 0
    assert not [k for k in feats if isinstance(k, tuple)]


def _two_resume_points(cache, tier, kv, sp, feats, prompt_end):
    """IDS with snapshots at prompt_end and at its end, released: the end written through (the reply end's tail), the
    snapshot at prompt_end a spare chunk end, which gets none (updated from main, where it was the prompt-end copy)."""
    n = len(IDS)
    pages, page_ids = kv.alloc_with_ids(n)
    p_slot, n_slot = sp.alloc(), sp.alloc()
    fill(kv, sp, pages, n_slot, seed=41)
    g = torch.Generator().manual_seed(42)
    sp.conv_all[:, p_slot] = torch.randn(sp.conv_all[:, p_slot].shape, generator=g).to(torch.bfloat16)
    sp.ssm_all[:, p_slot] = torch.randn(sp.ssm_all[:, p_slot].shape, generator=g)
    sp.aux["ple_conv"][p_slot] = torch.randn(5, generator=g)
    at_prompt, at_end = snapshot(kv, sp, pages[:prompt_end], p_slot), snapshot(kv, sp, pages, n_slot)
    feats["h"] = (3, 9, torch.randn(4, 2, generator=g))
    cache.insert(IDS, pages, two_snapshots(prompt_end, p_slot, n, n_slot, spare=True), page_ids=page_ids)
    tier.backup(cache, IDS, (n,))
    return at_prompt, at_end


def two_snapshots(prompt_end, p_slot, n, n_slot, spare=False):
    """A released request's snapshot at prompt_end (its last chunk end, or with `spare` an earlier chunk end) and its
    reply end (spare), each with the drafter's payload."""
    return [Snapshot(p_slot, prompt_end, {DRAFT: "h"}, spare=spare), Snapshot(n_slot, n, {DRAFT: "h"}, spare=True)]


def drop_interior(cache, sp):
    """Slot pressure takes the interior snapshot while the reply end past it is locked (a request continuing the
    conversation): the node keeps its pages."""
    (head,) = cache.root.children.values()
    (leaf,) = head.children.values()
    with cache._lock_node(leaf):
        cache.evict(need_slots=sp.free_count + 1)
    return head, leaf


def _rows(tier):
    return [json.loads(line) for line in open(tier.store.base / "restores.jsonl")]


def test_a_snapshot_dropped_while_its_pages_stay_restores_from_its_tail_alone(tmp_path):
    """The write-through's case: the GPU cache dropped the prompt-end snapshot for want of state slots and kept the
    pages. The restore reads the tail (its state) and nothing else, puts no pages, and needs none of the path's
    blocks on the SSD yet."""
    cache, tier, kv, sp, feats = engine(tmp_path)
    prompt_end = 2 * BLOCK + 10
    tier.staging_bytes = 1  # one object in flight: the tails go first and the blocks wait
    at_prompt, _ = _two_resume_points(cache, tier, kv, sp, feats, prompt_end)
    (head,) = cache.root.children.values()
    tier.through.clear()  # as if eviction had met the prompt end as a leaf and written its tail
    assert tier._write_tail(head, prompt_end, wait=True)
    tier.store.wait()
    tier.step(cache)
    assert head.tail.ready  # the prompt-end tail is on the SSD
    drop_interior(cache, sp)
    assert head.snapshot_slot is None and head.in_l1
    free = kv.free_count
    assert tier.restore(cache, IDS[:prompt_end] + [7]) == prompt_end
    (row,) = _rows(tier)
    assert row["objects"] == 1 and row["kv_tokens"] == 0 and tier.waited == 0 and kv.free_count == free
    assert same(restored(cache, kv, sp, IDS[:prompt_end]), at_prompt)
    settle(cache, tier)


def test_a_restore_reads_only_the_kv_past_what_l1_still_holds(tmp_path):
    """The GPU cache dropped the prompt-end snapshot (keeping its pages), then evicted the leaf past it to the SSD:
    only the leaf's 27 tokens of KV come back from the SSD, not the 522 still in L1."""
    cache, tier, kv, sp, feats = engine(tmp_path)
    prompt_end, n = 2 * BLOCK + 10, len(IDS)
    _, at_end = _two_resume_points(cache, tier, kv, sp, feats, prompt_end)
    settle(cache, tier)
    head, leaf = drop_interior(cache, sp)  # the interior snapshot goes
    cache.evict(need_pages=kv.free_count + 1)  # then the leaf: demoted, its tail already written
    assert head.in_l1 and head.snapshot_slot is None and not leaf.in_l1
    assert cache.match(IDS + [7]).length == 0
    free = kv.free_count
    assert tier.restore(cache, IDS + [7]) == n
    (row,) = _rows(tier)
    assert row["objects"] == 1 and row["kv_tokens"] == n - prompt_end and kv.free_count == free - (n - prompt_end)
    assert same(restored(cache, kv, sp, IDS), at_end)


def test_pages_beside_the_kv_travel_with_it(tmp_path):
    """A drafter's own per-page KV (Flash-Next's MTP layer) is stored with the target's KV blocks and tails, and a
    restore puts it back byte-exact into the pages the restored prefix gets."""
    kv = CpuKvPool(4096, CPU, target=TINY)
    kv.register_page_aux("index_keys", (3,), torch.bfloat16)
    sp = StatePool(8, CPU, target=TINY)
    sp.register_aux("ple_conv", (5,), torch.float32)
    draft = torch.zeros((kv.k[kv.layer_ids[0]].shape[0], 2, 4), dtype=torch.bfloat16)  # one row per page, like a KvPool
    cache = RadixCache(kv, sp, CPU)
    tier = SsdTier(SsdStore(tmp_path, IDENTITY, block=BLOCK), kv, sp, CPU, 2**30, min_tokens=BLOCK, min_hit=BLOCK,
                   extra_pages=[("draft.0.k", draft)])
    cache.ssd = tier
    pages, page_ids = kv.alloc_with_ids(len(IDS))
    slot = sp.alloc()
    fill(kv, sp, pages, slot, seed=21)
    draft[pages.to(torch.int64)] = torch.randn((len(IDS), 2, 4), generator=torch.Generator().manual_seed(21)).to(torch.bfloat16)
    want = draft[pages.to(torch.int64)].clone()
    cache.insert(IDS, pages, snaps({len(IDS): slot}, spare=True), page_ids=page_ids)
    tier.backup(cache, IDS)
    settle(cache, tier)
    evict_all(cache, kv)
    settle(cache, tier)
    draft.zero_()
    assert tier.restore(cache, IDS + [7]) == len(IDS)
    assert torch.equal(draft[cache.match(IDS).pages.to(torch.int64)].view(torch.uint8), want.view(torch.uint8))


def test_eviction_writes_no_tail_for_a_node_whose_descendants_are_stored(tmp_path):
    """Once a conversation's end is on the SSD, the snapshot at its prompt's end (or at a chunk checkpoint) is not
    worth a copy when eviction reaches that node: the conversation resumes from the end's tail."""
    cache, tier, kv, sp, feats = engine(tmp_path)
    prompt_end = 2 * BLOCK + 10
    _, at_end = _two_resume_points(cache, tier, kv, sp, feats, prompt_end)
    settle(cache, tier)
    (head,) = cache.root.children.values()
    assert head.snapshot_slot is not None and head.tail is None and tier.tails == 1  # only the end's, written through
    evict_all(cache, kv)
    settle(cache, tier)
    assert tier.tails == 1 and head.tail is None and not head.in_l1  # demoted with its blocks, no tail written
    assert tier.restore(cache, IDS + [7]) == len(IDS) and same(restored(cache, kv, sp, IDS), at_end)


def test_serving_eviction_that_drops_a_leaf_keeps_its_ancestors_blocks_queued(tmp_path):
    """Eviction drops a leaf it cannot copy for want of staging room. The blocks of the path above it stay queued
    for their own node, which keeps its pages, and the write-through queued for the leaf finds it gone: nothing is
    copied from the pages and slot the leaf gave back."""
    cache, tier, kv, sp, feats = engine(tmp_path)
    prompt_end = 2 * BLOCK + 10
    at_prompt, _ = _two_resume_points(cache, tier, kv, sp, feats, prompt_end)
    tier.staging_bytes = 1
    tier.staged = 1  # another copy holds the staging buffer
    cache.evict(need_pages=kv.free_count + 1)  # the leaf past the prompt end goes
    (head,) = cache.root.children.values()
    assert tier.evict_dropped == 1 and not head.children and head.in_l1
    assert len(tier.queue) == 2 and all(item[0] is head for item in tier.queue)
    tier.staged = 0
    settle(cache, tier)
    assert all(r.ready for r in head.blocks.values()) and tier.through_dropped == 1 and tier.tails == 0
    evict_all(cache, kv)  # the head is the leaf now: its tail goes out and it stays on the SSD
    settle(cache, tier)
    assert tier.restore(cache, IDS[:prompt_end] + [7]) == prompt_end
    assert same(restored(cache, kv, sp, IDS[:prompt_end]), at_prompt)


# ---- asynchronous restores (admission keeps serving while the reads land) ----

def land(cache, tier, ids, reserve=0, limit=200):
    """Admission's loop for one request: ask, and step the engine until its restore has landed. Returns the steps."""
    for steps in range(limit):
        if cache.restore_async(ids, reserve):
            return steps
        if tier._job is not None:
            tier._drive(tier._job)  # the engine scatters each chunk as it lands; it would keep serving meanwhile
        tier.step(cache)
    raise AssertionError("the restore never landed")


def locks(cache):
    """Every node's lock count (`lock_ref`: the locks at or below it)."""
    out, stack = [], list(cache.root.children.values())
    while stack:
        n = stack.pop()
        out.append(n.lock_ref)
        stack.extend(n.children.values())
    return out


def stored(tmp_path, ids=IDS, seed=1):
    cache, tier, kv, sp, feats = engine(tmp_path)
    before = release(cache, tier, kv, sp, feats, ids, seed=seed)
    settle(cache, tier)
    evict_all(cache, kv)
    settle(cache, tier)
    return cache, tier, kv, sp, feats, before


@pytest.mark.parametrize("ids", [IDS, LONG])
def test_an_asynchronous_restore_lands_byte_exact_and_holds_its_path_until_admission(tmp_path, ids):
    cache, tier, kv, sp, feats, before = stored(tmp_path, ids)
    if ids is LONG:
        tier._staging.close()
        tier._staging = ssd.Staging(2 * 4 * tier.block * tier._row, CPU)  # several chunks through two small halves
        tier.chunk_bytes = 2 * tier.block * tier._row
    assert not cache.restore_async(ids + [7]) and tier._job is not None and tier.async_restores == 1
    assert land(cache, tier, ids + [7]) >= 1 and tier.restores == 1 and tier._job is None
    assert same(restored(cache, kv, sp, ids), before)
    evict_all(cache, kv)  # held for its request: eviction cannot take it before admission
    assert cache.match(ids + [7]).length == len(ids)
    cache.release(ids + [7])  # admitted (the request holds it now) or gone
    assert not any(locks(cache))
    evict_all(cache, kv)
    assert cache.match(ids + [7]).length == 0


def test_while_a_restore_reads_others_are_served_and_its_path_survives_the_budget(tmp_path):
    cache, tier, kv, sp, feats, before = stored(tmp_path)
    tier._min_over_ranks = Peer(restore_done=False).min_over_ranks  # the other rank is still reading
    assert not cache.restore_async(IDS + [7])
    tier._drive(tier._job)
    other = list(range(9000, 9100))
    assert cache.restore_async(other + [1])  # nothing to restore: admitted at once
    tier.budget = 0  # the disk budget would drop every SSD-only leaf
    for _ in range(3):
        tier.step(cache)
    assert tier._job is not None and not cache.restore_async(IDS + [7]) and tier.restore_wait_steps >= 1
    tier._min_over_ranks = Peer().min_over_ranks  # now it landed there too
    tier.step(cache)
    assert cache.restore_async(IDS + [7]) and same(restored(cache, kv, sp, IDS), before)
    cache.release(IDS + [7])
    assert not any(locks(cache))


def test_the_disk_budget_keeps_the_tail_a_restore_reads_alone(tmp_path, monkeypatch):
    """The pages stayed and only the snapshot went, so the restore reads one tail: a copy of a node still in L1, the
    kind the budget's second pass deletes. The budget keeps it while the read is in flight, and the restore lands."""
    cache, tier, kv, sp, feats = engine(tmp_path)
    prompt_end = 2 * BLOCK + 10
    at_prompt, _ = _two_resume_points(cache, tier, kv, sp, feats, prompt_end)
    (head,) = cache.root.children.values()
    tier.through.clear()  # as if eviction had met the prompt end as a leaf and written its tail
    assert tier._write_tail(head, prompt_end, wait=True)
    settle(cache, tier)
    drop_interior(cache, sp)
    assert head.snapshot_slot is None and head.in_l1 and head.tail.ready
    gate, read = threading.Event(), tier.store.read_into
    monkeypatch.setattr(tier.store, "read_into", lambda *a: (gate.wait(10), read(*a))[1])
    assert not cache.restore_async(IDS[:prompt_end] + [7])
    tier.budget = 0  # over budget, and no SSD-only leaf to drop: only the copies of L1 nodes are left
    tier.step(cache)
    tier.store.wait()  # any delete the budget queued has run
    assert head.tail is not None and head.tail.ready
    gate.set()
    land(cache, tier, IDS[:prompt_end] + [7])
    assert tier.restore_fails == 0 and same(restored(cache, kv, sp, IDS[:prompt_end]), at_prompt)
    cache.release(IDS[:prompt_end] + [7])
    assert not any(locks(cache))


def test_a_second_restore_waits_for_the_first_and_for_its_request(tmp_path):
    """One restore at a time reads (they share the staging buffer), and none starts while a landed one still locks a
    slot for its request (`ssd.RESTORE_SLOTS`, in the slot floor): another request asking meanwhile goes ahead without
    one. Once the first request took its own lock, the next restore starts."""
    cache, tier, kv, sp, feats = engine(tmp_path)
    second = [t + 20000 for t in IDS]
    one = release(cache, tier, kv, sp, feats, IDS, seed=5)
    two = release(cache, tier, kv, sp, feats, second, seed=6)
    settle(cache, tier)
    evict_all(cache, kv)
    settle(cache, tier)
    assert not cache.restore_async(IDS + [7])
    assert not cache.restore_async(second + [7]) and tier.async_restores == 1  # waits: one restore at a time
    land(cache, tier, IDS + [7])
    assert cache.restore_async(second + [7]) and tier.async_restores == 1  # the landed one waits for its request
    cache.release(IDS + [7])  # its request locked it itself
    land(cache, tier, second + [7])
    assert tier.async_restores == 2
    assert same(restored(cache, kv, sp, IDS), one) and same(restored(cache, kv, sp, second), two)
    cache.release(second + [7])
    assert not any(locks(cache))


def test_a_request_cancelled_mid_read_leaves_its_restore_in_the_cache_unheld(tmp_path):
    cache, tier, kv, sp, feats, before = stored(tmp_path)
    assert not cache.restore_async(IDS + [7])
    cache.release(IDS + [7])  # cancelled while its reads are landing
    tier._drive(tier._job)
    tier.step(cache)
    assert tier._job is None and not tier._held and not any(locks(cache))
    assert same(restored(cache, kv, sp, IDS), before)  # still a cache hit for whoever comes next


def test_a_held_path_whose_request_never_comes_is_let_go(tmp_path, monkeypatch):
    monkeypatch.setattr(ssd, "HOLD_STEPS", 3)
    cache, tier, kv, sp, feats, _ = stored(tmp_path)
    land(cache, tier, IDS + [7])
    assert tier._held and any(locks(cache))
    for _ in range(5):
        tier.step(cache)
    assert not tier._held and not any(locks(cache))


def test_a_flush_during_a_restore_returns_its_pages_and_slot(tmp_path):
    cache, tier, kv, sp, feats, _ = stored(tmp_path)
    free = kv.free_count, sp.free_count
    assert not cache.restore_async(IDS + [7])
    cache.flush()
    assert tier._job is None and (kv.free_count, sp.free_count) == free and not cache.flushing
    assert cache.restore_async(IDS + [7]) and cache.match(IDS + [7]).length == 0  # flushed: nothing to restore


def test_an_asynchronous_restore_another_rank_could_not_read_fails_cleanly(tmp_path):
    cache, tier, kv, sp, feats, _ = stored(tmp_path)
    tier._min_over_ranks = Peer(unreadable={2}).min_over_ranks
    free = kv.free_count, sp.free_count
    assert not cache.restore_async(IDS + [7])
    tier._drive(tier._job)
    tier.step(cache)
    assert tier._job is None and tier.restore_fails == 1 and (kv.free_count, sp.free_count) == free
    assert not any(locks(cache)) and not [k for k in feats if isinstance(k, tuple)]
    assert cache.restore_async(IDS + [7]) and cache.match(IDS + [7]).length == 0  # the tail is lost: prefill instead


def test_a_synchronous_restore_finishes_the_one_in_flight_first(tmp_path):
    cache, tier, kv, sp, feats = engine(tmp_path)
    second = [t + 20000 for t in IDS]
    one = release(cache, tier, kv, sp, feats, IDS, seed=7)
    two = release(cache, tier, kv, sp, feats, second, seed=8)
    settle(cache, tier)
    evict_all(cache, kv)
    settle(cache, tier)
    assert not cache.restore_async(IDS + [7])
    assert tier.restore(cache, second + [7]) == len(second)  # shares the staging buffer: the first lands first
    assert tier._job is None and same(restored(cache, kv, sp, IDS), one) and same(restored(cache, kv, sp, second), two)
    cache.release(IDS + [7])
    assert not any(locks(cache))



# ---- a prompt's blocks, and its last chunk end's tail, go out while its request decodes ----

def prefilled(state, kv, sp, ids, seed):
    """Admit a request for ids and prefill all of it in one forward as the executor does: the forward writes the
    snapshots it places (`Executor._checkpoint_positions`: the prompt's last chunk end S), and once the chunk is in
    the prompt up to S goes into the tree and the request's lock moves there (`StateManager.checkpoint`); then sample
    its first token."""
    from sharar.engine.executor import Executor
    ex = Executor.__new__(Executor)
    ex.state = state
    req = Request(prompt_ids=list(ids), params=SamplingParams(max_new_tokens=4))
    state.admit(req, req.prompt_ids)
    n = req.prefill_len
    planned = state.prefill_checkpoints([(req, ex._checkpoint_positions(req, req.prefix_len, n))])
    fill(kv, sp, req.pages[req.prefix_len:], req.slot, seed)
    for _, _, slot in planned:  # the forward writes the state there (here: the same random state as the request's)
        sp.copy_slot(req.slot, slot)
    state.commit_prefill(req, n)
    state.checkpoint(req, n)
    req.push(7)
    return req


def anchor(n, start=0):
    """Where one forward from `start` places a prompt's last chunk end S: its last grid point before n (0: none)."""
    s = start + (n - 1 - start) // 64 * 64
    return s if s > start else 0


def owned_by_cache(kv, free, n):
    """Of the `free` pages the pool had, the cache holds exactly n; none came back twice."""
    return kv.free_count == free - n and len(set(kv._free)) == len(kv._free)


def test_a_prefilled_prompt_is_backed_up_before_its_request_finishes(tmp_path):
    """Updated from main (`cache_prompt` is gone): the checkpoint after the forward, not a second lock, puts the prompt
    up to S in the tree, and its whole blocks and S's tail go out while the request still runs; the pages past S stay
    the request's until release."""
    cache, tier, kv, sp, feats = engine(tmp_path)
    state, free = StateManager(kv, sp, cache), kv.free_count
    ids = list(range(5000, 5000 + 3 * BLOCK))  # whole blocks: S = 3 * BLOCK - 64, past the second block
    req = prefilled(state, kv, sp, ids, seed=1)
    s = anchor(len(ids))
    assert s == 3 * BLOCK - 64 and req.lock.node is cache.node_at(ids[:s]) and req.lock.node.state_lock_ref == 1
    assert len(cache.locks()) == 1 and req.page_ids[:s] == [-1] * s and -1 not in req.page_ids[s:]
    assert not req.private
    state.check()
    settle(cache, tier)
    assert tier.backed_blocks == 2 and tier.tails_through == 1  # S's blocks and tail, written while the request runs
    evict_all(cache, kv)
    assert cache.node_at(ids[:s]).in_l1 and cache.node_at(ids[:s]).snapshot is not None  # its request holds it
    before = snapshot(kv, sp, req.pages, req.slot)
    state.release(req)
    assert not any(locks(cache)) and owned_by_cache(kv, free, len(ids))
    settle(cache, tier)
    assert tier.backed_blocks == 3 and tier.tails_through == 2  # the last block and the end's tail; nothing twice
    evict_all(cache, kv)
    assert tier.restore(cache, ids + [7]) == len(ids)
    assert same(restored(cache, kv, sp, ids), before)


def test_a_later_turn_hands_over_only_the_prompt_past_its_cached_prefix(tmp_path):
    cache, tier, kv, sp, feats = engine(tmp_path)
    state, free = StateManager(kv, sp, cache), kv.free_count
    first = prefilled(state, kv, sp, list(range(5000, 5000 + BLOCK + 10)), seed=1)
    state.release(first)
    ids = first.prompt_ids + first.output_ids + list(range(9000, 9000 + 3 * BLOCK))
    req = prefilled(state, kv, sp, ids, seed=2)
    p = BLOCK + 10
    s = anchor(len(ids), start=p)
    assert req.prefix_len == p and req.lock.node is cache.node_at(ids[:s]) and len(cache.locks()) == 1
    assert req.page_ids[:s - p] == [-1] * (s - p) and -1 not in req.page_ids[s - p:]
    state.release(req)
    assert not any(locks(cache)) and owned_by_cache(kv, free, len(ids))
    settle(cache, tier)
    assert tier.backed_blocks == 4


def test_a_retracted_request_leaves_its_prompt_cached_to_its_resume_point_and_a_retry_resumes_there(tmp_path):
    """Updated from main: retraction keeps what the request checkpointed (the prompt up to S), so its retry resumes at
    S (main kept the prompt's whole blocks without a snapshot: the retry prefilled again)."""
    cache, tier, kv, sp, feats = engine(tmp_path)
    state, free = StateManager(kv, sp, cache), kv.free_count
    ids = list(range(5000, 5000 + 3 * BLOCK + 37))
    s = anchor(len(ids))
    req = prefilled(state, kv, sp, ids, seed=1)
    state.retract(req)
    assert req.lock is None and not any(locks(cache))
    assert owned_by_cache(kv, free, s)  # the cache keeps the prompt to S; the request freed the rest
    again = prefilled(state, kv, sp, ids, seed=1)  # resumes at S, and places no snapshot of its own past it
    assert again.prefix_len == s and again.lock.node is cache.node_at(ids[:s]) and len(cache.locks()) == 1
    state.release(again)
    assert not any(locks(cache)) and owned_by_cache(kv, free, len(ids))


def test_a_second_copy_checkpointed_onto_a_cached_snapshot_reads_the_trees_pages(tmp_path):
    """Two requests prefill the same prompt side by side. The second checkpoint finds S cached already: its snapshot
    copy is freed, its lock joins the first's at the same node, and its own copies of [0, S) are freed now, the
    request reading the tree's (SGLang's checkpoint repoints the request the same way)."""
    cache, tier, kv, sp, feats = engine(tmp_path)
    state, free = StateManager(kv, sp, cache), kv.free_count
    ids = list(range(5000, 5000 + 3 * BLOCK + 37))
    s = anchor(len(ids))
    a = Request(prompt_ids=list(ids), params=SamplingParams(max_new_tokens=4))
    state.admit(a, a.prompt_ids)  # admitted before the other prefilled: both prefill all of it
    b = prefilled(state, kv, sp, ids, seed=1)
    slots, pages = sp.free_count, kv.free_count
    state.prefill_checkpoints([(a, [(s, False)])])
    state.commit_prefill(a, a.prefill_len)
    state.checkpoint(a, a.prefill_len)
    assert a.lock.node is b.lock.node and a.lock.node.state_lock_ref == 2 and sp.free_count == slots
    assert a.page_ids[:s] == [-1] * s and kv.free_count == pages + s
    assert torch.equal(a.pages[:s], cache.path_pages(a.lock.node))
    state.check()
    for r in (a, b):
        r.cacheable = False
        state.release(r)
    assert not any(locks(cache)) and owned_by_cache(kv, free, s)


def test_without_an_ssd_tier_the_checkpoint_still_publishes_the_prompt_to_its_last_chunk_end(tmp_path):
    cache, tier, kv, sp, feats = engine(tmp_path)
    cache.ssd = None
    state = StateManager(kv, sp, cache)
    ids = list(range(5000, 5000 + 3 * BLOCK))
    s = anchor(len(ids))
    req = prefilled(state, kv, sp, ids, seed=1)
    assert req.lock.node is cache.node_at(ids[:s]) and cache.cached_tokens == s and req.page_ids[:s] == [-1] * s
    assert -1 not in req.page_ids[s:]


# ---- restores, resume tails and dead KV under pressure ----

def test_a_landed_restore_keeps_its_snapshot_under_slot_pressure_until_its_request_is_admitted(tmp_path):
    """A landed restore locks the restored node: slot pressure takes every other snapshot, even once another request's
    longer path makes the node interior, and leaves that one (and its drafter payload). Admission takes the request's
    own lock there first, then the restore's ends: the node is never unlocked."""
    cache, tier, kv, sp, feats, before = stored(tmp_path)
    ids = IDS + [7]  # the queued request's prompt
    land(cache, tier, ids[:-1])  # as `can_admit` asks for it
    node = cache.match(ids, touch=False).node
    assert node.depth == len(IDS) and node.state_lock_ref == 1 and [owner_kind(l.owner) for l in cache.locks()] == ["restore"]
    drafter = dict(node.snapshot.auxiliary)
    # A request admitted earlier with the same prefix publishes a longer path through the restored node.
    extra = IDS + [11, 12, 13, 14]
    pages, page_ids = kv.alloc_with_ids(len(extra))
    slot = sp.alloc()
    kv.free_ids(page_ids[:cache.insert(extra, pages, snaps({len(extra): slot}, spare=True), page_ids=page_ids)])
    assert node.children and node.state_lock_ref == 1
    cache.evict(need_slots=sp.num_slots)  # slot pressure takes every snapshot it can
    assert slot in sp._free and node.snapshot is not None and node.snapshot.auxiliary == drafter
    assert not cache.node_at(extra).in_l1  # the longer path's leaf went whole: stored, so demoted, not deleted
    assert same(restored(cache, kv, sp, IDS), before)
    order, inc, dec = [], cache.inc_lock_ref, cache.dec_lock_ref
    cache.inc_lock_ref = lambda n: (order.append(+1) if n is node else None, inc(n))[1]
    cache.dec_lock_ref = lambda n: (order.append(-1) if n is node else None, dec(n))[1]
    state = StateManager(kv, sp, cache)
    req = Request(prompt_ids=ids, params=SamplingParams(max_new_tokens=4))
    state.admit(req, req.prompt_ids)
    assert order == [1, -1], "the request's lock comes first, then the restore's ends"
    assert req.prefix_len == len(IDS) and req.lock.node is node and node.state_lock_ref == 1 and len(cache.locks()) == 1
    assert torch.equal(sp.ssm_all[:, req.slot], sp.ssm_all[:, node.snapshot.slot])
    state.check()
    req.cacheable = False
    state.release(req)
    assert not any(locks(cache))


def test_a_pending_tail_is_written_before_slot_pressure_takes_its_snapshot(tmp_path):
    """A primed prompt (one output token) leaves its last chunk end S, whose tail its checkpoint queued for a
    write-through, and its reply end E, a spare leaf below S. Slot pressure before the tier's next step takes E first
    (whole: its tail written as it goes, since staging has room) and then S, whose only child is now on the SSD: only
    its snapshot goes, and its tail is written first, from its slot (`dropping_snapshot`), instead of the queued write
    finding the slot gone. Re-sending the prompt restores it at S."""
    cache, tier, kv, sp, feats = engine(tmp_path)
    state = StateManager(kv, sp, cache)
    ids = list(range(5000, 5000 + 3 * BLOCK + 37))
    s = anchor(len(ids))
    req = prefilled(state, kv, sp, ids, seed=1)
    assert [e for _, e in tier.through] == [s]
    state.release(req)
    snode = cache.node_at(ids[:s])
    cache.evict(need_slots=sp.num_slots)  # no tier step in between: E's and S's writes are both still queued
    assert snode.tail is not None and tier.through_dropped == 0
    settle(cache, tier)
    assert snode.tail.ready and cache.match(ids[:-1], touch=False).length == 0
    assert tier.restore(cache, ids[:-1]) == s


def test_an_interior_snapshot_writes_its_pending_tail_before_it_goes(tmp_path):
    """The same with S interior (a locked path continues past it): slot pressure takes only S's snapshot and keeps its
    pages; the write-through still queued for it goes out first (`SsdTier.dropping_snapshot`)."""
    cache, tier, kv, sp, feats = engine(tmp_path)
    state = StateManager(kv, sp, cache)
    ids = list(range(5000, 5000 + 3 * BLOCK + 37))
    s = anchor(len(ids))
    req = prefilled(state, kv, sp, ids, seed=1)
    state.release(req)
    snode = cache.node_at(ids[:s])
    (leaf,) = snode.children.values()
    with cache._lock_node(leaf):
        cache.evict(need_slots=sp.num_slots)
        assert snode.snapshot is None and snode.in_l1 and snode.tail is not None and tier.through_dropped == 0
    settle(cache, tier)
    assert snode.tail.ready


def test_dead_kv_goes_when_the_disk_budget_deletes_the_tail_it_resumed_from(tmp_path):
    """A node whose snapshot slot pressure took kept its pages for its stored tail alone. When the disk budget deletes
    that copy, nothing can resume there: the node is dead KV and goes (`RadixCache._prune`)."""
    cache, tier, kv, sp, feats = engine(tmp_path)
    free = kv.free_count
    pages, page_ids = kv.alloc_with_ids(len(IDS))
    slot = sp.alloc()
    fill(kv, sp, pages, slot, seed=9)
    cache.insert(IDS, pages, snaps({len(IDS): slot}), page_ids=page_ids)
    tier.backup(cache, IDS)
    (node,) = cache.root.children.values()
    assert tier._write_tail(node, len(IDS), wait=True)
    settle(cache, tier)
    drop_snapshot(cache, sp, node)
    assert node.tail.ready and node.in_l1 and not node.children and not cache._dead(node)
    tier.budget = 0  # the budget deletes the copies of resident nodes: the tail goes, and with it the node
    tier.step(cache)
    assert cache.pruned == 1 and not cache.root.children and kv.free_count == free and cache.cached_tokens == 0


def test_dead_kv_goes_when_a_tail_write_fails(tmp_path, monkeypatch):
    """A resident node waiting on its tail write (its snapshot already taken) is dead once that write fails."""
    cache, tier, kv, sp, feats = engine(tmp_path)
    free = kv.free_count
    pages, page_ids = kv.alloc_with_ids(len(IDS))
    slot = sp.alloc()
    fill(kv, sp, pages, slot, seed=4)
    cache.insert(IDS, pages, snaps({len(IDS): slot}), page_ids=page_ids)
    (node,) = cache.root.children.values()
    real = tier.store._write

    def full_disk(header, tensors, buf):
        if header.kind == "tail":
            raise OSError(28, "No space left on device")
        return real(header, tensors, buf)
    monkeypatch.setattr(tier.store, "_write", full_disk)
    assert tier._write_tail(node, len(IDS), wait=True)
    drop_snapshot(cache, sp, node)
    assert not cache._dead(node)  # its tail is on its way
    tier.store.wait()
    tier._finish_writes()
    assert node.tail.nbytes < 0 and cache._dead(node) and tier.pending_prune() == [node]
    tier.step(cache)
    assert not cache.root.children and kv.free_count == free and cache.pruned == 1


# ---- queued backups and pruning ----

def test_queued_backups_survive_the_pruning_of_their_nodes(tmp_path):
    """Blocks queued for a path whose node is then pruned as dead KV (its snapshot gone, nothing below it, no tail)
    are dropped when their turn comes: never gathered from pages that went back to the pool and may hold another
    path's KV by then."""
    cache, tier, kv, sp, feats = engine(tmp_path)
    free = kv.free_count, sp.free_count
    pages, page_ids = kv.alloc_with_ids(len(IDS))
    slot = sp.alloc()
    fill(kv, sp, pages, slot, seed=7)
    cache.insert(IDS, pages, snaps({len(IDS): slot}), page_ids=page_ids)
    tier.backup(cache, IDS)
    (node,) = cache.root.children.values()
    assert len(tier.queue) == 2 and all(item[0] is node for item in tier.queue)
    drop_snapshot(cache, sp, node)
    cache._prune(node)  # dead KV: its pages go back while its blocks are still queued
    assert not cache.root.children and (kv.free_count, sp.free_count) == free
    other, other_ids = kv.alloc_with_ids(len(IDS))  # another path takes the pages
    settle(cache, tier)
    assert not tier.queue and tier.backed_blocks == 0 and not tier.inflight and tier.used == 0
    kv.free_ids(other_ids)
    cache.check()


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_randomized_tier_operations_keep_the_trees_invariants(tmp_path, seed):
    """Inserts sharing prefixes, backups with and without resume points, locks, slot and KV pressure, pruning and tier
    steps in random order, on small pools: after every operation `RadixCache.check` holds (counters, evictable totals,
    candidate sets, slot ownership, no dead KV but what the tier prunes next), with backups queued throughout."""
    rng = random.Random(seed)
    cache, tier, kv, sp, feats = engine(tmp_path / str(seed), pages=3 * BLOCK * 4, slots=10)
    bases = [list(range(1000 * b, 1000 * b + 2 * BLOCK)) for b in range(3)]
    locks, owner = [], object()
    for _ in range(200):
        op = rng.random()
        if op < 0.35:
            ids = rng.choice(bases)[:rng.randint(BLOCK, 2 * BLOCK)] + [50_000 + rng.randrange(1000) for _ in range(rng.randint(1, BLOCK))]
            cache.evict(need_pages=len(ids), need_slots=2)
            if kv.free_count < len(ids) or sp.free_count < 2:
                continue
            pages, page_ids = kv.alloc_with_ids(len(ids))
            cuts = sorted({rng.randint(1, len(ids) - 1), len(ids)})
            snapshots = [Snapshot(sp.alloc(), p, spare=rng.random() < 0.5) for p in cuts]
            new_start = cache.insert(ids, pages, snapshots, page_ids=page_ids)
            kv.free_ids(page_ids[:new_start])
            tier.backup(cache, ids, tuple(p for p in cuts if rng.random() < 0.5))
        elif op < 0.5 and cache.evictable_snapshots:
            node = rng.choice(list(cache.evictable_snapshots))
            locks.append(cache.lock(node, owner=owner))
        elif op < 0.6 and locks:
            cache.unlock(locks.pop(rng.randrange(len(locks))))
        elif op < 0.75:
            cache.evict(need_slots=sp.free_count + rng.randint(1, 3))
        elif op < 0.85:
            cache.evict(need_pages=kv.free_count + rng.randint(1, BLOCK))
        else:
            tier.step(cache)
        cache.check([owner], owned_slots=tier.owned_slots(), pending_prune=tier.pending_prune())
    for lock in locks:
        cache.unlock(lock)
    settle(cache, tier)
    cache.check(owned_slots=tier.owned_slots(), pending_prune=tier.pending_prune())
