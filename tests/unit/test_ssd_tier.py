"""The SSD tier end to end on CPU pools: the tier backs up a released
sequence's blocks, evicting its leaf writes the tail and demotes it, and the
next request restores it byte-exact (KV, page aux, snapshot, pool aux, drafter
payload) into fresh pages; the same works after a restart from the stored
headers alone, a corrupt object is a miss, and the budget deletes least
recently used SSD-only leaves."""
import dataclasses
import json

import pytest
import torch

from sharar.cache.kv_pool import KvPool
from sharar.cache.radix import RadixCache
from sharar.cache.ssd import SsdTier
from sharar.cache import ssd, ssd_store
from sharar.cache.ssd_store import BLOCK, SsdStore, memory_backed
from sharar.cache.state_pool import StatePool
from sharar.target import DEFAULT_TARGET

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
    cache.insert(ids, pages, {len(ids): slot}, {len(ids): {DRAFT: "h"}}, page_ids=page_ids)
    tier.backup(cache, ids)
    return before


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
    per_session = tier.used // 2
    tier.budget = per_session + per_session // 2  # room for one
    tier.step(cache)
    assert tier.used <= tier.budget and tier.deleted >= 2 and len(cache.root.children) == 1
    tier.store.wait()
    assert sum(1 for _ in tier.store.scan()) == 2  # one session left: its block and its tail


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
    sp.free(node.snapshot_slot)  # as snapshot-only eviction leaves it: no slot to write a tail from
    node.snapshot_slot = None
    used, sessions = tier.used, tier.sessions
    evict_all(cache, kv)
    settle(cache, tier)
    # the tail it was restored from is still its state at that end: the leaf stays on the SSD, and restores again
    assert cache.ssd_nodes == 1 and tier.used == used and tier.sessions == sessions
    assert tier.restore(cache, IDS + [7]) == len(IDS) and same(restored(cache, kv, sp, IDS), before)


def test_a_failed_tail_write_removes_the_unrestorable_leaf(tmp_path, monkeypatch):
    cache, tier, kv, sp, feats = engine(tmp_path)
    release(cache, tier, kv, sp, feats, IDS, seed=3)
    settle(cache, tier)
    real = tier.store._write

    def full_disk(header, tensors, ready):
        if header.kind == "tail":
            raise OSError(28, "No space left on device")
        return real(header, tensors, ready)
    monkeypatch.setattr(tier.store, "_write", full_disk)
    evict_all(cache, kv)
    settle(cache, tier)
    assert not cache.root.children and tier.store.write_errors == 1 and tier.restore(cache, IDS + [7]) == 0


LONG = list(range(5000, 5000 + 6 * BLOCK + 11))  # six full blocks and an 11-token tail


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


def test_tensor_parallel_a_restart_lists_only_sessions_every_rank_stored(tmp_path):
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


def test_only_the_answer_end_is_written_through(tmp_path):
    """A finished request writes through the tail at its end, which its conversation's next turn extends, and not
    the one at its prompt's end (about 144 MiB more per request on the 27B). The trade-off: when the tree drops that
    interior snapshot for want of state slots, a request ending at the prompt prefills again."""
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
    cache.insert(IDS, pages, {prompt_end: p_slot, n: n_slot}, {prompt_end: {DRAFT: "h"}, n: {DRAFT: "h"}}, page_ids=page_ids)
    tier.backup(cache, IDS, (n, prompt_end))
    settle(cache, tier)
    (head,) = cache.root.children.values()
    (leaf,) = head.children.values()
    assert tier.tails_through == 1 and head.tail is None and leaf.tail.ready and head.depth == prompt_end
    cache.evict(need_slots=sp.free_count + 1)  # slots short, pages ample: only the interior snapshot goes
    assert head.snapshot_slot is None and head.in_l1 and leaf.in_l1 and cache.snapshot_evictions == 1
    assert cache.match(IDS[:prompt_end] + [7]).length == 0
    assert tier.restore(cache, IDS[:prompt_end] + [7]) == 0  # no tail there: the request prefills
    assert tier.restore(cache, IDS + [7]) == 0 and cache.match(IDS + [7]).length == len(IDS)  # the end is still in L1
    del before
    evict_all(cache, kv)
    settle(cache, tier)
    assert tier.tails == 1  # the end's tail, written through: nothing twice, and none for the snapshot-less prompt end


def test_write_through_stays_within_the_staging_limit_and_skips_a_snapshot_dropped_meanwhile(tmp_path):
    """Writing a tail through copies a whole snapshot into pinned memory: requests finishing faster than the writer
    drains wait for staging room instead of each pinning a copy. The tier skips one whose snapshot the GPU cache drops
    while it waits, rather than write it from a slot it no longer owns."""
    cache, tier, kv, sp, feats = engine(tmp_path)
    tier.staging_bytes = 1  # one object in flight at a time
    sessions = [[s] + list(range(1000, 1000 + 2 * BLOCK + 9)) for s in (1, 2, 3)]
    befores = []
    for i, ids in enumerate(sessions):
        pages, page_ids = kv.alloc_with_ids(len(ids))
        slot = sp.alloc()
        fill(kv, sp, pages, slot, seed=30 + i)
        befores.append(snapshot(kv, sp, pages, slot))
        feats["h"] = (3, 9, torch.randn(4, 2, generator=torch.Generator().manual_seed(30 + i)))
        cache.insert(ids, pages, {len(ids): slot}, {len(ids): {DRAFT: "h"}}, page_ids=page_ids)
        tier.backup(cache, ids, (len(ids),))
    assert tier.staged == 0 and tier.tails_through == 0  # backup copies nothing as the requests finish
    tier.step(cache)
    assert tier.tails_through == 1 and len(tier.inflight) == 1
    third = cache.root.children[sessions[2][0]]
    sp.free(third.snapshot_slot)  # as snapshot-only eviction leaves it, while its write waits for room
    third.snapshot_slot = None
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
    assert tier.restore(cache, sessions[2] + [7]) == 0
    for ids, before in zip(sessions[:2], befores):
        assert tier.restore(cache, ids + [7]) == len(ids)
        assert same(restored(cache, kv, sp, ids), before)


def test_a_flush_forgets_the_deletions_the_old_epoch_owed(tmp_path, monkeypatch):
    """The tier owes a budget victim's writes still in flight a deletion when they land; a flush forgets those with
    the old epoch. Written again in the new epoch, the same session has the same keys, which must stay."""
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
def test_a_session_reinserted_before_its_discarded_writes_finish_reuses_them(tmp_path, monkeypatch, write_through):
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
        cache.insert(IDS[:cut], cache.match(IDS).pages[:cut], {cut: sp.alloc()})
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
    evict_all(cache, kv)
    assert tier.evict_dropped == 1 and not tier.inflight and cache.ssd_sessions() == 0
    tier.staged = 0
    monkeypatch.undo()
    settle(cache, tier)
    assert tier.restore(cache, LONG + [7]) == 0 and tier.used == 0


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
    """IDS with snapshots at prompt_end and at its end, released with both as resume points."""
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
    cache.insert(IDS, pages, {prompt_end: p_slot, n: n_slot}, {prompt_end: {DRAFT: "h"}, n: {DRAFT: "h"}}, page_ids=page_ids)
    tier.backup(cache, IDS, (n, prompt_end))
    return at_prompt, at_end


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
    cache.evict(need_slots=sp.free_count + 1)
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
    cache.evict(need_slots=sp.free_count + 1)  # the interior snapshot goes
    cache.evict(need_pages=kv.free_count + 1)  # then the leaf: demoted, its tail already written
    (head,) = cache.root.children.values()
    (leaf,) = head.children.values()
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
    cache.insert(IDS, pages, {len(IDS): slot}, page_ids=page_ids)
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
    out, stack = [], list(cache.root.children.values())
    while stack:
        n = stack.pop()
        out.append(n.lock)
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
    cache.release(ids + [7])  # admitted (the request's match holds it now) or gone
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


def test_a_second_session_waits_for_the_staging_buffer(tmp_path):
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
    land(cache, tier, second + [7])
    assert same(restored(cache, kv, sp, IDS), one) and same(restored(cache, kv, sp, second), two)
    cache.release(IDS + [7])
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


def test_admission_with_the_restore_off_restores_in_line(tmp_path):
    cache, tier, kv, sp, feats, before = stored(tmp_path)
    tier.async_restore = False
    assert cache.restore_async(IDS + [7]) and tier.async_restores == 0 and tier.restores == 1
    assert same(restored(cache, kv, sp, IDS), before) and not any(locks(cache))
