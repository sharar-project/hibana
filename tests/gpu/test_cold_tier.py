"""Cold tier: an evicted prefix leaf is parked in unified memory and restored
bit-exactly (pages, GDN snapshot, draft auxiliary) when its session returns."""
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("SHARAR_RUN_GPU") != "1" or not torch.cuda.is_available(), reason="GB10 only")


def _pools(pages=256, slots=6):
    from sharar.memory.kv_pool import KvPool
    from sharar.memory.radix import RadixCache
    from sharar.memory.state_pool import StatePool
    from sharar.memory.cold_tier import ColdTier

    dev = torch.device("cuda")
    kv, st = KvPool(pages, dev), StatePool(slots, dev)
    cache = RadixCache(kv, st, dev)
    cache.tier = ColdTier(kv, st, 1 * 2**30, dev)
    return kv, st, cache


def _fill_random(kv, st, pages, slot, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    for lid in kv.layer_ids:
        kv.k[lid][pages] = torch.randn(kv.k[lid][pages].shape, generator=g, device="cuda").to(kv.dtype)
        kv.v[lid][pages] = torch.randn(kv.v[lid][pages].shape, generator=g, device="cuda").to(kv.dtype)
    st.conv_all[:, slot] = torch.randn(st.conv_all[:, slot].shape, generator=g, device="cuda").to(st.conv_all.dtype)
    st.ssm_all[:, slot] = torch.randn(st.ssm_all[:, slot].shape, generator=g, device="cuda")


def _snapshot(kv, st, pages, slot):
    return ([(kv.k[lid][pages].clone(), kv.v[lid][pages].clone()) for lid in kv.layer_ids],
            st.conv_all[:, slot].clone(), st.ssm_all[:, slot].clone())


def test_park_and_restore_round_trip_is_exact():
    kv, st, cache = _pools()
    ids = list(range(1000, 1100))
    pages = kv.alloc(len(ids))
    slot = st.alloc()
    _fill_random(kv, st, pages, slot, 1)
    before = _snapshot(kv, st, pages, slot)
    seen = {}
    feats = torch.arange(64 * 8, device="cuda", dtype=torch.bfloat16).view(64, 8)
    cache.tier.register_codec("draft", lambda v: (v, 7, feats), lambda p, *, slot: seen.setdefault("imported", (p[0], p[1], p[2].clone())))
    cache.insert(ids, pages, {len(ids): slot}, {len(ids): {"draft": 42}})
    free_pages, free_slots = kv.free_count, st.free_count
    cache.evict(need_pages=kv.num_pages)  # forces the leaf out
    assert cache.match(ids, touch=False).length == 0
    assert kv.free_count == free_pages + len(ids) and st.free_count == free_slots + 1
    stats = cache.tier.stats()
    assert stats["entries"] == 1 and stats["parks"] == 1 and stats["parked_tokens"] == len(ids)
    for lid in kv.layer_ids:  # scribble over the freed pool memory so a stale read would be caught
        kv.k[lid].zero_(); kv.v[lid].zero_()
    st.conv_all.zero_(); st.ssm_all.zero_()
    assert cache.restore_parked(ids + [7]) == 1
    m = cache.match(ids, touch=False)
    assert m.length == len(ids) and m.snapshot_slot is not None
    after = _snapshot(kv, st, m.pages, m.snapshot_slot)
    for (k0, v0), (k1, v1) in zip(before[0], after[0]):
        assert torch.equal(k0.view(torch.uint8), k1.view(torch.uint8)) and torch.equal(v0.view(torch.uint8), v1.view(torch.uint8))
    assert torch.equal(before[1], after[1]) and torch.equal(before[2], after[2])
    assert seen["imported"][:2] == (42, 7) and torch.equal(seen["imported"][2], feats)
    assert m.node.auxiliary == {"draft": seen["imported"]}
    assert cache.tier.stats()["entries"] == 0 and cache.tier.stats()["restores"] == 1


def test_parked_child_needs_its_parent_and_lru_drop_frees_arena():
    kv, st, cache = _pools(pages=512, slots=8)
    a = list(range(10, 74))
    b = a + list(range(200, 264))
    pa, pb = kv.alloc(len(a)), kv.alloc(len(b) - len(a))
    sa, sb = st.alloc(), st.alloc()
    _fill_random(kv, st, pa, sa, 2); _fill_random(kv, st, pb, sb, 3)
    cache.insert(a, pa, {len(a): sa})
    cache.insert(b, torch.cat([pa, pb]), {len(b): sb})
    cache.evict(need_pages=kv.num_pages)  # child leaf first (whole path parked), then the parent (covered: rejected)
    stats = cache.tier.stats()
    assert stats["entries"] == 1 and stats["parked_tokens"] == len(b) and stats["rejects"] == 1
    assert cache.restore_parked(b) == 1
    m = cache.match(b, touch=False)
    assert m.length == len(b) and cache.tier.stats()["entries"] == 0
    assert torch.equal(kv.k[kv.layer_ids[0]][m.pages[:64]], kv.k[kv.layer_ids[0]][m.pages[:64]])  # pages valid
    # a parked entry survives its parent being evicted later, and restores only the missing suffix
    cache.evict(need_pages=kv.num_pages)
    assert cache.tier.stats()["entries"] == 1
    pa2, sa2 = kv.alloc(len(a)), st.alloc()
    _fill_random(kv, st, pa2, sa2, 4)
    cache.insert(a, pa2, {len(a): sa2})  # the parent path comes back on its own
    free_before = kv.free_count
    assert cache.restore_parked(b) == 1
    assert kv.free_count == free_before - (len(b) - len(a))  # only the suffix was allocated
    assert cache.match(b, touch=False).length == len(b)
    # LRU drop when the arena cannot fit a new entry: a fresh tier whose single slab holds one leaf
    from sharar.memory.cold_tier import ColdTier
    one = cache.tier._kv_row_bytes() * len(b) + cache.tier._state_bytes()
    cache.tier = ColdTier(kv, st, one + 4096, cache.device)
    cache.tier.arena.SLAB = one + 4096
    pb2, sb2 = kv.alloc(len(b) - len(a)), st.alloc(); _fill_random(kv, st, pb2, sb2, 6)
    cache.evict(need_pages=kv.num_pages)  # clears the tree (parks b into the fresh tier)
    cache.insert(b, torch.cat([kv.alloc(len(a)), pb2]), {len(b): sb2})
    c = list(range(300, 428)); pc, sc = kv.alloc(len(c)), st.alloc(); _fill_random(kv, st, pc, sc, 5)
    cache.insert(c, pc, {len(c): sc})
    cache.evict(need_pages=kv.num_pages)
    stats = cache.tier.stats()
    assert stats["drops"] >= 1 and stats["entries"] == 1 and stats["slab_gib"] > 0


def test_flush_clears_parked_entries():
    kv, st, cache = _pools()
    ids = list(range(5000, 5064))
    pages, slot = kv.alloc(len(ids)), st.alloc()
    cache.insert(ids, pages, {len(ids): slot})
    cache.evict(need_pages=kv.num_pages)
    assert cache.tier.stats()["entries"] == 1
    cache.flush()
    assert cache.tier.stats()["entries"] == 0 and cache.tier.arena.used_bytes == 0


def test_checkpoint_features_follow_park_restore_and_slot_reuse():
    from sharar.proposers.parallel import CheckpointFeatureCache
    kv, st, cache = _pools()
    features = CheckpointFeatureCache(st, 1024)
    data = torch.arange(512, device='cuda', dtype=torch.bfloat16).reshape(64, 8)
    cache.tier.register_codec('features', features.get,
                              lambda payload, *, slot: features.put(*payload, slot=slot))
    ids = list(range(6000, 6064))
    pages, slot = kv.alloc(64), st.alloc()
    handle = features.put(0, 64, data, slot=slot)
    cache.insert(ids, pages, {64: slot}, {64: {'features': handle}})
    cache.evict(need_pages=kv.num_pages)
    assert features.get(handle) is None and features.bytes == 0
    assert cache.tier.stats()['entries'] == 1
    reused = st.alloc()
    replacement = features.put(0, 64, data + 10, slot=reused)
    assert cache.restore_parked(ids) == 1
    match = cache.match(ids, touch=False)
    restored = match.node.auxiliary['features']
    assert features.slots[match.snapshot_slot] == restored
    assert torch.equal(features.get(restored)[2], data)
    assert torch.equal(features.get(replacement)[2], data + 10)
    cache.flush()
    st.free(reused)
    assert features.bytes == 0 and not features.entries


def _exact(before, kv, st, m):
    after = _snapshot(kv, st, m.pages, m.snapshot_slot)
    for (k0, v0), (k1, v1) in zip(before[0], after[0]):
        assert torch.equal(k0.view(torch.uint8), k1.view(torch.uint8)) and torch.equal(v0.view(torch.uint8), v1.view(torch.uint8))
    assert torch.equal(before[1], after[1]) and torch.equal(before[2], after[2])


def _parked_session(kv, st, cache, ids, seed):
    pages, slot = kv.alloc(len(ids)), st.alloc()
    _fill_random(kv, st, pages, slot, seed)
    before = _snapshot(kv, st, pages, slot)
    cache.insert(ids, pages, {len(ids): slot})
    cache.evict(need_pages=kv.num_pages)
    return before


def _scribble(kv, st):
    for lid in kv.layer_ids:
        kv.k[lid].zero_(); kv.v[lid].zero_()
    st.conv_all.zero_(); st.ssm_all.zero_()


def test_a_session_larger_than_a_slab_parks_in_chunks():
    kv, st, cache = _pools(pages=12000, slots=6)
    tier = cache.tier
    tier.arena.SLAB = tier._state_bytes() + 4096  # the smallest slab that holds a snapshot: ~4.6K tokens of KV
    tier.arena.nbytes = 8 * tier.arena.SLAB
    per = tier.arena.SLAB // tier._kv_row_bytes()
    ids = list(range(20000, 30000))
    before = _parked_session(kv, st, cache, ids, 11)
    (entry,) = tier.entries
    assert [chunk[0] for chunk in entry.chunks] == list(range(0, len(ids), per)) and len(entry.chunks) >= 3
    _scribble(kv, st)
    assert cache.restore_parked(ids + [1]) == 1
    _exact(before, kv, st, cache.match(ids, touch=False))


def test_a_dropped_session_comes_back_from_the_store(tmp_path):
    from sharar.memory.session_disk import SessionDisk
    kv, st, cache = _pools()
    feats = torch.arange(64 * 8, device="cuda", dtype=torch.bfloat16).view(64, 8)
    seen = {}
    disk = SessionDisk(tmp_path, 2**30, {"t": 1}, torch.device("cuda"))
    cache.tier.disk = disk
    cache.tier.register_codec("draft", lambda v: (v, 7, feats), lambda p, *, slot: seen.setdefault("imported", (p[0], p[1], p[2].clone())))
    ids = list(range(7000, 7200))
    pages, slot = kv.alloc(len(ids)), st.alloc()
    _fill_random(kv, st, pages, slot, 12)
    before = _snapshot(kv, st, pages, slot)
    cache.insert(ids, pages, {len(ids): slot}, {len(ids): {"draft": 42}})
    cache.evict(need_pages=kv.num_pages)
    (entry,) = cache.tier.entries
    disk.wait(entry)
    assert entry.on_disk and disk.stats()["records"] == 1
    cache.tier._drop_lru()  # the arena lets it go; the store keeps it
    assert not cache.tier.entries
    _scribble(kv, st)
    assert cache.restore_parked(ids + [3]) == 1
    _exact(before, kv, st, cache.match(ids, touch=False))
    assert seen["imported"][:2] == (42, 7) and torch.equal(seen["imported"][2], feats)
    assert cache.tier.stats()["disk_loads"] == 1 and disk.stats()["reads"] == 1


def test_a_restarted_server_restores_a_stored_session(tmp_path):
    from sharar.memory.cold_tier import ColdTier
    from sharar.memory.session_disk import SessionDisk
    kv, st, cache = _pools(pages=512)
    cache.tier.disk = SessionDisk(tmp_path, 2**30, {"t": 1}, torch.device("cuda"))
    ids = list(range(9000, 9300))
    before = _parked_session(kv, st, cache, ids, 13)
    cache.tier.disk.wait(cache.tier.entries[0])
    kv2, st2, cache2 = _pools(pages=512)  # new pools, arena and store: a restart
    cache2.tier = ColdTier(kv2, st2, 2**30, torch.device("cuda"), disk=SessionDisk(tmp_path, 2**30, {"t": 1}, torch.device("cuda")))
    assert cache2.tier.disk.stats()["records"] == 1
    assert cache2.restore_parked(ids + [5]) == 1
    _exact(before, kv2, st2, cache2.match(ids, touch=False))
    other = ColdTier(kv2, st2, 2**30, torch.device("cuda"), disk=SessionDisk(tmp_path, 2**30, {"t": 2}, torch.device("cuda")))
    assert other.find(ids + [5]) == []  # another identity never sees it


def test_shutdown_persists_cached_sessions(tmp_path):
    from sharar.engine.executor import Executor
    from sharar.memory.session_disk import SessionDisk
    kv, st, cache = _pools()
    cache.tier.disk = SessionDisk(tmp_path, 2**30, {"t": 1}, torch.device("cuda"))
    ids = list(range(40000, 40128))
    pages, slot = kv.alloc(len(ids)), st.alloc()
    _fill_random(kv, st, pages, slot, 14)
    cache.insert(ids, pages, {len(ids): slot})
    ex = Executor.__new__(Executor)
    ex.kv_pool, ex.state_pool = kv, st
    ex.state = type("S", (), {"cache": cache})()
    assert ex.persist_sessions() == 1 and cache.tier.disk.stats()["records"] == 1
