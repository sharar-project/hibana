"""The SSD tier's one-launch KV copies on the GPU (`_RowCopy`, kernels/csrc/kv_copy.cu), on the 27B's KV geometry
with page aux rows and a drafter's pages beside it: a backup writes the payload a torch gather writes, zero padding
included, and a prefix backed up, evicted and restored, in line or in the background, through one staging chunk or
several, and with part of its path still in L1, comes back byte-exact into its new pages; no copy writes a page it
does not list, page 0 included. Narrow page aux rows (2-byte words, padded segments) and a table ring of two
one-object slots (every launch split, every slot reused across both streams) restore the same."""
import dataclasses
import random

import pytest
import torch

from hibana.cache import ssd
from hibana.cache.kv_pool import KvPool
from hibana.cache.radix import RadixCache
from hibana.cache.snapshot import Snapshot
from hibana.cache.ssd import MIB, SsdTier
from hibana.cache.ssd_store import BLOCK, SsdStore, _seg_pad
from hibana.cache.state_pool import StatePool
from hibana.target import DEFAULT_TARGET

DEV = torch.device("cuda")
SMALL_STATE = dataclasses.replace(  # the recurrent state of a tiny model: the copies under test move KV rows only
    DEFAULT_TARGET, num_layers=4, layer_types=("linear_attention", "full_attention") * 2, linear_num_key_heads=1,
    linear_num_value_heads=2, linear_key_head_dim=4, linear_value_head_dim=4)
IDS = list(range(1000, 1000 + 6 * BLOCK + 37))  # six full blocks and a 37-token tail
ALIGNED = list(range(1000, 1000 + 6 * BLOCK))  # a tail with no KV rows of its own
SENTINEL = 0xA5


@pytest.fixture
def engine(tmp_path, monkeypatch):
    """Builds a tier on fresh pools (pages handed out in random order); afterwards lets go of its restore, worker and
    registered buffers whether or not the test passed."""
    made = []

    def build(chunk_bytes, narrow=False):
        kv = KvPool(4096, DEV)
        if narrow:  # 6-byte rows: 2-byte words, and a 37-token segment padded to 256 bytes
            kv.register_page_aux("index_keys", (3,), torch.bfloat16)
            monkeypatch.setattr(ssd._RowCopy, "SLOTS", 2)
            monkeypatch.setattr(ssd._RowCopy, "SLOT_BYTES", 8 * (2 + 2 * len(kv.layer_ids) + 2))  # one object a slot
        else:
            kv.register_page_aux("qsa.0", (2, 128), torch.bfloat16)
        kv._free = random.Random(0).sample(kv._free, len(kv._free))
        draft = torch.zeros((kv.num_pages, 1, 4, 256), dtype=torch.bfloat16, device=DEV)  # an MTP layer's KV
        sp = StatePool(4, DEV, target=SMALL_STATE)
        cache = RadixCache(kv, sp, DEV)
        tier = SsdTier(SsdStore(tmp_path / str(len(made)), {"target": "row-copy-test"}), kv, sp, DEV, 2**34,
                       min_tokens=BLOCK, min_hit=BLOCK, chunk_bytes=chunk_bytes, extra_pages=[("draft.0.k", draft)])
        cache.ssd = tier
        made.append((cache, tier))
        return cache, tier, kv, sp

    yield build
    torch.cuda.synchronize()
    for cache, tier in made:
        tier.abort(cache)
        tier._worker.shutdown(wait=True)
        tier._staging.close()
        tier._copy.tables.close()


def pools(tier):
    return [pool.view(torch.uint8).view(pool.shape[0], -1) for _, pool in tier._kv_targets()]


def fill(tier, pages, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    for pool in pools(tier):
        pool[pages.long()] = torch.randint(0, 256, (len(pages), pool.shape[1]), generator=g, device=DEV, dtype=torch.uint8)


def rows(tier, pages):
    return [pool[pages.long()].clone() for pool in pools(tier)]


def clear_all_but(tier, pages):
    """Zero every pool except `pages` (pages a restore takes may still hold an evicted leaf's rows), and put a
    sentinel in page 0."""
    for pool in pools(tier):
        kept = pool[pages.long()].clone()
        pool.zero_()
        pool[pages.long()] = kept
        pool[0] = SENTINEL


def untouched_elsewhere(tier, pages):
    keep = torch.ones(tier.kv_pool.num_pages, dtype=torch.bool, device=DEV)
    keep[pages.long()] = False
    keep[0] = False
    return all(bool((pool[0] == SENTINEL).all()) and not bool(pool[keep].any()) for pool in pools(tier))


def settle(cache, tier):
    for _ in range(100):
        tier.step(cache)
        tier.store.wait()
        if not tier.queue and not tier.through and not tier.inflight:
            return
    raise AssertionError("backup did not settle")


def land(cache, tier, ids):
    """Admission's loop for one request, as the scheduler runs it with nothing else to do."""
    for _ in range(1000):
        if cache.restore_async(ids):
            return
        tier.wait(0.05)
        tier.step(cache)
    raise AssertionError("the restore never landed")


def stored(build, ids, chunk_bytes, narrow=False):
    """ids backed up and evicted, every pool cleared."""
    cache, tier, kv, sp = build(chunk_bytes, narrow)
    pages, page_ids = kv.alloc_with_ids(len(ids))
    fill(tier, pages, seed=1)
    before = rows(tier, pages)
    cache.insert(ids, pages, [Snapshot(sp.alloc(), len(ids), spare=True)], page_ids=page_ids)
    tier.backup(cache, ids)
    settle(cache, tier)
    cache.evict(need_pages=kv.num_pages, need_slots=sp.num_slots)
    settle(cache, tier)
    clear_all_but(tier, pages[:0])
    return cache, tier, before


def restored_exactly(cache, tier, ids, before):
    torch.cuda.synchronize()
    m = cache.match(ids)
    return (m.length == len(ids) and all(torch.equal(a, b) for a, b in zip(rows(tier, m.pages), before))
            and untouched_elsewhere(tier, m.pages))


@pytest.mark.parametrize("chunk_bytes", [1, 256 * MIB], ids=["chunk-per-object", "one-chunk"])
@pytest.mark.parametrize("background", [False, True], ids=["in-line", "background"])
@torch.inference_mode()
def test_a_restore_puts_every_row_back_and_touches_no_other_page(engine, chunk_bytes, background):
    cache, tier, before = stored(engine, IDS, chunk_bytes)
    if background:
        land(cache, tier, IDS + [7])
        cache.release(IDS + [7])
    else:
        assert tier.restore(cache, IDS + [7]) == len(IDS)
    assert restored_exactly(cache, tier, IDS, before)


@pytest.mark.parametrize("ids", [IDS, ALIGNED], ids=["tail-rows", "block-aligned"])
@torch.inference_mode()
def test_narrow_rows_through_a_ring_of_one_object_slots_restore_the_same(engine, ids):
    cache, tier, before = stored(engine, ids, 1, narrow=True)
    assert tier._copy.word == 2
    land(cache, tier, ids + [7])
    cache.release(ids + [7])
    assert restored_exactly(cache, tier, ids, before)


@torch.inference_mode()
def test_a_restore_with_part_of_its_path_in_l1_starts_inside_a_block(engine):
    """The GPU cache dropped the prompt-end snapshot (keeping its pages) and evicted the leaf past it: the restore
    copies from the prompt's end on, its first object starting 10 rows into its block."""
    cache, tier, kv, sp = engine(1)
    prompt_end, n = 2 * BLOCK + 10, len(IDS)
    pages, page_ids = kv.alloc_with_ids(n)
    fill(tier, pages, seed=2)
    before = rows(tier, pages)
    cache.insert(IDS, pages, [Snapshot(sp.alloc(), prompt_end), Snapshot(sp.alloc(), n, spare=True)],
                 page_ids=page_ids)
    tier.backup(cache, IDS, (n, prompt_end))
    settle(cache, tier)
    (leaf,) = cache.root.children[IDS[0]].children.values()
    with cache._lock_node(leaf):  # the reply end is locked (a request continues it): only the interior snapshot goes
        cache.evict(need_slots=sp.free_count + 1)
    cache.evict(need_pages=kv.free_count + 1)  # then the leaf: demoted, its tail already written
    settle(cache, tier)
    (head,) = cache.root.children.values()
    assert head.in_l1 and cache.match(IDS + [7], touch=False).length == 0
    clear_all_but(tier, head.pages)
    assert tier.restore(cache, IDS + [7]) == n
    assert restored_exactly(cache, tier, IDS, before)


@pytest.mark.parametrize("narrow", [False, True], ids=["wide", "narrow"])
@torch.inference_mode()
def test_a_backup_writes_the_payload_a_torch_gather_writes(engine, narrow):
    cache, tier, kv, sp = engine(256 * MIB, narrow)
    pages, page_ids = kv.alloc_with_ids(len(IDS))
    fill(tier, pages, seed=3)
    cache.insert(IDS, pages, [Snapshot(sp.alloc(), len(IDS), spare=True)], page_ids=page_ids)
    (node,) = cache.root.children.values()
    spans = [(0, BLOCK), (2 * BLOCK, 3 * BLOCK), (5 * BLOCK, 6 * BLOCK)]  # apart, then a partial span alone
    with tier._copying():
        got = tier._gather_spans(node, spans) + tier._gather_spans(node, [(6 * BLOCK, len(IDS))])
    torch.cuda.synchronize()
    for (a, b), (tokens, segs, host) in zip(spans + [(6 * BLOCK, len(IDS))], got):
        assert tokens == IDS[a:b]
        at = 0
        for (name, seg), (_, pool) in zip(segs, tier._kv_targets()):
            n = seg.numel() * seg.element_size()
            assert torch.equal(seg.view(torch.uint8).cpu(), pool[pages[a:b].long()].contiguous().view(torch.uint8).cpu()), name
            assert not host[at + n:at + _seg_pad(n)].any(), f"{name}: padding not zero"
            at += _seg_pad(n)
