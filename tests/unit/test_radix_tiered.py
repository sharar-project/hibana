"""The tiered radix tree: with an SSD tier attached, L1 eviction demotes a
leaf instead of deleting it, L1 matching stops at SSD-only nodes, an SSD
match finds the deepest ready tail whose blocks are all ready, inserting a
path promotes SSD-only nodes back with the caller's pages, splits carry block
refs to the side that holds their last token, and SSD-only leaves leave the
tree only through the SSD tier's eviction or a flush."""
import torch

from hibana.cache.radix import RadixCache, SsdRef
from tests.unit.test_radix import FakePool, snaps

B = 4  # tokens per block in these tests


class FakeSsd:
    def __init__(self, keep=True):
        self.keep, self.flushes = keep, 0

    def evicting(self, cache, leaf, wait=True):
        return self.keep

    def clean(self, leaf):
        return True

    def fence(self):
        pass

    def flush(self):
        self.flushes += 1

    def abort(self, cache):
        pass

    def dropping_snapshot(self, cache, node):
        pass

    def pruning(self, node):
        pass


def make(keep=True):
    kv, st = FakePool(1000), FakePool(20)
    cache = RadixCache(kv, st, torch.device("cpu"))
    cache.ssd = FakeSsd(keep)
    return cache, kv, st


def cached(cache, kv, st, ids):
    pages = kv.alloc(len(ids))
    cache.insert(ids, pages, snaps({len(ids): st.alloc_slot()}), page_ids=pages.tolist())


def on_ssd(node, ends, tail=True):
    node.blocks = {e: SsdRef(f"b{e}", ready=True) for e in ends}
    node.tail = SsdRef("t", ready=True) if tail else None


def test_eviction_demotes_and_the_ssd_match_finds_the_tail():
    cache, kv, st = make()
    ids = list(range(10))
    cached(cache, kv, st, ids)
    (node,) = cache.root.children.values()
    on_ssd(node, [4, 8])
    free_before = kv.free_count
    cache.evict(need_pages=kv.free_count + 10)
    assert not node.in_l1 and node.snapshot_slot is None and cache.root.children  # demoted, still in the tree
    assert kv.free_count == free_before + 10 and cache.cached_tokens == 0 and cache.demotions == 1
    assert cache.ssd_prefixes() == 1
    assert cache.match(ids + [99]).length == 0  # L1 cannot use it
    hit = cache.ssd_match(ids + [99], 0, B)
    assert hit.length == 10 and [e for e, _ in hit.blocks] == [4, 8] and hit.path == [node]
    node.blocks[8].ready = False  # a block still being written: no resume point
    assert cache.ssd_match(ids + [99], 0, B) is None


def test_insert_promotes_ssd_only_nodes_back_into_l1():
    cache, kv, st = make()
    ids = list(range(10))
    cached(cache, kv, st, ids)
    (node,) = cache.root.children.values()
    on_ssd(node, [4, 8])
    cache.evict(need_pages=kv.free_count + 10)
    pages = kv.alloc(10)
    new_start = cache.insert(ids, pages, snaps({10: st.alloc_slot()}), page_ids=pages.tolist())
    assert new_start == 0 and node.in_l1 and node.pages.tolist() == pages.tolist() and cache.promotions == 1
    assert cache.match(ids + [99]).length == 10 and node.blocks  # still on the SSD too


def test_splitting_an_ssd_only_node_moves_block_refs_by_their_last_token():
    cache, kv, st = make()
    a = list(range(10))
    cached(cache, kv, st, a)
    (node,) = cache.root.children.values()
    on_ssd(node, [4, 8])
    cache.evict(need_pages=kv.free_count + 10)
    b = a[:6] + [50, 51]
    cached(cache, kv, st, b)  # splits the SSD-only node at 6 and promotes the head
    head = cache.root.children[0]
    assert head.key == a[:6] and head.in_l1 and set(head.blocks) == {4}
    (rest,) = [c for c in head.children.values() if not c.in_l1]
    assert rest.key == a[6:] and set(rest.blocks) == {8} and rest.tail is not None
    assert cache.ssd_match(a + [99], 6, B).length == 10  # L1 covers 6, the SSD the rest


def test_only_l1_leaves_are_demoted_and_ssd_leaves_are_evicted_separately():
    cache, kv, st = make()
    cached(cache, kv, st, list(range(10)))
    cached(cache, kv, st, list(range(6)) + [70, 71])
    head = cache.root.children[0]
    for c in head.children.values():
        on_ssd(c, [])
    cache.evict(need_pages=kv.free_count + 6)  # both leaves (4 + 2 tokens) go first; the head then has only SSD-only children
    assert all(not c.in_l1 for c in head.children.values())
    cache.evict(need_pages=kv.free_count + 6)
    assert not head.in_l1 and cache.cached_tokens == 0  # demoted, since its children need its key
    victim = cache.ssd_candidate()
    assert victim is not None and victim.parent is head and not victim.children
    cache.remove_ssd_leaf(victim)
    assert victim.key[0] not in head.children


def test_a_leaf_the_ssd_does_not_keep_is_deleted():
    cache, kv, st = make(keep=False)
    cached(cache, kv, st, list(range(10)))
    cache.evict(need_pages=kv.free_count + 10)
    assert not cache.root.children and cache.demotions == 0


def test_flush_drops_ssd_only_nodes_and_flushes_the_tier():
    cache, kv, st = make()
    cached(cache, kv, st, list(range(10)))
    cache.evict(need_pages=kv.free_count + 10)
    cached(cache, kv, st, [7, 8, 9])
    cache.flush()
    assert not cache.root.children and cache.ssd.flushes == 1 and cache.ssd_nodes == 0


def test_eviction_takes_a_stored_leaf_before_an_older_unstored_one():
    """With the SSD tier the least recently used leaf already stored goes first, so eviction does not wait for a copy;
    the least recently used of all goes only when no stored one is left."""
    cache, kv, st = make()
    cached(cache, kv, st, [1, 2, 3])  # older, its copy still queued
    cached(cache, kv, st, [5, 6, 7])  # newer, already stored
    older, newer = cache.root.children[1], cache.root.children[5]
    newer.last_access = older.last_access + 1
    cache.ssd.clean = lambda leaf: leaf is newer
    evicted = []
    cache.ssd.evicting = lambda c, leaf, wait=True: evicted.append(leaf) or True
    cache.evict(need_pages=kv.free_count + 1)
    assert evicted == [newer]
    cache.evict(need_pages=kv.free_count + 1)
    assert evicted == [newer, older]  # nothing stored left: the least recently used, dirty or not
