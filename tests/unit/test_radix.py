import torch

from sharar.cache.radix import RadixCache


class FakePool:
    def __init__(self, n):
        self._free = list(range(n - 1, 0, -1))
        self.freed = []

    @property
    def free_count(self):
        return len(self._free)

    def alloc(self, n=1):
        if n == 1 and not isinstance(n, torch.Tensor):
            pass
        out = [self._free.pop() for _ in range(n)]
        return torch.tensor(out, dtype=torch.int32)

    def alloc_slot(self):
        return self._free.pop()

    def alloc_with_ids(self, n):
        if n > self.free_count:
            raise MemoryError("pool exhausted")
        pages = self.alloc(n)
        return pages, pages.tolist()

    def free_ids(self, ids):
        self._free.extend(int(v) for v in ids)
        self.freed.extend(int(v) for v in ids)

    def free(self, x):
        if isinstance(x, torch.Tensor):
            self._free.extend(int(v) for v in x.tolist())
            self.freed.extend(int(v) for v in x.tolist())
        else:
            self._free.append(int(x))
            self.freed.append(int(x))


def make():
    kv, st = FakePool(1000), FakePool(20)
    return RadixCache(kv, st, torch.device("cpu")), kv, st


def test_insert_then_match_full_sequence():
    cache, kv, st = make()
    ids = [1, 2, 3, 4, 5, 6]
    pages = kv.alloc(6)
    slot = st.alloc_slot()
    assert cache.insert(ids, pages, {6: slot}) == 0
    m = cache.match(ids + [7, 8])
    assert m.length == 6 and m.snapshot_slot == slot and m.pages.tolist() == pages.tolist()
    # a shorter prompt that ends inside the node has no snapshot to resume from
    m2 = cache.match([1, 2, 3])
    assert m2.length == 0 and m2.raw_length == 3


def test_checkpoint_boundaries_split_nodes_and_match_partial():
    cache, kv, st = make()
    ids = list(range(100, 110))
    pages = kv.alloc(10)
    s4, s10 = st.alloc_slot(), st.alloc_slot()
    cache.insert(ids, pages, {4: s4, 10: s10})
    m = cache.match(ids[:7] + [999])
    assert m.length == 4 and m.snapshot_slot == s4 and m.pages.tolist() == pages[:4].tolist()
    m = cache.match(ids + [999])
    assert m.length == 10 and m.snapshot_slot == s10


def test_second_insert_reuses_prefix_and_reports_new_start():
    cache, kv, st = make()
    a = [1, 2, 3, 4]
    pa = kv.alloc(4)
    sa = st.alloc_slot()
    cache.insert(a, pa, {4: sa})
    b = [1, 2, 3, 4, 5, 6]
    pb = kv.alloc(6)
    sb = st.alloc_slot()
    new_start = cache.insert(b, pb, {6: sb})
    assert new_start == 4
    kv.free(pb[:new_start])  # caller frees duplicated prefix pages
    m = cache.match(b + [7])
    assert m.length == 6 and m.pages.tolist() == pa.tolist() + pb[4:].tolist()
    assert cache.cached_tokens == 6


def test_divergent_branch_splits_and_evicts_lru():
    cache, kv, st = make()
    a = [1, 2, 3, 4]
    cache.insert(a, kv.alloc(4), {4: st.alloc_slot()})
    b = [1, 2, 9, 9]
    cache.insert(b, kv.alloc(4), {4: st.alloc_slot()})
    assert cache.match(a + [0]).length == 4 and cache.match(b + [0]).length == 4
    # lock branch b (in use), evict: only a's leaf can go
    mb = cache.match(b + [0])
    cache.lock(mb.node, +1)
    kv._free = []  # pretend the pool is empty
    cache.evict(need_pages=1)
    assert cache.match(a + [0]).length == 0 and cache.match(b + [0]).length == 4
    assert cache.evictions == 1 and len(kv.freed) == 2  # a's two unique pages
    cache.lock(mb.node, -1)
    cache.flush()
    assert cache.cached_tokens == 0 and len(st._free) == 19


def test_new_snapshot_on_existing_prefix_keeps_its_matching_auxiliary():
    cache, kv, st = make()
    ids = [1, 2, 3, 4]
    cache.insert(ids, kv.alloc(4), {4: st.alloc_slot()}, {4: {'draft': 'end'}})
    slot = st.alloc_slot()
    pages = kv.alloc(4)
    duplicate = cache.insert(ids, pages, {2: slot}, {2: {'draft': 'middle'}})
    kv.free(pages[:duplicate])
    match = cache.match([1, 2, 9])
    assert match.snapshot_slot == slot and match.node.auxiliary == {'draft': 'middle'}
    # Discarding a duplicate snapshot must also discard its different features.
    pages = kv.alloc(4)
    duplicate = cache.insert(ids, pages, {2: st.alloc_slot()}, {2: {'draft': 'wrong'}})
    kv.free(pages[:duplicate])
    assert cache.match([1, 2, 9]).node.auxiliary == {'draft': 'middle'}
    cache.flush()
    assert st.free_count == 19 and kv.free_count == 999


def test_snapshot_pressure_preserves_kv_and_locked_resume_states():
    cache, kv, st = make()
    ids = [1, 2, 3, 4]
    first, last = st.alloc_slot(), st.alloc_slot()
    cache.insert(ids, kv.alloc(4), {2: first, 4: last}, {2: {'draft': 'old'}, 4: {'draft': 'new'}})
    match = cache.match(ids)
    cache.lock(match.node, 1)
    cache.evict(need_slots=19)
    assert first not in st.freed and last not in st.freed
    cache.lock(match.node, -1)
    cache.evict(need_slots=18)
    assert first in st.freed and last not in st.freed
    assert cache.cached_tokens == 4 and cache.snapshot_evictions == 1 and cache.evictions == 0
    assert cache.match(ids).node.auxiliary == {'draft': 'new'}
    cache.flush()
    assert st.free_count == 19 and kv.free_count == 999
