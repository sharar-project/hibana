"""Radix prefix cache over KV pages with GDN state snapshots.

A node holds a run of token IDs, the KV pages for those tokens, and optionally
a state snapshot slot: the conv/SSM state after the node's last token. For
a hybrid model a cached prefix is only usable up to a node that has a
snapshot, because attention pages alone cannot resume the recurrent layers.
Snapshots are taken at the end of every finished request and, when slots are
available, at checkpoint intervals inside long prefills.

The cache owns the pages and slots it references. Nodes on a matched path are
locked while a request uses them; eviction is LRU over unlocked leaves.

With an SSD tier attached (`ssd.py`), a node also records which of its KV
blocks and which session tail are on the SSD. Evicting such a leaf from L1
demotes it instead of deleting it: its pages and slot go back to the pools
and the node stays, SSD-only, so a later match can find a resume point on the
SSD past the L1 prefix. L1 stays prefix-closed: a node with pages always has
a parent with pages. The SSD tier evicts SSD-only leaves to stay within its
own budget.
"""

from __future__ import annotations

import time
from bisect import bisect_right
from dataclasses import dataclass, field

import torch


@dataclass
class SsdRef:
    key: str  # hex hash of the stored object (`ssd_store.py`)
    ready: bool = False  # written and renamed into place
    nbytes: int = 0  # the file's size once ready (0 before); -1 once its write failed or the file is deleted


@dataclass(eq=False)
class Node:
    key: list[int]
    pages: torch.Tensor | None  # int32 [len(key)]; None when the node is only on the SSD
    parent: "Node | None"
    children: dict[int, "Node"] = field(default_factory=dict)
    snapshot_slot: int | None = None
    auxiliary: dict = field(default_factory=dict)  # draft identity -> metadata for these pages
    page_ids: list[int] | None = None  # host ids of `pages` (frees without a device read-back)
    lock: int = 0
    last_access: float = field(default_factory=time.monotonic)
    blocks: dict[int, SsdRef] = field(default_factory=dict)  # KV blocks on the SSD whose last token is in this node, by absolute end
    tail: SsdRef | None = None  # a session tail on the SSD ending exactly at this node's end

    @property
    def in_l1(self) -> bool:
        return self.pages is not None

    @property
    def depth(self) -> int:
        d, n = 0, self
        while n is not None:
            d += len(n.key)
            n = n.parent
        return d


@dataclass
class SsdHit:
    length: int  # resume point: the end of a node with a ready tail
    path: list["Node"]  # root to that node (exclusive of the root)
    blocks: list[tuple[int, SsdRef]]  # (absolute end, ref) of the blocks to read, ascending
    tail: SsdRef
    l1_end: int  # the path's KV is in L1 up to here (a node end): only the blocks past it are read


@dataclass
class Match:
    length: int  # usable prefix length (ends at a snapshot)
    pages: torch.Tensor  # pages for the usable prefix
    snapshot_slot: int | None  # state after `length` tokens (None only when length == 0)
    node: "Node | None"  # deepest node used (locked by the caller)
    raw_length: int  # token match length ignoring snapshots (diagnostic)


class RadixCache:
    def __init__(self, kv_pool, state_pool, device: torch.device):
        self.kv_pool = kv_pool
        self.state_pool = state_pool
        self.device = device
        self.root = Node(key=[], pages=torch.empty(0, dtype=torch.int32, device=device), parent=None)
        self.hits = self.misses = self.evictions = 0
        self.snapshot_evictions = 0
        self.cached_tokens = 0
        self.flushing = False
        self.ssd = None  # optional SsdTier (`ssd.py`): with one, eviction can demote leaves instead of deleting them
        self.ssd_nodes = self.ssd_evictions = self.demotions = self.promotions = 0

    # ---- lookup ----
    def match(self, ids: list[int], *, touch: bool = True) -> Match:
        if self.flushing:
            self.misses += int(touch)
            return Match(0, self.root.pages, None, None, 0)
        node = self.root
        pos = 0
        best_node, best_len = None, 0
        path: list[Node] = []
        while pos < len(ids):
            child = node.children.get(ids[pos])
            if child is None or not child.in_l1:
                break
            k = child.key
            if ids[pos:pos + len(k)] == k:  # the whole node, in one comparison
                n = len(k)
            else:
                n = 0
                while n < len(k) and pos + n < len(ids) and k[n] == ids[pos + n]:
                    n += 1
            if n < len(k):
                pos += n  # partial node match: no snapshot here
                break
            pos += n
            node = child
            path.append(child)
            if child.snapshot_slot is not None:
                best_node, best_len = child, pos
        raw = pos
        if best_node is None or best_len == 0:
            self.misses += int(touch)
            return Match(0, torch.empty(0, dtype=torch.int32, device=self.device), None, None, raw)
        used = path[: path.index(best_node) + 1]
        pages = torch.cat([n.pages for n in used])
        if touch:
            now = time.monotonic()
            for n in used:
                n.last_access = now
            self.hits += 1
        return Match(best_len, pages, best_node.snapshot_slot, best_node, raw)

    def ssd_match(self, ids: list[int], l1_len: int, block: int, pending: bool = False) -> SsdHit | None:
        """The deepest resume point on the SSD past `l1_len`: the end of a node
        with a ready tail, every KV block covering [l1_len, that end) ready
        (the tail holds the last partial block). None when there is none.
        Blocks the path still holds in L1 (where eviction dropped a snapshot
        but kept the pages) are left out. With `pending`, objects still queued
        or being written also count."""
        usable = (lambda ref: ref is not None and (ref.ready or ref.nbytes >= 0)) if pending else (lambda ref: ref is not None and ref.ready)
        if self.flushing:
            return None
        node, pos, path, refs, best, resident = self.root, 0, [], {}, None, 0
        start = cursor = -1  # the KV to read starts past `start`; every block ending before `cursor` is usable
        while pos < len(ids):
            child = node.children.get(ids[pos])
            if child is None or ids[pos : pos + len(child.key)] != child.key:
                break
            pos += len(child.key)
            node = child
            path.append(child)
            refs.update(child.blocks)
            if child.in_l1 and resident == pos - len(child.key):  # L1 nodes are a prefix of any path
                resident = pos
            if pos <= l1_len:
                continue
            if max(l1_len, resident) != start:  # moves only within the L1 prefix, where no block is needed yet
                start = max(l1_len, resident)
                cursor = (start // block + 1) * block
            whole = pos // block * block
            while cursor <= whole and usable(refs.get(cursor)):  # each block is checked once, not once per tail
                cursor += block
            if cursor > whole and usable(child.tail):
                best = (pos, len(path), child.tail, start, whole)
        if best is None:
            return None
        pos, k, tail, start, whole = best
        needed = [(end, refs[end]) for end in range((start // block + 1) * block, whole + 1, block)]
        return SsdHit(pos, path[:k], needed, tail, start)

    # ---- SSD tier (no-ops without one) ----
    def restore(self, ids: list[int], reserve: int = 0) -> int:
        """Bring a session on the SSD back into L1 before it is matched, if the rest of ids and `reserve` more pages
        fit too; returns the tokens restored."""
        return self.ssd.restore(self, ids, reserve) if self.ssd is not None else 0

    def restore_async(self, ids: list[int], reserve: int = 0) -> bool:
        """Admission's form of `restore`: False while a restore for ids (or one ahead of it) is still reading, True
        when ids can be matched now. A restore that landed keeps its path locked until `release(ids)`."""
        if self.ssd is None:
            return True
        if not self.ssd.async_restore:
            self.ssd.restore(self, ids, reserve)
            return True
        return self.ssd.restore_async(self, ids, reserve)

    def release(self, ids: list[int]) -> None:
        if self.ssd is not None:
            self.ssd.release(self, ids)

    def backup(self, ids: list[int], resume_points: tuple[int, ...] = ()) -> None:
        if self.ssd is not None:
            self.ssd.backup(self, ids, resume_points)

    def step(self) -> None:
        """Once per engine loop iteration: apply finished SSD writes, trim to the SSD budget and start queued backups."""
        if self.ssd is not None:
            self.ssd.step(self)

    def lock(self, node: Node | None, delta: int) -> None:
        while node is not None and node is not self.root:
            node.lock += delta
            assert node.lock >= 0, "unbalanced cache release"
            node = node.parent
        if delta < 0 and self.flushing:
            self.flush()

    # ---- insertion ----
    def insert(self, ids: list[int], pages: torch.Tensor, snapshots: dict[int, int], auxiliary: dict | None = None, page_ids: list[int] | None = None) -> int:
        """Insert a full sequence. `pages` are its KV pages (the cache owns the
        ones it keeps), `snapshots` maps prefix length -> state slot (owned by
        the cache from now on) and forces node boundaries there.

        Returns `new_start`: tokens before it were already cached, so their
        pages in `pages[:new_start]` are duplicates the caller must free (unless
        they were the cache's own prefix pages to begin with)."""
        assert len(ids) == pages.shape[0]
        assert not self.flushing, "cannot insert while an active prefix is being flushed"
        node = self.root
        pos = 0
        boundaries = sorted(b for b in snapshots if 0 < b <= len(ids))
        new_start = None
        while pos < len(ids):
            i = bisect_right(boundaries, pos)
            next_b = boundaries[i] if i < len(boundaries) else len(ids)
            child = node.children.get(ids[pos])
            if child is None:
                if new_start is None:
                    new_start = pos
                new = Node(key=ids[pos:next_b], pages=pages[pos:next_b].clone(), parent=node)
                if page_ids is not None:
                    new.page_ids = list(page_ids[pos:next_b])
                    assert -1 not in new.page_ids, "new cache node must own its pages"
                new.snapshot_slot = snapshots.get(next_b)
                if new.snapshot_slot is not None:
                    new.auxiliary = (auxiliary or {}).get(next_b, {})
                node.children[ids[pos]] = new
                self.cached_tokens += next_b - pos
                node, pos = new, next_b
                continue
            k = child.key
            limit = min(len(k), next_b - pos)
            if limit == len(k) and ids[pos:pos + limit] == k:  # the whole node, in one comparison
                n = limit
            else:
                n = 0
                while n < limit and k[n] == ids[pos + n]:
                    n += 1
            if n < len(k):
                child = self._split(child, n)  # mismatch or boundary inside the node
            if not child.in_l1:  # an SSD-only node comes back into L1 with the caller's pages
                if new_start is None:
                    new_start = pos
                child.pages = pages[pos:pos + n].clone()
                if page_ids is not None:
                    child.page_ids = list(page_ids[pos:pos + n])
                    assert -1 not in child.page_ids, "promoted cache node must own its pages"
                self.cached_tokens += n
                self.promotions += 1
                self.ssd_nodes -= 1
            pos += n
            node = child
            node.last_access = time.monotonic()
            if pos in snapshots:
                if node.snapshot_slot is None:
                    node.snapshot_slot = snapshots[pos]
                    node.auxiliary = (auxiliary or {}).get(pos, {})
                else:
                    self.state_pool.free(snapshots[pos])
        if new_start is None:
            new_start = len(ids)
        return new_start

    def insert_ssd_path(self, ids: list[int], blocks: dict[int, SsdRef], tail: SsdRef) -> None:
        """Record a stored session at startup: add the path for ids (new nodes
        SSD-only), put each block ref on the node holding the block's last
        token and the tail on the node ending at len(ids)."""
        node, pos, path = self.root, 0, []
        while pos < len(ids):
            child = node.children.get(ids[pos])
            if child is None:
                child = Node(key=ids[pos:], pages=None, parent=node)
                node.children[ids[pos]] = child
                self.ssd_nodes += 1
                n = len(child.key)
            else:
                k, n = child.key, 0
                while n < len(k) and pos + n < len(ids) and k[n] == ids[pos + n]:
                    n += 1
                if n < len(k):
                    child = self._split(child, n)
            path.append((pos, child))
            node, pos = child, pos + n
        for end, ref in blocks.items():
            for start, n in path:
                if start < end <= start + len(n.key):
                    n.blocks.setdefault(end, ref)
                    break
        if node is not self.root and node.tail is None:
            node.tail = tail

    def _split(self, node: Node, n: int) -> Node:
        """Split `node` so that its first n tokens become a new parent node."""
        head = Node(key=node.key[:n], pages=None if node.pages is None else node.pages[:n], parent=node.parent, lock=node.lock,
                    last_access=node.last_access)
        if node.page_ids is not None:
            head.page_ids, node.page_ids = node.page_ids[:n], node.page_ids[n:]
        if node.blocks:
            cut = node.parent.depth + n
            head.blocks = {end: ref for end, ref in node.blocks.items() if end <= cut}
            node.blocks = {end: ref for end, ref in node.blocks.items() if end > cut}
        node.key = node.key[n:]
        if node.pages is not None:
            node.pages = node.pages[n:]
        if not head.in_l1:
            self.ssd_nodes += 1
        node.parent.children[head.key[0]] = head
        head.children[node.key[0]] = node
        node.parent = head
        return head

    # ---- eviction ----
    def evict(self, need_pages: int = 0, need_slots: int = 0) -> None:
        """Free unlocked leaves, least recently used first, until the pools can
        satisfy the request or nothing evictable remains."""
        while (self.kv_pool.free_count < need_pages) or (self.state_pool.free_count < need_slots):
            if self.kv_pool.free_count >= need_pages and (node := self._lru_candidate(snapshot_only=True)) is not None:
                # Recurrent capacity can run out while KV capacity is ample.
                # Drop an optional interior snapshot, retaining the radix path
                # and its newer resume points. Locked loans remain untouched.
                if self.ssd is not None:
                    self.ssd.fence()
                self.state_pool.free(node.snapshot_slot)
                node.snapshot_slot = None
                node.auxiliary = {}
                self.snapshot_evictions += 1
                continue
            leaf = self._lru_candidate()
            if leaf is None:
                break
            self._remove_leaf(leaf)

    def _walk(self, l1_only: bool = False):
        """Every node below the root, depth first; LRU ties go to the first in this order. With `l1_only`, only the
        nodes in L1, in the same order: L1 is prefix-closed, so no L1 node sits below an SSD-only one."""
        stack = [c for c in self.root.children.values() if c.in_l1 or not l1_only]
        while stack:
            n = stack.pop()
            stack.extend(c for c in n.children.values() if c.in_l1 or not l1_only)
            yield n

    def _lru_candidate(self, snapshot_only: bool = False) -> Node | None:
        """The least recently used unlocked leaf (with `snapshot_only`, an interior node holding a snapshot). With the
        SSD tier, the least recently used leaf already stored (`SsdTier.clean`) comes first, so eviction does not wait
        for a copy; the least recently used leaf of all only when none is."""
        best = clean = None
        tier = None if snapshot_only else self.ssd
        stack = [c for c in self.root.children.values() if c.in_l1]
        while stack:  # L1 nodes only: L1 is prefix-closed, so no SSD-only node has one below it
            n = stack.pop()
            kids = [c for c in n.children.values() if c.in_l1]
            stack.extend(kids)
            if n.lock > 0:
                continue
            if snapshot_only:
                if not kids or n.snapshot_slot is None:
                    continue
            elif kids:
                continue
            if (best is None or n.last_access < best.last_access
                    or (snapshot_only and n.last_access == best.last_access and n.depth < best.depth)):
                best = n
            if tier is not None and (clean is None or n.last_access < clean.last_access) and tier.clean(n):
                clean = n
        return clean or best

    def _remove_leaf(self, leaf: Node, demote: bool = True) -> None:
        if demote and self.ssd is not None and (self.ssd.evicting(self, leaf) or leaf.children):
            self._demote(leaf)  # its blocks and tail are on (or queued for) the SSD; SSD-only children need its key
            return
        del leaf.parent.children[leaf.key[0]]
        self._free_l1(leaf)

    def _demote(self, node: Node) -> None:
        """Free a node's L1 pages and slot; it stays in the tree, SSD-only."""
        self._free_l1(node)
        node.pages, node.page_ids, node.snapshot_slot, node.auxiliary = None, None, None, {}
        self.demotions += 1
        self.ssd_nodes += 1

    def _free_l1(self, node: Node) -> None:
        if self.ssd is not None:
            self.ssd.fence()
        if node.page_ids is not None:
            self.kv_pool.free_ids(node.page_ids)
        else:
            self.kv_pool.free(node.pages)
        if node.snapshot_slot is not None:
            self.state_pool.free(node.snapshot_slot)
        self.cached_tokens -= len(node.key)
        self.evictions += 1

    def ssd_candidate(self, unusable: bool = False, resident: bool = False) -> Node | None:
        """The least recently used node: an SSD-only leaf (with `unusable`, only
        one whose tail is missing or failed), or with `resident` an L1 node with
        a ready copy on the SSD, whose copies the budget can delete while the
        node keeps its pages."""
        best = None
        for n in self._walk(l1_only=resident):
            if resident:
                if not ((n.tail is not None and n.tail.ready) or any(r.ready for r in n.blocks.values())):
                    continue
            elif n.in_l1 or n.children or n.lock:  # locked: a restore is reading it
                continue
            if unusable and n.tail is not None and n.tail.nbytes >= 0:
                continue
            if best is None or n.last_access < best.last_access:
                best = n
        return best

    def ssd_sessions(self) -> int:
        """Sessions a request can resume from the SSD: nodes whose tail is written."""
        return sum(n.tail is not None and n.tail.ready for n in self._walk())

    def remove_ssd_leaf(self, node: Node) -> None:
        assert not node.in_l1 and not node.children
        del node.parent.children[node.key[0]]
        self.ssd_nodes -= 1
        self.ssd_evictions += 1

    def _drop_ssd_only(self) -> None:
        """Unlink every SSD-only subtree (a flush)."""
        stack = [self.root]
        while stack:
            n = stack.pop()
            for key, c in list(n.children.items()):
                if c.in_l1:
                    stack.append(c)
                else:
                    del n.children[key]
                    sub = [c]
                    while sub:
                        m = sub.pop()
                        self.ssd_nodes -= 1
                        sub.extend(m.children.values())

    def flush(self) -> None:
        # Hide the cache immediately, but retain pinned pages and snapshots
        # until their final reader releases them. Admission bypasses the cache
        # while that drain is pending.
        if self.ssd is not None and not self.flushing:  # once per flush, not again each time a reader releases its lock
            self.ssd.abort(self)  # a restore still reading: its pages and slot go back, and its path is no longer held
            self._drop_ssd_only()
            self.ssd.flush()  # a new epoch, so every ref left on the L1 nodes is stale
        while (leaf := self._lru_candidate()) is not None:
            self._remove_leaf(leaf, demote=False)
        self.flushing = bool(self.root.children)

    def stats(self) -> dict:
        out = {"hits": self.hits, "misses": self.misses, "evictions": self.evictions,
               "snapshot_evictions": self.snapshot_evictions, "cached_tokens": self.cached_tokens, "flush_pending": self.flushing}
        if self.ssd is not None:
            out["ssd"] = {"nodes": self.ssd_nodes, "demotions": self.demotions,
                          "promotions": self.promotions, "evictions": self.ssd_evictions, **self.ssd.stats()}
        return out
