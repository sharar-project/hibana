"""Radix prefix cache over KV pages with GDN state snapshots.

A node holds a run of token IDs, the KV pages for those tokens, and optionally
a state snapshot slot: the conv/SSM state after the node's last token. For
a hybrid model a cached prefix is only usable up to a node that has a
snapshot, because attention pages alone cannot resume the recurrent layers.
Snapshots are taken at the end of every finished request and, when slots are
available, at checkpoint intervals inside long prefills.

The cache owns the pages and slots it references. Nodes on a matched path are
locked while a request uses them; eviction is LRU over unlocked leaves.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch


@dataclass(eq=False)
class Node:
    key: list[int]
    pages: torch.Tensor  # int32 [len(key)]
    parent: "Node | None"
    children: dict[int, "Node"] = field(default_factory=dict)
    snapshot_slot: int | None = None
    auxiliary: dict = field(default_factory=dict)  # draft identity -> metadata for these pages
    page_ids: list[int] | None = None  # host ids of `pages` (frees without a device read-back)
    lock: int = 0
    last_access: float = field(default_factory=time.monotonic)

    @property
    def depth(self) -> int:
        d, n = 0, self
        while n is not None:
            d += len(n.key)
            n = n.parent
        return d


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
        self.tier = None  # optional ColdTier: evicted leaves are parked in unified memory (and on disk) and restored on demand

    # ---- cold tier ----
    def full_key(self, node: Node) -> list[int]:
        parts = []
        while node is not None and node is not self.root:
            parts.append(node.key)
            node = node.parent
        return [t for key in reversed(parts) for t in key]

    def _walk(self, key: list[int]) -> tuple[Node, torch.Tensor] | None:
        """Exact path for `key` (node boundaries must align); None if absent."""
        node, length, pages = self._present_prefix(key)
        return (node, pages) if length == len(key) else None

    def _present_prefix(self, key: list[int]) -> tuple[Node, int, torch.Tensor]:
        """Deepest node whose full path is a prefix of `key`, its length and pages."""
        node, pos, pages = self.root, 0, []
        while pos < len(key):
            child = node.children.get(key[pos])
            if child is None or key[pos : pos + len(child.key)] != child.key:
                break
            pos += len(child.key)
            pages.append(child.pages)
            node = child
        empty = torch.empty(0, dtype=torch.int32, device=self.device)
        return node, pos, (torch.cat(pages) if pages else empty)

    def restore_parked(self, ids: list[int]) -> int:
        """Bring back every parked leaf that is a prefix of `ids`, shortest
        first, so a later turn of a parked session matches as if it had never
        been evicted. Returns the number of leaves restored."""
        if self.tier is None or self.flushing or not (self.tier.entries or self.tier.disk is not None):
            return 0
        restored = 0
        for entry in self.tier.find(ids):
            if self._walk(entry.key) is not None:  # present again already
                self.tier.discard(entry)  # an arena entry is dropped; a stored record stays
                continue
            present, present_len, present_pages = self._present_prefix(entry.key)
            missing = len(entry.key) - present_len
            self.lock(present, 1)  # the surviving prefix must outlive the eviction below
            try:
                self.evict(need_pages=missing, need_slots=1)
                if self.kv_pool.free_count < missing or self.state_pool.free_count < 1:
                    continue
                pages, page_ids = self.kv_pool.alloc_with_ids(missing)
                slot = self.state_pool.alloc()
                auxiliary = self.tier.restore(entry, pages, slot, start=present_len)  # arena entry or stored record
                if auxiliary is None:  # the stored record could not be read
                    self.kv_pool.free_ids(page_ids)
                    self.state_pool.free(slot)
                    continue
                self.insert(entry.key, torch.cat([present_pages, pages]), {len(entry.key): slot}, {len(entry.key): auxiliary},
                            page_ids=[-1] * present_len + page_ids)
                restored += 1
            finally:
                self.lock(present, -1)
        return restored

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
            if child is None:
                break
            k = child.key
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
        pages = torch.cat([n.pages for n in path[: path.index(best_node) + 1]])
        if touch:
            now = time.monotonic()
            for n in path[: path.index(best_node) + 1]:
                n.last_access = now
            self.hits += 1
        return Match(best_len, pages, best_node.snapshot_slot, best_node, raw)

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
            next_b = next((b for b in boundaries if b > pos), len(ids))
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
            n = 0
            while n < limit and k[n] == ids[pos + n]:
                n += 1
            if n < len(k):
                child = self._split(child, n)  # mismatch or boundary inside the node
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

    def _split(self, node: Node, n: int) -> Node:
        """Split `node` so that its first n tokens become a new parent node."""
        head = Node(key=node.key[:n], pages=node.pages[:n], parent=node.parent, lock=node.lock, last_access=node.last_access)
        if node.page_ids is not None:
            head.page_ids, node.page_ids = node.page_ids[:n], node.page_ids[n:]
        node.key = node.key[n:]
        node.pages = node.pages[n:]
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
                self.state_pool.free(node.snapshot_slot)
                node.snapshot_slot = None
                node.auxiliary = {}
                self.snapshot_evictions += 1
                continue
            leaf = self._lru_candidate()
            if leaf is None:
                break
            self._remove_leaf(leaf)

    def _lru_candidate(self, snapshot_only: bool = False) -> Node | None:
        best = None
        stack = [self.root]
        while stack:
            n = stack.pop()
            stack.extend(n.children.values())
            if n is self.root or n.lock > 0:
                continue
            if snapshot_only:
                if not n.children or n.snapshot_slot is None:
                    continue
            elif n.children:
                continue
            if (best is None or n.last_access < best.last_access
                    or (snapshot_only and n.last_access == best.last_access and n.depth < best.depth)):
                best = n
        return best

    def _remove_leaf(self, leaf: Node, park: bool = True) -> None:
        if park and self.tier is not None and leaf.snapshot_slot is not None:
            # The whole path is parked (self-contained entry) while it is still
            # linked; copies run before the frees below, on the same stream as
            # any later reuse of the pages.
            key = self.full_key(leaf)
            _, path_pages = self._walk(key)
            self.tier.park(key, path_pages, leaf.snapshot_slot, leaf.auxiliary)
        del leaf.parent.children[leaf.key[0]]
        if leaf.page_ids is not None:
            self.kv_pool.free_ids(leaf.page_ids)
        else:
            self.kv_pool.free(leaf.pages)
        if leaf.snapshot_slot is not None:
            self.state_pool.free(leaf.snapshot_slot)
        self.cached_tokens -= len(leaf.key)
        self.evictions += 1

    def flush(self) -> None:
        # Hide the cache immediately, but retain pinned pages and snapshots
        # until their final reader releases them. Admission bypasses the cache
        # while that drain is pending.
        if self.tier is not None:
            self.tier.clear()  # a flush invalidates parked sessions too
        while (leaf := self._lru_candidate()) is not None:
            self._remove_leaf(leaf, park=False)
        self.flushing = bool(self.root.children)

    def stats(self) -> dict:
        out = {"hits": self.hits, "misses": self.misses, "evictions": self.evictions,
               "snapshot_evictions": self.snapshot_evictions, "cached_tokens": self.cached_tokens, "flush_pending": self.flushing}
        if self.tier is not None:
            out["cold_tier"] = self.tier.stats()
        return out
