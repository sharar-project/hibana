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
        self.cached_tokens = 0
        self.flushing = False

    # ---- lookup ----
    def match(self, ids: list[int]) -> Match:
        if self.flushing:
            self.misses += 1
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
            self.misses += 1
            return Match(0, torch.empty(0, dtype=torch.int32, device=self.device), None, None, raw)
        pages = torch.cat([n.pages for n in path[: path.index(best_node) + 1]])
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
    def insert(self, ids: list[int], pages: torch.Tensor, snapshots: dict[int, int], auxiliary: dict | None = None) -> int:
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
                else:
                    self.state_pool.free(snapshots[pos])
        if new_start is None:
            new_start = len(ids)
        return new_start

    def _split(self, node: Node, n: int) -> Node:
        """Split `node` so that its first n tokens become a new parent node."""
        head = Node(key=node.key[:n], pages=node.pages[:n], parent=node.parent, lock=node.lock, last_access=node.last_access)
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
            leaf = self._lru_leaf()
            if leaf is None:
                break
            self._remove_leaf(leaf)

    def _lru_leaf(self) -> Node | None:
        best = None
        stack = [self.root]
        while stack:
            n = stack.pop()
            stack.extend(n.children.values())
            if n is self.root or n.children or n.lock > 0:
                continue
            if best is None or n.last_access < best.last_access:
                best = n
        return best

    def _remove_leaf(self, leaf: Node) -> None:
        del leaf.parent.children[leaf.key[0]]
        self.kv_pool.free(leaf.pages)
        if leaf.snapshot_slot is not None:
            self.state_pool.free(leaf.snapshot_slot)
        self.cached_tokens -= len(leaf.key)
        self.evictions += 1

    def flush(self) -> None:
        # Hide the cache immediately, but retain pinned pages and snapshots
        # until their final reader releases them. Admission bypasses the cache
        # while that drain is pending.
        while (leaf := self._lru_leaf()) is not None:
            self._remove_leaf(leaf)
        self.flushing = bool(self.root.children)

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "evictions": self.evictions, "cached_tokens": self.cached_tokens, "flush_pending": self.flushing}
