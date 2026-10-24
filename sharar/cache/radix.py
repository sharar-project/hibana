"""Radix prefix cache over KV pages with GDN state snapshots.

A node holds a run of token IDs, the KV pages for those tokens, and optionally a `Snapshot` (`snapshot.py`): the
conv/SSM state after the node's last token and the drafters' metadata for it. For a hybrid model a cached prefix is
only usable up to a node that has a snapshot, because attention pages alone cannot resume the recurrent layers. Where
snapshots go is the executor's decision (`Executor._checkpoint_positions`); the tree keeps them, protects what is
locked and evicts the rest.

Protection (SGLang's lock_ref). The only way to keep anything is a `TreeLock` (`lock(node, owner=...)`, ended by
`unlock`): one per running request (`req.lock`), one per in-flight or landed SSD restore (`job.lock`) and short-lived
admission probes (`_lock_node`). A lock on node X adds 1 to `lock_ref` on X and on every ancestor (the KV of X's whole
path stays in L1) and 1 to `state_lock_ref` on X alone (X's snapshot stays). `inc_lock_ref` and `dec_lock_ref` are the
counters' only writers, and the evictable totals move with them: `evictable_size` / `protected_size` (L1 KV tokens
with lock_ref 0 / above 0) and `state_evictable_size` / `state_protected_size` (L1 snapshots with state_lock_ref 0 /
above 0), which admission budgets without evicting anything (`fits`).

Eviction. Short of state slots, an unlocked snapshot goes, the lowest `get_priority` first (spare ones, then the least
recently used): an unlocked leaf goes whole, any other node keeps its KV and loses only its snapshot. Short of KV
pages, unlocked leaves go (`evictable_leaves`), the least recently used first. Recency is a logical tick
(`snapshot.tick`), so tensor-parallel ranks choose alike. Either way the dead KV left behind goes too (`_prune`): a node
no lock reaches, with no snapshot, no children and no stored tail, can never serve a resume. Nodes are never merged.

With an SSD tier attached (`ssd.py`), a node also records which of its KV blocks are on the SSD, and whether its
tail is. Evicting such a leaf from L1 demotes it instead of deleting it: its pages and slot go back to the pools and
the node stays, SSD-only, so a later match can find a resume point on the SSD past the L1 prefix. L1 stays
prefix-closed: a node with pages always has a parent with pages. The SSD tier evicts SSD-only leaves to stay within
its own budget. Every site that frees pages or a slot tells the tier first (`evicting`, `dropping_snapshot`,
`pruning`, `fence`), because its queued copies may still read them.
"""

from __future__ import annotations

import contextlib
import weakref
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

import torch

from sharar.cache.snapshot import GRID, Snapshot, get_priority, tick


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
    snapshot: Snapshot | None = None  # the state after this node's last token, and its drafter metadata
    page_ids: list[int] | None = None  # host ids of `pages` (frees without a device read-back)
    lock_ref: int = 0  # locks at this node or below it: its KV stays (written only by inc_lock_ref / dec_lock_ref)
    state_lock_ref: int = 0  # locks on this node: its snapshot stays (written only by inc_lock_ref / dec_lock_ref)
    last_access: int = field(default_factory=tick)  # KV recency (a logical tick): each resume along the path, each insert
    blocks: dict[int, SsdRef] = field(default_factory=dict)  # KV blocks on the SSD whose last token is in this node, by absolute end
    tail: SsdRef | None = None  # the tail of a prefix stored on the SSD, ending exactly at this node's end

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

    @property
    def snapshot_slot(self) -> int | None:
        return self.snapshot.slot if self.snapshot is not None else None


def owner_kind(owner) -> str:
    """A lock's owner, for stats and checks: "request", "restore" (an SSD restore job) or "probe" (admission)."""
    return owner if isinstance(owner, str) else getattr(owner, "lock_kind", "request")


class TreeLock:
    """A lock on a node (SGLang's lock_ref, `RadixCache.lock`): the node's snapshot and the KV of its whole path stay
    until `RadixCache.unlock`. `node` is None once unlocked."""

    __slots__ = ("node", "owner")

    def __init__(self, node: "Node", owner):
        self.node, self.owner = node, owner

    def __repr__(self) -> str:
        return f"TreeLock({owner_kind(self.owner)} at {self.node.depth if self.node is not None else None})"


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
    snapshot: Snapshot | None  # the snapshot after `length` tokens (None only when length == 0)
    node: "Node | None"  # deepest node used: the caller locks it (`RadixCache.lock`)
    raw_length: int  # token match length ignoring snapshots: KV the tree holds for this prefix
    # SGLang's mamba_branching_seqlen: the KV match's end when it runs at least a grid past the matched snapshot, else
    # 0. The prefill forward that contains it snapshots there, rounded down to its own grid (the branching point).
    branching_seqlen: int = 0

    @property
    def snapshot_slot(self) -> int | None:
        return self.snapshot.slot if self.snapshot is not None else None


class RadixCache:
    def __init__(self, kv_pool, state_pool, device: torch.device):
        self.kv_pool = kv_pool
        self.state_pool = state_pool
        self.device = device
        self.root = Node(key=[], pages=torch.empty(0, dtype=torch.int32, device=device), parent=None)
        self._locks: dict[TreeLock, None] = {}  # every live lock, in the order taken
        # The evictable totals and candidates (dicts as ordered sets: every rank iterates them alike).
        self._evictable_tokens = self._protected_tokens = 0  # L1 KV tokens with lock_ref == 0 / > 0
        self._protected_snapshots = 0  # L1 snapshots with state_lock_ref > 0
        self._spare_snapshots = 0  # L1 snapshots marked spare (for `stats`, every engine step: no walk)
        self.evictable_leaves: dict[Node, None] = {}  # L1 nodes with lock_ref 0 and no L1 child: KV pressure's candidates
        self.evictable_snapshots: dict[Node, None] = {}  # L1 nodes with a snapshot and state_lock_ref 0: slot pressure's
        self.branching = weakref.WeakSet()  # snapshots placed at a branching point, to count resumes from them
        self.hits = self.misses = self.evictions = 0
        self.snapshot_evictions = 0  # snapshots slot pressure took from a node that keeps its KV
        self.pruned = 0  # dead-KV nodes freed (`_prune`)
        self.full_reprefills = 0  # admissions that resumed from nothing although the tree held >= 64 tokens of their KV
        self.branching_resumes = 0  # admissions that resumed from a branching point
        self.evicted: Counter = Counter()  # (reason, spare) of every snapshot the cache gave up
        self.cached_tokens = 0
        self.flushing = False
        self.ssd = None  # optional SsdTier (`ssd.py`): with one, eviction can demote leaves instead of deleting them
        self.ssd_nodes = self.ssd_evictions = self.demotions = self.promotions = 0

    # ---- lookup ----
    def match(self, ids: list[int], *, touch: bool = True) -> Match:
        """The deepest snapshot on ids' path. With `touch` (an admission resuming there), the path's nodes count as
        used for KV recency, and the matched snapshot as resumed from (`Snapshot.last_access`; no longer spare); nodes
        walked past keep their snapshots' recency."""
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
            if child.snapshot is not None:
                best_node, best_len = child, pos
        raw = pos
        branching = raw if raw - best_len >= GRID else 0
        if best_node is None or best_len == 0:
            self.misses += int(touch)
            return Match(0, torch.empty(0, dtype=torch.int32, device=self.device), None, None, raw, branching)
        used = path[: path.index(best_node) + 1]
        pages = torch.cat([n.pages for n in used])
        if touch:
            now = tick()
            for n in used:
                n.last_access = now
            best_node.snapshot.last_access = now
            if best_node.snapshot.spare:  # a request needs it after all
                best_node.snapshot.spare = False
                self._spare_snapshots -= 1
            self.hits += 1
        return Match(best_len, pages, best_node.snapshot, best_node, raw, branching)

    def node_at(self, ids: list[int]) -> Node | None:
        """The node whose path from the root is exactly ids, or None."""
        node, pos = self.root, 0
        while pos < len(ids):
            child = node.children.get(ids[pos])
            if child is None or ids[pos:pos + len(child.key)] != child.key:
                return None
            node, pos = child, pos + len(child.key)
        return node

    @staticmethod
    def path_pages(node: Node) -> torch.Tensor:
        """The L1 pages of node's whole path, root first."""
        parts = []
        while node.parent is not None:
            parts.append(node.pages)
            node = node.parent
        return torch.cat(parts[::-1]) if parts else node.pages

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
        """Bring a stored prefix back into L1 before it is matched, if the rest of ids and `reserve` more pages
        fit too; returns the tokens restored."""
        return self.ssd.restore(self, ids, reserve) if self.ssd is not None else 0

    def restore_async(self, ids: list[int], reserve: int = 0) -> bool:
        """The queue head's prefetch (`StateManager.prefetch`): False while a restore for ids (or another one) is still
        reading, True when ids can be matched now. A restore that landed locks its node until `release(ids)`."""
        if self.ssd is None:
            return True
        return self.ssd.restore_async(self, ids, reserve)

    def release(self, ids: list[int]) -> None:
        """The request a landed restore locked its node for has its own lock there now, or is gone: the restore
        unlocks."""
        if self.ssd is not None:
            self.ssd.release(self, ids)

    def backup(self, ids: list[int], resume_points: tuple[int, ...] = ()) -> None:
        if self.ssd is not None:
            self.ssd.backup(self, ids, resume_points)

    def step(self, prefilling: bool = False) -> None:
        """Once per engine loop iteration: apply finished SSD writes, trim to the SSD budget and start queued backups
        (more of them while a request is prefilling)."""
        if self.ssd is not None:
            self.ssd.step(self, prefilling)

    # ---- locks ----
    def lock(self, node: Node | None, *, owner) -> TreeLock | None:
        """Keep node's snapshot and its whole path's KV for `owner` (a request, an SSD restore job, or "probe") until
        `unlock`. None on no node (nothing matched): nothing to keep."""
        if node is None or node is self.root:
            return None
        lock = TreeLock(node, owner)
        self.inc_lock_ref(node)
        self._locks[lock] = None
        return lock

    def unlock(self, lock: TreeLock | None) -> None:
        """End a lock; idempotent (None, or a lock already ended). Dead KV it kept goes now (`_prune`)."""
        if lock is None or lock.node is None:
            return
        node, lock.node = lock.node, None
        del self._locks[lock]
        self.dec_lock_ref(node)
        if self.flushing:
            self.flush()
        else:
            self._prune(node)

    @contextlib.contextmanager
    def _lock_node(self, node: Node | None):
        """A probe lock for the duration of a check (SGLang's PrefillAdder._lock_node): admission budgets with its
        candidate's own match counted as protected, not evictable."""
        lock = self.lock(node, owner="probe")
        try:
            yield lock
        finally:
            self.unlock(lock)

    def locks(self) -> list[TreeLock]:
        return list(self._locks)

    def inc_lock_ref(self, node: Node) -> None:
        if node.state_lock_ref == 0 and node.snapshot is not None:
            del self.evictable_snapshots[node]
            self._protected_snapshots += 1
        node.state_lock_ref += 1
        while node is not self.root:
            if node.lock_ref == 0:
                if node.in_l1:
                    self._evictable_tokens -= len(node.key)
                    self._protected_tokens += len(node.key)
                self.evictable_leaves.pop(node, None)
            node.lock_ref += 1
            node = node.parent

    def dec_lock_ref(self, node: Node) -> None:
        node.state_lock_ref -= 1
        assert node.state_lock_ref >= 0, "unbalanced lock"
        if node.state_lock_ref == 0 and node.snapshot is not None:
            self.evictable_snapshots[node] = None
            self._protected_snapshots -= 1
        while node is not self.root:
            node.lock_ref -= 1
            assert node.lock_ref >= 0, "unbalanced lock"
            if node.lock_ref == 0:
                if node.in_l1:
                    self._protected_tokens -= len(node.key)
                    self._evictable_tokens += len(node.key)
                self._update_leaf(node)
            node = node.parent

    # ---- the evictable totals (admission budgets with them, without evicting) ----
    def evictable_size(self) -> int:
        """L1 KV tokens (pages) no lock reaches: eviction can free all of them, leaf by leaf."""
        return self._evictable_tokens

    def protected_size(self) -> int:
        return self._protected_tokens

    def state_evictable_size(self) -> int:
        """L1 snapshots no lock sits on: slot pressure can free each one's slot."""
        return len(self.evictable_snapshots)

    def state_protected_size(self) -> int:
        return self._protected_snapshots

    def fits(self, pages: int, slots: int) -> bool:
        """Whether `pages` KV pages and `slots` state slots are free or can be freed by eviction, without evicting."""
        return (self.kv_pool.free_count + self._evictable_tokens >= pages
                and self.state_pool.free_count + len(self.evictable_snapshots) >= slots)

    def _update_leaf(self, node: Node) -> None:
        """Keep node's place in `evictable_leaves` exact after its lock_ref, residency or children changed."""
        if (node is not self.root and node.in_l1 and node.lock_ref == 0
                and not any(c.in_l1 for c in node.children.values())):
            self.evictable_leaves[node] = None
        else:
            self.evictable_leaves.pop(node, None)

    def _entered_l1(self, node: Node) -> None:
        """A new node, or an SSD-only one promoted, now holds pages."""
        n = len(node.key)
        self.cached_tokens += n
        if node.lock_ref:
            self._protected_tokens += n
        else:
            self._evictable_tokens += n
        self.evictable_leaves.pop(node.parent, None)
        self._update_leaf(node)

    # ---- snapshots ----
    def _set_snapshot(self, node: Node, snap: Snapshot) -> None:
        assert node.snapshot is None and node.in_l1
        node.snapshot = snap
        self._spare_snapshots += snap.spare
        if node.state_lock_ref:
            self._protected_snapshots += 1
        else:
            self.evictable_snapshots[node] = None

    def _take_snapshot(self, node: Node) -> Snapshot | None:
        """Detach node's snapshot (the caller frees or keeps its slot)."""
        snap, node.snapshot = node.snapshot, None
        if snap is not None:
            self._spare_snapshots -= snap.spare
            if node.state_lock_ref:
                self._protected_snapshots -= 1
            else:
                del self.evictable_snapshots[node]
        return snap

    def _free_slot(self, snap: Snapshot) -> None:
        """Give a snapshot's slot back (after the SSD tier's copies that may read it) and mark it dead."""
        if self.ssd is not None:
            self.ssd.fence()
        self.state_pool.free(snap.slot)
        snap.slot = -1
        snap.auxiliary = {}

    # ---- insertion ----
    def insert(self, ids: list[int], pages: torch.Tensor, snapshots: Iterable[Snapshot] = (), page_ids: list[int] | None = None) -> int:
        """Insert a full sequence. `pages` are its KV pages (the cache owns the ones it keeps); each of `snapshots`
        (owned by the cache from now on, with its slot and drafter metadata) forces a node boundary at its position.
        A snapshot at a node that has one already is a duplicate: the tree keeps its own, spare only if both are and
        as recent as the newer, and the duplicate's slot goes back.

        Returns `new_start`: tokens before it were already cached, so their
        pages in `pages[:new_start]` are duplicates the caller must free (unless
        they were the cache's own prefix pages to begin with)."""
        assert len(ids) == pages.shape[0]
        assert not self.flushing, "cannot insert while an active prefix is being flushed"
        snaps = {s.position: s for s in snapshots}
        assert all(0 < p <= len(ids) for p in snaps), "a snapshot inside the inserted sequence"
        node = self.root
        pos = 0
        boundaries = sorted(snaps)
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
                node.children[ids[pos]] = new
                self._entered_l1(new)
                if next_b in snaps:
                    self._set_snapshot(new, snaps[next_b])
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
                self._entered_l1(child)
                self.promotions += 1
                self.ssd_nodes -= 1
            pos += n
            node = child
            node.last_access = tick()
            if pos in snaps:
                snap = snaps[pos]
                if node.snapshot is None:
                    self._set_snapshot(node, snap)
                else:  # a duplicate: the tree keeps its own, which another request may be reading
                    old = node.snapshot
                    if old.spare and not snap.spare:
                        old.spare = False
                        self._spare_snapshots -= 1
                    old.last_access = max(old.last_access, snap.last_access)
                    if snap in self.branching:
                        self.branching.add(old)
                    self._free_slot(snap)
        if new_start is None:
            new_start = len(ids)
        return new_start

    def insert_ssd_path(self, ids: list[int], blocks: dict[int, SsdRef], tail: SsdRef) -> None:
        """Record a stored prefix at startup: add the path for ids (new nodes
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
        """Split `node` so that its first n tokens become a new parent node. The head carries the path's locks
        (lock_ref) and none of its own; the original node, which still ends where it did, keeps its snapshot and the
        locks on it. The evictable totals do not change: both parts share the node's residency and lock_ref."""
        head = Node(key=node.key[:n], pages=None if node.pages is None else node.pages[:n], parent=node.parent,
                    lock_ref=node.lock_ref, last_access=node.last_access)
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
    def evict(self, need_pages: int = 0, need_slots: int = 0, *, spare_only: bool = False) -> None:
        """Free cache entries until the pools can satisfy the request or nothing evictable remains. Short of KV pages:
        unlocked leaves, least recently used first. Short of state slots: unlocked snapshots by `get_priority`; with
        `spare_only` (a spare private snapshot's slot), only spare ones. Nothing a lock protects is ever freed, and the
        dead KV left behind goes too (`_prune`)."""
        while True:
            if self.kv_pool.free_count < need_pages and (leaf := self._lru_candidate()) is not None:
                self._evict_leaf(leaf, "kv")
            elif (self.state_pool.free_count < need_slots and (node := self._lru_candidate(snapshot_only=True)) is not None
                  and (node.snapshot.spare or not spare_only)):
                self._evict_snapshot(node, "slots")
            else:
                return

    def _lru_candidate(self, snapshot_only: bool = False, *, clean_first: bool = True) -> Node | None:
        """The next victim. With `snapshot_only`, the node whose unlocked snapshot has the lowest `get_priority`;
        otherwise the least recently used unlocked leaf (`evictable_leaves`), with the SSD tier and `clean_first` the
        least recently used one already stored (`SsdTier.clean`) first, so eviction rarely waits for a copy."""
        if snapshot_only:
            return min(self.evictable_snapshots, key=lambda n: get_priority(n.snapshot), default=None)
        best = clean = None
        tier = self.ssd if clean_first else None
        for n in self.evictable_leaves:
            if best is None or n.last_access < best.last_access:
                best = n
            if tier is not None and (clean is None or n.last_access < clean.last_access) and tier.clean(n):
                clean = n
        return clean or best

    def evict_snapshot(self, node: Node, reason: str) -> None:
        """Give up `node`'s unlocked snapshot now (a request's superseded spare chunk end), as slot pressure would."""
        self._evict_snapshot(node, reason)

    def _evict_snapshot(self, node: Node, reason: str) -> None:
        """Take `node`'s snapshot. An unlocked leaf goes whole, pages too (demoted, with the SSD tier, when its tail is
        stored or has room now): its KV resumes nothing without the snapshot. Any other node keeps its KV for the locks
        or the nodes below it (an L1 leaf above SSD-only nodes, too: KV pressure demotes it, copying the blocks they
        need, where slot pressure never waits for the writers)."""
        assert node.state_lock_ref == 0 and node.snapshot is not None, "only an unlocked snapshot is evicted"
        if node.lock_ref == 0 and not node.children:
            self._evict_leaf(node, reason)
            return
        if self.ssd is not None:
            self.ssd.dropping_snapshot(self, node)  # a non-spare snapshot's queued tail goes out if staging has room
        snap = self._take_snapshot(node)
        self.evicted[reason, snap.spare] += 1
        self._free_slot(snap)
        self.snapshot_evictions += 1
        self._prune(node)

    def _evict_leaf(self, leaf: Node, reason: str) -> None:
        assert leaf.lock_ref == 0, "eviction never frees locked KV"
        if leaf.snapshot is not None:
            self.evicted[reason, leaf.snapshot.spare] += 1
        parent = leaf.parent
        self._remove_leaf(leaf, wait=reason == "kv")  # waiting for staging room is for KV pressure, never a slot's
        self._prune(leaf if parent.children.get(leaf.key[0]) is leaf else parent)  # demoted: still in the tree

    @staticmethod
    def _dead(node: Node) -> bool:
        """KV that can no longer serve a resume: no lock reaches it, and it has no snapshot, nothing below it, and no
        tail stored or on its way to the SSD."""
        return (node.lock_ref == 0 and node.snapshot is None and not node.children
                and (node.tail is None or node.tail.nbytes < 0))

    def _prune(self, node: Node | None) -> None:
        """Free dead KV from `node` up its path: each dead node (`_dead`) goes, with any stored blocks of its own, and
        its parent is looked at next. Called after every removal, snapshot eviction, demotion, unlock and SSD tail
        loss."""
        while node is not None and node is not self.root and node.parent is not None and self._dead(node):
            parent = node.parent
            if parent.children.get(node.key[0]) is not node:
                return  # already gone
            if self.ssd is not None:
                self.ssd.pruning(node)  # its stored objects, if any, go with it
            del parent.children[node.key[0]]
            if node.in_l1:
                self._free_l1(node, evicted=False)
            else:
                self.ssd_nodes -= 1
            self.pruned += 1
            node = parent

    def _remove_leaf(self, leaf: Node, demote: bool = True, wait: bool = True) -> None:
        assert leaf.lock_ref == 0, "eviction never frees locked KV"
        if demote and self.ssd is not None and (self.ssd.evicting(self, leaf, wait=wait) or leaf.children):
            self._demote(leaf)  # its blocks and tail are on (or queued for) the SSD; SSD-only children need its key
            return
        del leaf.parent.children[leaf.key[0]]
        self._free_l1(leaf)  # gone: a write-through still queued for it finds nothing to copy

    def _demote(self, node: Node) -> None:
        """Free a node's L1 pages and slot; it stays in the tree, SSD-only."""
        self._free_l1(node)
        self.demotions += 1
        self.ssd_nodes += 1

    def _free_l1(self, node: Node, evicted: bool = True) -> None:
        """Give an unlocked node's pages and slot back; it is SSD-only from now on (or already out of the tree)."""
        assert node.lock_ref == 0, "eviction never frees locked KV"
        if self.ssd is not None:
            self.ssd.fence()
        if node.page_ids is not None:
            self.kv_pool.free_ids(node.page_ids)
        else:
            self.kv_pool.free(node.pages)
        snap = self._take_snapshot(node)
        if snap is not None:
            self._free_slot(snap)
        self.cached_tokens -= len(node.key)
        self._evictable_tokens -= len(node.key)
        node.pages = node.page_ids = None
        self.evictable_leaves.pop(node, None)
        if node.parent is not None:
            self._update_leaf(node.parent)
        self.evictions += int(evicted)

    def _walk(self, l1_only: bool = False):
        """Every node below the root, depth first. With `l1_only`, only the nodes in L1, in the same order: L1 is
        prefix-closed, so no L1 node sits below an SSD-only one."""
        stack = [c for c in self.root.children.values() if c.in_l1 or not l1_only]
        while stack:
            n = stack.pop()
            stack.extend(c for c in n.children.values() if c.in_l1 or not l1_only)
            yield n

    def ssd_candidate(self, unusable: bool = False, resident: bool = False, skip: Node | None = None) -> Node | None:
        """The least recently used node: an SSD-only leaf (with `unusable`, only
        one whose tail is missing or failed), or with `resident` an L1 node with
        a ready copy on the SSD, whose copies the budget can delete while the
        node keeps its pages. Never `skip` (the node a restore is reading)."""
        best = None
        for n in self._walk(l1_only=resident):
            if n is skip:
                continue
            if resident:
                if not ((n.tail is not None and n.tail.ready) or any(r.ready for r in n.blocks.values())):
                    continue
            elif n.in_l1 or n.children or n.lock_ref:  # locked: a restore is reading it
                continue
            if unusable and n.tail is not None and n.tail.nbytes >= 0:
                continue
            if best is None or n.last_access < best.last_access:
                best = n
        return best

    def ssd_prefixes(self) -> int:
        """Prefixes a request can resume from the SSD: nodes whose tail is written."""
        return sum(n.tail is not None and n.tail.ready for n in self._walk())

    def remove_ssd_leaf(self, node: Node) -> None:
        """The SSD budget drops a stored leaf; the dead KV it leaves above goes too (`_prune`)."""
        assert not node.in_l1 and not node.children and node.lock_ref == 0
        parent = node.parent
        del parent.children[node.key[0]]
        self.ssd_nodes -= 1
        self.ssd_evictions += 1
        self._prune(parent)

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
        # Hide the cache immediately, but keep locked pages and snapshots until their locks end. Admission bypasses the
        # cache while that drain is pending, and no snapshot is placed or inserted.
        if self.ssd is not None and not self.flushing:  # once per flush, not again each time a lock ends
            self.ssd.abort(self)  # a restore still reading: its pages and slot go back, and its lock ends
            self._drop_ssd_only()
            self.ssd.flush()  # a new epoch, so every ref left on the L1 nodes is stale
        while (leaf := self._lru_candidate(clean_first=False)) is not None:
            if leaf.snapshot is not None:
                self.evicted["flush", leaf.snapshot.spare] += 1
            self._remove_leaf(leaf, demote=False)
        self.flushing = bool(self.root.children)

    # ---- invariants ----
    def check(self, owners: Iterable = (), owned_slots: Iterable[tuple[str, int]] = (),
              pending_prune: Iterable[Node] = ()) -> None:
        """Assert the cache's invariants (the unit tests run it after every scheduler step):
        - lock_ref(n) == state_lock_ref(n) + sum(lock_ref(child)) everywhere, L1 and SSD-only;
        - the state_lock_refs add up to the live locks, each on a node in the tree, and every lock's owner is live
          (`owners`: the admitted requests and the SSD restores; no probe outlives its check);
        - a locked node has a snapshot, unless every lock on it is an SSD restore still reading its tail;
        - the evictable totals, `evictable_leaves` and `evictable_snapshots` equal a fresh recompute;
        - every committed state slot has exactly one owner: a node's snapshot, one of `owned_slots` ((owner, slot):
          requests' own slots, a restore's slot in flight), or the free list;
        - no dead KV is left (`_dead`), but for nodes the SSD tier will prune at its next step (`pending_prune`).
        Eviction itself asserts that it never frees KV with lock_ref > 0 or a snapshot with state_lock_ref > 0."""
        live = {id(o) for o in owners}
        locked = Counter()
        kinds_by_node: dict[Node, set] = {}
        for lock in self._locks:
            assert lock.node is not None, "an ended lock is registered"
            kind = owner_kind(lock.owner)
            assert kind != "probe", "a probe lock outlived its check"
            assert id(lock.owner) in live, ("a lock whose owner is gone", lock)
            locked[lock.node] += 1
            kinds_by_node.setdefault(lock.node, set()).add(kind)
        seen, slots, pending = set(), [], set(pending_prune)
        evictable = protected = protected_snaps = spare = 0
        leaves, unlocked_snaps = set(), set()
        for n in self._walk():
            seen.add(n)
            assert n.lock_ref == n.state_lock_ref + sum(c.lock_ref for c in n.children.values()), \
                ("lock_ref is not the locks at or below the node", n.depth, n.lock_ref, n.state_lock_ref)
            assert n.state_lock_ref == locked.get(n, 0), ("state_lock_ref is not the locks on the node", n.depth)
            if n.state_lock_ref:
                assert n.snapshot is not None or kinds_by_node[n] == {"restore"}, ("a locked node without a snapshot", n.depth)
            if n.in_l1:
                if n.lock_ref:
                    protected += len(n.key)
                else:
                    evictable += len(n.key)
                    if not any(c.in_l1 for c in n.children.values()):
                        leaves.add(n)
            if n.snapshot is not None:
                assert n.in_l1, "an SSD-only node with a snapshot"
                assert n.snapshot.position == n.depth, ("a snapshot off its node's end", n.snapshot.position, n.depth)
                assert n.snapshot.slot > 0, "a dead snapshot in the tree"
                slots.append(("node", n.snapshot.slot))
                spare += n.snapshot.spare
                if n.state_lock_ref:
                    protected_snaps += 1
                else:
                    unlocked_snaps.add(n)
            if not self.flushing:
                assert n in pending or not self._dead(n), ("dead KV left in the tree", n.depth)
        assert all(n in seen for n in locked), "a lock on a node that left the tree"
        assert (evictable, protected) == (self._evictable_tokens, self._protected_tokens), \
            ("KV totals out of date", evictable, protected, self._evictable_tokens, self._protected_tokens)
        assert evictable + protected == self.cached_tokens, ("cached_tokens out of date", self.cached_tokens)
        assert leaves == set(self.evictable_leaves), "evictable_leaves out of date"
        assert unlocked_snaps == set(self.evictable_snapshots), "evictable_snapshots out of date"
        assert protected_snaps == self._protected_snapshots, "state_protected_size out of date"
        assert spare == self._spare_snapshots, "the spare count out of date"
        assert protected_snaps <= len(self._locks), "more locked snapshots than locks"
        owners_of = Counter(s for _, s in slots) + Counter(s for _, s in owned_slots)
        free = Counter(self.state_pool._free)
        both = owners_of + free
        assert all(c == 1 for c in both.values()), ("a state slot with two owners", [s for s, c in both.items() if c > 1])
        assert set(both) == set(range(1, self.state_pool.num_slots)), \
            ("a state slot with no owner", sorted(set(range(1, self.state_pool.num_slots)) - set(both)))

    def stats(self) -> dict:
        by_owner = Counter(owner_kind(lock.owner) for lock in self._locks)
        snapshots = len(self.evictable_snapshots) + self._protected_snapshots
        evicted: dict[str, dict[str, int]] = {}
        for (reason, was_spare), count in sorted(self.evicted.items()):
            evicted.setdefault(reason, {})["spare" if was_spare else "not_spare"] = count
        out = {"hits": self.hits, "misses": self.misses, "evictions": self.evictions,
               "snapshot_evictions": self.snapshot_evictions, "cached_tokens": self.cached_tokens, "flush_pending": self.flushing,
               "locks": {"total": len(self._locks), **dict(sorted(by_owner.items()))},
               "snapshots": {"spare": self._spare_snapshots, "not_spare": snapshots - self._spare_snapshots},
               "evicted": evicted, "pruned": self.pruned, "full_reprefills": self.full_reprefills,
               "branching_resumes": self.branching_resumes,
               "evictable": {"kv_tokens": self._evictable_tokens, "kv_tokens_protected": self._protected_tokens,
                             "snapshots": len(self.evictable_snapshots), "snapshots_protected": self._protected_snapshots}}
        if self.ssd is not None:
            out["ssd"] = {"nodes": self.ssd_nodes, "demotions": self.demotions,
                          "promotions": self.promotions, "evictions": self.ssd_evictions, **self.ssd.stats()}
        return out
