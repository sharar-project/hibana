"""The ownership boundary for request KV, recurrent state and the request's cache lock.

Kernels may write through supplied tensor views. Only StateManager transfers
ownership, advances the processed prefix, or returns request resources.

A request's life in the prefix cache (docs/cache_design.md, "Protection and eviction"):
- `can_admit` decides without changing anything: the candidate's uncached suffix, its decode reserve and the running
  requests' outstanding reserve against the free and evictable KV, and a slot of its own against the free and evictable
  slots, with its own match held by a probe lock so it is not counted as evictable (FIFO, one request at a time);
- `admit` matches the prompt and locks the matched node (`req.lock`, a `radix.TreeLock`) before it evicts anything for
  the request's own slot (`req.slot`) and pages; the matched snapshot is copied into that slot, and the drafter reads its
  metadata at the first prefill chunk (`req.match`);
- each prefill forward writes the snapshots `Executor._checkpoint_positions` places into the request's private
  snapshots (`req.private`, allocated by `prefill_checkpoints`). Once the drafter ingested the chunk,
  `checkpoint(req, up_to)` inserts them into the tree and moves the lock to the deepest (lock the new node, then unlock
  the old one), so `req.private` is empty between scheduler steps;
- `release` inserts the whole sequence with the request's own slot as the reply end, then unlocks; `retract` unlocks
  without inserting, so a retry resumes at the last checkpoint.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import torch

from sharar.cache.snapshot import GRID, Snapshot

if TYPE_CHECKING:
    from sharar.engine.request import Request
    from sharar.cache.kv_pool import KvPool
    from sharar.cache.state_pool import StatePool
    from sharar.cache.radix import Match, RadixCache, TreeLock


@dataclass
class Reservation:
    pages: torch.Tensor
    ids: list[int]
    full_pages: torch.Tensor | None = None


@dataclass(eq=False, kw_only=True)
class RequestState:
    slot: int = -1  # the request's own state slot: the state after committed_len tokens; the reply end at release
    pages: torch.Tensor | None = None
    prefix_len: int = 0  # the matched prefix, the cache's; the rest is request-owned until checkpointed
    committed_len: int = 0  # prefix represented by both KV and recurrent state
    match: Match | None = None  # what admission resumed from, for the drafter's first chunk (cleared by `checkpoint`)
    branching_seqlen: int = 0  # Match.branching_seqlen while prefilling: where its branching point goes (0: none)
    lock: TreeLock | None = None  # the request's one lock: its matched node, then its deepest checkpoint (None: nothing)
    private: dict[int, Snapshot] = field(default_factory=dict)  # this forward's snapshots by position (empty between steps)
    reservation: Reservation | None = None
    page_ids: list[int] = field(default_factory=list)  # host ids of the pages after prefix_len (-1: the cache owns it)
    cacheable: bool = True
    cache_enabled: bool = False
    released: bool = False


class StateManager:
    def __init__(self, kv_pool: KvPool, state_pool: StatePool, cache: RadixCache | None = None):
        self.kv = kv_pool
        self.recurrent = state_pool
        self.cache = cache
        self.requests: dict[int, Request] = {}  # admitted requests still owning resources, by id(): for `check`

    def can_admit(self, ids: list[int], reserve: int = 0) -> bool:
        """Whether the request with ids can be admitted now. Pure: nothing is evicted, restored or allocated. Its
        uncached suffix and `reserve` decode pages (its own and, from the scheduler, the running requests' outstanding
        reserve) must fit in the free and evictable KV, and one slot of its own in the free and evictable slots, while a
        probe lock keeps its own matched path and snapshot out of the evictable totals (SGLang's _lock_node): `admit`
        locks them before it evicts. The decision holds for one request at a time, checked and then admitted."""
        if self.cache is None:
            return self.kv.free_count >= len(ids) + reserve and self.recurrent.free_count >= 1
        match = self.cache.match(ids[:-1], touch=False)  # the last token must produce logits
        with self.cache._lock_node(match.node):
            return self.cache.fits(len(ids) - match.length + reserve, 1)

    def prefetch(self, ids: list[int], reserve: int = 0) -> bool:
        """The queue head's SSD restore, before `can_admit` (`SsdTier.restore_async`): False while a stored prefix is
        still coming back for it (the engine serves everyone else meanwhile), True when the head can be checked now. A
        restore starts only when its pages and slot, the rest of the prompt and `reserve` fit by the same pure count."""
        if self.cache is None:
            return True
        return self.cache.restore_async(ids[:-1], reserve + 1)

    def forget(self, ids: list[int]) -> None:
        """A queued request is gone: a restore locking its path unlocks."""
        if self.cache is not None:
            self.cache.release(ids[:-1])

    def ensure_pages(self, count: int) -> None:
        if self.kv.free_count < count and self.cache is not None:
            self.cache.evict(need_pages=count)
        if self.kv.free_count < count:
            raise MemoryError(f"KV pool exhausted: need {count}, free {self.kv.free_count}")

    def admit(self, req: Request, ids: list[int]) -> None:
        """Match the cached prefix and lock it for the request before anything is evicted for it: the lock keeps the
        path's KV and the matched snapshot, which admission copies now and the drafter reads at the first prefill chunk,
        until `checkpoint` moves it. A landed SSD restore that locked the node for this request unlocks after the
        request's lock is taken. A failure after the match unlocks."""
        if req.slot >= 0 or req.pages is not None or req.reservation is not None or req.released:
            raise ValueError("request already owns resources or has been released")
        cache = self.cache
        match, lock, slot, own = None, None, -1, None
        try:
            if cache is not None:
                cache.restore(ids[:-1])  # a direct caller's restore; nothing once the scheduler's prefetch landed one
                match = cache.match(ids[:-1])  # last token must produce logits
                lock = cache.lock(match.node, owner=req)
                cache.release(ids[:-1])  # a landed restore that locked this path for the request: the request's lock now
                cache.evict(need_pages=len(ids) - match.length, need_slots=1)
            prefix = match.length if match is not None else 0
            slot = self.recurrent.alloc()
            if prefix:
                self.recurrent.copy_slot(match.snapshot.slot, slot)
            own, own_ids = self.kv.alloc_with_ids(len(ids) - prefix)
            pages = torch.cat([match.pages, own]) if prefix else own
        except Exception:
            if own is not None:
                self.kv.free_ids(own_ids)
            if slot >= 0:
                self.recurrent.free(slot)
            if cache is not None:
                cache.unlock(lock)
            raise
        if cache is not None:
            if not prefix and match.raw_length >= GRID:
                cache.full_reprefills += 1  # the tree held its KV, but no snapshot it could resume from
            if prefix and match.snapshot in cache.branching:
                cache.branching_resumes += 1
        req.match = match if prefix else None
        req.branching_seqlen = match.branching_seqlen if match is not None else 0
        req.slot, req.pages, req.lock = slot, pages, lock
        req.page_ids = list(own_ids)
        req.private = {}
        self.requests[id(req)] = req
        req.cache_enabled = cache is not None
        req.prefix_len = req.committed_len = req.prefill_pos = prefix

    def _private_snapshot(self, req: Request, position: int, spare: bool) -> int | None:
        """A private snapshot at `position` for the coming forward: its slot, or None when there is none to take. A
        spare one takes only a free slot or a spare snapshot's; any other one the lowest-priority unlocked snapshot's
        (`radix.get_priority`). A locked snapshot is never taken."""
        if position in req.private:
            return req.private[position].slot
        if self.recurrent.free_count < 1:
            self.cache.evict(need_slots=1, spare_only=spare)
        if self.recurrent.free_count < 1:
            return None
        slot = self.recurrent.alloc()
        req.private[position] = Snapshot(slot, position, spare=spare)
        return slot

    def prefill_checkpoints(self, boundaries: list[tuple[Request, list[tuple[int, bool]]]]) -> list[tuple[int, int, int]]:
        """Allocate the private snapshots each row's forward places ((position, spare), from
        `Executor._checkpoint_positions`, in each row's own order: the message boundary and the last chunk end, then
        the branching point, then a spare chunk end), round-robin over the rows, so one long prompt cannot take every
        row's slots. Returns (row, position, slot) for each one allocated; a slot may be missing (none to take)."""
        if self.cache is None or self.cache.flushing:
            return []
        planned = []
        for rank in range(max((len(p) for _, p in boundaries), default=0)):
            for row, (req, positions) in enumerate(boundaries):
                if rank < len(positions):
                    pos, spare = positions[rank]
                    slot = self._private_snapshot(req, pos, spare)
                    if slot is not None:
                        planned.append((row, pos, slot))
        return planned

    def discard_private(self, req: Request) -> None:
        """A forward failed: the private snapshots it would have written are never written, on every row of it."""
        for snap in req.private.values():
            self.recurrent.free(snap.slot)
        req.private = {}

    def commit_prefill(self, req: Request, end: int) -> None:
        if not req.committed_len <= end <= len(req.pages):
            raise ValueError("prefill commit outside owned pages")
        req.prefill_pos = req.committed_len = end

    def checkpoint(self, req: Request, up_to: int) -> None:
        """Insert the request's prefix up to its deepest written private snapshot at or before `up_to`, with every
        written one, and move its lock there: after each prefill chunk the drafter has ingested
        (`Executor._finish_*prefill`, both paths), and at release (the whole sequence, with the reply end). A private
        snapshot is written once the forward that fills it has committed (position <= committed_len); any other one is
        freed, so nothing private outlives the step. The request's own pages become the cache's where the tree had
        none; where the tree had the span already (another request's copy), the request reads the tree's pages from now
        on and frees its own. With an SSD tier, the prefix's whole blocks and the non-spare snapshots' tails are queued
        for a write-through. Then the lock moves, with nothing allocated or evicted in between: lock the new node, unlock
        the old one. Last, the request's previous spare chunk end goes (one spare chunk end per request in the tree).
        Nothing is inserted while the cache flushes, or for a request that may not be cached."""
        req.match = None  # the drafter read it at the first chunk
        private, req.private = req.private, {}
        limit = min(up_to, req.committed_len)
        written = sorted((s for p, s in private.items() if p <= limit), key=lambda s: s.position)
        for snap in private.values():
            if snap.position > limit:
                self.recurrent.free(snap.slot)
        cache = self.cache
        if cache is None or cache.flushing or not req.cacheable or not req.cache_enabled or not written:
            for snap in written:
                self.recurrent.free(snap.slot)
            return
        p = written[-1].position
        seq = (req.prompt_ids + req.output_ids)[:p]
        bp = req.branching_seqlen
        for snap in written:  # the one serving as its branching point: resumes from it are counted (`cache_stats`)
            if bp and snap.position <= bp < snap.position + GRID:
                cache.branching.add(snap)
        owned = [-1] * req.prefix_len + req.page_ids
        new_start = min(cache.insert(seq, req.pages[:p], written, page_ids=owned[:p]), p)
        node = cache.node_at(seq)
        assert node is not None and node.snapshot is not None, "a checkpoint ends at a snapshot"
        lo = req.prefix_len
        dup = [q for q in req.page_ids[:new_start - lo] if q >= 0]
        if dup:  # the tree had part of [lo, new_start) already: read its pages (locked below) and free the duplicates
            req.pages = torch.cat([cache.path_pages(node)[:new_start], req.pages[new_start:]])
            self.kv.free_ids(dup)
            req.page_ids[:new_start - lo] = [-1] * (new_start - lo)
        req.page_ids[new_start - lo:p - lo] = [-1] * (p - new_start)  # the cache's from now on
        cache.backup(seq, tuple(s.position for s in written if not s.spare))
        old = req.lock
        if old is not None and old.node is node:
            return
        superseded = old.node if old is not None else None
        assert superseded is None or _below(node, superseded), "a lock moves down its request's own path"
        req.lock = cache.lock(node, owner=req)
        cache.unlock(old)
        if (superseded is not None and superseded.snapshot is not None and superseded.snapshot.spare
                and superseded.state_lock_ref == 0):
            cache.evict_snapshot(superseded, "superseded")

    def reserve(self, requests: list[Request], width: int) -> torch.Tensor:
        if width < 1 or any(r.reservation is not None or r.finished for r in requests):
            raise ValueError("invalid or overlapping reservation")
        self.ensure_pages(len(requests) * width)
        pages, ids = self.kv.alloc_with_ids(len(requests) * width)
        for i, req in enumerate(requests):
            lo, hi = i * width, (i + 1) * width
            req.reservation = Reservation(pages[lo:hi], ids[lo:hi])
        return pages

    def step_pages(self, req: Request) -> torch.Tensor:
        """Borrow the committed prefix plus the pending reservation for a forward."""
        if req.reservation is None:
            raise ValueError("forward requires a reservation")
        if req.reservation.full_pages is None:
            req.reservation.full_pages = torch.cat([req.pages, req.reservation.pages])
        return req.reservation.full_pages

    def commit(self, req: Request, count: int, *, ssm=None, conv=None, aux: dict | None = None) -> None:
        """Keep count verify inputs, or publish a successful in-place decode.

        ssm/conv are borrowed snapshots after input count-1; together they
        describe exactly the prefix that owns the retained pages.
        """
        pending = req.reservation
        if pending is None or not 1 <= count <= len(pending.ids):
            raise ValueError("commit outside reservation")
        if (ssm is None) != (conv is None):
            raise ValueError("recurrent commit needs both SSM and conv state")
        pages = pending.full_pages if count == len(pending.ids) and pending.full_pages is not None else torch.cat([req.pages, pending.pages[:count]])
        if ssm is not None:
            self.recurrent.ssm_all[:, req.slot].copy_(ssm)
            self.recurrent.conv_all[:, req.slot].copy_(conv)
            for name, value in (aux or {}).items():
                self.recurrent.aux[name][req.slot].copy_(value)
        self._publish(req, count, pages)

    def _validate_reconstruction(self, requests, replay, conv_windows, verified_steps: int, counts=None) -> list[int]:
        """The batch-wide checks shared by the host and device commit paths; returns the slots."""
        batch = len(requests)
        if (not 1 <= batch <= replay.batch or (counts is not None and len(counts) != batch)
                or type(verified_steps) is not int or not 1 <= verified_steps <= replay.steps):
            raise ValueError("invalid reconstruction batch")
        slots = [r.slot for r in requests]
        if len(set(slots)) != batch or any(not 0 < slot < self.recurrent.num_slots for slot in slots):
            raise ValueError("reconstruction requires unique private slots")
        conv = self.recurrent.conv_all
        if (conv_windows.ndim != conv.ndim + 1 or conv_windows.shape[0] != conv.shape[0]
                or conv_windows.shape[1] < batch or conv_windows.shape[3:] != conv.shape[2:]
                or conv_windows.dtype != conv.dtype or conv_windows.device != conv.device):
            raise ValueError("invalid reconstruction convolution windows")
        for row, req in enumerate(requests):
            pending = req.reservation
            if (req.finished or req.released or req.pages is None or len(req.pages) != req.committed_len
                    or pending is None or len(pending.ids) < verified_steps or conv_windows.shape[2] < verified_steps):
                raise ValueError("reconstruction commit outside reservation")
            if counts is not None:
                count = counts[row]
                if type(count) is not int or not 1 <= count <= min(len(pending.ids), verified_steps, conv_windows.shape[2]):
                    raise ValueError("reconstruction commit outside reservation")
        return slots

    def commit_reconstructed(self, requests, counts, replay, conv_windows, *, verified_steps: int, aux_windows: dict | None = None) -> None:
        """Validate the whole batch before allowing reconstruction into private slots.

        Prepare retained page tensors before mutation. A failed in-place commit
        poisons every affected request; rollback can free pages but cannot undo
        recurrent writes. Publication and proposer feedback follow all writes.
        """
        slots = self._validate_reconstruction(requests, replay, conv_windows, verified_steps, counts)
        conv = self.recurrent.conv_all
        pages = self._retained_pages(requests, counts)
        # Two aligned rows: Triton specializes kernels on pointer alignment, and a [2, B] row view
        # of counts sits at 4*B bytes, which recompiled the reconstruction per batch size.
        metadata = torch.zeros((2, (len(slots) + 3) // 4 * 4), dtype=torch.int32, pin_memory=conv.device.type == "cuda")
        metadata[0, :len(slots)] = torch.tensor(slots, dtype=torch.int32)
        metadata[1, :len(counts)] = torch.tensor(counts, dtype=torch.int32)
        metadata = metadata.to(conv.device, non_blocking=True)
        try:
            replay.run(self.recurrent, metadata[0, :len(slots)], metadata[1, :len(counts)])
            for row, (req, count) in enumerate(zip(requests, counts)):
                conv[:, req.slot].copy_(conv_windows[:, row, count - 1])
                for name, win in (aux_windows or {}).items():
                    self.recurrent.aux[name][req.slot].copy_(win[row, count - 1])
            for req, count, retained in zip(requests, counts, pages):
                self._publish(req, count, retained)
        except BaseException as exc:
            self.poison(requests, exc)

    def authorize_reconstruction(self, requests, replay, conv_windows, *, verified_steps: int) -> list[int]:
        """Validate a batch whose counts the device will decide (`engine/accept_graphs.py`).

        Every count the device can produce lies in [0, verified_steps]; the
        reservation covers verified_steps pages and the windows verified_steps
        rows, so the only host checks left are ownership and uniqueness, made
        before any in-place write. Returns the private slots in batch order.
        """
        return self._validate_reconstruction(requests, replay, conv_windows, verified_steps)

    def publish_reconstructed(self, requests, counts) -> None:
        """Publish counts the device decided after `authorize_reconstruction` and the writes landed."""
        for req, count in zip(requests, counts):
            pending = req.reservation
            if pending is None or type(count) is not int or not 1 <= count <= len(pending.ids):
                raise ValueError("reconstruction commit outside reservation")
        for req, count, retained in zip(requests, counts, self._retained_pages(requests, counts)):
            self._publish(req, count, retained)

    def poison(self, requests, exc: BaseException):
        """A failure after in-place recurrent writes: nothing in the batch may be cached or retried."""
        for req in requests:
            req.cacheable = False
        if isinstance(exc, MemoryError):
            # The scheduler retries pool-pressure MemoryError. A partially
            # written recurrent slot requires failure cleanup instead.
            raise RuntimeError("reconstruction failed after state mutation") from exc
        raise exc

    def _retained_pages(self, requests, counts) -> list[torch.Tensor]:
        return [r.reservation.full_pages if count == len(r.reservation.ids) and r.reservation.full_pages is not None
                else torch.cat([r.pages, r.reservation.pages[:count]]) for r, count in zip(requests, counts)]


    def _publish(self, req, count, pages) -> None:
        self.kv.free_ids(req.reservation.ids[count:])
        req.page_ids += req.reservation.ids[:count]
        req.pages = pages
        req.committed_len += count
        req.reservation = None

    def rollback(self, req: Request) -> None:
        """Discard pages, not in-place kernel writes; retract before retrying.

        Verification preserves the committed slot until acceptance. A failed
        decode or direct reconstruction may have changed it, so an abandoned
        transaction must never be cached.
        """
        if req.reservation is not None:
            req.cacheable = False
            self.kv.free_ids(req.reservation.ids)
            req.reservation = None

    def retract(self, req: Request) -> None:
        """Give everything back without inserting: the lock ends, and the request's own pages and slot are freed. What
        it checkpointed stays in the tree, so its retry resumes at its last checkpoint."""
        self.rollback(req)
        self._drop(req)
        req.prefix_len = req.committed_len = req.prefill_pos = 0

    def release(self, req: Request, attach: Callable[[Request, Snapshot], None] | None = None) -> None:
        """Finish a request: its whole sequence (prompt and reply, but its last token) goes into the tree with the
        request's own slot as the reply end (a spare snapshot; `attach`, the drafter's `prepare_release`, puts its
        metadata in it first), the reply end's SSD tail is queued for a write-through, then the lock ends."""
        if req.released:
            return
        if req.reservation is not None:
            # In-place decode could have partially advanced a slot on failure.
            req.cacheable = False
            self.rollback(req)
        n = len(req.prompt_ids) + len(req.output_ids[:-1])  # the whole sequence but its last token
        if (self.cache is not None and not self.cache.flushing and req.cacheable and req.cache_enabled
                and req.slot >= 0 and req.pages is not None and req.prefill_done
                and req.committed_len == n and n and len(req.pages) >= n):
            end = Snapshot(req.slot, n, spare=True)  # the reply end
            if attach is not None:
                attach(req, end)
            req.private[n], req.slot = end, -1
            self.checkpoint(req, n)
            self.cache.backup((req.prompt_ids + req.output_ids[:-1]), (n,))
        self._drop(req)
        req.released = True

    def _drop(self, req: Request) -> None:
        """The lock ends (after `checkpoint`, which needs the locked path); what the request still owns goes back."""
        self.requests.pop(id(req), None)
        if self.cache is not None:
            self.cache.unlock(req.lock)  # and prunes what it left dead
        req.lock = req.match = None
        if req.pages is not None:
            self.kv.free_ids([p for p in req.page_ids if p >= 0])  # host ids: no device read-back
        if req.slot >= 0:
            self.recurrent.free(req.slot)
        self.discard_private(req)
        req.pages, req.slot = None, -1
        req.page_ids = []

    def flush(self) -> None:
        if self.cache is not None:
            self.cache.flush()

    def check(self) -> None:
        """The cache's invariants (`RadixCache.check`) with what the running requests own, between scheduler steps:
        - no request owns a private snapshot (they live only inside a prefill step);
        - each request has at most its one lock, its own, on its own sequence's path, no deeper than its progress;
        - a request reads the tree's pages exactly up to its lock (`page_ids` -1 there) and owns every page after;
        - every KV page has exactly one owner: a tree node, a request, a reservation, an SSD restore, or the free list."""
        cache = self.cache
        if cache is None:
            return
        ssd = cache.ssd
        owned_slots, pages = [], Counter()
        mine = Counter(id(lock.owner) for lock in cache.locks() if id(lock.owner) in self.requests)
        for key, req in self.requests.items():
            assert not req.private, ("a private snapshot outlived its forward", sorted(req.private))
            if req.slot >= 0:
                owned_slots.append(("request", req.slot))
            assert mine.get(key, 0) == (req.lock is not None), ("a request with other than its one lock", mine.get(key, 0))
            depth = 0
            if req.lock is not None:
                node = req.lock.node
                assert req.lock.owner is req and node.snapshot is not None, "a request's lock sits at a snapshot"
                depth = node.depth
                assert req.prefix_len <= depth <= max(req.committed_len, req.prefix_len), "a lock past the request's progress"
                seq = (req.prompt_ids + req.output_ids)[:depth]
                assert cache.node_at(seq) is node, "a lock off its request's own path"
                assert torch.equal(req.pages[:depth].cpu(), cache.path_pages(node).cpu()), "a request reads other pages than its lock's"
            if req.pages is not None:
                shared = depth - req.prefix_len if req.lock is not None else 0
                assert req.page_ids[:shared] == [-1] * shared and -1 not in req.page_ids[shared:], \
                    ("a request owns pages up to its lock, or not past it", shared)
                pages.update(p for p in req.page_ids if p >= 0)
            if req.reservation is not None:
                pages.update(req.reservation.ids)
        for n in cache._walk(l1_only=True):
            pages.update(n.page_ids if n.page_ids is not None else n.pages.tolist())
        if ssd is not None:
            pages.update(ssd.owned_pages())
        both = pages + Counter(self.kv._free)
        assert all(c == 1 for c in both.values()), ("a KV page with two owners", sorted(p for p, c in both.items() if c > 1)[:8])
        assert set(both) == set(range(1, self.kv.num_pages)), \
            ("a KV page with no owner", sorted(set(range(1, self.kv.num_pages)) - set(both))[:8])
        cache.check([*self.requests.values(), *(ssd.live_jobs() if ssd is not None else [])],
                    owned_slots + (ssd.owned_slots() if ssd is not None else []),
                    ssd.pending_prune() if ssd is not None else ())


def _below(node, ancestor) -> bool:
    """Whether `ancestor` is on node's path (node itself excluded)."""
    n = node.parent
    while n is not None and n is not ancestor:
        n = n.parent
    return n is ancestor
