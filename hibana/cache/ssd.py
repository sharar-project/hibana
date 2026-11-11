"""SSD tier: decides what goes to the NVMe and brings it back.

The tiered radix tree (`radix.py`) is the only record of what is on disk;
`ssd_store.py` only stores objects. This module moves data between them:

- Backup. A request's prompt enters the tree at each of its prefill
  checkpoints (`StateManager.checkpoint`: chunk ends, its message boundary,
  its branching point, its last chunk end), and its whole sequence once it
  finishes. Each time, for at least `min_tokens`, `backup()` queues each of
  its full KV blocks (`store.block` tokens) not yet stored, so a long prompt's
  blocks go out while it still prefills and decodes, and a write-through tail
  at each non-spare snapshot it placed (message boundary, last chunk end,
  branching point) and, at the finish, at its reply end: a snapshot the GPU
  cache later drops for want of state slots stays restorable. A spare chunk
  end gets no tail. `step()` (every engine loop iteration) gathers queued tails,
  then queued blocks, from the pool pages and slots into pinned host memory,
  up to `step_bytes` while no request is prefilling and as much as staging
  takes while one is, blocks leaving room for one tail within `staging_bytes`
  in flight, and hands them to the store's writer threads.
  An object larger than that limit goes out alone. Gathers that eviction under
  KV pressure, a restore or shutdown need at once wait for room; slot pressure
  never waits (`dropping_snapshot`: a non-spare snapshot's queued tail goes out
  first only if staging has room now). `step()` skips a queued tail whose
  snapshot the GPU cache dropped first.
- Eviction. Before the tree frees an L1 leaf's pages it calls `evicting()`,
  which gathers every queued block touching the leaf and, for a leaf of at
  least `min_tokens` with a snapshot, writes its tail (its last partial block,
  the GDN snapshot at its end, the pool's per-slot aux and the drafter
  payload) unless a write-through already did, under KV pressure waiting for
  staging room for the tail if it must. The tree then keeps the leaf SSD-only
  instead of deleting it.
- Restore. The queue head's prefetch (`restore_async()`, from
  `StateManager.prefetch`, before admission's pure check) finds the deepest
  ready tail past the L1 prefix (first finishing its writes if it was evicted
  moments ago). It starts only if the restored KV, the rest of the prompt and
  the decode reserve, and two slots, are free or evictable by admission's own
  count (`RadixCache.fits`), with the request's L1 match and the stored node
  locked first. It takes the pages and a slot, locks the stored node
  (`RadixCache.lock`, the restore job its owner), and streams the blocks past
  the KV still in L1, then the tail, through a host buffer the GPU reads in
  place, in two halves of `chunk_bytes`: a worker thread reads one chunk with
  O_DIRECT (a small pool in parallel) into one half while the engine keeps
  serving everyone else and scatters the previous chunk from the other
  straight into the fresh pages and slot (one kernel launch, `_RowCopy`).
  Once the reads landed on every rank, `step()` imports the tail's state and
  inserts the path back into L1 (a non-spare snapshot), still locked by the
  job until its request takes its own lock (the request first, then the job
  unlocks), HOLD_STEPS pass, or the request is cancelled (`release`). A
  landed restore counts as locked in admission's totals and in the pool plan's
  slot floor (`RESTORE_SLOTS`): no other restore starts while one waits for
  its request. `restore()` does the same with the engine waiting (admission's
  own call, for callers without the prefetch). When the GPU
  cache dropped only a snapshot and kept its pages, the restore reads only the
  tail: its state goes into a slot and the pages stay where they are. The
  chunk size, not the restore, bounds the buffer, so the pool plan reserves a
  constant (`memory_bound`).
- Budget. Completed writes count against `budget_bytes`; over it, `step()`
  deletes the least recently used SSD-only leaves, then the stored copies of
  GPU-resident nodes. A node left with nothing to resume from (a lost or failed
  tail, deleted copies) is pruned as dead KV (`RadixCache._prune`).

Only the engine thread calls this module or touches the tree; the store's
writers report back through completions that `step()` drains.

Tensor parallel: each rank stores its own heads of every stored prefix on its own
Spark, and every rank's tree must take the same steps (the schedulers do).
The ranks settle what one rank cannot decide alone with a minimum over ranks:
a write counts as complete once it completed on every rank (and as failed if
it failed on any), a restore goes ahead only if every rank read every object
intact, and a restart lists only the prefixes every rank stored. On one rank,
each agreement returns that rank's own answer.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import time
from array import array
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import queue
import threading

import torch

from hibana.cache.radix import Node, RadixCache, SsdHit, SsdRef
from hibana.cache.snapshot import Snapshot
from hibana.cache.ssd_store import SEG_ALIGN, Header, SsdStore, _pad, _seg_pad, block_hash, tail_hash
from hibana.distributed import all_reduce_min, common_values

MIB = 2**20
GIB = 2**30
STAGING_BYTES = 512 * MIB  # host copies of backups in flight to the writers
CHUNK_BYTES = 256 * MIB  # each half of the restore staging buffer
HOLD_STEPS = 1000  # a landed asynchronous restore locks its node this many engine steps at most for its request
# State slots restores lock beyond the running requests' own (the pool plan's slot floor, `Ledger.plan`): one restore
# reads at a time, and none starts while a landed one waits for its request, which then locks that slot itself.
RESTORE_SLOTS = 1


@dataclass(eq=False)
class _Restore:
    """One restore from `_begin` to `_finish`: what it reads, and the pages and slot it fills."""
    ids: list[int]
    matched: int  # the L1 match before it
    l1: int  # the KV before this is still in L1
    r: int  # the resume point it restores to
    node: Node  # the path's last node, locked (`lock`) until the restore is done (or its request admitted)
    resident: list  # the pages of the path's L1 prefix
    parts: list
    plan: tuple
    t0: float
    pages: torch.Tensor | None = None  # fresh pages for the KV it reads
    page_ids: list[int] | None = None
    slot: int = -1  # a fresh slot for the tail's state
    dst: torch.Tensor | None = None  # the pages as int64 indices, for the scatter
    future: object = None
    result: tuple | None = None
    orphan: bool = False  # its request is gone: it lands in the cache, held by nobody
    landed: queue.Queue = field(default_factory=queue.Queue)  # chunks read and checked: (index, objects, parts, cursor, rows)
    issued: list = field(default_factory=list)  # per chunk, set once its scatter is queued on the GPU
    freed: dict = field(default_factory=dict)  # chunk -> CUDA event: its scatter has read its half
    t_put: float = 0.0  # the engine thread's time issuing the scatter
    lock: object = None  # the restore's `radix.TreeLock` on `node`, the job its owner
    lock_kind = "restore"  # `radix.owner_kind`


class Staging:
    """One reusable host buffer the GPU reads in place: the restore buffer, which
    restores read into with O_DIRECT, and the copy tables (`_RowCopy`). On GB10
    (integrated GPU, unified memory) it registers its mapping with CUDA, and the
    device pointer is the host address: kernels read it directly, with no host
    memcpy and no host-to-device copy. Without CUDA it is a plain mapping."""

    def __init__(self, nbytes: int, device: torch.device):
        import mmap
        self.nbytes = nbytes
        self.mm = mmap.mmap(-1, nbytes)  # anonymous and page-aligned, as O_DIRECT needs
        self.view = memoryview(self.mm)
        self.rt = None
        if device.type != "cuda":
            self.base = torch.frombuffer(self.mm, dtype=torch.uint8)
            return
        import ctypes
        rt = _cudart()
        ptr = ctypes.addressof(ctypes.c_char.from_buffer(self.mm))
        if rt.cudaHostRegister(ctypes.c_void_p(ptr), ctypes.c_size_t(nbytes), ctypes.c_uint(2)) != 0:  # cudaHostRegisterMapped
            raise RuntimeError("cudaHostRegister failed for an SSD tier host buffer")
        self.rt, self.ptr = rt, ptr
        dev = ctypes.c_void_p()
        if rt.cudaHostGetDevicePointer(ctypes.byref(dev), ctypes.c_void_p(ptr), ctypes.c_uint(0)) != 0:
            self.close()
            raise RuntimeError("cudaHostGetDevicePointer failed for an SSD tier host buffer")
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (dev.value, False), "version": 3, "strides": None}
        self.base = torch.as_tensor(self, device=device)

    def close(self) -> None:
        """Free the mapping now: the CUDA tensor's deleter holds this object,
        a cycle the garbage collector cannot see."""
        self.base = None
        self.__dict__.pop("__cuda_array_interface__", None)
        if self.rt is not None:
            import ctypes
            self.rt.cudaHostUnregister(ctypes.c_void_p(self.ptr))
            self.rt = None
        self.view.release()
        try:
            self.mm.close()
        except BufferError:  # a view still exported (a caller kept a segment): the mapping goes with it
            pass


class _RowCopy:
    """The tier's KV copies on the GPU: one launch moves a restore chunk's rows from the staging buffer into their
    pages, or a backup gather's rows from their pages into pinned payloads, for every pool the tier stores
    (kernels/csrc/kv_copy.cu). The engine thread shares the GIL with the tier's I/O threads and the API, and a torch
    op lets it go: per pool, an op could wait milliseconds to get it back. Each launch's table goes into a slot of a
    small registered host buffer the GPU reads in place, so issuing a copy takes no torch op until its event."""

    SLOTS = 16  # tables in flight; a slot is rewritten once the launch that read it has finished
    SLOT_BYTES = 256 * 1024  # one launch's table: about 900 objects on the 27B; a larger copy takes several launches

    def __init__(self, targets, device: torch.device):
        from hibana.kernels.kv_copy import module
        self.kernel = module().kv_copy
        pools = [pool for _, pool in targets]
        assert all(pool.is_contiguous() for pool in pools), "a page's rows are one contiguous run of every pool"
        self.names = [name for name, _ in targets]
        self.row_bytes = [pool[0].numel() * pool.element_size() for pool in pools]
        self.word = 16  # the widest word (1 to 16 bytes) every row and pool start allows
        for v in self.row_bytes + [pool.data_ptr() for pool in pools]:
            self.word = math.gcd(self.word, v)
        self.width = 2 + len(pools)
        table = array("q", [v for pool, n in zip(pools, self.row_bytes) for v in (pool.data_ptr(), n)])
        self.pools = torch.frombuffer(table, dtype=torch.int64).to(device)
        self.tables = Staging(self.SLOTS * self.SLOT_BYTES, device)
        self.words = memoryview(self.tables.mm).cast("q")
        self.events: list = [None] * self.SLOTS
        self.next = 0

    def put_rows(self, base: int, got, parts, dst: torch.Tensor) -> None:
        """`SsdTier._put_rows`: the KV rows of every object of `got` (headers and payload offsets from `base`, the
        staging buffer's device address) into pages dst, in order."""
        table, n, most, cursor, layout, at = array("q"), 0, 0, 0, None, None
        for (h, off), (_, _, start, lo, hi) in zip(got, parts):
            a, b = lo - start, hi - start
            if b <= a:
                continue
            if h.segments != layout:  # whole blocks share one layout: look its offsets up once
                where = {name: seg_off for name, _, _, seg_off, _ in h.layout()}
                at, layout = [where[name] for name in self.names], h.segments
            table.append(b - a)
            table.append(cursor)
            table.extend([base + off + seg_off + a * row for seg_off, row in zip(at, self.row_bytes)])
            cursor, n, most = cursor + b - a, n + 1, max(most, b - a)
        self._launch(table, n, most, dst, to_pool=True)

    def gather(self, pages: torch.Tensor, starts: list[int], rows: int, hosts: list[int], offsets: list[int]) -> None:
        """`SsdTier._gather_spans`: `rows` rows from pages[start:] for each start, into the pinned payload at each of
        `hosts`, every pool's rows at its offset."""
        table = array("q")
        for start, host in zip(starts, hosts):
            table.append(rows)
            table.append(start)
            table.extend([host + o for o in offsets])
        self._launch(table, len(starts), rows, pages, to_pool=False)

    def _launch(self, table, n: int, most: int, pages: torch.Tensor, to_pool: bool) -> None:
        stream = torch.cuda.current_stream()
        per = self.SLOT_BYTES // (8 * self.width)  # objects one slot holds
        for first in range(0, n, per):
            slot, self.next = self.next, (self.next + 1) % self.SLOTS
            if (done := self.events[slot]) is not None:
                done.synchronize()  # the launch SLOTS launches ago, long finished in practice
            part = table[first * self.width:(first + per) * self.width]
            at = slot * self.SLOT_BYTES // 8
            self.words[at:at + len(part)] = part
            self.kernel(self.pools.data_ptr(), self.tables.base.data_ptr() + slot * self.SLOT_BYTES, pages.data_ptr(),
                        pages.dtype == torch.int64, self.width - 2, len(part) // self.width, most, self.word, to_pool,
                        stream.cuda_stream)
            self.events[slot] = torch.cuda.Event()
            self.events[slot].record(stream)


def _cudart():
    import ctypes
    import glob
    import os
    for path in glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib", "libcudart.so*")) + ["libcudart.so"]:
        try:
            return ctypes.CDLL(path)
        except OSError:
            continue
    raise RuntimeError("CUDA runtime library not found")


def _dtype_name(t: torch.Tensor) -> str:
    return str(t.dtype).removeprefix("torch.")


def _digest(ids: list[int]) -> int:
    """A stored prefix's tokens as one number every rank computes alike (stored keys are per rank)."""
    return int.from_bytes(hashlib.sha256(array("q", ids).tobytes()).digest()[:16], "little")


def _path(anchor: Node):
    """(node, start, end) for each node from anchor up to the root's child; one depth walk gives every end."""
    n, e = anchor, anchor.depth
    while n.parent is not None:
        s = e - len(n.key)
        yield n, s, e
        n, e = n.parent, s


def _pinned(nbytes: int) -> int:
    """What a pinned host copy of `nbytes` holds: PyTorch's caching host allocator rounds it up to a power of two."""
    return 1 << max(nbytes - 1, 0).bit_length()


def _charge(sizes, cuda: bool) -> int:
    """What host copies of `sizes` bytes hold against the staging limit."""
    return sum(_pinned(n) for n in sizes) if cuda else sum(sizes)


def _to_host(t: torch.Tensor, cuda: bool) -> torch.Tensor:
    host = torch.empty(t.shape, dtype=t.dtype, pin_memory=cuda)
    host.copy_(t, non_blocking=cuda)
    return host


class SsdTier:
    def __init__(self, store: SsdStore, kv_pool, state_pool, device: torch.device, budget_bytes: int, min_tokens: int = 4096,
                 min_hit: int = 2048, step_bytes: int = 64 * MIB, staging_bytes: int = STAGING_BYTES,
                 chunk_bytes: int = CHUNK_BYTES, trace: bool = False, extra_pages=()):
        self.block = store.block  # tokens per stored KV block (part of the store's identity)
        self.trace = trace  # one line per restore in restores.jsonl beside the store (SHARAR_SSD_TRACE)
        self.store, self.kv_pool, self.state_pool, self.device = store, kv_pool, state_pool, device
        self.budget, self.min_tokens, self.min_hit = int(budget_bytes), int(min_tokens), int(min_hit)
        self.step_bytes, self.staging_bytes, self.chunk_bytes = int(step_bytes), int(staging_bytes), int(chunk_bytes)
        self.codecs: dict = {}  # drafter identity -> (export, import)
        self.queue: deque = deque()  # (anchor node, block end, hash, parent hash) awaiting a gather
        self.through: deque = deque()  # (node, its end) of resume points awaiting a write-through
        self.inflight: dict[tuple[str, str], tuple[SsdRef, int]] = {}  # submitted, not yet complete: ref, staging bytes
        self.doomed: set[tuple[str, str]] = set()  # in flight for a node that is gone: its file goes once the write completes
        self.landed: deque = deque()  # completed writes, in order, not yet complete on every rank
        self.staged = 0
        self.used = 0
        self.prefixes = 0  # written tails: prefixes a request can resume from the SSD
        self._failed_tail = False
        self.backed_blocks = self.tails = self.tails_through = self.through_dropped = self.restores = self.restore_fails = self.forced = self.deleted = 0
        self.restored_tokens = self.restored_kv_tokens = self.waited = self.restore_bytes = 0
        self.restore_read_seconds = self.restore_put_seconds = 0.0
        self.restore_seconds = 0.0
        self.evict_seconds = self.evict_max = self.settle_seconds = self.settle_max = 0.0  # the engine blocked on backups
        self.step_seconds = self.step_max = 0.0  # the engine thread in step(): completions, trimming, gathering copies
        self.patient = False  # also wait for room to copy blocks (shutdown); serving drops a leaf whose blocks do not fit
        self.evict_dropped = 0  # leaves eviction dropped instead of waiting for the writers
        self.room_waits = self.room_wait_writes = 0  # waits for staging room, and the writes they waited for
        self.room_wait_seconds = self.room_wait_max = 0.0
        self._pool = ThreadPoolExecutor(8, thread_name_prefix="sharar-ssd-read")  # parallel O_DIRECT reads keep the drive's queue full
        # backup copies run on their own stream, so they overlap the model's kernels instead of sitting between them;
        # every free of pages or a slot first makes the main stream wait for them (`fence`)
        self._copies = torch.cuda.Stream(device) if device.type == "cuda" else None
        self.extra_pages = list(extra_pages)  # (name, per-page tensor) beside the KV, e.g. the MTP draft layer's KV
        self._row = kv_pool.row_bytes + sum(t[0].numel() * t.element_size() for _, t in self.extra_pages)
        self._copy = _RowCopy(self._kv_targets(), device) if device.type == "cuda" else None
        # two halves, each chunk_bytes or one whole tail (a block of KV, the slot's state and a drafter payload), reused by
        # every restore; sized here, never grown mid-restore, where a failure on one rank would leave the other waiting
        tail = self.block * self._row + state_pool.bytes_per_slot + 32 * MIB
        self._staging = Staging(2 * max(self.chunk_bytes, _pad(tail)), device)
        self._layouts: dict = {}  # expected segment layouts by (kind, rows, aux counts); the pools' geometry is fixed
        self._payloads: dict = {}  # payload layouts by row count (`_payload`)
        # what the largest tail holds against the staging limit (a block of KV less a token, a slot's state, and the
        # largest drafter payload written so far): background gathers leave it free for eviction's tail
        self._tail_room = self._tail_charge(self.block - 1, [t for _, t in self._state(0)])
        # asynchronous restores (`restore_async`): one at a time; a worker thread reads, the engine thread scatters
        self._worker = ThreadPoolExecutor(1, thread_name_prefix="sharar-ssd-restore")
        self._job: _Restore | None = None  # the restore in flight
        self._held: list = []  # (job, since) of landed restores, locking their nodes until their requests are admitted
        self._tail_nodes: dict[str, Node] = {}  # tails in flight: key -> the node they are for (pruned if the write fails)
        self._orphans: list[Node] = []  # nodes whose stored tail was lost or failed: pruned at the next step (`_trim`)
        self._staging_free = None  # an event: the last restore's state import has read the staging buffer
        self._steps = 0  # engine steps (`step` calls), the same count on every rank
        self.async_restores = self.restore_wait_steps = 0
        self.restore_blocked_seconds = 0.0  # the engine thread inside restores: all of a synchronous one, the start and finish of an asynchronous one

    @staticmethod
    def memory_bound(target, block: int, extra_page_bytes: int = 0, staging_bytes: int = STAGING_BYTES,
                     chunk_bytes: int = CHUNK_BYTES) -> int:
        """Pinned host memory the tier can hold, for the pool plan: the restore
        buffer's two halves (each chunk_bytes, or one whole tail object if that
        is larger: a block of KV and a snapshot), the backup staging (at least
        one whole object), the snapshot a tail gather copies and the copy
        tables (`_RowCopy`). The context length does not change it: restores
        stream through the halves chunk by chunk."""
        tail = block * (target.kv_bytes_per_token + extra_page_bytes) + target.state_bytes_per_slot + 32 * MIB  # + header, draft state
        return (2 * max(chunk_bytes, tail) + max(staging_bytes, tail) + target.state_bytes_per_slot
                + _RowCopy.SLOTS * _RowCopy.SLOT_BYTES)

    @contextlib.contextmanager
    def _copying(self):
        """Run backup copies on the copy stream, after the work that wrote their pages."""
        if self._copies is None:
            yield
            return
        self._copies.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._copies):
            yield

    def fence(self) -> None:
        """Before the tree frees pages or a slot: the main stream waits (on the device) for the backup copies that
        may read them."""
        if self._copies is not None:
            torch.cuda.current_stream().wait_stream(self._copies)

    def register_codec(self, identity, export, import_) -> None:
        self.codecs[identity] = (export, import_)

    # ---- geometry ----
    def _hashes(self, ids: list[int]) -> list[bytes]:
        """The chain hash after each full block of ids, in order."""
        out, h = [], self.store.root
        for end in range(self.block, len(ids) + 1, self.block):
            h = block_hash(h, ids[end - self.block:end])
            out.append(h)
        return out

    @staticmethod
    def _owner(anchor: Node, end: int) -> Node:
        """The node on anchor's path whose span holds token end-1 (0 < end <= anchor.depth)."""
        return next(n for n, s, e in _path(anchor) if s < end <= e)

    @staticmethod
    def _span(anchor: Node, lo: int, hi: int) -> tuple[list[int], torch.Tensor] | None:
        """Tokens and L1 pages for [lo, hi) on anchor's path, or None if a node
        in that range is no longer in L1."""
        parts = []
        for n, s, e in _path(anchor):
            if e > lo and s < hi:
                if not n.in_l1:
                    return None
                a, b = max(lo, s) - s, min(hi, e) - s
                parts.append((n.key[a:b], n.pages[a:b]))
            if s <= lo:
                break
        parts.reverse()
        return [t for k, _ in parts for t in k], torch.cat([p for _, p in parts])

    # ---- backup ----
    def backup(self, cache: RadixCache, ids: list[int], resume_points: tuple[int, ...] = ()) -> None:
        """Queue the full blocks of a sequence that just entered the tree, and a tail at each of `resume_points`
        (lengths where the tree holds a snapshot of it) for a write-through: a prompt's non-spare snapshots at its
        checkpoints (`StateManager.checkpoint`: message boundary, last chunk end, branching point) and a finished
        sequence's reply end at its release. A conversation's next turn, or the same prompt sent again, resumes at
        one of those, so those tails are the ones restores use; a spare chunk end (about 144 MiB on the 27B in FP32)
        gets none. Queued, not copied now: a snapshot is large, and `step()` keeps the copies within the staging
        limit."""
        if len(ids) < self.min_tokens or cache.flushing:
            return
        anchor, pos = cache.root, 0
        while pos < len(ids):  # the deepest node of this exact path
            child = anchor.children.get(ids[pos])
            if child is None or ids[pos:pos + len(child.key)] != child.key:
                break
            anchor, pos = child, pos + len(child.key)
        if pos != len(ids):
            return
        self._queue_blocks(anchor, ids)
        for end in sorted(set(resume_points)):
            node = self._ending_at(anchor, end)
            if (end >= self.min_tokens and node is not None and node.snapshot is not None
                    and (node.tail is None or node.tail.nbytes < 0)):
                self.through.append((node, end))

    def _pending_ref(self, kind: str, key: str) -> SsdRef | None:
        """Reattach a write a removed prefix left in flight. One key has one
        write and one staging allocation, even if the prefix comes back
        before that write completes."""
        pending = self.inflight.get((kind, key))
        if pending is None:
            return None
        ref, _ = pending
        self.doomed.discard((kind, key))
        ref.nbytes = 0
        return ref

    def _queue_blocks(self, anchor: Node, ids: list[int]) -> None:
        """Record a ref for every full block of ids (the tokens of anchor's path) and queue those not yet stored.
        One walk of the path finds each block's node, and the ref of a block the tree already records holds its hash
        (keys are content-addressed), so a later turn of a conversation hashes only its new blocks."""
        spans = list(_path(anchor))[::-1]  # (node, start, end), root side first; the last ends at len(ids)
        h, j = self.store.root, 0
        for end in range(self.block, len(ids) + 1, self.block):
            while spans[j][2] < end:
                j += 1
            owner, parent = spans[j][0], h
            ref = owner.blocks.get(end)
            h = bytes.fromhex(ref.key) if ref is not None else block_hash(h, ids[end - self.block:end])
            if ref is None or ref.nbytes < 0:
                pending = self._pending_ref("kv", h.hex())
                owner.blocks[end] = pending if pending is not None else SsdRef(h.hex())
                if pending is None:
                    self.queue.append((anchor, end, h, parent))

    def _hash_to(self, anchor: Node, end: int) -> bytes | None:
        """The chain hash of anchor's path through `end` (a block boundary): the key of the block the tree records
        there, else the hash of the path's tokens from the root. None if the path is no longer in L1."""
        if end == 0:
            return self.store.root
        ref = self._owner(anchor, end).blocks.get(end)
        if ref is not None:
            return bytes.fromhex(ref.key)
        span = self._span(anchor, 0, end)
        return None if span is None else self._hashes(span[0])[-1]

    @staticmethod
    def _ending_at(anchor: Node, end: int) -> Node | None:
        """The node on anchor's path whose span ends exactly at end."""
        return next((n for n, _, e in _path(anchor) if e == end), None)

    def _gather(self, anchor: Node, lo: int, hi: int) -> tuple[list[int], list[tuple[str, torch.Tensor]], torch.Tensor] | None:
        got = self._gather_spans(anchor, [(lo, hi)])
        return None if got is None else got[0]

    def _payload(self, rows: int):
        """The file payload of `rows` rows of every pool: per pool (name, pool, offset, bytes), the padded total,
        whether it has padding (zero, as a file has) and the header's segment spec. The pools' geometry is fixed, so
        each row count is laid out once."""
        got = self._payloads.get(rows)
        if got is None:
            layout, total = [], 0
            for name, pool in self._kv_targets():
                n = rows * pool[0].numel() * pool.element_size()
                layout.append((name, pool, total, n))
                total += _seg_pad(n)
            spec = [[name, _dtype_name(pool), [rows, *pool.shape[1:]], n] for name, pool, _, n in layout]
            got = self._payloads[rows] = (layout, total, any(n % SEG_ALIGN for *_, n in layout), spec)
        return got

    def _gather_spans(self, anchor: Node, spans: list[tuple[int, int]], views: bool = True):
        """For each (lo, hi) on anchor's path, all of one length: tokens, segments (with `views`; else None) and the
        one host buffer they are views of, laid out as the file's payload (`_payload`); each span goes out in one
        write. On the GPU one launch writes every span's payload into its pinned buffer (`_RowCopy`); on the CPU each
        pool is gathered once for all spans together. None if a span is no longer in L1."""
        lo, hi = min(a for a, _ in spans), max(b for _, b in spans)
        span = self._span(anchor, lo, hi)
        if span is None:
            return None
        tokens, pages = span
        rows = spans[0][1] - spans[0][0]
        layout, total, padded, _ = self._payload(rows)
        empty = torch.zeros if padded else torch.empty
        if self._copy is not None:
            hosts = [empty(total, dtype=torch.uint8, pin_memory=True) for _ in spans]
            self._copy.gather(pages, [a - lo for a, _ in spans], rows, [h.data_ptr() for h in hosts], [at for _, _, at, _ in layout])
        else:
            contiguous = all(a == lo + i * rows and b - a == rows for i, (a, b) in enumerate(spans))
            idx = (pages if contiguous else torch.cat([pages[a - lo:b - lo] for a, b in spans])).to(torch.int64)
            dev = empty((len(spans), total), dtype=torch.uint8, device=self.device)
            for _, pool, at, n in layout:
                src = pool.view(torch.uint8)  # every pool as raw bytes
                if len(spans) == 1:
                    torch.index_select(src, 0, idx, out=dev[0, at:at + n].view(rows, *src.shape[1:]))
                else:
                    dev[:, at:at + n].copy_(src.index_select(0, idx).view(len(spans), n))
            hosts = [_to_host(dev[i], False) for i in range(len(spans))]
        return [(tokens[a - lo:b - lo], [(name, host[at:at + n].view(pool.dtype).view(rows, *pool.shape[1:]))
                                         for name, pool, at, n in layout] if views else None, host)
                for (a, b), host in zip(spans, hosts)]

    def _submit(self, kind: str, key: bytes, parent: bytes, start: int, tokens: list[int], segs, ref: SsdRef, scalars=None,
                packed: torch.Tensor | None = None, spec=None, held: list[int] | None = None) -> None:
        """Queue an object's write. `packed`: one host buffer holding every segment in payload layout (`_gather`),
        described by `spec` (`_payload`) instead of `segs`. `held`: the sizes of the host buffers `segs` are views
        of, when they share some."""
        ready = None
        if self.device.type == "cuda":
            ready = torch.cuda.Event(blocking=True)  # writers wait on it: sleep, don't spin a core each
            ready.record()
        if spec is None:
            spec = [[name, _dtype_name(t), list(t.shape), t.numel() * t.element_size()] for name, t in segs]
        else:
            spec = [[name, dtype, list(shape), n] for name, dtype, shape, n in spec]  # each header owns its spec
        # the pinned memory the copies hold: the caching host allocator rounds each up to a power of two
        sizes = [packed.numel()] if packed is not None else held if held is not None else [s[3] for s in spec]
        pinned = _charge(sizes, self.device.type == "cuda")
        header = Header(kind, key.hex(), parent.hex(), start, tokens, spec, scalars or {})
        self.inflight[(kind, key.hex())] = (ref, pinned)
        self.staged += pinned
        self.store.write(header, [packed] if packed is not None else [t for _, t in segs], ready)

    def _room(self, nbytes: int, wait: bool) -> bool:
        """Whether a copy of `nbytes` fits within the staging limit (asked before
        copying). An object larger than the limit can go alone; eviction,
        restores and shutdown wait for older copies instead of exceeding it,
        for the fewest of the oldest writes in flight that free the room."""
        if not (self.staged and self.staged + nbytes > self.staging_bytes):
            return True
        if not wait:
            return False
        t0, over, k = time.perf_counter(), self.staged + nbytes - self.staging_bytes, 0
        for _, staged in self.inflight.values():  # in submission order, the order completions apply in
            k += 1
            over -= staged
            if over <= 0:
                break
        self.store.wait_first(k - len(self.landed))  # `landed` holds the oldest that finished here already
        self._finish_writes()  # every rank has its own first k, so the agreement applies at least k everywhere
        assert self.staged == 0 or self.staged + nbytes <= self.staging_bytes
        spent = time.perf_counter() - t0
        self.room_waits += 1
        self.room_wait_writes += k
        self.room_wait_seconds += spent
        self.room_wait_max = max(self.room_wait_max, spent)
        return True

    def _block_charge(self) -> int:
        """What one block's host copy holds against the staging limit."""
        return _charge([self._payload(self.block)[1]], self.device.type == "cuda")

    def _gather_block(self, item, wait: bool = False) -> bool:
        return self._gather_blocks([item], wait) == 1

    def _gather_blocks(self, items, wait: bool = False, limit: int | None = None) -> int:
        """Copy out queued blocks, those of one path in one gather, as many as the staging limit (or `limit`, below
        it) takes, waiting for room with `wait`. Returns how many of items are done (copied, or no longer needing
        it): a prefix."""
        live, seen = [], set()  # (position in items, item, owner, ref); a block queued twice goes out once
        for i, item in enumerate(items):
            anchor, end, h, _ = item
            owner = self._owner(anchor, end)
            ref = owner.blocks.get(end)
            if not (ref is None or ref.key != h.hex() or ref.nbytes < 0 or ref.ready or ("kv", ref.key) in self.inflight
                    or ref.key in seen):
                live.append((i, item, owner, ref))
                seen.add(ref.key)
        cost, j = self._block_charge(), 0
        cap = self.staging_bytes if limit is None else limit
        while j < len(live):
            fit = len(live) - j if self.staged + (len(live) - j) * cost <= cap else max(0, (cap - self.staged) // cost)
            if fit == 0 and self.staged == 0:
                fit = 1  # an object larger than the limit goes alone
            if fit == 0:
                if not wait:
                    return live[j][0]
                got = one = segs = packed = None  # the last batch's host copies are the writers' now: room counts them gone
                self._room(min(len(live) - j, max(1, cap // cost)) * cost, True)
                continue
            batch, j = live[j:j + fit], j + fit
            groups: dict[int, list] = {}
            for entry in batch:  # one gather per path
                groups.setdefault(id(entry[1][0]), []).append(entry)
            for group in groups.values():
                anchor = group[0][1][0]
                with self._copying():
                    got = self._gather_spans(anchor, [(e[1][1] - self.block, e[1][1]) for e in group], views=False)
                    spec = self._payload(self.block)[3]
                    for k, (_, (_, end, h, parent), owner, ref) in enumerate(group):
                        one = got[k] if got is not None else self._gather(anchor, end - self.block, end)
                        if one is None:  # part of the block left L1 while it sat in the queue: drop its ref
                            del owner.blocks[end]
                            continue
                        tokens, segs, packed = one
                        self._submit("kv", h, parent, end - self.block, tokens, segs, ref, packed=packed,
                                     spec=spec if segs is None else None)
                        self.backed_blocks += 1
        return len(items)

    def _touches(self, item, node: Node, start: int, end: int) -> bool:
        """Whether a queued block has a token in node's span [start, end) on its own path."""
        anchor, block_end, _, _ = item
        if block_end <= start or block_end - self.block >= end:
            return False
        n = anchor
        while n is not None and n is not node:  # is node on the item's path?
            n = n.parent
        return n is node

    # ---- eviction ----
    def evicting(self, cache: RadixCache, leaf: Node, wait: bool = True) -> bool:
        """The tree calls this before it frees an L1 leaf's pages. Gathers the
        queued blocks of its whole path while its pages exist, in one batch (a
        leaf kept without them could not be restored, and its parent would wait
        for room when its turn came), and writes its tail unless the leaf has
        descendants on the SSD: those carry their own tails, and the
        conversation resumes from them, so an interior snapshot (a prompt's end,
        a chunk checkpoint) is not worth its copy. True keeps the leaf SSD-only.

        While serving, eviction waits to copy blocks only for a leaf whose
        SSD-only descendants need them: otherwise, when the staging buffer
        cannot take the leaf's blocks now, the leaf is dropped as it would be
        without the tier (counted in `evict_dropped`). Waiting there stalls
        every running request behind fdatasyncs, which costs more than the
        prefix is worth when the cache is under that much pressure. It does wait
        for room for a tail, at most for the writes already in flight: without
        its tail no turn can resume from the prefix, so each ancestor would in
        turn need a tail of its own and be dropped with blocks already stored,
        and the next turn would prefill all of it again. Background block backups
        leave room for one tail, but the leaf's own queued blocks, copied just
        before it, or a write-through tail can take that room. Shutdown
        (`patient`) waits for blocks too, so every cached prefix is persisted.
        Without `wait` (slot pressure, which takes only a leaf with no children)
        nothing waits: a tail without room now is not written."""
        t0 = time.perf_counter()
        end = leaf.depth
        missing, node = end // self.block, leaf
        while node.parent is not None:
            missing -= sum(ref.nbytes >= 0 for ref in node.blocks.values())
            node = node.parent
        if missing > 0 and end >= self.min_tokens:  # the disk budget may have dropped copies whose pages stayed in L1
            self._queue_blocks(leaf, self._span(leaf, 0, end)[0])  # the leaf's whole path is in L1
        pending, keep = [], deque()
        for item in self.queue:  # gather the queued blocks of its whole path now: kept, it needs every one of them
            (pending if self._touches(item, leaf, 0, end) else keep).append(item)
        self.queue = keep
        patient = self.patient or bool(leaf.children)  # stored descendants need this leaf's blocks
        done = self._gather_blocks(pending, wait=patient)
        self.forced += done
        copied = done == len(pending)  # else no staging room: its pages are about to go, so its blocks cannot be stored
        if not copied:  # the ancestors keep their pages: their blocks stay queued, or their refs would look pending forever
            start = end - len(leaf.key)
            self.queue.extendleft(reversed([(leaf.parent, e, h, p) for _, e, h, p in pending[done:] if e <= start]))
        written = leaf.tail is not None and leaf.tail.nbytes >= 0  # written through already: its snapshot may be gone
        # a true leaf's tail is due; one with SSD-only descendants needs none, unless it is not spare (where a turn
        # that departs before those descendants resumes: its write-through may still be queued)
        due = (leaf.snapshot is not None and end >= self.min_tokens
               and (not leaf.children or not leaf.snapshot.spare))
        keep = copied and (written or (due and self._write_tail(leaf, end, wait=wait)))
        if not keep and (written or due):
            self.evict_dropped += 1
        if not keep and not leaf.children:
            self._discard_node(leaf)
        spent = time.perf_counter() - t0  # the gathers above, and waits for staging room behind the writers
        self.evict_seconds += spent
        self.evict_max = max(self.evict_max, spent)
        return keep

    def dropping_snapshot(self, cache: RadixCache, node: Node) -> None:
        """Slot pressure is about to take `node`'s snapshot and keep its KV. A non-spare snapshot (a message boundary,
        a last chunk end, a branching point) whose tail still waits in the write-through queue gets it written now if
        staging has room now: once the slot is free, the tail could never be written. Slot pressure never waits for the
        writers, so without room the point is dropped on purpose (`step()` counts it in `tails_through_dropped`), as is
        a spare snapshot's queued tail (a reply end's)."""
        end = node.depth
        if (not node.snapshot.spare and end >= self.min_tokens and (node.tail is None or node.tail.nbytes < 0)
                and any(n is node for n, _ in self.through)):
            if self._write_tail(node, end, wait=False):
                self.tails_through += 1
        self.fence()

    def pruning(self, node: Node) -> None:
        """The tree deletes `node` as dead KV (`RadixCache._prune`): its stored objects go with it."""
        self._discard_node(node)

    def owned_slots(self) -> list[tuple[str, int]]:
        """The state slot the restore in flight fills (`RadixCache.check`): neither free nor any node's yet."""
        return [("restore", self._job.slot)] if self._job is not None else []

    def owned_pages(self) -> list[int]:
        """The KV pages the restore in flight fills (`StateManager.check`): neither free nor any node's yet."""
        return list(self._job.page_ids) if self._job is not None else []

    def live_jobs(self) -> list:
        """The restores whose locks may be in the tree: the one in flight and the landed ones (`RadixCache.check`)."""
        return [job for job, _ in self._held] + ([self._job] if self._job is not None else [])

    def pending_prune(self) -> list[Node]:
        """Nodes whose tail was lost or failed since the last step: `_trim` prunes them (`RadixCache.check`)."""
        return list(self._orphans)

    def clean(self, leaf: Node) -> bool:
        """Whether evicting `leaf` needs no copy now: its full blocks are on the SSD or already copied out, and so is
        its tail if it holds a snapshot worth one. The tree evicts clean leaves first (least recently used among
        them), so eviction rarely waits for the writers; a dirty leaf goes only when no clean one is left."""
        end = leaf.depth
        if end < self.min_tokens:
            return True  # never stored: dropping it costs nothing
        if (leaf.snapshot is not None and (not leaf.children or not leaf.snapshot.spare)
                and (leaf.tail is None or leaf.tail.nbytes < 0)):
            return False  # its tail is due (`evicting`)
        start = end - len(leaf.key)
        for e in range((start // self.block + 1) * self.block, end + 1, self.block):
            ref = leaf.blocks.get(e)
            if ref is None or ref.nbytes < 0 or (not ref.ready and ("kv", ref.key) not in self.inflight):
                return False
        return True

    def _state(self, slot: int) -> list[tuple[str, torch.Tensor]]:
        sp = self.state_pool
        return ([("conv", sp.conv_all[:, slot]), ("ssm", sp.ssm_all[:, slot])]
                + [(f"pool.{name}", t[slot]) for name, t in sp.aux.items()])

    def _tail_charge(self, rows: int, state: list[torch.Tensor]) -> int:
        """What a tail of `rows` KV rows (one host buffer) and the `state` tensors (one each) holds against the
        staging limit, as `_submit` charges it."""
        return _charge(self._tail_sizes(rows, state), self.device.type == "cuda")

    def _tail_sizes(self, rows: int, state: list[torch.Tensor]) -> list[int]:
        return ([self._payload(rows)[1]] if rows else []) + [t.numel() * t.element_size() for t in state]

    def _write_tail(self, leaf: Node, end: int, wait: bool = False) -> bool:
        if leaf.tail is not None and leaf.tail.nbytes >= 0:
            return True  # already written (or on its way)
        whole = end // self.block * self.block
        span = self._span(leaf, whole, end) if end > whole else ([], None)
        last = self._hash_to(leaf, whole) if span is not None else None
        if last is None:
            return False
        tokens = span[0]
        key = tail_hash(last, tokens)
        pending = self._pending_ref("tail", key.hex())
        if pending is not None:
            leaf.tail = pending
            self._tail_nodes[key.hex()] = leaf
            return True
        cuda = self.device.type == "cuda"
        state = self._state(leaf.snapshot.slot)
        scalars = {}
        for identity, value in leaf.snapshot.auxiliary.items():
            codec = self.codecs.get(identity)
            payload = codec[0](value) if codec is not None else None
            if payload is None:
                continue
            scalars[str(identity)] = [x for x in payload if not torch.is_tensor(x)]
            state += [(f"aux.{identity}.{i}", x) for i, x in enumerate(t for t in payload if torch.is_tensor(t))]
        nbytes = self._tail_charge(end - whole, [t for _, t in state])
        self._tail_room = max(self._tail_room, self._tail_charge(self.block - 1, [t for _, t in state]))
        if not self._room(nbytes, wait):
            return False
        # `_queue_blocks` queues the full blocks; gather the partial block only once the whole tail has room.
        with self._copying():
            got = self._gather(leaf, whole, end) if end > whole else (tokens, [], None)
            if got is None:
                return False
            tokens, segs, _ = got
            segs += [(name, _to_host(t, cuda)) for name, t in state]
            leaf.tail = SsdRef(key.hex())
            self._tail_nodes[key.hex()] = leaf
            self._submit("tail", key, last, whole, tokens, segs, leaf.tail, scalars,
                         held=self._tail_sizes(end - whole, [t for _, t in state]))
        self.tails += 1
        return True

    def _delete(self, kind: str, ref: SsdRef) -> None:
        self.store.delete(kind, ref.key)
        self.used -= ref.nbytes
        self.prefixes -= kind == "tail" and ref.ready
        ref.ready, ref.nbytes = False, -1
        self.deleted += 1

    def _discard(self, kind: str, ref: SsdRef | None) -> None:
        """Drop a ref whose node is leaving the tree: delete its object now, or
        once its write completes if it is still in flight."""
        if ref is None:
            return
        if ref.ready:
            self._delete(kind, ref)
        elif (kind, ref.key) in self.inflight:
            self.doomed.add((kind, ref.key))
            ref.nbytes = -1

    def _discard_node(self, node: Node) -> None:
        self._discard("tail", node.tail)
        for ref in node.blocks.values():
            self._discard("kv", ref)

    def _lost(self, kind: str, ref: SsdRef) -> None:
        """A ref whose object turned out missing, corrupt or a collision: it
        leaves the budget and the disk, and no restore reads it again."""
        if ref.nbytes >= 0:
            self._delete(kind, ref)

    # ---- tensor parallel ----
    def _min_over_ranks(self, question: str, values: list[int]) -> list[int]:
        """The element-wise minimum of `values` over tensor-parallel ranks. Every rank asks the same `question` at the
        same point; `question` only names the agreement, for tests that play another rank."""
        return all_reduce_min(*values) if values else []

    def _held_by_all(self, values: list[int]) -> list[int]:
        return common_values(values)

    def _largest(self, sizes: list[int]) -> list[int]:
        """Each object's size as the largest any rank stored: the payloads match, but each rank's header carries its
        own CRC, so a header that crosses a 4 KiB page makes one rank's file a page longer. Every rank then reads
        into a span its own file fits and counts the same bytes against the budget."""
        return [-v for v in self._min_over_ranks("sizes", [-n for n in sizes])]

    def _completions(self) -> list[tuple[str, str, bool, int]]:
        """Writes complete on every rank, as (kind, key, ok, bytes). Each rank queues the same writes in the same order
        and its store hands completions back in that order, so the first n completions on every rank are the same
        objects (n: the fewest any rank has). One that failed on any rank failed on all, and the ranks where it landed
        delete their file; bytes is the largest any rank wrote (`_largest`). It asks the other ranks only while writes
        are in flight, which every rank sees alike."""
        self.landed.extend(self.store.completions())
        if not self.inflight:
            return []
        n = self._min_over_ranks("landed", [len(self.landed)])[0]
        got = [self.landed.popleft() for _ in range(n)]
        # one agreement for both: written on every rank (min of 0/1) and the largest size (min of the negated sizes)
        both = self._min_over_ranks("outcomes", [int(ok) for _, _, ok, _ in got] + [-nbytes for *_, nbytes in got])
        written, sizes = both[:n], [-v for v in both[n:]]
        out = []
        for (kind, key, ok, _), everywhere, size in zip(got, written, sizes):
            if ok and not everywhere:
                self.store.delete(kind, key)
            out.append((kind, key, bool(everywhere), size if everywhere else 0))
        return out

    # ---- per step ----
    def _finish_writes(self) -> None:
        """Apply completed writes to their refs and free their staging copies, without removing any node (`_trim`
        does that): eviction can wait here for staging room while it still holds the leaf it is evicting."""
        for kind, key, ok, nbytes in self._completions():
            ref, staged = self.inflight.pop((kind, key))
            self.staged -= staged
            node = self._tail_nodes.pop(key, None) if kind == "tail" else None
            if node is not None and not ok:
                self._orphans.append(node)  # nothing to resume from there now, perhaps: `_trim` prunes it
            if (kind, key) in self.doomed:
                self.doomed.discard((kind, key))
                if ok:
                    self.store.delete(kind, key)
                    self.deleted += 1
                continue
            ref.ready, ref.nbytes = ok, (nbytes if ok else -1)
            self.used += nbytes if ok else 0
            self.prefixes += kind == "tail" and ok
            self._failed_tail |= kind == "tail" and not ok

    def _trim(self, cache: RadixCache) -> None:
        orphans, self._orphans = self._orphans, []
        for node in orphans:  # a lost or failed tail: the node may be dead KV now
            cache._prune(node)
        if self._failed_tail:  # a full disk: no restore can use an SSD-only leaf without a tail
            while (dead := cache.ssd_candidate(unusable=True)) is not None:
                self._drop_leaf(cache, dead)
            self._failed_tail = False
        while self.used > self.budget and (victim := cache.ssd_candidate()) is not None:
            self._drop_leaf(cache, victim)
        reading = self._job.node if self._job is not None else None  # a restore of a resident path reads its tail alone
        while self.used > self.budget and (victim := cache.ssd_candidate(resident=True, skip=reading)) is not None:
            self._discard_node(victim)
            victim.blocks.clear()
            victim.tail = None
            cache._prune(victim)  # its tail was all it could resume from: dead KV now

    def _drop_leaf(self, cache: RadixCache, node: Node) -> None:
        self._discard_node(node)
        cache.remove_ssd_leaf(node)

    def _gather_queued(self, wait: bool = False, prefilling: bool = False) -> None:
        """Queued write-through tails, up to `step_bytes` a step, then queued blocks: up to `step_bytes` of payload
        while no request is prefilling (copies there take DRAM time from the decode), as many as the staging limit
        takes while one is (seconds of compute they hide behind), and every one with `wait` (shutdown). Background
        blocks leave room for one tail, which eviction needs at once."""
        budget, spent, size = (self.staging_bytes if prefilling else self.step_bytes), 0, self._payload(self.block)[1]
        room = self.staging_bytes - self._tail_room
        limit = None if wait or room < self._block_charge() else room  # keep a tail's room only if a block fits beside it
        while self.through and (wait or spent < self.step_bytes):  # tails first: a dropped snapshot is gone
            node, end = self.through[0]
            if node.tail is not None and node.tail.nbytes >= 0:  # eviction wrote it meanwhile, or backup queued it twice
                self.through.popleft()
                continue
            if node.snapshot is None:  # the GPU cache dropped it before there was room
                self.through.popleft()
                self.through_dropped += 1
                continue
            if not self._write_tail(node, end, wait=wait):
                break  # leave it queued until there is staging room
            self.through.popleft()
            self.tails_through += 1
            spent += self.inflight[("tail", node.tail.key)][1]
        while self.queue and (wait or spent < budget):
            take = len(self.queue) if wait else max(1, min(len(self.queue), (budget - spent) // size))
            items = [self.queue[i] for i in range(take)]
            done = self._gather_blocks(items, wait=wait, limit=limit)
            for _ in range(done):
                self.queue.popleft()
            spent += done * size
            if done < take:
                break

    def step(self, cache: RadixCache, prefilling: bool = False) -> None:
        """Each engine step; `prefilling`: a request is mid-prefill as the step starts (scheduler state, the same on
        every rank)."""
        t0 = time.perf_counter()
        self._steps += 1
        self._poll(cache)
        self._finish_writes()
        self._trim(cache)
        self._gather_queued(prefilling=prefilling)
        spent = time.perf_counter() - t0
        self.step_seconds += spent
        self.step_max = max(self.step_max, spent)

    def drain(self, cache: RadixCache) -> None:
        """Gather every queued tail and block within the staging bound, then wait (shutdown)."""
        self._gather_queued(wait=True)
        self.store.wait()
        self._finish_writes()
        self._trim(cache)
        self.store.wait()  # again, for the deletes the completions and the budget queued

    # ---- restore ----
    def _settle_for(self, cache: RadixCache, ids: list[int], l1: int) -> SsdHit | None:
        """The deepest resume point on the SSD past l1. If a deeper one is still in flight (evicted moments ago),
        finish its writes now rather than re-prefill: gather the path's queued blocks, wait for the writers, apply the
        completions. A ready resume point is also a pending one, so with none pending there is none."""
        pending = cache.ssd_match(ids, l1, self.block, pending=True)
        if pending is None:
            return None
        ready = cache.ssd_match(ids, l1, self.block)
        if ready is not None and ready.length >= pending.length:
            return ready
        t0 = time.perf_counter()
        needed = {ref.key for _, ref in pending.blocks if not ref.ready}
        forced, keep = [], deque()
        for item in self.queue:  # blocks of this path still queued (eviction normally gathered them already)
            (forced if item[2].hex() in needed else keep).append(item)
        self.queue = keep
        for item in forced:
            self._gather_block(item, wait=True)
        self.store.wait()
        self._finish_writes()
        spent = time.perf_counter() - t0
        self.settle_seconds += spent
        self.settle_max = max(self.settle_max, spent)
        self.waited += 1
        return cache.ssd_match(ids, l1, self.block)

    def restore(self, cache: RadixCache, ids: list[int], reserve: int = 0) -> int:
        """Bring the deepest SSD resume point for ids back into L1 now, if the pool can also hold the rest of ids and
        `reserve` more pages: otherwise the cache would only evict what it restored and read it again. Returns the
        tokens restored (0 when there is nothing worth restoring). The engine waits: nothing else runs meanwhile
        (`restore_async` lets it run)."""
        if self._landed(ids) is not None:
            return 0  # an asynchronous restore already brought it back
        if self._job is not None and self._worth(cache, ids):
            self._join(cache)  # the staging buffer is the in-flight restore's until it lands
        t0 = time.perf_counter()
        job = self._begin(cache, ids, reserve)
        if job is None:
            return 0
        job.future = self._worker.submit(self._read, job)
        self._drive(job)
        job.result = job.future.result()
        restored = self._finish(cache, job, keep=False)
        self.restore_blocked_seconds += time.perf_counter() - t0
        return restored

    def restore_async(self, cache: RadixCache, ids: list[int], reserve: int = 0) -> bool:
        """The queue head's prefetch, before admission's pure check (`StateManager.prefetch`): True when the request
        with ids can be checked now (nothing worth restoring, no room for a restore, another restore's landed path still
        waiting for its request, or its own restore has landed and locks the path in L1 until `release`), False while a
        restore is in flight: its own, started here if it needs one, or another one holding the staging buffer. A
        worker thread reads, and the engine thread scatters each chunk as it lands while it keeps serving everyone
        else; `step()` finishes the restore once it landed on every rank. Every rank asks alike (the trees are the
        same), so every rank starts the same restores."""
        if self._landed(ids) is not None:
            return True
        job = self._job
        if job is not None and job.ids == ids:
            self.restore_wait_steps += 1
            return False
        if self._held or not self._worth(cache, ids):
            return True  # at most one restore locks a slot for its request (`RESTORE_SLOTS`)
        if job is not None:
            self.restore_wait_steps += 1
            return False  # one restore at a time: they share the staging buffer and the drive
        t0 = time.perf_counter()
        job = self._begin(cache, ids, reserve)
        if job is not None:
            job.future = self._worker.submit(self._read, job)
            self._job = job
            self.async_restores += 1
        self.restore_blocked_seconds += time.perf_counter() - t0
        return job is None

    def release(self, cache: RadixCache, ids: list[int]) -> None:
        """The request a landed restore locked its node for was admitted (its own lock, taken first, is there now) or
        is gone."""
        for i, (job, _) in enumerate(self._held):
            if job.ids == ids:
                del self._held[i]
                cache.unlock(job.lock)
                return
        if self._job is not None and self._job.ids == ids:
            self._job.orphan = True  # it still lands in the cache, held by nobody

    def wait(self, timeout: float) -> None:
        """Nothing else to run: scatter what landed and wait up to `timeout` seconds for the in-flight restore's next
        chunk (no-op without one)."""
        job = self._job
        if job is not None:
            self._scatter_landed(job)
            try:
                job.landed.put(job.landed.get(timeout=timeout))  # put back for the scatter (each chunk carries its offsets)
            except queue.Empty:
                pass

    def _landed(self, ids: list[int]):
        return next((job.node for job, _ in self._held if job.ids == ids), None)

    def _worth(self, cache: RadixCache, ids: list[int]) -> bool:
        """Whether ids has an SSD resume point worth reading (counting writes still in flight)."""
        m = cache.match(ids, touch=False)
        hit = cache.ssd_match(ids, m.length, self.block, pending=True)
        return hit is not None and hit.length - m.length >= self.min_hit

    def _begin(self, cache: RadixCache, ids: list[int], reserve: int) -> "_Restore | None":
        """Plan a restore and take what it needs: a lock on the stored node it reads (its L1 prefix stays resident;
        eviction, the disk budget and flush leave its stored nodes alone), and fresh pages and a slot. It starts only
        when the restored KV, the rest of ids and `reserve` pages, and two slots (the restored snapshot's and its
        request's own), are free or evictable (`RadixCache.fits`), counted with the request's L1 match and the stored
        node locked: admission's own count, so the restore never evicts its request's resume point, and the request
        is not then refused for what its restore took. None when there is nothing worth restoring or no room."""
        m = cache.match(ids, touch=False)
        hit = self._settle_for(cache, ids, m.length)
        if hit is None or hit.length - m.length < self.min_hit:
            return None
        l1, r = hit.l1_end, hit.length  # L1 still holds the KV before l1 (at least the matched prefix): read only [l1, r)
        whole = r // self.block * self.block
        parts = [("kv", ref, end - self.block, max(end - self.block, l1), min(end, whole)) for end, ref in hit.blocks]
        parts = [p for p in parts if p[3] < p[4]] + [("tail", hit.tail, whole, max(whole, l1), r)]
        plan = self._plan_chunks(parts)
        if 2 * plan[2] > self._staging.nbytes:  # an object larger than a half: every rank plans from the same sizes
            self.restore_fails += 1
            return None
        job = _Restore(ids=ids, matched=m.length, l1=l1, r=r, node=hit.path[-1], resident=[], parts=parts, plan=plan,
                       t0=time.perf_counter(), issued=[threading.Event() for _ in plan[3]])
        need = len(ids) - l1 + reserve  # the restored KV, the rest of ids and the reserve
        with cache._lock_node(m.node):  # the request's own L1 resume point is not evicted for its restore
            job.lock = cache.lock(job.node, owner=job)
            try:
                if not cache.fits(need, 2):
                    cache.unlock(job.lock)
                    return None
                cache.evict(need_pages=need, need_slots=2)
                job.pages, job.page_ids = self.kv_pool.alloc_with_ids(r - l1)
                job.slot = self.state_pool.alloc()
            except BaseException:
                if job.page_ids is not None:
                    self.kv_pool.free_ids(job.page_ids)
                cache.unlock(job.lock)
                raise
        job.resident = [n.pages for n in hit.path if n.in_l1]  # a prefix of the path, locked
        job.dst = job.pages.to(torch.int64)
        return job

    def _read(self, job: "_Restore"):
        """The worker thread: read the restore's objects chunk by chunk with O_DIRECT into the two halves of the
        staging buffer (in parallel within a chunk), check each, and hand it to the engine thread (`landed`), which
        scatters it: a chunk goes into a half only once the scatter of the one before it there has read it. Returns
        (the tail's header and offset, the parts this rank found missing, corrupt or a collision, bytes read, seconds
        waiting on reads)."""
        sizes, stride, _, chunks = job.plan
        st = self._staging
        half = st.nbytes // 2
        if self._staging_free is not None:
            self._staging_free.synchronize()  # the last restore's state import has read the buffer these reads refill

        def submit(chunk, h):
            def one(i, off):
                kind, ref, *_ = job.parts[i]
                got = self.store.read_into(kind, ref.key, st.view[off:off + _pad(sizes[i])])
                return None if got is None else (got[0], off + got[1])
            return [(i, self._pool.submit(one, i, h * half + rel)) for i, rel in chunk]

        parts, ids = job.parts, job.ids
        tail, lost, cursor, t_read = None, set(), 0, 0.0
        inflight = {0: submit(chunks[0], 0)}
        for k in range(len(chunks)):
            h = k % 2
            if k + 1 < len(chunks):  # keep the drive busy: the next chunk reads into the other half as soon as
                if k >= 1:  # the chunk before it there has been scattered, before waiting on this one
                    job.issued[k - 1].wait()
                    if (ev := job.freed.get(k - 1)) is not None:
                        ev.synchronize()
                inflight[k + 1] = submit(chunks[k + 1], 1 - h)
            t = time.perf_counter()
            got = [(i, f.result()) for i, f in inflight.pop(k)]
            t_read += time.perf_counter() - t
            lost = {i for i, g in got if g is None or not self._valid_layout(g[0]) or g[0].start != parts[i][2]
                    or g[0].end != (parts[i][2] + self.block if parts[i][0] == "kv" else parts[i][4])
                    or g[0].tokens != ids[parts[i][2]:parts[i][2] + len(g[0].tokens)]}
            if lost:
                for later in inflight.values():  # let the reads into the other half land: the next restore reuses it
                    for _, f in later:
                        f.result()
                break
            chunk_parts = [parts[i] for i, _ in got]
            rows = sum(hi - lo for _, _, _, lo, hi in chunk_parts)
            job.landed.put((k, [g for _, g in got], chunk_parts, cursor, rows))
            cursor += rows
            if got[-1][0] == len(parts) - 1:  # the tail, the last object: its state goes in once every rank has it all
                tail = got[-1][1]
        return tail, lost, sum(sizes) if not lost else 0, t_read

    def _scatter_landed(self, job: "_Restore", scatter: bool = True) -> None:
        """The engine thread: scatter every chunk that landed into the restore's pages, on the main stream (ahead of
        any kernel that reads them), and let the reader reuse its half once the scatter has read it. Without
        `scatter` (an abort), only let the reader go on."""
        t = time.perf_counter()
        cuda = self.device.type == "cuda"
        while True:
            try:
                k, got, chunk_parts, cursor, rows = job.landed.get_nowait()
            except queue.Empty:
                break
            if scatter:
                self._put_rows(got, chunk_parts, job.dst[cursor:cursor + rows], job.plan[1])
                if cuda:
                    job.freed[k] = torch.cuda.Event()
                    job.freed[k].record()
            job.issued[k].set()
        job.t_put += time.perf_counter() - t

    def _drive(self, job: "_Restore", scatter: bool = True) -> None:
        """Scatter as chunks land until the reader is done (a synchronous restore, a join, an abort)."""
        while True:
            done = job.future.done()
            self._scatter_landed(job, scatter)
            if done:
                return
            try:
                job.landed.put(job.landed.get(timeout=0.05))  # wake when a chunk lands; put back for the scatter
            except queue.Empty:
                pass

    def _finish(self, cache: RadixCache, job: "_Restore", keep: bool) -> int:
        """On every rank at the same point, once the reads landed everywhere: import the tail's state if every rank
        read every object intact, insert the path into L1 with the restored snapshot (not spare: a request asked for
        it) and, with `keep`, keep it locked for its request. Returns the tokens restored."""
        t0 = time.perf_counter()
        tail, lost, nbytes, t_read = job.result
        t_put = job.t_put
        intact = self._min_over_ranks("read", [int(i not in lost) for i in range(len(job.parts))])
        inserted = False
        try:
            if all(intact):
                st = self._staging
                aux = self._put_state((tail[0], self.store.segments(tail[0], st.base, tail[1])), job.slot)
                if self.device.type == "cuda":
                    self._staging_free = torch.cuda.Event()
                    self._staging_free.record()  # the next restore's reads wait for this import before refilling it
                ids, l1, r = job.ids, job.l1, job.r
                new_start = cache.insert(ids[:r], torch.cat([*job.resident, job.pages]),
                                         [Snapshot(job.slot, r, auxiliary=aux)], page_ids=[-1] * l1 + job.page_ids)
                if new_start > l1:  # another request brought part of the path back meanwhile: those pages are duplicates
                    self.kv_pool.free_ids(job.page_ids[:new_start - l1])
                inserted = True
            else:
                for (kind, ref, *_), ok in zip(job.parts, intact):
                    if not ok:
                        self._lost(kind, ref)
                self.restore_fails += 1
        finally:
            if not inserted:
                self.kv_pool.free_ids(job.page_ids)
                self.state_pool.free(job.slot)
            if inserted and keep and not job.orphan:
                self._held.append((job, self._steps))
            else:
                cache.unlock(job.lock)  # and, after a lost tail, prunes what has nothing to resume from now
        if not inserted:
            return 0
        total = time.perf_counter() - job.t0
        self.restores += 1
        self.restored_tokens += job.r - job.matched
        self.restored_kv_tokens += job.r - job.l1
        self.restore_seconds += total
        self.restore_bytes += nbytes
        self.restore_read_seconds += t_read
        self.restore_put_seconds += t_put
        if keep:
            self.restore_blocked_seconds += time.perf_counter() - t0 + job.t_put
        if self.trace:
            self._trace({"tokens": job.r - job.matched, "kv_tokens": job.r - job.l1, "bytes": nbytes, "objects": len(job.parts),
                         "read_ms": round(1000 * t_read, 1), "read_gbps": round(nbytes / max(t_read, 1e-9) / 1e9, 2),
                         "scatter_ms": round(1000 * t_put, 1), "total_ms": round(1000 * total, 1),
                         "e2e_gbps": round(nbytes / max(total, 1e-9) / 1e9, 2), "async": keep})
        return job.r - job.matched

    def _poll(self, cache: RadixCache) -> None:
        """Each engine step while a restore is in flight: finish it once its reads landed on every rank."""
        job = self._job
        if job is not None:
            done = job.future.done()  # before scattering: whatever it read by then gets scattered now
            self._scatter_landed(job)
            if self._min_over_ranks("restore", [int(done)])[0]:
                self._job = None
                job.result = job.future.result()
                self._finish(cache, job, keep=True)
        while self._held and self._steps - self._held[0][1] > HOLD_STEPS:  # its request never came for it (counted in
            job, _ = self._held.pop(0)  # steps, which every rank takes alike, so every rank lets go at the same one)
            cache.unlock(job.lock)

    def _join(self, cache: RadixCache) -> None:
        """Wait for the in-flight restore and finish it (every rank calls this at the same point)."""
        job, self._job = self._job, None
        self._drive(job)
        job.result = job.future.result()
        self._finish(cache, job, keep=True)

    def abort(self, cache: RadixCache) -> None:
        """Before a flush or at shutdown: wait for the in-flight restore's reads, return its pages and slot without
        inserting anything, and end every restore's lock. Every rank does the same, with no agreement needed."""
        job, self._job = self._job, None
        if job is not None:
            self._drive(job, scatter=False)
            job.future.result()
            self.kv_pool.free_ids(job.page_ids)
            self.state_pool.free(job.slot)
            cache.unlock(job.lock)
        while self._held:
            held, _ = self._held.pop()
            cache.unlock(held.lock)

    def _trace(self, row: dict) -> None:
        import json
        try:
            with open(self.store.base / "restores.jsonl", "a") as f:
                f.write(json.dumps({"t": round(time.time(), 3), **row}) + "\n")
        except OSError:
            pass

    def _plan_chunks(self, parts):
        """Where each object of a restore lands in the staging buffer: KV blocks
        at one fixed stride (so a single strided view covers a run of them), the
        tail after them, packed into chunks that each fit one half. Returns
        (sizes, stride, half, chunks of [(part index, offset within the half)]).
        Every ready ref records its file's size (the largest any rank wrote); the
        read of a file deleted behind the tree's back finds it gone and fails the restore."""
        sizes = [ref.nbytes for _, ref, *_ in parts]
        stride = max([_pad(n) for n in sizes[:-1]], default=0)
        widths = [stride] * (len(parts) - 1) + [_pad(sizes[-1])]
        half = max(self.chunk_bytes, *widths)
        chunks, cur, off = [], [], 0
        for i, w in enumerate(widths):
            if cur and off + w > half:
                chunks.append(cur)
                cur, off = [], 0
            cur.append((i, off))
            off += w
        chunks.append(cur)
        return sizes, stride, half, chunks

    def _kv_targets(self):
        kvp = self.kv_pool
        out = [(f"kv.{j}.{t}", pool) for j, lid in enumerate(kvp.layer_ids) for t, pool in (("k", kvp.k[lid]), ("v", kvp.v[lid]))]
        return out + [(f"paux.{name}", pool) for name, pool in kvp.page_aux.items()] + self.extra_pages

    def _valid_layout(self, header: Header) -> bool:
        """The store checks that the segments add up; this checks them against
        the pools' geometry before a scatter or state import trusts a segment."""
        rows = len(header.tokens)
        if header.start % self.block or (rows != self.block if header.kind == "kv" else rows >= self.block):
            return False
        sp = self.state_pool
        layout = (header.kind, rows, len(self.kv_pool.page_aux), len(sp.aux))
        if (expected := self._layouts.get(layout)) is None:
            expected = {name: (_dtype_name(t), [rows, *t.shape[1:]]) for name, t in self._kv_targets()} if rows else {}
            if header.kind == "tail":
                expected.update({name: (_dtype_name(t), [t.shape[0], *t.shape[2:]])
                                 for name, t in (("conv", sp.conv_all), ("ssm", sp.ssm_all))})
                expected.update({f"pool.{name}": (_dtype_name(t), list(t.shape[1:]))
                                 for name, t in sp.aux.items()})
            self._layouts[layout] = expected
        actual = {name: (dtype, shape) for name, dtype, shape, _ in header.segments if not name.startswith("aux.")}
        return actual == expected

    def _put_rows(self, got, parts, dst: torch.Tensor, stride: int) -> None:
        """The KV rows of every object into pages dst, read in place from the
        staging buffer: on the GPU in one launch (`_RowCopy`); on the CPU, whole
        blocks sharing one layout in one scatter per segment over a strided view
        of all of them, the rest object by object."""
        if self._copy is not None:
            self._copy.put_rows(self._staging.base.data_ptr(), got, parts, dst)
            return
        base, targets = self._staging.base, self._kv_targets()
        pieces = [(h, off, lo - start, hi - start) for (h, off), (_, _, start, lo, hi) in zip(got, parts)]
        cursor, i = 0, 0
        while i < len(pieces):
            h, off, a, b = pieces[i]
            j = i
            if h.kind == "kv" and a == 0 and b == self.block:  # a run of whole blocks with the same layout
                while (j + 1 < len(pieces) and pieces[j + 1][0].kind == "kv" and pieces[j + 1][2:] == (0, self.block)
                       and pieces[j + 1][1] - pieces[j][1] == stride and pieces[j + 1][0].segments == h.segments):
                    j += 1
            if j > i:
                n = j - i + 1
                rows = dst[cursor:cursor + n * self.block].view(n, self.block)
                specs = {name: (seg_off, nbytes) for name, _, _, seg_off, nbytes in h.layout()}
                for name, pool in targets:
                    seg_off, nbytes = specs[name]
                    row = nbytes // self.block
                    src = base.as_strided((n, self.block, row), (stride, row, 1), base.storage_offset() + off + seg_off)
                    pool.view(torch.uint8).view(pool.shape[0], -1)[rows] = src
                cursor += n * self.block
            else:
                if b > a:
                    views = self.store.segments(h, base, off)
                    for name, pool in targets:
                        pool.view(torch.uint8).index_copy_(0, dst[cursor:cursor + b - a], views[name].view(torch.uint8)[a:b])
                    cursor += b - a
            i = j + 1

    def _put_state(self, got, slot: int) -> dict:
        header, tensors = got
        sp = self.state_pool
        sp.conv_all[:, slot].copy_(tensors["conv"])
        sp.ssm_all[:, slot].copy_(tensors["ssm"])
        for name, t in sp.aux.items():
            t[slot].copy_(tensors[f"pool.{name}"])
        aux = {}
        for identity, (_, import_) in self.codecs.items():
            scalars = header.scalars.get(str(identity))
            if scalars is None:
                continue
            views, i = [], 0
            while f"aux.{identity}.{i}" in tensors:
                views.append(tensors[f"aux.{identity}.{i}"].to(self.device).clone())  # the staging buffer is reused
                i += 1
            value = import_((*scalars, *views), slot=slot)
            if value is not None:
                aux[identity] = value
        return aux

    # ---- startup and flush ----
    def rebuild(self, cache: RadixCache) -> int:
        """Rebuild SSD-only paths from the stored headers: list every tail
        whose chain of blocks is complete as a resumable path, and delete the
        blocks no tail reaches. Tensor parallel: list only the prefixes every
        rank stored, in the same order on every rank. Returns the number of
        tails listed."""
        blocks, tails = {}, []
        for header, size in self.store.scan():
            if not self._valid_layout(header):
                self.store.delete(header.kind, header.key)
            elif header.kind == "kv":
                blocks[header.key] = (header, size)
            else:
                tails.append((header, size))
        candidates = []
        for tail, size in tails:
            chain, parent = [], tail.parent
            while parent != self.store.root.hex():
                got = blocks.get(parent)
                if got is None:
                    break
                chain.append(got)
                parent = got[0].parent
            else:
                chain.reverse()
                ids = [t for h, _ in chain for t in h.tokens] + tail.tokens
                if [h.start for h, _ in chain] == list(range(0, len(chain) * self.block, self.block)) and tail.start == len(chain) * self.block:
                    candidates.append((_digest(ids), ids, chain, tail, size))
                    continue
            self.store.delete("tail", tail.key)  # no restore can use an incomplete or misaligned chain
        everywhere = set(self._held_by_all([c[0] for c in candidates]))
        for _, _, _, tail, _ in (c for c in candidates if c[0] not in everywhere):
            self.store.delete("tail", tail.key)  # another rank lacks it: no rank could restore it
        listing = sorted((c for c in candidates if c[0] in everywhere), key=lambda c: c[0])
        sizes = iter(self._largest([s for _, _, chain, _, size in listing for s in [*(s for _, s in chain), size]]))
        kept = {}  # block key -> size
        for _, ids, chain, tail, _ in listing:
            refs = {h.end: SsdRef(h.key, True, kept.setdefault(h.key, next(sizes))) for h, _ in chain}
            size = next(sizes)
            cache.insert_ssd_path(ids, refs, SsdRef(tail.key, True, size))
            self.used += size
        self.used += sum(kept.values())
        for key in blocks.keys() - kept.keys():
            self.store.delete("kv", key)
        self.prefixes = cache.ssd_prefixes()
        return len(listing)

    def flush(self) -> None:
        self.queue.clear()
        self.through.clear()
        self.store.flush()
        self.inflight.clear()
        self.doomed.clear()  # their files went with the old epoch; the same keys written again must stay
        self.store.completions()  # store.flush waited for the writers: every rank drops the same completions
        self.landed.clear()
        self.staged = self.used = self.prefixes = 0
        self._failed_tail = False

    def stats(self) -> dict:
        return {"prefixes": self.prefixes, "used_gib": round(self.used / GIB, 3), "budget_gib": round(self.budget / GIB, 1), "block_tokens": self.block, "queued_blocks": len(self.queue),
                "staged_mib": round(self.staged / MIB, 1), "backed_blocks": self.backed_blocks, "tails": self.tails, "tails_written_through": self.tails_through,
                "tails_through_dropped": self.through_dropped, "queued_tails": len(self.through),
                "forced_gathers": self.forced, "evict_dropped": self.evict_dropped, "restores": self.restores, "restore_fails": self.restore_fails,
                "restored_tokens": self.restored_tokens, "restored_kv_tokens": self.restored_kv_tokens, "waited_for_writes": self.waited, "deleted": self.deleted,
                "evict_ms_total": round(1000 * self.evict_seconds, 1),
                "evict_ms_max": round(1000 * self.evict_max, 1), "settle_ms_total": round(1000 * self.settle_seconds, 1),
                "settle_ms_max": round(1000 * self.settle_max, 1), "room_waits": self.room_waits,
                "room_wait_writes": self.room_wait_writes, "room_wait_ms_total": round(1000 * self.room_wait_seconds, 1),
                "room_wait_ms_max": round(1000 * self.room_wait_max, 1),
                "step_ms_total": round(1000 * self.step_seconds, 1), "step_ms_max": round(1000 * self.step_max, 1),
                "restore_ms_avg": round(1000 * self.restore_seconds / max(1, self.restores), 1),
                "async_restores": self.async_restores, "restore_wait_steps": self.restore_wait_steps,
                "restore_blocked_ms_total": round(1000 * self.restore_blocked_seconds, 1),
                "restore_gib": round(self.restore_bytes / GIB, 2),
                "restore_read_gbps": round(self.restore_bytes / max(self.restore_read_seconds, 1e-9) / 1e9, 2),
                "restore_scatter_ms_avg": round(1000 * self.restore_put_seconds / max(1, self.restores), 1),
                "restore_e2e_gbps": round(self.restore_bytes / max(self.restore_seconds, 1e-9) / 1e9, 2), "store": self.store.stats()}
