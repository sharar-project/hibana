"""SSD tier: decides what goes to the NVMe and brings it back.

The tiered radix tree (`radix.py`) is the only record of what is on disk;
`ssd_store.py` only stores objects. This module moves data between them:

- Backup. When a finished request's sequence of at least `min_tokens` enters
  the tree, `backup()` queues each of its full KV blocks (`store.block` tokens)
  not yet stored, and the tails at its resume points (the sequence's end and
  its prompt's end) for a write-through: a snapshot the GPU cache later drops
  for want of state slots stays restorable. `step()` (every engine loop
  iteration) gathers queued tails, then queued blocks, from the pool pages
  and slots into pinned host memory, up to `step_bytes` per step and
  `staging_bytes` in flight, and hands them to the store's writer threads.
  An object larger than that limit goes out alone. Gathers that eviction, a
  restore or shutdown need at once wait for room. `step()` skips a queued
  tail whose snapshot the GPU cache dropped first.
- Eviction. Before the tree frees an L1 leaf's pages it calls `evicting()`,
  which gathers every queued block touching the leaf and, for a leaf of at
  least `min_tokens` with a snapshot, writes its tail (its last partial block,
  the GDN snapshot at its end, the pool's per-slot aux and the drafter
  payload) unless a write-through already did. The tree then keeps the leaf
  SSD-only instead of deleting it.
- Restore. At admission `restore()` finds the deepest ready tail past the L1
  prefix (first finishing its writes if it was evicted moments ago) and
  streams the blocks past the KV still in L1, then the tail, through a host
  buffer the GPU reads in place, in two halves of `chunk_bytes`: a small
  thread pool reads one chunk with O_DIRECT into one half while the GPU
  scatters the previous chunk from the other straight into fresh pages and a
  slot. The engine waits for the restore: other requests do not run
  meanwhile. `restore()` then inserts the path back into L1. When the GPU
  cache dropped only a snapshot and kept its pages, the restore reads only the
  tail: its state goes into a slot and the pages stay where they are. The
  chunk size, not the restore, bounds the buffer, so the pool plan reserves a
  constant (`memory_bound`).
- Budget. Completed writes count against `budget_bytes`; over it, `step()`
  deletes the least recently used SSD-only leaves, then the stored copies of
  GPU-resident nodes.

Only the engine thread calls this module or touches the tree; the store's
writers report back through completions that `step()` drains.

Tensor parallel: each rank stores its own heads of every session on its own
Spark, and every rank's tree must take the same steps (the schedulers do).
The ranks settle what one rank cannot decide alone with a minimum over ranks:
a write counts as complete once it completed on every rank (and as failed if
it failed on any), a restore goes ahead only if every rank read every object
intact, and a restart lists only the sessions every rank stored. On one rank,
each agreement returns that rank's own answer.
"""

from __future__ import annotations

import hashlib
import time
from array import array
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import torch

from sharar.cache.radix import Node, RadixCache, SsdHit, SsdRef
from sharar.cache.ssd_store import Header, SsdStore, _pad, block_hash, tail_hash
from sharar.distributed import all_reduce_min, common_values

MIB = 2**20
GIB = 2**30
STAGING_BYTES = 512 * MIB  # host copies of backups in flight to the writers
CHUNK_BYTES = 256 * MIB  # each half of the restore staging buffer


class Staging:
    """One reusable host buffer that restores read into with O_DIRECT and the
    GPU reads in place. On GB10 (integrated GPU, unified memory) it registers
    its mapping with CUDA, and the device pointer is the host address: the
    scatter kernels read it directly, with no host memcpy and no host-to-device
    copy. Without CUDA it is a plain mapping."""

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
            raise RuntimeError("cudaHostRegister failed for the SSD staging buffer")
        self.rt, self.ptr = rt, ptr
        dev = ctypes.c_void_p()
        if rt.cudaHostGetDevicePointer(ctypes.byref(dev), ctypes.c_void_p(ptr), ctypes.c_uint(0)) != 0:
            self.close()
            raise RuntimeError("cudaHostGetDevicePointer failed for the SSD staging buffer")
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
    """A stored session's tokens as one number every rank computes alike (stored keys are per rank)."""
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
        self.sessions = 0  # written tails: sessions a request can resume from the SSD
        self._failed_tail = False
        self.backed_blocks = self.tails = self.tails_through = self.through_dropped = self.restores = self.restore_fails = self.forced = self.deleted = 0
        self.restored_tokens = self.restored_kv_tokens = self.waited = self.restore_bytes = 0
        self.restore_read_seconds = self.restore_put_seconds = 0.0
        self.restore_seconds = 0.0
        self.evict_seconds = self.evict_max = self.settle_seconds = self.settle_max = 0.0  # the engine blocked on backups
        self._pool = ThreadPoolExecutor(8, thread_name_prefix="sharar-ssd-read")  # parallel O_DIRECT reads keep the drive's queue full
        self.extra_pages = list(extra_pages)  # (name, per-page tensor) beside the KV, e.g. the MTP draft layer's KV
        self._row = kv_pool.row_bytes + sum(t[0].numel() * t.element_size() for _, t in self.extra_pages)
        # two halves, each chunk_bytes or one whole tail (a block of KV, the slot's state and a drafter payload), reused by
        # every restore; sized here, never grown mid-restore, where a failure on one rank would leave the other waiting
        tail = self.block * self._row + state_pool.bytes_per_slot + 32 * MIB
        self._staging = Staging(2 * max(self.chunk_bytes, _pad(tail)), device)
        self._layouts: dict = {}  # expected segment layouts by (kind, rows, aux counts); the pools' geometry is fixed

    @staticmethod
    def memory_bound(target, block: int, extra_page_bytes: int = 0, staging_bytes: int = STAGING_BYTES,
                     chunk_bytes: int = CHUNK_BYTES) -> int:
        """Pinned host memory the tier can hold, for the pool plan: the restore
        buffer's two halves (each chunk_bytes, or one whole tail object if that
        is larger: a block of KV and a snapshot), the backup staging (at least
        one whole object) and the snapshot a tail gather copies. The context
        length does not change it: restores stream through the halves chunk by
        chunk."""
        tail = block * (target.kv_bytes_per_token + extra_page_bytes) + target.state_bytes_per_slot + 32 * MIB  # + header, draft state
        return 2 * max(chunk_bytes, tail) + max(staging_bytes, tail) + target.state_bytes_per_slot

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
        """Queue the full blocks of a sequence that just entered the tree, and the tails at `resume_points`
        (lengths where the tree holds a snapshot of it: its end, its prompt's end) for a write-through. The tree
        drops an interior snapshot when state slots run out while pages remain, which eviction never sees; a
        write-through keeps it restorable. Only these two: every chunk checkpoint would cost a snapshot each.
        Queued, not copied now: a snapshot is large, and `step()` keeps the copies within the staging limit."""
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
            if (end >= self.min_tokens and node is not None and node.snapshot_slot is not None
                    and (node.tail is None or node.tail.nbytes < 0)):
                self.through.append((node, end))

    def _pending_ref(self, kind: str, key: str) -> SsdRef | None:
        """Reattach a write a removed session left in flight. One key has one
        write and one staging allocation, even if the session comes back
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
        (keys are content-addressed), so a later turn of a session hashes only its new blocks."""
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

    def _gather(self, anchor: Node, lo: int, hi: int) -> tuple[list[int], list[tuple[str, torch.Tensor]]] | None:
        span = self._span(anchor, lo, hi)
        if span is None:
            return None
        tokens, pages = span
        idx = pages.to(torch.int64)
        cuda = self.device.type == "cuda"
        out = []
        for name, pool in self._kv_targets():
            src = pool.view(torch.uint8) if name.startswith("kv.") else pool  # gather FP8 K/V as raw bytes
            out.append((name, _to_host(src.index_select(0, idx), cuda).view(pool.dtype)))
        return tokens, out

    def _submit(self, kind: str, key: bytes, parent: bytes, start: int, tokens: list[int], segs, ref: SsdRef, scalars=None) -> None:
        ready = None
        if self.device.type == "cuda":
            ready = torch.cuda.Event(blocking=True)  # writers wait on it: sleep, don't spin a core each
            ready.record()
        spec = [[name, _dtype_name(t), list(t.shape), t.numel() * t.element_size()] for name, t in segs]
        # the pinned memory the copies hold: the caching host allocator rounds each up to a power of two
        pinned = sum(_pinned(s[3]) for s in spec) if self.device.type == "cuda" else sum(s[3] for s in spec)
        header = Header(kind, key.hex(), parent.hex(), start, tokens, spec, scalars or {})
        self.inflight[(kind, key.hex())] = (ref, pinned)
        self.staged += pinned
        self.store.write(header, [t for _, t in segs], ready)

    def _room(self, nbytes: int, wait: bool) -> bool:
        """Whether a copy of `nbytes` fits within the staging limit (asked before
        copying). An object larger than the limit can go alone; eviction,
        restores and shutdown wait for older copies instead of exceeding it."""
        if self.staged and self.staged + nbytes > self.staging_bytes:
            if not wait:
                return False
            self.store.wait()
            self._finish_writes()  # every rank has finished its writers before agreeing on completions
            assert self.staged == 0
        return True

    def _gather_block(self, item, wait: bool = False) -> bool:
        anchor, end, h, parent = item
        owner = self._owner(anchor, end)
        ref = owner.blocks.get(end)
        if ref is None or ref.key != h.hex() or ref.nbytes < 0 or ref.ready or ("kv", ref.key) in self.inflight:
            return True
        if not self._room(self.block * self._row, wait):
            return False
        got = self._gather(anchor, end - self.block, end)
        if got is None:  # part of the block left L1 while it sat in the queue: drop its ref
            del owner.blocks[end]
            return True
        tokens, segs = got
        self._submit("kv", h, parent, end - self.block, tokens, segs, ref)
        self.backed_blocks += 1
        return True

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
    def evicting(self, cache: RadixCache, leaf: Node) -> bool:
        """The tree calls this before it frees an L1 leaf's pages. Gathers the
        queued blocks with a token in the leaf while its pages exist, and writes
        its tail; True keeps the leaf SSD-only."""
        t0 = time.perf_counter()
        end = leaf.depth
        missing, node = end // self.block, leaf
        while node.parent is not None:
            missing -= sum(ref.nbytes >= 0 for ref in node.blocks.values())
            node = node.parent
        if missing > 0 and end >= self.min_tokens:  # the disk budget may have dropped copies whose pages stayed in L1
            self._queue_blocks(leaf, self._span(leaf, 0, end)[0])  # the leaf's whole path is in L1
        start, pending, keep = end - len(leaf.key), [], deque()
        for item in self.queue:  # gather the blocks with a token in this leaf now, while its pages exist
            (pending if self._touches(item, leaf, start, end) else keep).append(item)
        self.queue = keep
        for item in pending:
            self._gather_block(item, wait=True)
            self.forced += 1
        written = leaf.tail is not None and leaf.tail.nbytes >= 0  # written through already: its snapshot may be gone
        keep = written or (leaf.snapshot_slot is not None and end >= self.min_tokens and self._write_tail(leaf, end, wait=True))
        if not keep and not leaf.children:
            self._discard_node(leaf)
        spent = time.perf_counter() - t0  # the gathers above, and waits for staging room behind the writers
        self.evict_seconds += spent
        self.evict_max = max(self.evict_max, spent)
        return keep

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
            return True
        sp, slot, cuda = self.state_pool, leaf.snapshot_slot, self.device.type == "cuda"
        state = [("conv", sp.conv_all[:, slot]), ("ssm", sp.ssm_all[:, slot])]
        state += [(f"pool.{name}", t[slot]) for name, t in sp.aux.items()]
        scalars = {}
        for identity, value in leaf.auxiliary.items():
            codec = self.codecs.get(identity)
            payload = codec[0](value) if codec is not None else None
            if payload is None:
                continue
            scalars[str(identity)] = [x for x in payload if not torch.is_tensor(x)]
            state += [(f"aux.{identity}.{i}", x) for i, x in enumerate(t for t in payload if torch.is_tensor(t))]
        nbytes = (end - whole) * self._row + sum(t.numel() * t.element_size() for _, t in state)
        if not self._room(nbytes, wait):
            return False
        # `_queue_blocks` queues the full blocks; gather the partial block only once the whole tail has room.
        got = self._gather(leaf, whole, end) if end > whole else (tokens, [])
        if got is None:
            return False
        tokens, segs = got
        segs += [(name, _to_host(t, cuda)) for name, t in state]
        leaf.tail = SsdRef(key.hex())
        self._submit("tail", key, last, whole, tokens, segs, leaf.tail, scalars)
        self.tails += 1
        return True

    def _delete(self, kind: str, ref: SsdRef) -> None:
        self.store.delete(kind, ref.key)
        self.used -= ref.nbytes
        self.sessions -= kind == "tail" and ref.ready
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
        written = self._min_over_ranks("written", [int(ok) for _, _, ok, _ in got])
        sizes = self._largest([nbytes for *_, nbytes in got])
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
            if (kind, key) in self.doomed:
                self.doomed.discard((kind, key))
                if ok:
                    self.store.delete(kind, key)
                    self.deleted += 1
                continue
            ref.ready, ref.nbytes = ok, (nbytes if ok else -1)
            self.used += nbytes if ok else 0
            self.sessions += kind == "tail" and ok
            self._failed_tail |= kind == "tail" and not ok

    def _trim(self, cache: RadixCache) -> None:
        if self._failed_tail:  # a full disk: no restore can use an SSD-only leaf without a tail
            while (dead := cache.ssd_candidate(unusable=True)) is not None:
                self._drop_leaf(cache, dead)
            self._failed_tail = False
        while self.used > self.budget and (victim := cache.ssd_candidate()) is not None:
            self._drop_leaf(cache, victim)
        while self.used > self.budget and (victim := cache.ssd_candidate(resident=True)) is not None:
            self._discard_node(victim)
            victim.blocks.clear()
            victim.tail = None

    def _drop_leaf(self, cache: RadixCache, node: Node) -> None:
        self._discard_node(node)
        cache.remove_ssd_leaf(node)

    def _gather_queued(self, wait: bool = False) -> None:
        spent = 0
        while self.through and (wait or spent < self.step_bytes):  # tails first: a dropped snapshot is gone
            node, end = self.through[0]
            if node.tail is not None and node.tail.nbytes >= 0:  # eviction wrote it meanwhile, or backup queued it twice
                self.through.popleft()
                continue
            if node.snapshot_slot is None:  # the GPU cache dropped it before there was room
                self.through.popleft()
                self.through_dropped += 1
                continue
            if not self._write_tail(node, end, wait=wait):
                break  # leave it queued until there is staging room
            self.through.popleft()
            self.tails_through += 1
            spent += self.inflight[("tail", node.tail.key)][1]
        while self.queue and (wait or spent < self.step_bytes):
            if not self._gather_block(self.queue[0], wait=wait):
                break
            self.queue.popleft()
            spent += self.block * self._row

    def step(self, cache: RadixCache) -> None:
        self._finish_writes()
        self._trim(cache)
        self._gather_queued()

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
        """Bring the deepest SSD resume point for ids back into L1, if the pool can also hold the rest of ids and
        `reserve` more pages: otherwise the cache would only evict what it restored and read it again. Returns the
        tokens restored (0 when there is nothing worth restoring)."""
        m = cache.match(ids, touch=False)
        hit = self._settle_for(cache, ids, m.length)
        if hit is None or hit.length - m.length < self.min_hit:
            return 0
        t0 = time.perf_counter()
        l1, r = hit.l1_end, hit.length  # L1 still holds the KV before l1 (at least the matched prefix): read only [l1, r)
        whole = r // self.block * self.block
        resident = [n for n in hit.path if n.in_l1]  # a prefix of the path
        held = resident[-1] if resident else None  # locked, with its ancestors: the eviction below keeps their pages
        cache.lock(held, 1)
        pages = page_ids = slot = None
        try:
            need = len(ids) - l1 + reserve  # the restored KV, the rest of ids and the reserve
            cache.evict(need_pages=need, need_slots=2)  # one slot for the restored path, one spare for the request itself
            if self.kv_pool.free_count < need or self.state_pool.free_count < 2:
                return 0
            parts = [("kv", ref, end - self.block, max(end - self.block, l1), min(end, whole)) for end, ref in hit.blocks]
            parts = [p for p in parts if p[3] < p[4]] + [("tail", hit.tail, whole, max(whole, l1), r)]
            plan = self._plan_chunks(parts)
            pages, page_ids = self.kv_pool.alloc_with_ids(r - l1)
            slot = self.state_pool.alloc()
            aux, nbytes, t_read, t_put = self._read_and_put(parts, plan, ids, pages.to(torch.int64), slot)
            if aux is None:  # an object turned out missing, corrupt or a collision: insert nothing
                self.restore_fails += 1
                return 0
            prefix = [n.pages for n in resident]
            new_start = cache.insert(ids[:r], torch.cat([*prefix, pages]), {r: slot}, {r: aux}, page_ids=[-1] * l1 + page_ids)
            assert new_start == l1  # every path node past l1 is SSD-only, so the insert takes all the restored pages
            pages = slot = None  # owned by the cache now
        finally:
            if pages is not None:  # not inserted: failed or raised
                self.kv_pool.free_ids(page_ids)
            if slot is not None:
                self.state_pool.free(slot)
            cache.lock(held, -1)
        total = time.perf_counter() - t0
        self.restores += 1
        self.restored_tokens += r - m.length
        self.restored_kv_tokens += r - l1
        self.restore_seconds += total
        self.restore_bytes += nbytes
        self.restore_read_seconds += t_read
        self.restore_put_seconds += t_put
        if self.trace:
            self._trace({"tokens": r - m.length, "kv_tokens": r - l1, "bytes": nbytes, "objects": len(parts), "read_ms": round(1000 * t_read, 1),
                         "read_gbps": round(nbytes / max(t_read, 1e-9) / 1e9, 2), "scatter_ms": round(1000 * t_put, 1),
                         "total_ms": round(1000 * total, 1), "e2e_gbps": round(nbytes / max(total, 1e-9) / 1e9, 2)})
        return r - m.length

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

    def _read_and_put(self, parts, plan, ids: list[int], dst: torch.Tensor, slot: int):
        """Read the restore's objects chunk by chunk with O_DIRECT into the two
        halves of the staging buffer, in parallel within a chunk, and scatter
        each chunk into pages dst (the tail's state into slot) while the next
        chunk reads into the other half. Returns (auxiliary dict, bytes read,
        seconds waiting on reads, seconds scattering), or a None auxiliary if
        an object is missing, corrupt or a collision on any rank: the ranks
        agree before importing the tail's state, so no rank's drafter keeps
        features of a restore another rank failed."""
        sizes, stride, half, chunks = plan
        if 2 * half > self._staging.nbytes:  # an object larger than a half: every rank plans from the same sizes, so all skip
            return None, 0, 0.0, 0.0
        st, cuda = self._staging, self.device.type == "cuda"
        half = st.nbytes // 2
        freed = [torch.cuda.Event(), torch.cuda.Event()] if cuda else None  # per half: its last scatter has read it

        def submit(chunk, h):
            def one(i, off):
                kind, ref, *_ = parts[i]
                got = self.store.read_into(kind, ref.key, st.view[off:off + _pad(sizes[i])])
                return None if got is None else (got[0], off + got[1])
            return [(i, self._pool.submit(one, i, h * half + rel)) for i, rel in chunk]

        tail, lost, cursor, t_read, t_put = None, set(), 0, 0.0, 0.0
        inflight = {0: submit(chunks[0], 0)}
        for k in range(len(chunks)):
            h = k % 2
            if k + 1 < len(chunks):  # keep the drive busy: the next chunk reads into the other half as soon as
                if cuda:  # that half's last scatter is done, before waiting on this chunk
                    freed[1 - h].synchronize()
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
            t = time.perf_counter()
            chunk_parts = [parts[i] for i, _ in got]
            rows = sum(hi - lo for _, _, _, lo, hi in chunk_parts)
            self._put_rows([g for _, g in got], chunk_parts, dst[cursor:cursor + rows], stride)
            cursor += rows
            if got[-1][0] == len(parts) - 1:  # the tail, the last object: its state goes in once every rank has it all
                tail = got[-1][1]
            if cuda:
                freed[h].record()
            t_put += time.perf_counter() - t
        intact = self._min_over_ranks("read", [int(i not in lost) for i in range(len(parts))])
        aux = None
        if all(intact):
            t = time.perf_counter()
            aux = self._put_state((tail[0], self.store.segments(tail[0], st.base, tail[1])), slot)
            t_put += time.perf_counter() - t
        else:
            for (kind, ref, *_), ok in zip(parts, intact):
                if not ok:
                    self._lost(kind, ref)
        if cuda:
            t = time.perf_counter()
            torch.cuda.current_stream().synchronize()  # the kernels read the staging buffer, which the next restore refills
            t_put += time.perf_counter() - t
        return aux, sum(sizes) if aux is not None else 0, t_read, t_put

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
        staging buffer. Whole blocks sharing one layout go in one scatter per
        segment over a strided view of all of them; the rest object by object."""
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
        blocks no tail reaches. Tensor parallel: list only the sessions every
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
        self.sessions = cache.ssd_sessions()
        return len(listing)

    def flush(self) -> None:
        self.queue.clear()
        self.through.clear()
        self.store.flush()
        self.inflight.clear()
        self.doomed.clear()  # their files went with the old epoch; the same keys written again must stay
        self.store.completions()  # store.flush waited for the writers: every rank drops the same completions
        self.landed.clear()
        self.staged = self.used = self.sessions = 0
        self._failed_tail = False

    def stats(self) -> dict:
        return {"sessions": self.sessions, "used_gib": round(self.used / GIB, 3), "budget_gib": round(self.budget / GIB, 1), "block_tokens": self.block, "queued_blocks": len(self.queue),
                "staged_mib": round(self.staged / MIB, 1), "backed_blocks": self.backed_blocks, "tails": self.tails, "tails_written_through": self.tails_through,
                "tails_through_dropped": self.through_dropped, "queued_tails": len(self.through),
                "forced_gathers": self.forced, "restores": self.restores, "restore_fails": self.restore_fails,
                "restored_tokens": self.restored_tokens, "restored_kv_tokens": self.restored_kv_tokens, "waited_for_writes": self.waited, "deleted": self.deleted,
                "evict_ms_total": round(1000 * self.evict_seconds, 1),
                "evict_ms_max": round(1000 * self.evict_max, 1), "settle_ms_total": round(1000 * self.settle_seconds, 1),
                "settle_ms_max": round(1000 * self.settle_max, 1),
                "restore_ms_avg": round(1000 * self.restore_seconds / max(1, self.restores), 1),
                "restore_gib": round(self.restore_bytes / GIB, 2),
                "restore_read_gbps": round(self.restore_bytes / max(self.restore_read_seconds, 1e-9) / 1e9, 2),
                "restore_scatter_ms_avg": round(1000 * self.restore_put_seconds / max(1, self.restores), 1),
                "restore_e2e_gbps": round(self.restore_bytes / max(self.restore_seconds, 1e-9) / 1e9, 2), "store": self.store.stats()}
