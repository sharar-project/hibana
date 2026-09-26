"""Cold session tier in GB10 unified memory.

When the radix cache evicts a leaf (its KV pages, its GDN snapshot slot and
the draft's auxiliary state) the data used to be lost, and the next turn of
that session re-prefilled everything: 14 s for a 30K-token history. On GB10
the GPU and CPU share one DRAM, and a CUDA-managed allocation is readable by
GPU kernels at about 100 GB/s, so an evicted leaf can be *parked* instead:
its pages are gathered straight into a managed arena (about 12 ms for 30K
tokens) and restored into fresh pool pages when the session comes back.

Measured on this GB10 (2026-09-07, `docs/cold-tier.md`): device->managed
100 GB/s, managed->device 108 GB/s, page gather into managed 93 GB/s, but
attention *over* managed KV runs 3x slower than over device pages, which is
why the tier only parks and restores; it never serves attention. The arena is
touched once at startup (first touch costs ~330 ms per GiB) and then reused.

The tier owns nothing of the pools: park() copies out of pages/slots the
caller is about to free, restore() copies into pages/slots the caller has
just allocated. All copies are kernels on the engine's stream, so they are
ordered with respect to later reuse of the freed pages.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

GIB = 2**30


class _ManagedBlock:
    """cudaMallocManaged through the CUDA runtime torch ships, exposed with the
    CUDA array interface so torch can wrap it without copying (no cupy)."""

    def __init__(self, nbytes: int):
        import ctypes
        import glob
        import os

        candidates = glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib", "libcudart.so*"))
        candidates += glob.glob("/usr/local/cuda/lib64/libcudart.so*") + ["libcudart.so"]
        last = None
        for path in candidates:
            try:
                self.rt = ctypes.CDLL(path)
                break
            except OSError as exc:
                last = exc
        else:
            raise RuntimeError(f"CUDA runtime library not found for managed memory: {last}")
        ptr = ctypes.c_void_p()
        rc = self.rt.cudaMallocManaged(ctypes.byref(ptr), ctypes.c_size_t(nbytes), ctypes.c_uint(1))  # cudaMemAttachGlobal
        if rc != 0:
            raise MemoryError(f"cudaMallocManaged({nbytes}) failed with CUDA error {rc}")
        self.ptr, self.nbytes = ptr, nbytes
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (ptr.value, False), "version": 3, "strides": None}

    def __del__(self):
        try:
            self.rt.cudaFree(self.ptr)
        except Exception:
            pass


class ManagedArena:
    """CUDA-managed memory in slabs, allocated on demand up to a budget and
    released when empty. Chunks never span slabs. A slab is prefetched to the
    GPU when created (1 ms; a fill kernel instead page-faults the whole slab
    and leaves it reading 5x slower, measured on this GB10)."""

    SLAB = 2 * GIB

    def __init__(self, nbytes: int, device: torch.device):
        self.nbytes = int(nbytes)
        self.device = device
        self.slabs: list[dict] = []  # {"block", "buffer", "used": [(offset, size)]}
        self.floor_bytes = 16 * GIB  # keep the host guard's floor plus a margin available

    @property
    def slab_bytes(self) -> int:
        return len(self.slabs) * self.SLAB

    @property
    def used_bytes(self) -> int:
        return sum(s for slab in self.slabs for _, s in slab["used"])

    def _new_slab(self) -> dict | None:
        if self.slab_bytes + self.SLAB > max(self.nbytes, self.SLAB):
            return None
        from sharar.cache.ledger import mem_available_bytes

        if mem_available_bytes() - self.SLAB < self.floor_bytes + 4 * GIB:
            return None  # the host is tight: do not grow the tier
        block = _ManagedBlock(self.SLAB)
        block.rt.cudaMemPrefetchAsync(block.ptr, ctypes_size(self.SLAB), ctypes_int(0), ctypes_ptr(0))
        slab = {"block": block, "buffer": torch.as_tensor(block, device=self.device), "used": []}
        self.slabs.append(slab)
        return slab

    def alloc(self, size: int) -> tuple[int, int] | None:
        """Returns (slab index, offset) or None when nothing fits."""
        size = (size + 255) // 256 * 256
        if size > self.SLAB:
            return None
        for i, slab in enumerate(self.slabs):
            cursor = 0
            for j, (off, sz) in enumerate(slab["used"]):
                if off - cursor >= size:
                    slab["used"].insert(j, (cursor, size))
                    return i, cursor
                cursor = off + sz
            if self.SLAB - cursor >= size:
                slab["used"].append((cursor, size))
                return i, cursor
        slab = self._new_slab()
        if slab is None:
            return None
        slab["used"].append((0, size))
        return len(self.slabs) - 1, 0

    def free(self, handle: tuple[int, int]) -> None:
        i, offset = handle
        slab = self.slabs[i]
        slab["used"] = [(o, s) for o, s in slab["used"] if o != offset]

    def release_empty(self) -> int:
        """Drop empty slabs (except the first) so the host gets the memory back."""
        keep, released = [], 0
        for i, slab in enumerate(self.slabs):
            if slab["used"] or i == 0:
                keep.append(slab)
            else:
                released += 1
        # handles are (slab index, offset): renumbering is only safe when the
        # dropped slabs are the trailing ones, so drop only those.
        while len(self.slabs) > 1 and not self.slabs[-1]["used"]:
            self.slabs.pop()
        return released

    def view(self, handle: tuple[int, int], nbytes: int, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
        i, offset = handle
        return self.slabs[i]["buffer"][offset : offset + nbytes].view(dtype).view(*shape)


def ctypes_size(n):
    import ctypes
    return ctypes.c_size_t(n)


def ctypes_int(n):
    import ctypes
    return ctypes.c_int(n)


def ctypes_ptr(n):
    import ctypes
    return ctypes.c_void_p(n)


@dataclass
class Parked:
    key: list[int]  # full token prefix from the root
    own: int  # tokens owned by the parked leaf (the last `own` of key)
    handles: list[tuple[int, int]]  # arena chunks: the KV chunks, then the state chunk
    nbytes: int
    chunks: list[tuple[int, torch.Tensor, dict]]  # (first token, [layers, 2, n, ...] KV view, {page aux name: [n, ...] view}), each within one slab
    conv: torch.Tensor | None
    ssm: torch.Tensor | None
    auxiliary: dict = field(default_factory=dict)  # identity -> (scalars, views); "__pool__" -> ((), [(name, view)])
    ready: torch.cuda.Event | None = None
    last_access: float = field(default_factory=time.monotonic)
    on_disk: bool = False  # a complete copy exists in the session store


class ColdTier:
    def __init__(self, kv_pool, state_pool, budget_bytes: int, device: torch.device, log=None, disk=None):
        self.kv_pool, self.state_pool, self.device = kv_pool, state_pool, device
        self.arena = ManagedArena(budget_bytes, device)
        self.disk = disk  # optional SessionDisk: parked entries are written through and restored after drops or restarts
        self.entries: list[Parked] = []
        self.codecs: dict = {}  # identity -> (export, import)
        self.parks = self.restores = self.drops = self.rejects = self.loads = 0
        self.load_seconds = 0.0
        self.last_reject = None
        self.park_seconds = self.restore_seconds = 0.0
        if log:
            log(f"  cold tier: up to {budget_bytes / GIB:.1f} GiB of managed memory for evicted sessions (2 GiB slabs on demand)")

    # ---- sizes ----
    def _kv_only_row_bytes(self) -> int:
        k = self.kv_pool.k[self.kv_pool.layer_ids[0]]
        return k[0].numel() * k.element_size() * 2 * len(self.kv_pool.layer_ids)

    def _kv_row_bytes(self) -> int:
        """Bytes one token's page holds: K/V of every layer plus the page aux rows."""
        return self._kv_only_row_bytes() + sum(t[0].numel() * t.element_size() for t in self.kv_pool.page_aux.values())

    def _chunk_views(self, handle: tuple[int, int], n: int) -> tuple[torch.Tensor, dict]:
        """A chunk's KV view followed by one view per page aux tensor."""
        kvp = self.kv_pool
        k0 = kvp.k[kvp.layer_ids[0]]
        kv_bytes = n * self._kv_only_row_bytes()
        kv = self.arena.view(handle, kv_bytes, k0.dtype, (len(kvp.layer_ids), 2, n, *k0.shape[1:]))
        slab_i, offset = handle
        offset += kv_bytes
        aux = {}
        for name, t in kvp.page_aux.items():
            nbytes = n * t[0].numel() * t.element_size()
            aux[name] = self.arena.view((slab_i, offset), nbytes, t.dtype, (n, *t.shape[1:]))
            offset += nbytes
        return kv, aux

    def _state_bytes(self) -> int:
        sp = self.state_pool
        return (sp.conv_all[:, 0].numel() * sp.conv_all.element_size() + sp.ssm_all[:, 0].numel() * sp.ssm_all.element_size()
                + sum(t[0].numel() * t.element_size() for t in getattr(sp, "aux", {}).values()))

    def register_codec(self, identity, export, import_) -> None:
        """Import receives ``slot=`` naming the newly owned recurrent slot."""
        self.codecs[identity] = (export, import_)

    # ---- park ----
    def park(self, key: list[int], pages: torch.Tensor, slot: int | None, auxiliary: dict | None) -> bool:
        """Copy a whole path (root to the evicted leaf: `key` and the pages of
        every node on it) out before the leaf's pages/slot are freed. The
        entry is self-contained, so parents evicted later cannot strand it.
        Returns False (parks nothing) when it does not fit even after dropping
        older entries, or when a parked entry already covers this key."""
        own = int(pages.shape[0])
        if own == 0 or slot is None or own != len(key):
            self.rejects += 1
            self.last_reject = f"shape: pages {own}, key {len(key)}, slot {slot}"
            return False
        if any(len(e.key) >= own and e.key[:own] == key for e in self.entries):
            self.rejects += 1  # already covered by a parked descendant
            self.last_reject = "covered"
            return False
        # Auxiliary payloads: (identity, tuple of scalars, list of device tensors). Copied into the arena too.
        aux_items = []
        for identity, value in (auxiliary or {}).items():
            codec = self.codecs.get(identity)
            if codec is None:
                continue
            payload = codec[0](value)
            if payload is None:
                continue
            scalars = tuple(x for x in payload if not torch.is_tensor(x))
            tensors = [x for x in payload if torch.is_tensor(x)]
            aux_items.append((identity, scalars, tensors))
        aux_bytes = sum((x.numel() * x.element_size() + 255) // 256 * 256 for _, _, ts in aux_items for x in ts)
        row = self._kv_row_bytes()
        state_bytes = self._state_bytes() + aux_bytes
        nbytes = own * row + state_bytes
        if nbytes > self.arena.nbytes or state_bytes > self.arena.SLAB:
            self.rejects += 1
            self.last_reject = f"too large: {nbytes / GIB:.2f} GiB"
            return False
        per_chunk = self.arena.SLAB // row  # a long session spans slabs in token chunks
        sizes = [min(per_chunk, own - s) for s in range(0, own, per_chunk)]
        handles = self._alloc_all([n * row for n in sizes] + [state_bytes])
        if handles is None:
            self.rejects += 1  # nothing left to drop and the host or the budget refuses a slab
            self.last_reject = f"no slab: {nbytes / GIB:.2f} GiB, slabs {len(self.arena.slabs)}"
            return False
        t0 = time.perf_counter()
        kvp = self.kv_pool
        idx = pages.to(torch.int64)
        chunks, start = [], 0
        for handle, n in zip(handles, sizes):
            kv, aux = self._chunk_views(handle, n)
            for j, lid in enumerate(kvp.layer_ids):
                torch.index_select(kvp.k[lid], 0, idx[start:start + n], out=kv[j, 0])
                torch.index_select(kvp.v[lid], 0, idx[start:start + n], out=kv[j, 1])
            for name, view in aux.items():
                torch.index_select(kvp.page_aux[name], 0, idx[start:start + n], out=view)
            chunks.append((start, kv, aux))
            start += n
        conv, ssm, aux = self._state_views(handles[-1], aux_items)
        sp = self.state_pool
        conv.copy_(sp.conv_all[:, slot])
        ssm.copy_(sp.ssm_all[:, slot])
        for name, v in aux.get("__pool__", ((), []))[1]:
            v.copy_(sp.aux[name][slot])
        for identity, scalars, tensors in aux_items:
            for x, v in zip(tensors, aux[identity][1]):
                v.copy_(x)
        ev = torch.cuda.Event()
        ev.record()
        entry = Parked(list(key), own, handles, nbytes, chunks, conv, ssm, aux, ev)
        self.entries.append(entry)
        self.parks += 1
        self.park_seconds += time.perf_counter() - t0
        if self.disk is not None:
            self.disk.submit(entry)  # written through in the background; drops wait for it
        return True

    def _state_views(self, handle: tuple[int, int], aux_items) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Views for the snapshot, the pool's per-slot auxiliary state and the
        codec payload tensors, laid out in one arena chunk."""
        slab_i, base = handle
        sp = self.state_pool
        conv_n = sp.conv_all[:, 0].numel() * sp.conv_all.element_size()
        conv = self.arena.view((slab_i, base), conv_n, sp.conv_all.dtype, tuple(sp.conv_all[:, 0].shape))
        ssm_n = sp.ssm_all[:, 0].numel() * sp.ssm_all.element_size()
        ssm = self.arena.view((slab_i, base + conv_n), ssm_n, sp.ssm_all.dtype, tuple(sp.ssm_all[:, 0].shape))
        cursor = base + conv_n + ssm_n
        pool_aux = []
        for name, t in getattr(sp, "aux", {}).items():
            n = t[0].numel() * t.element_size()
            pool_aux.append((name, self.arena.view((slab_i, cursor), n, t.dtype, tuple(t[0].shape))))
            cursor += n
        aux = {"__pool__": ((), pool_aux)} if pool_aux else {}
        for identity, scalars, tensors in aux_items:
            views = []
            for x in tensors:
                n = x.numel() * x.element_size()
                views.append(self.arena.view((slab_i, cursor), n, x.dtype, tuple(x.shape)))
                cursor += (n + 255) // 256 * 256
            aux[identity] = (scalars, views)
        return conv, ssm, aux

    def _alloc_all(self, sizes: list[int]) -> list[tuple[int, int]] | None:
        """Every chunk or none, dropping least recently used entries to make room."""
        while True:
            handles = []
            for size in sizes:
                handle = self.arena.alloc(size)
                if handle is None:
                    break
                handles.append(handle)
            if len(handles) == len(sizes):
                return handles
            for handle in handles:
                self.arena.free(handle)
            if not self.entries:
                return None
            self._drop_lru()

    def _drop_lru(self) -> None:
        if not self.entries:
            raise MemoryError("cold tier arena cannot fit this entry")
        victim = min(self.entries, key=lambda e: e.last_access)
        self._drop(victim)

    def discard(self, entry) -> None:
        if isinstance(entry, Parked):
            self._drop(entry)

    def clear(self) -> None:
        for entry in list(self.entries):
            self._drop(entry)
        if self.disk is not None:
            self.disk.clear()  # a flush invalidates stored sessions too

    def _drop(self, entry: Parked) -> None:
        if entry.ready is not None:
            entry.ready.synchronize()  # the arena bytes are reused: the park copy must have landed
        if self.disk is not None:
            self.disk.wait(entry)  # and the write-through must have read them
        self.entries.remove(entry)
        self._free(entry)
        self.drops += 1

    def _free(self, entry: Parked) -> None:
        for handle in entry.handles:
            self.arena.free(handle)
        self.arena.release_empty()

    # ---- find / restore ----
    def find(self, ids: list[int]) -> list:
        """Parked entries whose key is a prefix of ids, shortest first: in the
        arena (Parked) or only in the session store (its index records)."""
        hits = [e for e in self.entries if len(e.key) <= len(ids) and ids[: len(e.key)] == e.key]
        if self.disk is not None:
            present = {tuple(e.key) for e in self.entries}
            hits += [r for r in self.disk.find(ids) if tuple(r.key) not in present]
        hits.sort(key=lambda e: len(e.key))
        return hits

    def restore(self, entry, pages: torch.Tensor, slot: int, start: int = 0) -> dict | None:
        """Copy the entry's tokens [start:] into freshly allocated pages (one
        per token) and its snapshot into `slot`, then drop it from the arena
        (a stored copy stays in the session store). A stored record is read
        straight from its file instead. Returns the auxiliary dict for the
        re-inserted node, or None when a stored record could not be read."""
        if not isinstance(entry, Parked):
            return self._restore_record(entry, pages, slot, start)
        t0 = time.perf_counter()
        assert int(pages.shape[0]) == entry.own - start
        kvp = self.kv_pool
        idx = pages.to(torch.int64)
        for first, kv, aux in entry.chunks:
            lo, hi = max(first, start), first + kv.shape[2]
            if lo >= hi:
                continue
            dst = idx[lo - start:hi - start]
            for j, lid in enumerate(kvp.layer_ids):
                # index_copy_ has no fp8 kernel: scatter through a byte view of the same storage
                kvp.k[lid].view(torch.uint8).index_copy_(0, dst, kv[j, 0, lo - first:].view(torch.uint8))
                kvp.v[lid].view(torch.uint8).index_copy_(0, dst, kv[j, 1, lo - first:].view(torch.uint8))
            for name, view in aux.items():
                kvp.page_aux[name].index_copy_(0, dst, view[lo - first:])
        sp = self.state_pool
        sp.conv_all[:, slot].copy_(entry.conv)
        sp.ssm_all[:, slot].copy_(entry.ssm)
        aux = {}
        for identity, (scalars, views) in entry.auxiliary.items():
            if identity == "__pool__":
                for name, v in views:
                    sp.aux[name][slot].copy_(v)
                continue
            codec = self.codecs.get(identity)
            if codec is not None:
                value = codec[1]((*scalars, *views), slot=slot)
                if value is not None:
                    aux[identity] = value
        torch.cuda.current_stream().synchronize()  # the arena chunks are freed below; make sure the reads landed
        if self.disk is not None:
            self.disk.wait(entry)
        self.entries.remove(entry)
        self._free(entry)
        self.restores += 1
        self.restore_seconds += time.perf_counter() - t0
        return aux

    def _restore_record(self, record, pages: torch.Tensor, slot: int, start: int) -> dict | None:
        t0 = time.perf_counter()
        kvp, sp = self.kv_pool, self.state_pool
        kv = [t for lid in kvp.layer_ids for t in (kvp.k[lid], kvp.v[lid])]
        state = {"conv": sp.conv_all[:, slot], "ssm": sp.ssm_all[:, slot],
                 **{f"pool.{name}": t[slot] for name, t in getattr(sp, "aux", {}).items()}}
        payloads = self.disk.restore_into(record, start, pages, kv, state, record.aux_specs(self.codecs), page_aux=kvp.page_aux)
        if payloads is None:
            return None
        aux = {}
        for identity, (scalars, tensors) in payloads.items():
            value = self.codecs[identity][1]((*scalars, *tensors), slot=slot)
            if value is not None:
                aux[identity] = value
        self.loads += 1
        self.load_seconds += time.perf_counter() - t0
        return aux

    def stats(self) -> dict:
        return {
            "entries": len(self.entries), "parked_tokens": sum(e.own for e in self.entries),
            "used_gib": round(self.arena.used_bytes / GIB, 3), "slab_gib": round(self.arena.slab_bytes / GIB, 3), "budget_gib": round(self.arena.nbytes / GIB, 3),
            "parks": self.parks, "restores": self.restores, "drops": self.drops, "rejects": self.rejects,
            "last_reject": self.last_reject,
            "park_ms_avg": round(1000 * self.park_seconds / max(1, self.parks), 2),
            "restore_ms_avg": round(1000 * self.restore_seconds / max(1, self.restores), 2),
            "disk_loads": self.loads, "disk_load_ms_avg": round(1000 * self.load_seconds / max(1, self.loads), 2),
            **({"sessions": self.disk.stats()} if self.disk is not None else {}),
        }
