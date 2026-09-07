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
    """One CUDA-managed allocation carved into variable-size chunks."""

    def __init__(self, nbytes: int, device: torch.device):
        self.nbytes = int(nbytes)
        self.device = device
        block = _ManagedBlock(self.nbytes)
        self.buffer = torch.as_tensor(block, device=device)  # zero-copy view of the managed memory
        self._keep = block
        self.buffer.fill_(0)  # first touch now, not on the eviction path
        self._used: list[tuple[int, int]] = []  # (offset, size) sorted by offset

    def alloc(self, size: int) -> int | None:
        size = (size + 255) // 256 * 256
        cursor = 0
        for i, (off, sz) in enumerate(self._used):
            if off - cursor >= size:
                self._used.insert(i, (cursor, size))
                return cursor
            cursor = off + sz
        if self.nbytes - cursor >= size:
            self._used.append((cursor, size))
            return cursor
        return None

    def free(self, offset: int) -> None:
        self._used = [(o, s) for o, s in self._used if o != offset]

    @property
    def used_bytes(self) -> int:
        return sum(s for _, s in self._used)

    def view(self, offset: int, nbytes: int, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
        return self.buffer[offset : offset + nbytes].view(dtype).view(*shape)


@dataclass
class Parked:
    key: list[int]  # full token prefix from the root
    own: int  # tokens owned by the parked leaf (the last `own` of key)
    offset: int
    nbytes: int
    kv: torch.Tensor  # [layers, own, ...] view into the arena
    conv: torch.Tensor | None
    ssm: torch.Tensor | None
    auxiliary: dict = field(default_factory=dict)  # identity -> exported payload (host side)
    ready: torch.cuda.Event | None = None
    last_access: float = field(default_factory=time.monotonic)


class ColdTier:
    def __init__(self, kv_pool, state_pool, budget_bytes: int, device: torch.device, log=None):
        self.kv_pool, self.state_pool, self.device = kv_pool, state_pool, device
        self.arena = ManagedArena(budget_bytes, device)
        self.entries: list[Parked] = []
        self.codecs: dict = {}  # identity -> (export, import)
        self.parks = self.restores = self.drops = self.rejects = 0
        self.park_seconds = self.restore_seconds = 0.0
        if log:
            log(f"  cold tier: {budget_bytes / GIB:.1f} GiB managed arena for evicted sessions")

    # ---- sizes ----
    def _kv_row_bytes(self) -> int:
        k = self.kv_pool.k[self.kv_pool.layer_ids[0]]
        return k[0].numel() * k.element_size() * 2 * len(self.kv_pool.layer_ids)

    def _state_bytes(self) -> int:
        sp = self.state_pool
        return sp.conv_all[:, 0].numel() * sp.conv_all.element_size() + sp.ssm_all[:, 0].numel() * sp.ssm_all.element_size()

    def register_codec(self, identity, export, import_) -> None:
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
            return False
        if any(len(e.key) >= own and e.key[:own] == key for e in self.entries):
            self.rejects += 1  # already covered by a parked descendant
            return False
        nbytes = own * self._kv_row_bytes() + self._state_bytes()
        if nbytes > self.arena.nbytes:
            self.rejects += 1
            return False
        while (offset := self.arena.alloc(nbytes)) is None:
            self._drop_lru()
        t0 = time.perf_counter()
        kvp = self.kv_pool
        layers = kvp.layer_ids
        k0 = kvp.k[layers[0]]
        per_layer = k0[0].numel() * k0.element_size()
        kv = self.arena.view(offset, own * per_layer * 2 * len(layers), k0.dtype, (len(layers), 2, own, *k0.shape[1:]))
        idx = pages.to(torch.int64)
        for j, lid in enumerate(layers):
            torch.index_select(kvp.k[lid], 0, idx, out=kv[j, 0])
            torch.index_select(kvp.v[lid], 0, idx, out=kv[j, 1])
        base = offset + own * per_layer * 2 * len(layers)
        sp = self.state_pool
        conv_n = sp.conv_all[:, 0].numel() * sp.conv_all.element_size()
        conv = self.arena.view(base, conv_n, sp.conv_all.dtype, tuple(sp.conv_all[:, 0].shape))
        ssm = self.arena.view(base + conv_n, sp.ssm_all[:, 0].numel() * sp.ssm_all.element_size(), sp.ssm_all.dtype, tuple(sp.ssm_all[:, 0].shape))
        conv.copy_(sp.conv_all[:, slot])
        ssm.copy_(sp.ssm_all[:, slot])
        aux = {}
        for identity, value in (auxiliary or {}).items():
            codec = self.codecs.get(identity)
            if codec is not None:
                payload = codec[0](value)
                if payload is not None:
                    aux[identity] = payload
        ev = torch.cuda.Event()
        ev.record()
        self.entries.append(Parked(list(key), own, offset, nbytes, kv, conv, ssm, aux, ev))
        self.parks += 1
        self.park_seconds += time.perf_counter() - t0
        return True

    def _drop_lru(self) -> None:
        if not self.entries:
            raise MemoryError("cold tier arena cannot fit this entry")
        victim = min(self.entries, key=lambda e: e.last_access)
        self._drop(victim)

    def discard(self, entry: Parked) -> None:
        self._drop(entry)

    def clear(self) -> None:
        for entry in list(self.entries):
            self._drop(entry)

    def _drop(self, entry: Parked) -> None:
        if entry.ready is not None:
            entry.ready.synchronize()  # the arena bytes are reused: the park copy must have landed
        self.entries.remove(entry)
        self.arena.free(entry.offset)
        self.drops += 1

    # ---- find / restore ----
    def find(self, ids: list[int]) -> list[Parked]:
        """Parked entries whose key is a prefix of ids, shortest first."""
        hits = [e for e in self.entries if len(e.key) <= len(ids) and ids[: len(e.key)] == e.key]
        hits.sort(key=lambda e: len(e.key))
        return hits

    def restore(self, entry: Parked, pages: torch.Tensor, slot: int, start: int = 0) -> dict:
        """Copy the entry's tokens [start:] into freshly allocated pages (one
        per token) and its snapshot into `slot`, then drop it. Returns the
        auxiliary dict for the re-inserted node."""
        t0 = time.perf_counter()
        assert int(pages.shape[0]) == entry.own - start
        kvp = self.kv_pool
        idx = pages.to(torch.int64)
        for j, lid in enumerate(kvp.layer_ids):
            # index_copy_ has no fp8 kernel: scatter through a byte view of the same storage
            kvp.k[lid].view(torch.uint8).index_copy_(0, idx, entry.kv[j, 0, start:].view(torch.uint8))
            kvp.v[lid].view(torch.uint8).index_copy_(0, idx, entry.kv[j, 1, start:].view(torch.uint8))
        sp = self.state_pool
        sp.conv_all[:, slot].copy_(entry.conv)
        sp.ssm_all[:, slot].copy_(entry.ssm)
        aux = {}
        for identity, payload in entry.auxiliary.items():
            codec = self.codecs.get(identity)
            if codec is not None:
                value = codec[1](payload)
                if value is not None:
                    aux[identity] = value
        torch.cuda.current_stream().synchronize()  # the arena chunk is freed below; make sure the reads landed
        self.entries.remove(entry)
        self.arena.free(entry.offset)
        self.restores += 1
        self.restore_seconds += time.perf_counter() - t0
        return aux

    def stats(self) -> dict:
        return {
            "entries": len(self.entries), "parked_tokens": sum(e.own for e in self.entries),
            "used_gib": round(self.arena.used_bytes / GIB, 3), "budget_gib": round(self.arena.nbytes / GIB, 3),
            "parks": self.parks, "restores": self.restores, "drops": self.drops, "rejects": self.rejects,
            "park_ms_avg": round(1000 * self.park_seconds / max(1, self.parks), 2),
            "restore_ms_avg": round(1000 * self.restore_seconds / max(1, self.restores), 2),
        }
