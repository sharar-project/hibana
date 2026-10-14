# SPDX-License-Identifier: Apache-2.0
"""Prefix-lookup n-gram embeddings (the PLE before decoder layer 1).

Each token reads 16 rows of a 320M-row FP8 table (8 heads for bigrams, 8 for trigrams; 160 bytes
per row): hash(token, previous tokens) mod the head's vocabulary plus the head's offset. The table
stays on the host as memory maps of the checkpoint's shards; the page cache holds the hot rows. On two Sparks
each rank instead holds its half of the heads in device memory and the ranks all-gather the rows (#178).
The rows project to a key per residual stream and one value; a per-stream gate from the key and
the stream's own norm scales the value, and a depth-wise causal convolution over time (kernel 4,
dilation 3) adds a short-range mix. The convolution's nine-row window per sequence is per-slot
state next to the GDN states.
"""

from __future__ import annotations

import ctypes
import json
import mmap
import os
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROW_BYTES = 160
_PAGE = mmap.PAGESIZE
_libc = ctypes.CDLL(None, use_errno=True)
_libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]


_NATIVE_SRC = Path(__file__).resolve().parents[2] / "kernels" / "csrc" / "ple_gather.cpp"
_native_module = False  # not yet tried


def _native():
    """The native gather (kernels/csrc/ple_gather.cpp), or None when it cannot be built here."""
    global _native_module
    if _native_module is False:
        try:
            import hashlib
            from torch.utils.cpp_extension import load
            revision = hashlib.sha256(_NATIVE_SRC.read_bytes()).hexdigest()[:16]
            _native_module = load(name=f"sharar_ple_gather_{revision}", sources=[str(_NATIVE_SRC)], extra_cflags=["-O3"], verbose=False)
        except Exception as exc:  # no compiler, no ninja: the Python gather below
            import warnings
            warnings.warn(f"PLE native gather unavailable ({exc}); using the Python gather")
            _native_module = None
    return _native_module


def _residency_visible(path: Path) -> bool:
    """Whether `mincore` reports `path`'s page-cache residency. Linux (5.0+) reports it only for files the caller owns or
    may open for writing; for others it says every page is resident (a root-owned file seen from a container's user, a
    shared read-only model directory)."""
    effective = os.access in os.supports_effective_ids
    return os.stat(path).st_uid == os.geteuid() or os.access(path, os.W_OK, effective_ids=effective)


class PleTable:
    """The FP8 shards on the host: memory maps for bulk gathers plus parallel `pread`s for the rows a
    decode step needs. A token's sixteen rows are scattered over the 51 GB table, so a serial gather
    pays sixteen NVMe latencies when they are not in the page cache (several milliseconds per step);
    reading them from threads overlaps the misses. Rows the page cache holds skip the threads.

    Without `RWF_NOWAIT` (a filesystem that refuses it, as ZFS does, or a Python built without `preadv2`, as uv's
    standalone CPython is; #166) the reads cannot say which rows they would find cached, and ZFS's own cache (the ARC,
    often a few GiB) holds few of them. Then the gather asks `mincore` which rows' pages the page cache holds under the
    memory maps, copies those from the maps, and reads the others through the maps from the threads, so the page
    cache, which can grow into free memory, keeps the rows decoding reads again. Where `mincore` cannot see the page
    cache (files this process neither owns nor may write), every row goes to the threads."""

    THREADS = 32
    NATIVE_THREADS = 64  # the native gather's pool: enough reads in flight to overlap the misses' NVMe latency
    native = True  # the native gather when it builds (tests switch it off to compare)

    def __init__(self, snap: Path, weight_map: dict[str, str], name: str, shards: int):
        from concurrent.futures import ThreadPoolExecutor
        self.maps: list[np.memmap] = []
        self.fds: list[int] = []
        self.offsets: list[int] = []  # byte offset of each shard's first row within its file
        self._pool = ThreadPoolExecutor(max_workers=self.THREADS)
        self._pread = os.pread
        self._nowait = hasattr(os, "preadv") and hasattr(os, "RWF_NOWAIT")  # page-cache reads that never block
        self.hits = self.misses = 0  # rows read from the page cache / through the threads
        self._prefetcher = None  # one background thread for `prefetch`, started on first use
        starts = []
        rows = 0
        opened: dict[str, tuple[int, dict]] = {}
        for i in range(shards):
            tensor = f"{name}.shard_{i}.weight"
            file = weight_map[tensor]
            if file not in opened:
                with open(snap / file, "rb") as fh:
                    (n,) = struct.unpack("<Q", fh.read(8))
                    header = json.loads(fh.read(n))
                opened[file] = (8 + n, header)
            base, header = opened[file]
            meta = header[tensor]
            if meta["dtype"] != "F8_E4M3" or meta["shape"][1] != ROW_BYTES:
                raise ValueError(f"unexpected PLE shard layout for {tensor}: {meta}")
            begin, end = meta["data_offsets"]
            count = meta["shape"][0]
            if end - begin != count * ROW_BYTES:
                raise ValueError(f"PLE shard {tensor} byte extent mismatch")
            self.maps.append(np.memmap(snap / file, dtype=np.uint8, mode="r", offset=base + begin, shape=(count, ROW_BYTES)))
            self.fds.append(os.open(snap / file, os.O_RDONLY))
            self.offsets.append(base + begin)
            starts.append(rows)
            rows += count
        self.starts = np.asarray(starts, dtype=np.int64)
        self.rows = rows
        self.bases = np.asarray([m.ctypes.data for m in self.maps], dtype=np.int64)  # each shard's first row in memory
        for m in self.maps:
            m._mmap.madvise(mmap.MADV_RANDOM)  # a fault reads its own page, not a readahead window of neighbours
        self._vec = (ctypes.c_ubyte * 2)()
        self._mincore = all(_residency_visible(snap / file) for file in opened)  # else every row through the threads
        self._fds_t = torch.tensor(self.fds, dtype=torch.int32)
        self._offsets_t = torch.tensor(self.offsets, dtype=torch.int64)

    def gather(self, rows: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """rows: int64 array of any shape -> uint8 [..., ROW_BYTES] (into `out` when given, contiguous).

        Rows in the page cache are read serially without blocking (`RWF_NOWAIT`: a few microseconds
        each); only the misses go to the threads, where their NVMe latencies overlap. Handing every
        row to the threads cost about 1.1 ms for a verification's 80 rows with all of them cached (#36).
        Without `RWF_NOWAIT` the page cache under the memory maps answers instead (`_from_maps`)."""
        flat = rows.reshape(-1)
        n = flat.shape[0]
        out = np.empty((n, ROW_BYTES), dtype=np.uint8) if out is None else out.reshape(n, ROW_BYTES)
        shard = np.searchsorted(self.starts, flat, side="right") - 1
        local = flat - self.starts[shard]
        native = _native() if self.native else None
        if native is not None and n:
            # every row from a native pool without the GIL: a non-blocking page-cache read, else a blocking one (#172)
            hits, misses, errors = native.gather(self._fds_t, self._offsets_t, torch.from_numpy(np.ascontiguousarray(shard, dtype=np.int64)),
                                                 torch.from_numpy(np.ascontiguousarray(local, dtype=np.int64)), torch.from_numpy(out),
                                                 self.NATIVE_THREADS)
            if errors:
                raise OSError(f"PLE gather: {errors} of {n} rows could not be read")
            self.hits += hits
            self.misses += misses
            return out.reshape(*rows.shape, ROW_BYTES)
        fds, offsets = self.fds, self.offsets
        pread = self._pread
        misses = []
        start = 0
        if self._nowait:
            preadv, flag = os.preadv, os.RWF_NOWAIT
            for i in range(n):
                s = int(shard[i])
                try:
                    if preadv(fds[s], [out[i]], offsets[s] + int(local[i]) * ROW_BYTES, flag) != ROW_BYTES:
                        misses.append(i)  # a row across a page boundary, half cached
                except BlockingIOError:
                    misses.append(i)
                except OSError:  # the filesystem does not take RWF_NOWAIT: the memory maps answer from here on
                    self._nowait = False
                    start = i
                    break
        mapped = not self._nowait
        if mapped:
            addrs = (self.bases[shard] + local * ROW_BYTES).tolist()
            misses += self._from_maps(addrs, out, start) if self._mincore else range(start, n)
        self.hits += n - len(misses)
        self.misses += len(misses)

        def read(part) -> None:
            if mapped:  # a fault through the map fills the page cache; memmove runs without the GIL, so faults overlap
                base = out.ctypes.data
                for i in part:
                    ctypes.memmove(base + i * ROW_BYTES, addrs[i], ROW_BYTES)
                return
            for i in part:
                s = int(shard[i])
                out[i] = np.frombuffer(pread(fds[s], ROW_BYTES, offsets[s] + int(local[i]) * ROW_BYTES), dtype=np.uint8)

        if len(misses) == 1:
            read(misses)
        elif misses:
            chunk = max(1, (len(misses) + self.THREADS - 1) // self.THREADS)
            list(self._pool.map(read, [misses[lo:lo + chunk] for lo in range(0, len(misses), chunk)]))
        return out.reshape(*rows.shape, ROW_BYTES)

    def _from_maps(self, addrs: list[int], out: np.ndarray, start: int) -> list[int]:
        """Copy rows start.. whose pages the page cache holds (`mincore` on the memory maps) from the maps into `out`;
        returns the others."""
        vec, mincore, memmove, base = self._vec, _libc.mincore, ctypes.memmove, out.ctypes.data
        misses = []
        for i in range(start, len(addrs)):
            addr = addrs[i]
            page = addr & -_PAGE
            span = addr + ROW_BYTES - page  # a row may cross into the next page
            if mincore(page, span, vec) == 0 and vec[0] & 1 and (span <= _PAGE or vec[1] & 1):
                memmove(base + i * ROW_BYTES, addr, ROW_BYTES)
            else:
                misses.append(i)
        return misses

    def prefetch(self, rows: np.ndarray) -> None:
        """Start reading `rows` into the page cache without waiting (#178): `POSIX_FADV_WILLNEED` on each distinct page
        they touch, from a background thread. The kernel queues the reads and returns, so a later gather of the
        same rows finds them cached instead of waiting on the NVMe."""
        rows = np.array(rows, dtype=np.int64).reshape(-1)  # a copy: the caller's array may change
        if rows.size == 0:
            return
        if self._prefetcher is None:
            from concurrent.futures import ThreadPoolExecutor
            self._prefetcher = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ple-prefetch")
        self._prefetcher.submit(self._advise, rows)

    def _advise(self, rows: np.ndarray) -> None:
        """The distinct pages of `rows` (both pages of a row across a page boundary), each advised once."""
        flat = np.unique(rows)
        shard = np.searchsorted(self.starts, flat, side="right") - 1
        first = np.asarray(self.offsets, dtype=np.int64)[shard] + (flat - self.starts[shard]) * ROW_BYTES
        pages = np.concatenate([first & -_PAGE, (first + ROW_BYTES - 1) & -_PAGE])
        keys = np.unique(np.stack([np.concatenate([shard, shard]), pages], 1), axis=0)
        fds, advise, page, need = self.fds, os.posix_fadvise, _PAGE, os.POSIX_FADV_WILLNEED
        for s, offset in keys.tolist():
            advise(fds[s], offset, page, need)

    def copy_rows(self, first: int, out: torch.Tensor, chunk_bytes: int = 256 << 20) -> None:
        """Rows first .. first + len(out) into `out` (uint8 [n, ROW_BYTES], any device), shard by shard: large sequential
        reads into a pinned buffer (the maps are MADV_RANDOM, which would fault one page at a time), and each range's
        pages leave the page cache once copied."""
        n = out.shape[0]
        step = max(1, chunk_bytes // ROW_BYTES)
        stage = torch.empty((step, ROW_BYTES), dtype=torch.uint8, pin_memory=out.is_cuda)
        flat = memoryview(stage.numpy().reshape(-1))
        done = 0
        for s in range(len(self.maps)):
            lo, count = int(self.starts[s]), self.maps[s].shape[0]
            a, b = max(first, lo), min(first + n, lo + count)
            if a >= b:
                continue
            fd, start = self.fds[s], self.offsets[s] + (a - lo) * ROW_BYTES
            os.posix_fadvise(fd, start, (b - a) * ROW_BYTES, os.POSIX_FADV_SEQUENTIAL)
            for i in range(a, b, step):
                j = min(i + step, b)
                want, got = (j - i) * ROW_BYTES, 0
                while got < want:
                    r = os.preadv(fd, [flat[got:want]], start + (i - a) * ROW_BYTES + got)
                    if r <= 0:
                        raise OSError(f"PLE shard {s}: short read at row {i}")
                    got += r
                out[i - first:j - first].copy_(stage[:j - i])
                done += j - i
            os.posix_fadvise(fd, start, (b - a) * ROW_BYTES, os.POSIX_FADV_DONTNEED)
        if done != n:
            raise ValueError(f"PLE rows {first}..{first + n} run past the table ({self.rows} rows)")

    def gather_mapped(self, rows: np.ndarray) -> np.ndarray:
        """The memory-map gather (serial page faults); kept as the reference for tests."""
        flat = rows.reshape(-1)
        out = np.empty((flat.shape[0], ROW_BYTES), dtype=np.uint8)
        shard = np.searchsorted(self.starts, flat, side="right") - 1
        for s in np.unique(shard):
            sel = np.nonzero(shard == s)[0]
            out[sel] = self.maps[s][flat[sel] - self.starts[s]]
        return out.reshape(*rows.shape, ROW_BYTES)


class ResidentRows:
    """A tensor-parallel rank's share of the PLE heads in device memory (#178). Rank r of n holds heads
    [r * heads / n, (r + 1) * heads / n), whose rows are one contiguous region of the table (each head's vocabulary
    follows the last's): 23.8 GiB of the 47.7 on each of two Sparks. A forward's rows gather on the device and the
    ranks all-gather them, so every rank sees the bytes the host gather reads and nothing waits on the disk."""

    def __init__(self, table: PleTable, offsets: torch.Tensor, vocab_sizes: torch.Tensor, rank: int, size: int,
                 device: torch.device, log=print):
        from sharar.distributed import div_even
        heads = len(vocab_sizes)
        per = div_even(heads, size)
        self.heads, self.per, self.h0 = heads, per, rank * per
        h1 = self.h0 + per
        if any(int(offsets[h + 1]) != int(offsets[h] + vocab_sizes[h]) for h in range(self.h0, h1 - 1)):
            raise ValueError("PLE heads of one rank must be contiguous in the table")
        self.base = int(offsets[self.h0])
        n = int(offsets[h1 - 1] + vocab_sizes[h1 - 1]) - self.base
        started = time.perf_counter()
        self.rows = torch.empty((n, ROW_BYTES), dtype=torch.uint8, device=device)
        table.copy_rows(self.base, self.rows)
        log(f"  PLE: heads {self.h0}-{h1 - 1} resident on this rank ({n * ROW_BYTES / 2**30:.1f} GiB, "
            f"{time.perf_counter() - started:.0f}s)")

    def gather(self, ids: torch.Tensor) -> torch.Tensor:
        """ids int64 [N, heads] (table rows, host or device) -> device uint8 [N, heads, ROW_BYTES]: this rank's heads
        from its rows, the others' through one all-gather."""
        from sharar.distributed import all_gather
        local = ids[:, self.h0:self.h0 + self.per].to(self.rows.device, non_blocking=True) - self.base
        part = self.rows.index_select(0, local.reshape(-1)).view(ids.shape[0], self.per * ROW_BYTES)
        return all_gather(part).view(ids.shape[0], self.heads, ROW_BYTES)


class PleLayer:
    def __init__(self, w, snap: Path, prefix: str, device: torch.device, target, linear_cls, precision: str):
        self.device = device
        self.hc, self.hidden, self.eps = target.hc_count, target.hidden_size, target.rms_norm_eps
        self.ngram, self.heads_per_ngram, self.kernel = target.ple_ngram, target.ple_heads_per_ngram, target.ple_conv_kernel
        self.heads = (self.ngram - 1) * self.heads_per_ngram
        self.eos = target.eos_token_id if hasattr(target, "eos_token_id") else None
        self.key_proj = linear_cls.load(w, [f"{prefix}.key_proj.weight"], device, precision)  # [hc*H, ple_dim]
        self.value_proj = linear_cls.load(w, [f"{prefix}.value_proj.weight"], device, precision)  # [H, ple_dim]
        self.norm_key = w.get(f"{prefix}.norm_key.weight").to(device)
        self.norm_query = w.get(f"{prefix}.norm_query.weight").to(device)
        self.norm_conv = w.get(f"{prefix}.norm_conv.weight").to(device)
        self.conv_weight = w.get(f"{prefix}.conv1d.weight").to(device).reshape(-1, self.kernel)  # [hc*H, kernel]
        e = f"{prefix}.ple_embedding"
        self.multipliers = w.get(f"{e}.layer_multipliers").to(torch.int64)  # host
        self.vocab_sizes = w.get(f"{e}.ngram_heads_vocab_sizes").to(torch.int64)
        self.offsets = w.get(f"{e}.ngram_heads_offsets").to(torch.int64)
        self.scale = w.get(f"{e}.ngram_embedding.weight_scale").float().reshape(()).to(device)
        self.table = PleTable(snap, w._map, f"{e}.ngram_embedding", shards=128)
        self.resident = None
        from sharar.distributed import get_tp_info
        from sharar.engine.config import current
        tp = get_tp_info()
        if tp.size > 1 and current().ple_resident != "0":
            self.resident = ResidentRows(self.table, self.offsets, self.vocab_sizes, tp.rank, tp.size, device)
        self.state_len = (self.kernel - 1) * self.ngram  # conv window rows kept per sequence
        self._scratch: torch.Tensor | None = None  # stand-alone (no pool) window

    @property
    def bytes(self) -> int:
        return self.key_proj.bytes + self.value_proj.bytes + (self.norm_key.numel() * 3 + self.conv_weight.numel()) * 2

    # ---- table rows ----
    def ngram_ids(self, ids: torch.Tensor, history: torch.Tensor | None, eos: int) -> torch.Tensor:
        """ids int64 [T] (host), history int64 [2] (the two tokens before ids[0]; None = EOS) -> int64 [T, heads].

        Follows Qwen4ExpNGramEmbedding._hash_contexts: the token n back is itself when no EOS lies
        between it and the current token, else EOS; back-1 is always t-1, back-2 is t-2 unless t-1
        or t-2 is EOS.
        """
        h = torch.full((2,), eos, dtype=torch.int64) if history is None else history.to(torch.int64)
        ext = torch.cat([h, ids.to(torch.int64)])
        return self._hash(ids.to(torch.int64), ext[1:-1], ext[:-2], eos)

    def token_ids(self, ids: torch.Tensor, histories: torch.Tensor, eos: int) -> torch.Tensor:
        """One token per row, each after its own two tokens: ids int64 [N], histories int64 [N, 2] (host) ->
        int64 [N, heads]; the rows `ngram_ids` gives each token inside its sequence (a verification column)."""
        return self._hash(ids.to(torch.int64), histories[:, 1].to(torch.int64), histories[:, 0].to(torch.int64), eos)

    def _hash(self, cur: torch.Tensor, back1: torch.Tensor, back2: torch.Tensor, eos: int) -> torch.Tensor:
        if self.ngram != 3:
            raise NotImplementedError("the PLE hash is spelled out for 3-grams")
        back2 = torch.where((back1 == eos) | (back2 == eos), torch.full_like(back2, eos), back2)
        shifted = [cur, back1, back2]
        blocks = []
        for n in range(2, self.ngram + 1):
            mix = shifted[0] * self.multipliers[0]
            for pos in range(1, n):
                mix = torch.bitwise_xor(mix, shifted[pos] * self.multipliers[pos])
            lo, hi = (n - 2) * self.heads_per_ngram, (n - 1) * self.heads_per_ngram
            blocks.append(torch.remainder(mix[:, None], self.vocab_sizes[lo:hi][None]) + self.offsets[lo:hi][None])
        return torch.cat(blocks, dim=-1)

    def rows_for(self, ids: torch.Tensor, seqs: list[tuple[int, int]], histories: torch.Tensor | None, eos: int) -> torch.Tensor:
        """Gather the table rows for `ids` (host int64 [T]) laid out as `seqs` (start, length) -> device uint8 [T, heads, ROW_BYTES]."""
        parts = []
        for i, (start, length) in enumerate(seqs):
            hist = None if histories is None else histories[i]
            parts.append(self.ngram_ids(ids[start:start + length], hist, eos))
        return self.device_rows(torch.cat(parts))

    def device_rows(self, ids: torch.Tensor) -> torch.Tensor:
        """Table rows ids (int64 [N, heads] on the host) -> device uint8 [N, heads, ROW_BYTES]: from the ranks'
        resident shares when they hold the table (#178), else the host gather and an upload."""
        if self.resident is not None:
            return self.resident.gather(ids)
        return torch.from_numpy(self.table.gather(ids.numpy())).to(self.device, non_blocking=False)

    # ---- state ----
    STATE_NAME = "ple_conv"

    def register(self, pool) -> torch.Tensor:
        return pool.register_aux(self.STATE_NAME, (self.hc * self.hidden, self.state_len), torch.bfloat16)

    def _states(self, ctx) -> torch.Tensor:
        """[total_slots, hc*H, state_len] bf16: the pool's auxiliary tensor (cleared, copied and parked with the GDN states)."""
        if ctx is not None and ctx.state_pool is not None:
            return self.register(ctx.state_pool)
        if self._scratch is None:
            self._scratch = torch.zeros((1, self.hc * self.hidden, self.state_len), dtype=torch.bfloat16, device=self.device)
        return self._scratch

    # ---- forward ----
    def forward(self, hyper: torch.Tensor, input_ids: torch.Tensor, ctx, eos: int) -> torch.Tensor:
        t = hyper.shape[0]
        if ctx is None or ctx.mode == "prefill":
            if ctx is not None and ctx.batched:
                seqs = [(s, n) for s, n in zip([0, *np.cumsum(ctx.seq_lens)[:-1].tolist()], ctx.seq_lens)]
                prefixes, slots = list(ctx.seq_prefix_lens), ctx.seq_slots.tolist()
            else:
                seqs = [(0, t)]
                prefixes = [ctx.prefix_len if ctx is not None else 0]
                slots = [ctx.slot if ctx is not None and ctx.state_pool is not None else 0]
        elif ctx.mode == "decode":
            seqs = [(i, 1) for i in range(t)]
            prefixes, slots = [1] * t, None  # every decode row continues its slot's state (indexed on the device)
        else:  # verify: B blocks of verify_steps tokens, each continuing its slot's state without committing
            s_len = ctx.verify_steps
            seqs = [(i * s_len, s_len) for i in range(t // s_len)]
            prefixes, slots = [1] * len(seqs), None
        side = None if ctx is None else ctx.extras  # FlashNextSideInputs (model/flashnext/extras.py)
        histories = None if side is None else side.ple_history
        from sharar.kernels.flashnext_fused import grouped_gemma_rms  # Triton; imported here so host tools can use PleTable
        rows = side.ple_rows if side is not None and side.ple_rows is not None else self.rows_for(input_ids.cpu(), seqs, histories, eos)
        emb = (rows.view(torch.float8_e4m3fn).to(torch.bfloat16) * self.scale).to(torch.bfloat16).reshape(t, -1)  # [T, ple_dim]
        key = self.key_proj(emb)  # [T, hc*H]
        value = self.value_proj(emb)  # [T, H]
        key_n = grouped_gemma_rms(key, self.norm_key, self.eps, self.hidden).unflatten(-1, (self.hc, self.hidden))
        query_n = grouped_gemma_rms(hyper, self.norm_query, self.eps, self.hidden).unflatten(-1, (self.hc, self.hidden))
        gate = (key_n.float() * query_n.float()).sum(-1, keepdim=True) / (self.hidden ** 0.5)
        gate = torch.sigmoid(gate.abs().clamp_min(1e-6).sqrt() * gate.sign())
        gated = (gate * value.float().unsqueeze(-2)).flatten(-2)  # [T, hc*H] fp32
        gated_n = grouped_gemma_rms(gated.to(torch.bfloat16), self.norm_conv, self.eps, self.hidden)  # bf16
        conv = self._conv(gated_n, seqs, prefixes, slots, ctx)
        return (gated + F.silu(conv.float())).to(hyper.dtype)

    def _conv(self, x: torch.Tensor, seqs, prefixes, slots, ctx) -> torch.Tensor:
        """Depth-wise causal convolution over time with dilation `ngram`, continuing each sequence's window."""
        states = self._states(ctx)
        k, d, s_len = self.kernel, self.ngram, self.state_len
        out = torch.empty_like(x)
        if ctx is not None and ctx.mode == "verify":
            steps = ctx.verify_steps
            b = x.shape[0] // steps
            prev = states[ctx.seq_slots.to(torch.int64)]  # [B, C, s_len], read only
            full = torch.cat([prev, x.view(b, steps, -1).transpose(1, 2)], dim=-1)  # [B, C, s_len + steps]
            conv = F.conv1d(full, self.conv_weight.unsqueeze(1), dilation=d, groups=x.shape[1])  # [B, C, steps]
            windows = ctx.verify_aux_windows.get(self.STATE_NAME)
            if windows is not None:  # state after step j = the s_len rows ending at token j
                windows[:b, :steps].copy_(full.unfold(2, s_len, 1)[:, :, 1:].permute(0, 2, 1, 3))
            return conv.transpose(1, 2).reshape(x.shape[0], -1).to(x.dtype)
        if ctx is not None and ctx.mode == "decode":
            slot_t = ctx.slots.to(torch.int64)
            window = torch.cat([states[slot_t], x.unsqueeze(-1)], dim=-1)  # [B, C, s_len + 1]
            acc = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.float32, device=x.device)
            for j in range(k):
                acc += self.conv_weight[:, j].float() * window[:, :, j * d].float()
            states[slot_t] = window[:, :, 1:]
            return acc.to(x.dtype)
        checkpoints = []
        if ctx is not None and ctx.checkpoint_positions:
            checkpoints = list(zip(ctx.checkpoint_positions, ctx.checkpoint_slots, ctx.checkpoint_seqs or [0] * len(ctx.checkpoint_slots)))
        for i, (start, length) in enumerate(seqs):
            slot = slots[i]
            prev = states[slot] if prefixes[i] > 0 else torch.zeros_like(states[0])
            full = torch.cat([prev.transpose(0, 1), x[start:start + length]], dim=0)  # [s_len + length, C]
            conv = F.conv1d(full.transpose(0, 1).unsqueeze(0), self.conv_weight.unsqueeze(1), dilation=d, groups=x.shape[1])
            out[start:start + length] = conv.squeeze(0).transpose(0, 1)
            for pos, ck_slot, seq in checkpoints:
                if seq == i:
                    local = pos - prefixes[i]
                    states[ck_slot] = full[local:local + s_len].transpose(0, 1)
            if ctx is not None and ctx.state_pool is not None:
                states[slot] = full[-s_len:].transpose(0, 1)
        return out
