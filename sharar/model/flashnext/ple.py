# SPDX-License-Identifier: Apache-2.0
"""Prefix-lookup n-gram embeddings (the PLE before decoder layer 1).

Each token reads 16 rows of a 320M-row FP8 table (8 heads for bigrams, 8 for trigrams; 160 bytes
per row): hash(token, previous tokens) mod the head's vocabulary plus the head's offset. The table
stays on the host as memory maps of the checkpoint's shards; the page cache holds the hot rows.
The rows project to a key per residual stream and one value; a per-stream gate from the key and
the stream's own norm scales the value, and a depth-wise causal convolution over time (kernel 4,
dilation 3) adds a short-range mix. The convolution's nine-row window per sequence is per-slot
state next to the GDN states.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROW_BYTES = 160


class PleTable:
    """The FP8 shards on the host: memory maps for bulk gathers plus parallel `pread`s for the rows a
    decode step needs. A token's sixteen rows are scattered over the 51 GB table, so a serial gather
    pays sixteen NVMe latencies when they are not in the page cache (several milliseconds per step);
    reading them from threads overlaps the misses. Rows the page cache holds skip the threads."""

    THREADS = 32

    def __init__(self, snap: Path, weight_map: dict[str, str], name: str, shards: int):
        from concurrent.futures import ThreadPoolExecutor
        self.maps: list[np.memmap] = []
        self.fds: list[int] = []
        self.offsets: list[int] = []  # byte offset of each shard's first row within its file
        self._pool = ThreadPoolExecutor(max_workers=self.THREADS)
        self._pread = os.pread
        self._nowait = hasattr(os, "preadv") and hasattr(os, "RWF_NOWAIT")  # page-cache reads that never block
        self.hits = self.misses = 0  # rows read from the page cache / through the threads
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

    def gather(self, rows: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """rows: int64 array of any shape -> uint8 [..., ROW_BYTES] (into `out` when given, contiguous).

        Rows in the page cache are read serially without blocking (`RWF_NOWAIT`: a few microseconds
        each); only the misses go to the threads, where their NVMe latencies overlap. Handing every
        row to the threads cost about 1.1 ms for a verification's 80 rows with all of them cached (#36)."""
        flat = rows.reshape(-1)
        n = flat.shape[0]
        out = np.empty((n, ROW_BYTES), dtype=np.uint8) if out is None else out.reshape(n, ROW_BYTES)
        shard = np.searchsorted(self.starts, flat, side="right") - 1
        local = flat - self.starts[shard]
        fds, offsets = self.fds, self.offsets
        pread = self._pread
        misses = []
        if self._nowait:
            preadv, flag = os.preadv, os.RWF_NOWAIT
            for i in range(n):
                s = int(shard[i])
                try:
                    if preadv(fds[s], [out[i]], offsets[s] + int(local[i]) * ROW_BYTES, flag) != ROW_BYTES:
                        misses.append(i)  # a row across a page boundary, half cached
                except BlockingIOError:
                    misses.append(i)
                except OSError:  # the filesystem does not take RWF_NOWAIT: every row through the threads
                    self._nowait = False
                    misses.extend(range(i, n))
                    break
        else:
            misses = range(n)
        self.hits += n - len(misses)
        self.misses += len(misses)

        def read(part) -> None:
            for i in part:
                s = int(shard[i])
                out[i] = np.frombuffer(pread(fds[s], ROW_BYTES, offsets[s] + int(local[i]) * ROW_BYTES), dtype=np.uint8)

        if len(misses) == 1:
            read(misses)
        elif misses:
            chunk = max(1, (len(misses) + self.THREADS - 1) // self.THREADS)
            list(self._pool.map(read, [misses[lo:lo + chunk] for lo in range(0, len(misses), chunk)]))
        return out.reshape(*rows.shape, ROW_BYTES)

    def gather_mapped(self, rows: np.ndarray) -> np.ndarray:
        """The memory-map gather (serial page faults); kept as the reference for tests."""
        flat = rows.reshape(-1)
        out = np.empty((flat.shape[0], ROW_BYTES), dtype=np.uint8)
        shard = np.searchsorted(self.starts, flat, side="right") - 1
        for s in np.unique(shard):
            sel = np.nonzero(shard == s)[0]
            out[sel] = self.maps[s][flat[sel] - self.starts[s]]
        return out.reshape(*rows.shape, ROW_BYTES)


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
        if self.ngram != 3:
            raise NotImplementedError("the PLE hash is spelled out for 3-grams")
        h = torch.full((2,), eos, dtype=torch.int64) if history is None else history.to(torch.int64)
        ext = torch.cat([h, ids.to(torch.int64)])
        back1, back2 = ext[1:-1], ext[:-2]
        back2 = torch.where((back1 == eos) | (back2 == eos), torch.full_like(back2, eos), back2)
        shifted = [ids.to(torch.int64), back1, back2]
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
        rows = torch.cat(parts).numpy()
        return torch.from_numpy(self.table.gather(rows)).to(self.device, non_blocking=False)

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
