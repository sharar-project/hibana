"""Flash-Next's side inputs for a forward: the PLE table rows and the sparse-attention bookkeeping.

The engine hands every forward a `ForwardContext`; what Flash-Next needs beyond
it lives in one `FlashNextSideInputs` value on `ctx.extras`, built by
`FlashNextExtras` (the model's `forward_extras()`):

- the two tokens before each sequence (the PLE's n-gram history) for prefill,
  and the gathered table rows of every decode or verify row (read on the host
  from the NVMe table, or on two Sparks from each rank's resident share);
- the indexer's per-sequence layout (eager), or its graph-safe key-maintenance
  rows (graphs), plus the page table a selecting graph reads past the budget.

The engine only calls `for_prefill` / `for_decode` / `for_verify` on the eager
path, `buffers` / `fill` / `selecting` around CUDA graphs, and `variant` to pick
the dense or selecting graph.
"""

from __future__ import annotations

import dataclasses
import weakref
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class FlashNextSideInputs:
    # PLE: the two tokens before each sequence's new tokens, int64 [B, 2] on the host (None = EOS history), and
    # optionally the table rows already gathered for the forward's tokens (uint8 [T, heads, row_bytes] on the device).
    ple_history: torch.Tensor | None = None
    ple_rows: torch.Tensor | None = None
    # Indexer, eager: per sequence in token order, (first new position, new tokens, every page through its last new token).
    qsa_seqs: list | None = None
    # Indexer, graph-safe key maintenance: int32 [rows, ratio] pages of each row's group through itself, and [rows]
    # the page that receives the compressed key (page 0, the dummy, when the row does not complete a group).
    qsa_members: torch.Tensor | None = None
    qsa_writes: torch.Tensor | None = None
    # Indexer, graph-safe selection (some row past the budget): the flat page table the attention plan reads,
    # its per-sequence offsets, each row's sequence and the static group capacity.
    qsa_table: torch.Tensor | None = None
    qsa_indptr: torch.Tensor | None = None
    qsa_row_seq: torch.Tensor | None = None
    qsa_gcap: int = 0
    # Opt-in eager-prefill plan shared by attention layers within this forward only.
    qsa_prefill_plan: object | None = None
    qsa_optimized_prefill: bool = False


class FlashNextExtras:
    def __init__(self, model):
        from sharar.engine.config import current
        t = model.target
        self.model, self.ple, self.device, self.eos = model, model.ple, model.device, model.eos
        self.budget, self.ratio = t.indexer_budget, t.indexer_compress_ratio
        self.variants = ("dense", "selecting") if self.budget and current().qsa_sparse_graphs else ("dense",)
        self._staging: torch.Tensor | None = None  # pinned PLE rows of a streamed verification (`_stream_verify_rows`)
        self._prefetched = weakref.WeakSet()  # requests whose prompt rows `_prefetch_prompts` has asked for
        self._staged = torch.cuda.Event() if self.device.type == "cuda" else None

    # ---- dense or selecting ----
    def variant(self, seqs) -> str:
        """Dense while every row sees at most the indexer budget (exact); otherwise the selecting path."""
        dense = not self.budget or all(first + n - 1 < self.budget + self.ratio - 1 for first, n, _ in seqs)
        return "dense" if dense else "selecting"

    # ---- PLE rows ----
    def _history(self, req, start: int) -> list[int]:
        """The two tokens before absolute position `start` of the request (EOS before the sequence)."""
        full = req.prompt_ids + req.output_ids
        return [full[i] if i >= 0 else self.eos for i in (start - 2, start - 1)]

    def _decode_rows(self, reqs) -> torch.Tensor:
        """Table rows for each request's pending decode token (host gather) -> device uint8 [B, heads, row_bytes]."""
        tokens = torch.tensor([r.output_ids[-1] for r in reqs], dtype=torch.int64)
        histories = torch.tensor([self._history(r, r.context_len - 1) for r in reqs], dtype=torch.int64)
        return self.ple.rows_for(tokens, [(i, 1) for i in range(len(reqs))], histories, self.eos)

    def _verify_rows(self, reqs, drafts: list[list[int]]) -> torch.Tensor:
        """Table rows for each request's verify block (its pending token then the drafted ones) -> [B*(k+1), heads, row_bytes]."""
        k = len(drafts[0])
        tokens = torch.tensor([[r.output_ids[-1], *d] for r, d in zip(reqs, drafts)], dtype=torch.int64).reshape(-1)
        histories = torch.tensor([self._history(r, r.context_len - 1) for r in reqs], dtype=torch.int64)
        return self.ple.rows_for(tokens, [(i * (k + 1), k + 1) for i in range(len(reqs))], histories, self.eos)

    def _stream_verify_rows(self, out: torch.Tensor, reqs, columns, k: int) -> None:
        """`_verify_rows` column by column into `out` (device uint8 [B*(k+1), heads, row_bytes]): the pending
        tokens' rows while the first draft step runs, drafted token j's while step j+1 runs (`DraftColumns`),
        so only the last column's read follows the drafts. One upload from pinned memory."""
        from sharar.model.flashnext.ple import ROW_BYTES
        b, heads = len(reqs), self.ple.heads
        if self.ple.resident is not None:  # the rows are in device memory: row ids per column, then one gather
            ids = torch.empty((b, k + 1, heads), dtype=torch.int64)
            seqs = [self._history(r, r.context_len - 1) + [r.output_ids[-1]] for r in reqs]
            for j in range(k + 1):
                if j:
                    for seq, token in zip(seqs, columns.column(j - 1)):
                        seq.append(token)
                tokens = torch.tensor([seq[-1] for seq in seqs], dtype=torch.int64)
                ids[:, j] = self.ple.token_ids(tokens, torch.tensor([seq[-3:-1] for seq in seqs], dtype=torch.int64), self.eos)
            out.copy_(self.ple.device_rows(ids.reshape(-1, heads)))
            return
        if self._staging is None or self._staging.shape[0] < b * (k + 1):
            self._staging = torch.empty((b * (k + 1), heads, ROW_BYTES), dtype=torch.uint8, pin_memory=self._staged is not None)
        elif self._staged is not None:
            self._staged.synchronize()  # the previous upload has read the staging rows (it precedes the last verify)
        host = self._staging[:b * (k + 1)].numpy().reshape(b, k + 1, heads, ROW_BYTES)
        # each request's two tokens before the block, its pending token, then the drafted columns as they land
        seqs = np.empty((b, k + 3), dtype=np.int64)
        seqs[:, :3] = [self._history(r, r.context_len - 1) + [r.output_ids[-1]] for r in reqs]
        for j in range(k + 1):
            if j:
                seqs[:, j + 2] = columns.column(j - 1)
            host[:, j] = self.ple.table.gather(self.ple.column_ids(seqs[:, j + 2], seqs[:, j + 1], seqs[:, j], self.eos))
        out.copy_(self._staging[:b * (k + 1)], non_blocking=True)
        if self._staged is not None:
            self._staged.record()

    # ---- eager forwards ----
    def for_prefill(self, chunks) -> FlashNextSideInputs:
        """chunks: (request, first new position, new tokens, every page through the last new token) per sequence."""
        side = FlashNextSideInputs()
        if self.ple is not None:
            side.ple_history = torch.tensor([self._history(req, start) for req, start, _, _ in chunks], dtype=torch.int64)
            self._prefetch_prompts(chunks)
        if self.budget:
            side.qsa_seqs = [(start, n, pages) for _, start, n, pages in chunks]
        return side

    def _prefetch_prompts(self, chunks) -> None:
        """When a request's first prefill chunk runs, start reading the rows of the rest of its prompt from the NVMe,
        so its later chunks find them in the page cache while this one computes. Not with resident heads."""
        if self.ple.resident is not None:
            return
        for req, start, n, _ in chunks:
            end = start + n
            if req in self._prefetched or end >= len(req.prompt_ids):
                continue
            self._prefetched.add(req)
            ids = torch.tensor(req.prompt_ids[end:], dtype=torch.int64)
            history = torch.tensor(self._history(req, end), dtype=torch.int64)
            self.ple.table.prefetch(self.ple.ngram_ids(ids, history, self.eos).numpy())

    def for_decode(self, reqs, seqs) -> FlashNextSideInputs:
        return FlashNextSideInputs(ple_rows=self._decode_rows(reqs) if self.ple is not None else None,
                                   qsa_seqs=seqs if self.budget else None)

    def for_verify(self, reqs, drafts: torch.Tensor, seqs) -> FlashNextSideInputs:
        return FlashNextSideInputs(ple_rows=self._verify_rows(reqs, drafts.tolist()) if self.ple is not None else None,
                                   qsa_seqs=seqs if self.budget else None)

    # ---- CUDA graphs ----
    def buffers(self, rows: int) -> FlashNextSideInputs:
        """The static side inputs of a graph over `rows` rows."""
        from sharar.model.flashnext.ple import ROW_BYTES
        d = self.device
        side = FlashNextSideInputs()
        if self.ple is not None:
            side.ple_rows = torch.zeros((rows, self.ple.heads, ROW_BYTES), dtype=torch.uint8, device=d)
        if self.budget:
            side.qsa_members = torch.zeros((rows, self.ratio), dtype=torch.int32, device=d)
            side.qsa_writes = torch.zeros(rows, dtype=torch.int32, device=d)
        return side

    def fill(self, side: FlashNextSideInputs, reqs, drafts: torch.Tensor | None, seqs, rows: int, columns=None) -> None:
        """Copy one step's values into a graph's static side inputs: `rows` real rows (padded rows keep page 0).
        The indexer rows come first: they need only positions and pages, while a verification's PLE rows
        wait for the drafted tokens, so the host launches them while the draft graphs still run.
        `columns` (`DraftColumns`, sequential drafts) lets the PLE rows follow the draft step by step."""
        if side.qsa_members is not None:
            from sharar.model.flashnext.indexer import FlashNextIndexer
            members, writes = FlashNextIndexer.static_rows(seqs, self.ratio, self.device)
            side.qsa_members[:rows].copy_(members)
            side.qsa_writes[:rows].copy_(writes)
            side.qsa_members[rows:].zero_()
            side.qsa_writes[rows:].zero_()
        if side.ple_rows is not None:
            if drafts is None:
                side.ple_rows[:rows].copy_(self._decode_rows(reqs))
            elif columns is not None:
                self._stream_verify_rows(side.ple_rows[:rows], reqs, columns, drafts.shape[1])
            else:
                side.ple_rows[:rows].copy_(self._verify_rows(reqs, drafts.tolist()))

    def selecting(self, side: FlashNextSideInputs, table: torch.Tensor, indptr: torch.Tensor, row_seq: torch.Tensor,
                  capacity: int) -> FlashNextSideInputs:
        """The selecting variant of a graph's side inputs: rows find their pages through the plan's own buffers."""
        return dataclasses.replace(side, qsa_table=table, qsa_indptr=indptr, qsa_row_seq=row_seq, qsa_gcap=capacity // self.ratio)
