"""Flash-Next's side inputs for a forward (#47): the PLE table rows and the sparse-attention bookkeeping.

The engine hands every forward a `ForwardContext`; what Flash-Next needs beyond
it lives in one `FlashNextSideInputs` value on `ctx.extras`, built by
`FlashNextExtras` (the model's `forward_extras()`):

- the two tokens before each sequence (the PLE's n-gram history) for prefill,
  and the gathered table rows of every decode or verify row (read on the host
  from the NVMe table);
- the indexer's per-sequence layout (eager), or its graph-safe key-maintenance
  rows (graphs), plus the page table a selecting graph reads past the budget.

The engine only calls `for_prefill` / `for_decode` / `for_verify` on the eager
path, `buffers` / `fill` / `selecting` around CUDA graphs, and `variant` to pick
the dense or selecting graph. The values are the ones the executor computed
itself before this module existed.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

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


class FlashNextExtras:
    def __init__(self, model):
        from sharar.engine.config import current
        t = model.target
        self.model, self.ple, self.device, self.eos = model, model.ple, model.device, model.eos
        self.budget, self.ratio = t.indexer_budget, t.indexer_compress_ratio
        self.variants = ("dense", "selecting") if self.budget and current().qsa_sparse_graphs else ("dense",)

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

    # ---- eager forwards ----
    def for_prefill(self, chunks) -> FlashNextSideInputs:
        """chunks: (request, first new position, new tokens, every page through the last new token) per sequence."""
        side = FlashNextSideInputs()
        if self.ple is not None:
            side.ple_history = torch.tensor([self._history(req, start) for req, start, _, _ in chunks], dtype=torch.int64)
        if self.budget:
            side.qsa_seqs = [(start, n, pages) for _, start, n, pages in chunks]
        return side

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

    def fill(self, side: FlashNextSideInputs, reqs, drafts: torch.Tensor | None, seqs, rows: int) -> None:
        """Copy one step's values into a graph's static side inputs: `rows` real rows (padded rows keep page 0)."""
        if side.ple_rows is not None:
            side.ple_rows[:rows].copy_(self._decode_rows(reqs) if drafts is None else self._verify_rows(reqs, drafts.tolist()))
        if side.qsa_members is not None:
            from sharar.model.flashnext.indexer import FlashNextIndexer
            members, writes = FlashNextIndexer.static_rows(seqs, self.ratio, self.device)
            side.qsa_members[:rows].copy_(members)
            side.qsa_writes[:rows].copy_(writes)
            side.qsa_members[rows:].zero_()
            side.qsa_writes[rows:].zero_()

    def selecting(self, side: FlashNextSideInputs, table: torch.Tensor, indptr: torch.Tensor, row_seq: torch.Tensor,
                  capacity: int) -> FlashNextSideInputs:
        """The selecting variant of a graph's side inputs: rows find their pages through the plan's own buffers."""
        return dataclasses.replace(side, qsa_table=table, qsa_indptr=indptr, qsa_row_seq=row_seq, qsa_gcap=capacity // self.ratio)
