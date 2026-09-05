"""Fixed-batch MTP graphs, including the first pending-token draft step."""

from __future__ import annotations

import torch

from sharar import config
from sharar.engine.graphs import capture_sizes, check_context_capacity
from sharar.engine.spec_graphs import _prefill_wrapper


class DraftGraphs:
    """One-row-per-request graphs for every step, including the pending input."""

    def __init__(self, proposer):
        self.proposer = proposer
        self.mtp = proposer.head
        self.device = proposer.context.device
        self.k = proposer.k
        self.max_context = proposer.context.max_context + proposer.k
        self.sizes = capture_sizes(proposer.context.max_batch)
        self._pool = torch.cuda.graph_pool_handle()
        self._ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self._st: dict[int, dict] = {}
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}

    def _static(self, b: int) -> dict:
        d = self.device
        st = {
            "tokens": torch.zeros(b, dtype=torch.int64, device=d),
            "hidden": torch.zeros(b, config.HIDDEN_SIZE, dtype=torch.bfloat16, device=d),
            "positions": torch.zeros(b, dtype=torch.int64, device=d),
            "pages": torch.zeros(b, dtype=torch.int32, device=d),
            "qo_indptr": torch.arange(0, b + 1, dtype=torch.int32, device=d),
            "kv_indptr": torch.zeros(b + 1, dtype=torch.int32, device=d),
            "kv_indices": torch.zeros(b * self.max_context, dtype=torch.int32, device=d),
            "last_page_len": torch.ones(b, dtype=torch.int32, device=d),
        }
        st["wrapper"] = _prefill_wrapper(self._ws, st)
        return st

    def _plan(self, st: dict, kv_all: list[torch.Tensor]) -> None:
        check_context_capacity([p.shape[0] for p in kv_all], self.max_context)
        d = self.device
        b = len(kv_all)
        indptr = torch.tensor([0] + list(torch.tensor([p.shape[0] for p in kv_all]).cumsum(0).tolist()), dtype=torch.int32)  # host: no sync in plan()
        st["wrapper"].plan(
            torch.arange(0, b + 1, dtype=torch.int32), indptr, torch.cat(kv_all), torch.ones(b, dtype=torch.int32),
            config.NUM_ATTENTION_HEADS, config.NUM_KV_HEADS, config.HEAD_DIM, 1,
            causal=True, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16, sm_scale=config.HEAD_DIM**-0.5,
        )

    def _forward(self, st: dict) -> tuple[torch.Tensor, torch.Tensor]:
        mtp = self.mtp
        saved = mtp._attn
        mtp._attn = st["wrapper"]
        try:
            d = mtp.forward_static(st["tokens"], st["hidden"], st["positions"], st["pages"])
        finally:
            mtp._attn = saved
        return d, mtp.logits(d).argmax(-1)

    def capture(self, log=None) -> None:
        for b in self.sizes:
            st = self._static(b)
            self._plan(st, [torch.zeros(2, dtype=torch.int32, device=self.device) for _ in range(b)])
            for _ in range(2):
                self._forward(st)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self._pool):
                st["out_hidden"], st["out_argmax"] = self._forward(st)
            self._st[b] = st
            self._graphs[b] = g
            torch.cuda.synchronize()
            if log:
                log(f"  captured draft graph for batch {b}")

    def pick(self, b: int) -> int:
        for s in self.sizes:
            if s >= b:
                return s
        raise ValueError(b)

    def run(self, tokens: torch.Tensor, hidden: torch.Tensor, positions: torch.Tensor, pages: torch.Tensor, kv_all: list[torch.Tensor]):
        b = tokens.shape[0]
        size = self.pick(b)
        st = self._st[size]
        st["tokens"][:b].copy_(tokens)
        st["hidden"][:b].copy_(hidden)
        st["positions"][:b].copy_(positions)
        st["pages"][:b].copy_(pages)
        if size > b:
            st["tokens"][b:].zero_()
            st["positions"][b:].zero_()
            st["pages"][b:].zero_()
        pad = [torch.zeros(2, dtype=torch.int32, device=self.device) for _ in range(size - b)]
        self._plan(st, list(kv_all) + pad)
        self._graphs[size].replay()
        return st["out_hidden"][:b], st["out_argmax"][:b]
