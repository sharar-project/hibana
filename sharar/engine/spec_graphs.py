"""CUDA graphs for the speculative path.

Verify: one graph per batch size over B*(k+1) rows (target forward, lm_head,
argmax). Draft: one graph per (batch size, step j >= 2) for the single-row
chain steps; step 1 ingests a variable number of rows and stays eager. Both
use FlashInfer wrappers in graph mode with fixed index buffers; plan() runs on
the host before every replay.
"""

from __future__ import annotations

import flashinfer
import torch

from sharar import config
from sharar.engine.context import ForwardContext
from sharar.memory.kv_pool import KV_DTYPE

SIZES = (1, 2, 4)


def _prefill_wrapper(ws: torch.Tensor, st: dict) -> flashinfer.BatchPrefillWithPagedKVCacheWrapper:
    return flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws, "NHD", use_cuda_graph=True,
        qo_indptr_buf=st["qo_indptr"], paged_kv_indptr_buf=st["kv_indptr"],
        paged_kv_indices_buf=st["kv_indices"], paged_kv_last_page_len_buf=st["last_page_len"],
    )


class VerifyGraphs:
    def __init__(self, spec, max_context: int = 32768 + 64):
        self.spec = spec
        self.ex = spec.ex
        self.device = spec.device
        self.s_len = spec.steps
        self.max_context = max_context
        self._pool = torch.cuda.graph_pool_handle()
        self._ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self._st: dict[int, dict] = {}
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}

    def _static(self, b: int) -> dict:
        d, s = self.device, self.s_len
        st = {
            "tokens": torch.zeros(b * s, dtype=torch.int64, device=d),
            "positions": torch.zeros(b * s, dtype=torch.int64, device=d),
            "pages": torch.zeros(b * s, dtype=torch.int32, device=d),
            "slots": torch.zeros(b, dtype=torch.int32, device=d),
            "qo_indptr": torch.arange(0, b * s + 1, s, dtype=torch.int32, device=d),
            "kv_indptr": torch.zeros(b + 1, dtype=torch.int32, device=d),
            "kv_indices": torch.zeros(b * self.max_context, dtype=torch.int32, device=d),
            "last_page_len": torch.ones(b, dtype=torch.int32, device=d),
        }
        st["wrapper"] = _prefill_wrapper(self._ws, st)
        st["ctx"] = ForwardContext(
            mode="verify", kv_pool=self.ex.kv_pool, state_pool=self.ex.state_pool, positions=st["positions"],
            pages=st["pages"], seq_len=b * s, cu_seqlens=st["qo_indptr"], seq_slots=st["slots"],
            seq_prefix_lens=[0] * b, seq_lens=[s] * b, verify_steps=s,
            verify_scratch=torch.arange(b, dtype=torch.int32, device=d), inter_ssm=self.spec.inter_ssm,
            paged_prefill_wrapper=st["wrapper"],
        )
        return st

    def _plan(self, st: dict, kv_all: list[torch.Tensor]) -> None:
        d = self.device
        indptr = torch.tensor([0] + list(torch.tensor([p.shape[0] for p in kv_all]).cumsum(0).tolist()), dtype=torch.int32, device=d)
        st["wrapper"].plan(
            st["qo_indptr"], indptr, torch.cat(kv_all), st["last_page_len"],
            config.NUM_ATTENTION_HEADS, config.NUM_KV_HEADS, config.HEAD_DIM, 1,
            causal=True, q_data_type=torch.bfloat16, kv_data_type=KV_DTYPE, sm_scale=config.HEAD_DIM**-0.5,
        )

    def capture(self, log=None) -> None:
        for b in SIZES:
            st = self._static(b)
            self._st[b] = st
            # dummy plan: each row attends to page 0 plus its own rows (pages 0)
            self._plan(st, [torch.zeros(self.s_len + 1, dtype=torch.int32, device=self.device) for _ in range(b)])
            for _ in range(2):
                out = self.ex.model.forward(st["tokens"], st["ctx"])
                self.ex.model.logits(out["norm"]).argmax(-1)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self._pool):
                out = self.ex.model.forward(st["tokens"], st["ctx"])
                st["norm"] = out["norm"]
                st["argmax"] = self.ex.model.logits(out["norm"]).argmax(-1)
            self._graphs[b] = g
            torch.cuda.synchronize()
            if log:
                log(f"  captured verify graph for batch {b}")

    def pick(self, b: int) -> int:
        for s in SIZES:
            if s >= b:
                return s
        raise ValueError(b)

    def run(self, tokens: torch.Tensor, positions: torch.Tensor, pages: torch.Tensor, slots: list[int], kv_all: list[torch.Tensor]):
        """Returns (norm rows [b*s, H], argmax [b*s]) for the real rows; padded
        rows use slot 0 and page 0."""
        b = len(slots)
        size = self.pick(b)
        st = self._st[size]
        s = self.s_len
        n = b * s
        st["tokens"][:n].copy_(tokens)
        st["positions"][:n].copy_(positions)
        st["pages"][:n].copy_(pages)
        st["slots"][:b].copy_(torch.tensor(slots, dtype=torch.int32))
        if size > b:
            st["tokens"][n:].zero_()
            st["positions"][n:].zero_()
            st["pages"][n:].zero_()
            st["slots"][b:].zero_()
        pad_kv = [torch.zeros(s + 1, dtype=torch.int32, device=self.device) for _ in range(size - b)]
        self._plan(st, list(kv_all) + pad_kv)
        st["ctx"].verify_conv_windows = st["ctx"].verify_conv_windows  # filled by the graph
        self._graphs[size].replay()
        return st["norm"][:n], st["argmax"][:n], st["ctx"]


class DraftGraphs:
    """Graphs for draft steps j = 2..k at each batch size: one row per request."""

    def __init__(self, spec, max_context: int = 32768 + 64):
        self.spec = spec
        self.mtp = spec.mtp
        self.device = spec.device
        self.k = spec.k
        self.max_context = max_context
        self._pool = torch.cuda.graph_pool_handle()
        self._ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self._st: dict[tuple[int, int], dict] = {}
        self._graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}

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
        d = self.device
        indptr = torch.tensor([0] + list(torch.tensor([p.shape[0] for p in kv_all]).cumsum(0).tolist()), dtype=torch.int32, device=d)
        st["wrapper"].plan(
            st["qo_indptr"], indptr, torch.cat(kv_all), st["last_page_len"],
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
        for b in SIZES:
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
        for s in SIZES:
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
