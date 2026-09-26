"""Fixed-shape target verification graphs shared by every proposer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sharar.engine import graph_set
from sharar.engine.context import ForwardContext
from sharar.engine.graphs import _qsa_static, capture_sizes, check_context_capacity, fill_qsa, sparse_ctx, sparse_graphs

if TYPE_CHECKING:
    import flashinfer

def _prefill_wrapper(ws: torch.Tensor, st: dict) -> flashinfer.BatchPrefillWithPagedKVCacheWrapper:
    import flashinfer

    return flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws, "NHD", use_cuda_graph=True,
        qo_indptr_buf=st["qo_indptr"], paged_kv_indptr_buf=st["kv_indptr"],
        paged_kv_indices_buf=st["kv_indices"], paged_kv_last_page_len_buf=st["last_page_len"],
    )


class VerifyGraphs:
    def __init__(self, spec, depth=None):
        self.spec = spec
        self.ex = spec.ex
        self.device = spec.device
        depth = spec.k if depth is None else depth
        self.s_len = depth + 1
        self.max_context = spec.ex.max_context + depth
        self.sizes = capture_sizes(spec.ex.max_batch)
        self._pool = torch.cuda.graph_pool_handle()
        self._ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self.sparse = sparse_graphs(spec.ex.target)
        self._st: dict[int, dict] = {}
        self._graphs: dict[tuple[int, bool], torch.cuda.CUDAGraph] = {}

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
            "ple_rows": (torch.zeros((b * s, self.ex._ple.heads, 160), dtype=torch.uint8, device=d) if getattr(self.ex, "_ple", None) is not None else None),
            **_qsa_static(self.ex.target, b * s, d),
        }
        st["wrapper"] = _prefill_wrapper(self._ws, st)
        st["ctx"] = ForwardContext(
            feature_request=self.spec.proposer.requirements.features,
            mode="verify", kv_pool=self.ex.kv_pool, state_pool=self.ex.state_pool, positions=st["positions"],
            pages=st["pages"], seq_len=b * s, cu_seqlens=st["qo_indptr"], seq_slots=st["slots"],
            seq_prefix_lens=[0] * b, seq_lens=[s] * b, verify_steps=s,
            verify_scratch=torch.arange(b, dtype=torch.int32, device=d), inter_ssm=self.spec.inter_ssm,
            verify_replay=self.spec.replay.layer_inputs if self.spec.replay is not None else {},
            verify_conv_windows=self.spec.conv_windows, verify_aux_windows=self.spec.aux_windows,
            paged_prefill_wrapper=st["wrapper"], ple_rows=st["ple_rows"],
            qsa_members=st["qsa_members"], qsa_writes=st["qsa_writes"],
        )
        if self.sparse:
            st["row_seq"] = torch.arange(b * s, device=d) // s
            st["ctx_sparse"] = sparse_ctx(st["ctx"], st["kv_indices"], st["kv_indptr"], st["row_seq"], self.max_context,
                                          self.ex.target.indexer_compress_ratio)
        return st

    def _plan(self, st: dict, kv_all: list[torch.Tensor]) -> None:
        check_context_capacity([p.shape[0] for p in kv_all], self.max_context)
        d = self.device
        b, s = len(kv_all), self.s_len
        # Host index pointers: plan() reads them on the CPU (a device tensor would sync).
        indptr = torch.tensor([0] + list(torch.tensor([p.shape[0] for p in kv_all]).cumsum(0).tolist()), dtype=torch.int32)
        self.ex.attention.paged_prefill(st["wrapper"], torch.arange(0, b * s + 1, s, dtype=torch.int32), indptr, torch.cat(kv_all),
                                        torch.ones(b, dtype=torch.int32), causal=True, guard=False, page_size=1)  # graph-bound buffers

    def capture(self, log=None) -> None:
        for b in self.sizes:
            st = self._static(b)
            self._st[b] = st
            # dummy plan: each row attends to page 0 plus its own rows (pages 0)
            self._plan(st, [torch.zeros(self.s_len + 1, dtype=torch.int32, device=self.device) for _ in range(b)])

            def warm():
                self.ex.model.logits(self.ex.model.forward(st["tokens"], st["ctx"])["norm"]).argmax(-1)

            def body():
                out = self.ex.model.forward(st["tokens"], st["ctx"])
                st["norm"] = out["norm"]
                st["features"] = out.get("features")
                logits = self.ex.model.logits(out["norm"])
                st["argmax"] = logits.argmax(-1)
                st["logits"] = logits  # materialized by the graph anyway; traces read the target's top-k from it
            self._graphs[b, False] = graph_set.capture(self._pool, body, warmup=warm)
            if self.sparse:  # the selecting variant writes the same outputs (device acceptance reads them)
                self._graphs[b, True] = graph_set.capture(self._pool, lambda: self._sparse_body(st))
            if log:
                log(f"  captured {'dense and selecting ' if self.sparse else ''}verify graph for batch {b}")

    @torch.inference_mode()  # the dense capture's outputs are inference tensors
    def _sparse_body(self, st: dict) -> None:
        out = self.ex.model.forward(st["tokens"], st["ctx_sparse"])
        st["norm"].copy_(out["norm"])
        for lid, rows in (out.get("features") or {}).items():
            st["features"][lid].copy_(rows)
        logits = self.ex.model.logits(out["norm"])
        st["logits"].copy_(logits)
        st["argmax"].copy_(logits.argmax(-1))

    def pick(self, b: int) -> int:
        return graph_set.pick(self.sizes, b)

    def run(self, tokens: torch.Tensor, positions: torch.Tensor, pages: torch.Tensor, slots: torch.Tensor, kv_all: list[torch.Tensor], *, sampled=False,
            ple_rows: torch.Tensor | None = None, qsa: tuple[torch.Tensor, torch.Tensor] | None = None, sparse: bool = False):
        """Returns (norm rows [b*s, H], argmax or logits [b*s(, V)], ctx, features,
        logits [b*s, V]) for the real rows; padded rows use slot 0 and page 0.
        The logits view is overwritten by the next replay. `sparse` replays the
        selecting graph (some row past the indexer budget)."""
        b = slots.shape[0]
        size = self.pick(b)
        st = self._st[size]
        s = self.s_len
        n = b * s
        if ple_rows is not None:
            st["ple_rows"][:n].copy_(ple_rows)
        fill_qsa(st, qsa, n)
        st["tokens"][:n].copy_(tokens)
        st["positions"][:n].copy_(positions)
        st["pages"][:n].copy_(pages)
        st["slots"][:b].copy_(slots)
        if size > b:
            st["tokens"][n:].zero_()
            st["positions"][n:].zero_()
            st["pages"][n:].zero_()
            st["slots"][b:].zero_()
        pad_kv = [torch.zeros(s + 1, dtype=torch.int32, device=self.device) for _ in range(size - b)]
        self._plan(st, list(kv_all) + pad_kv)
        self._graphs[size, sparse].replay()
        layers = {lid: value[:n] for lid, value in st["features"].items()} if st["features"] else None
        return st["norm"][:n], st["logits" if sampled else "argmax"][:n], st["ctx"], layers, st["logits"][:n]
