"""CUDA graphs for the decode step at fixed batch sizes.

One graph per batch size in CAPTURE_SIZES (two for a sparse-attention target: the
dense one, used while every row is inside the indexer budget, and one that selects
on the device, `kernels/qsa_select.py`). A decode batch of b rows is padded
to the smallest captured size with dummy rows that use state slot 0 and KV
page 0, both reserved for that purpose. FlashInfer's decode wrapper is built
with use_cuda_graph=True and fixed index buffers; plan() runs before every
replay (host side) and run() is inside the graph.
"""

from __future__ import annotations

import dataclasses
import os

import torch

from sharar.engine.context import ForwardContext
from sharar.engine.contracts import FeatureRequest
from sharar.engine.profile import ServingProfile
from sharar.cache.kv_pool import KV_DTYPE

CAPTURE_SIZES = (1, 2, 4)


def capture_sizes(max_batch: int) -> tuple[int, ...]:
    return ServingProfile(max_batch=max_batch).graph_sizes


def _qsa_static(target, rows: int, device) -> dict:
    """Index-key maintenance buffers for a sparse-attention target (none otherwise)."""
    if not target.indexer_budget:
        return {"qsa_members": None, "qsa_writes": None}
    return {"qsa_members": torch.zeros((rows, target.indexer_compress_ratio), dtype=torch.int32, device=device),
            "qsa_writes": torch.zeros(rows, dtype=torch.int32, device=device)}


def fill_qsa(st: dict, qsa, rows: int) -> None:
    """Copy one step's (members, writes) into the graph's buffers; padded rows use page 0."""
    if st.get("qsa_members") is None:
        return
    if qsa is None:
        raise ValueError("a sparse-attention target's graph needs its index-key rows")
    st["qsa_members"][:rows].copy_(qsa[0])
    st["qsa_writes"][:rows].copy_(qsa[1])
    st["qsa_members"][rows:].zero_()
    st["qsa_writes"][rows:].zero_()


def sparse_graphs(target) -> bool:
    """Whether a sparse-attention target also captures selecting graphs (SHARAR_QSA_SPARSE_GRAPHS=0: rows
    past the budget run eagerly)."""
    return bool(target.indexer_budget) and os.environ.get("SHARAR_QSA_SPARSE_GRAPHS", "1") != "0"


def sparse_ctx(ctx: ForwardContext, table: torch.Tensor, indptr: torch.Tensor, row_seq: torch.Tensor, capacity: int, ratio: int) -> ForwardContext:
    """The selecting variant of a graph's context: rows find their pages through the plan's own buffers."""
    return dataclasses.replace(ctx, qsa_table=table, qsa_indptr=indptr, qsa_row_seq=row_seq, qsa_gcap=capacity // ratio)


def check_context_capacity(counts: list[int], capacity: int) -> None:
    if not counts or any(n < 1 or n > capacity for n in counts):
        raise ValueError(f"KV lengths {counts} exceed graph capacity {capacity}")


class DecodeGraphs:
    def __init__(self, executor):
        self.ex = executor
        self.device = executor.device
        self.sizes = capture_sizes(executor.max_batch)
        self.max_context = executor.max_context
        self._pool = torch.cuda.graph_pool_handle()
        self.sparse = sparse_graphs(executor.target)
        self._graphs: dict[tuple[int, bool], torch.cuda.CUDAGraph] = {}
        self._static: dict[int, dict] = {}
        self._wrappers: dict[int, flashinfer.BatchDecodeWithPagedKVCacheWrapper] = {}
        self._ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)

    def _make_static(self, b: int) -> dict:
        d = self.device
        return {
            "tokens": torch.zeros(b, dtype=torch.int64, device=d),
            "positions": torch.zeros(b, dtype=torch.int64, device=d),
            "slots": torch.zeros(b, dtype=torch.int32, device=d),
            "out_pages": torch.zeros(b, dtype=torch.int32, device=d),
            "indptr": torch.zeros(b + 1, dtype=torch.int32, device=d),
            "indices": torch.zeros(b * self.max_context, dtype=torch.int32, device=d),
            "last_page_len": torch.ones(b, dtype=torch.int32, device=d),
            "last_page_len_host": torch.ones(b, dtype=torch.int32),
            "logits": None,
            "ple_rows": (torch.zeros((b, self.ex._ple.heads, 160), dtype=torch.uint8, device=d) if getattr(self.ex, "_ple", None) is not None else None),
            **_qsa_static(self.ex.target, b, d),
            "row_seq": torch.arange(b, dtype=torch.int64, device=d),
        }

    def _plan(self, b: int, counts: list[int], pages: list[torch.Tensor]) -> None:
        check_context_capacity(counts, self.max_context)
        st = self._static[b]
        indptr = torch.tensor([0] + list(torch.tensor(counts).cumsum(0).tolist()), dtype=torch.int32)  # host: no sync in plan()
        indices = torch.cat(pages)
        self._wrappers[b].plan(
            indptr,
            indices,
            st["last_page_len_host"],
            self.ex.target.num_attention_heads,
            self.ex.target.num_kv_heads,
            self.ex.target.head_dim,
            self.ex.kv_pool.page_size,
            pos_encoding_mode="NONE",
            q_data_type=torch.bfloat16,
            kv_data_type=KV_DTYPE,
            sm_scale=self.ex.target.head_dim**-0.5,
        )

    def _ctx(self, b: int, sparse: bool = False) -> ForwardContext:
        st = self._static[b]
        ctx = ForwardContext(
            feature_request=self.ex.spec.proposer.requirements.features if self.ex.spec is not None else FeatureRequest(),
            mode="decode",
            kv_pool=self.ex.kv_pool,
            state_pool=self.ex.state_pool,
            positions=st["positions"],
            slots=st["slots"],
            out_pages=st["out_pages"],
            decode_wrapper=self._wrappers[b],
            batch_size=b,
            ple_rows=st["ple_rows"],
            qsa_members=st["qsa_members"],
            qsa_writes=st["qsa_writes"],
        )
        if sparse:
            ctx = sparse_ctx(ctx, st["indices"], st["indptr"], st["row_seq"], self.max_context, self.ex.target.indexer_compress_ratio)
        return ctx

    def capture(self, log=print) -> None:
        import flashinfer
        for b in self.sizes:
            st = self._make_static(b)
            self._static[b] = st
            self._wrappers[b] = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                self._ws,
                "NHD",
                use_cuda_graph=True,
                use_tensor_cores=True,
                paged_kv_indptr_buffer=st["indptr"],
                paged_kv_indices_buffer=st["indices"],
                paged_kv_last_page_len_buffer=st["last_page_len"],
            )
            # Dummy rows: slot 0, one page (page 0). Position 0 keeps rope in range.
            dummy_pages = [torch.zeros(1, dtype=torch.int32, device=self.device) for _ in range(b)]
            self._plan(b, [1] * b, dummy_pages)
            for sparse in (False, True) if self.sparse else (False,):
                ctx = self._ctx(b, sparse)
                # Warm up (Triton/FlashInfer JIT, allocator) then capture.
                for _ in range(2):
                    out = self.ex.model.forward(st["tokens"], ctx)
                    self.ex.model.logits(out["norm"])
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=self._pool):
                    out = self.ex.model.forward(st["tokens"], ctx)
                    st["logits", sparse] = self.ex.model.logits(out["norm"])
                    st["out", sparse] = out
                self._graphs[b, sparse] = g
                torch.cuda.synchronize()
                if log:
                    log(f"  captured {'selecting ' if sparse else ''}decode graph for batch {b}")

    def pick(self, b: int) -> int:
        for s in self.sizes:
            if s >= b:
                return s
        raise ValueError(f"batch {b} exceeds captured sizes {self.sizes}")

    def run(self, tokens: list[int], positions: list[int], slots: list[int], out_pages: torch.Tensor, pages: list[torch.Tensor], *, return_features=False,
            ple_rows: torch.Tensor | None = None, qsa: tuple[torch.Tensor, torch.Tensor] | None = None, sparse: bool = False):
        """Returns logits [b, V] for the b real rows; `sparse` replays the selecting graph."""
        b = len(tokens)
        size = self.pick(b)
        st = self._static[size]
        pad = size - b
        if ple_rows is not None:
            st["ple_rows"][:b].copy_(ple_rows)
        fill_qsa(st, qsa, b)
        st["tokens"].copy_(torch.tensor(tokens + [0] * pad, dtype=torch.int64))
        st["positions"].copy_(torch.tensor(positions + [0] * pad, dtype=torch.int64))
        st["slots"].copy_(torch.tensor(slots + [0] * pad, dtype=torch.int32))
        st["out_pages"][:b].copy_(out_pages)
        if pad:
            st["out_pages"][b:].zero_()
        counts = [p.shape[0] for p in pages] + [1] * pad
        all_pages = list(pages) + [torch.zeros(1, dtype=torch.int32, device=self.device) for _ in range(pad)]
        self._plan(size, counts, all_pages)
        self._graphs[size, sparse].replay()
        out, logits = st["out", sparse], st["logits", sparse]
        if return_features:
            features = {"norm": out["norm"][:b]}
            if "features" in out:
                features["features"] = {lid: rows[:b] for lid, rows in out["features"].items()}
            return logits[:b], features
        return logits[:b]
