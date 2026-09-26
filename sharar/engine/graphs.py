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

import torch

from sharar.engine import graph_set
from sharar.engine.context import ForwardContext
from sharar.engine.contracts import FeatureRequest
from sharar.engine.profile import ServingProfile

CAPTURE_SIZES = (1, 2, 4)


def capture_sizes(max_batch: int) -> tuple[int, ...]:
    return ServingProfile(max_batch=max_batch).graph_sizes


def graph_variants(executor) -> tuple[str, ...]:
    """The graph variants to capture: the model's side inputs' (Flash-Next past its indexer budget adds
    "selecting"; SHARAR_QSA_SPARSE_GRAPHS=0 leaves those rows eager), else only "dense"."""
    return executor.extras.variants if executor.extras is not None else ("dense",)


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
        self.variants = graph_variants(executor)
        self._graphs: dict[tuple[int, str], torch.cuda.CUDAGraph] = {}
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
            "side": self.ex.extras.buffers(b) if self.ex.extras is not None else None,  # the model's static side inputs
            "row_seq": torch.arange(b, dtype=torch.int64, device=d),
        }

    def _plan(self, b: int, counts: list[int], pages: list[torch.Tensor]) -> None:
        check_context_capacity(counts, self.max_context)
        st = self._static[b]
        indptr = torch.tensor([0] + list(torch.tensor(counts).cumsum(0).tolist()), dtype=torch.int32)  # host: no sync in plan()
        indices = torch.cat(pages)
        self.ex.attention.decode(self._wrappers[b], indptr, indices, st["last_page_len_host"], guard=False)  # graph-bound buffers

    def _ctx(self, b: int, variant: str = "dense") -> ForwardContext:
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
            extras=st["side"],
        )
        if variant == "selecting":
            ctx = dataclasses.replace(ctx, extras=self.ex.extras.selecting(st["side"], st["indices"], st["indptr"], st["row_seq"], self.max_context))
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
            for variant in self.variants:
                ctx = self._ctx(b, variant)

                def warm():
                    self.ex.model.logits(self.ex.model.forward(st["tokens"], ctx)["norm"])

                def body():
                    out = self.ex.model.forward(st["tokens"], ctx)
                    st["logits", variant] = self.ex.model.logits(out["norm"])
                    st["out", variant] = out
                self._graphs[b, variant] = graph_set.capture(self._pool, body, warmup=warm)
                if log:
                    log(f"  captured {'selecting ' if variant == 'selecting' else ''}decode graph for batch {b}")

    def pick(self, b: int) -> int:
        return graph_set.pick(self.sizes, b)

    def run(self, tokens: list[int], positions: list[int], slots: list[int], out_pages: torch.Tensor, pages: list[torch.Tensor], *, return_features=False,
            side=None, variant: str = "dense"):
        """Returns logits [b, V] for the b real rows; `side` is (requests, seqs) for the model's side inputs,
        `variant` the graph to replay ("selecting": some row past Flash-Next's indexer budget)."""
        b = len(tokens)
        size = self.pick(b)
        st = self._static[size]
        pad = size - b
        if st["side"] is not None:
            self.ex.extras.fill(st["side"], side[0], None, side[1], b)
        st["tokens"].copy_(torch.tensor(tokens + [0] * pad, dtype=torch.int64))
        st["positions"].copy_(torch.tensor(positions + [0] * pad, dtype=torch.int64))
        st["slots"].copy_(torch.tensor(slots + [0] * pad, dtype=torch.int32))
        st["out_pages"][:b].copy_(out_pages)
        if pad:
            st["out_pages"][b:].zero_()
        counts = [p.shape[0] for p in pages] + [1] * pad
        all_pages = list(pages) + [torch.zeros(1, dtype=torch.int32, device=self.device) for _ in range(pad)]
        self._plan(size, counts, all_pages)
        self._graphs[size, variant].replay()
        out, logits = st["out", variant], st["logits", variant]
        if return_features:
            features = {"norm": out["norm"][:b]}
            if "features" in out:
                features["features"] = {lid: rows[:b] for lid, rows in out["features"].items()}
            return logits[:b], features
        return logits[:b]
