"""CUDA graphs for the decode step at fixed batch sizes.

One graph per batch size in CAPTURE_SIZES. A decode batch of b rows is padded
to the smallest captured size with dummy rows that use state slot 0 and KV
page 0, both reserved for that purpose. FlashInfer's decode wrapper is built
with use_cuda_graph=True and fixed index buffers; plan() runs before every
replay (host side) and run() is inside the graph.
"""

from __future__ import annotations

import torch

from sharar import config
from sharar.engine.context import ForwardContext
from sharar.engine.profile import ServingProfile
from sharar.memory.kv_pool import KV_DTYPE

CAPTURE_SIZES = (1, 2, 4)


def capture_sizes(max_batch: int) -> tuple[int, ...]:
    return ServingProfile(max_batch=max_batch).graph_sizes


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
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
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
            config.NUM_ATTENTION_HEADS,
            config.NUM_KV_HEADS,
            config.HEAD_DIM,
            self.ex.kv_pool.page_size,
            pos_encoding_mode="NONE",
            q_data_type=torch.bfloat16,
            kv_data_type=KV_DTYPE,
            sm_scale=config.HEAD_DIM**-0.5,
        )

    def _ctx(self, b: int) -> ForwardContext:
        st = self._static[b]
        return ForwardContext(
            mode="decode",
            kv_pool=self.ex.kv_pool,
            state_pool=self.ex.state_pool,
            positions=st["positions"],
            slots=st["slots"],
            out_pages=st["out_pages"],
            decode_wrapper=self._wrappers[b],
            batch_size=b,
        )

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
            ctx = self._ctx(b)
            # Warm up (Triton/FlashInfer JIT, allocator) then capture.
            for _ in range(2):
                out = self.ex.model.forward(st["tokens"], ctx)
                self.ex.model.logits(out["norm"])
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self._pool):
                out = self.ex.model.forward(st["tokens"], ctx)
                st["logits"] = self.ex.model.logits(out["norm"])
            self._graphs[b] = g
            torch.cuda.synchronize()
            if log:
                log(f"  captured decode graph for batch {b}")

    def pick(self, b: int) -> int:
        for s in self.sizes:
            if s >= b:
                return s
        raise ValueError(f"batch {b} exceeds captured sizes {self.sizes}")

    def run(self, tokens: list[int], positions: list[int], slots: list[int], out_pages: torch.Tensor, pages: list[torch.Tensor]) -> torch.Tensor:
        """Returns logits [b, V] for the b real rows."""
        b = len(tokens)
        size = self.pick(b)
        st = self._static[size]
        pad = size - b
        st["tokens"].copy_(torch.tensor(tokens + [0] * pad, dtype=torch.int64))
        st["positions"].copy_(torch.tensor(positions + [0] * pad, dtype=torch.int64))
        st["slots"].copy_(torch.tensor(slots + [0] * pad, dtype=torch.int32))
        st["out_pages"][:b].copy_(out_pages)
        if pad:
            st["out_pages"][b:].zero_()
        counts = [p.shape[0] for p in pages] + [1] * pad
        all_pages = list(pages) + [torch.zeros(1, dtype=torch.int32, device=self.device) for _ in range(pad)]
        self._plan(size, counts, all_pages)
        self._graphs[size].replay()
        return st["logits"][:b]
