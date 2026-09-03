"""Single-process executor: prefill one request, decode a batch, generate.

Eager path (Phase 2 step 3). CUDA graphs wrap the decode step later in the
phase; the ForwardContext is already built from tensors that can be static.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import flashinfer
import torch

from sharar import config
from sharar.engine.context import ForwardContext
from sharar.engine.graphs import DecodeGraphs
from sharar.engine.sampler import Sampler, SamplingParams
from sharar.memory.kv_pool import KV_DTYPE, KvPool
from sharar.memory.state_pool import StatePool
from sharar.model.qwen38 import Qwen38


@dataclass
class Request:
    prompt_ids: list[int]
    params: SamplingParams
    output_ids: list[int] = field(default_factory=list)
    slot: int = -1
    pages: torch.Tensor | None = None  # [context_len] int32
    finished: bool = False
    finish_reason: str = ""
    prefill_seconds: float = 0.0

    @property
    def context_len(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)

    def push(self, token: int) -> None:
        self.output_ids.append(token)
        if not self.params.ignore_eos and token in self.params.stop_token_ids:
            self.finished, self.finish_reason = True, "stop"
        elif len(self.output_ids) >= self.params.max_new_tokens:
            self.finished, self.finish_reason = True, "length"


class Executor:
    def __init__(self, model: Qwen38, num_pages: int = 65536, num_slots: int = 8, max_batch: int = 4):
        self.model = model
        self.device = model.device
        self.kv_pool = KvPool(num_pages, self.device)
        self.state_pool = StatePool(num_slots, self.device)
        self.sampler = Sampler(self.device)
        self.max_batch = max_batch
        self._decode_ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        # The CUDA-core decode kernel rejects GQA group size 6 (24/4 heads); the
        # tensor-core path is what the reference runs for this model too.
        self._decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(self._decode_ws, "NHD", use_tensor_cores=True)
        self.graphs: DecodeGraphs | None = None
        self.use_graphs = False

    def capture_graphs(self, log=print) -> None:
        self.graphs = DecodeGraphs(self)
        self.graphs.capture(log=log)
        self.use_graphs = True

    # ---- lifecycle ----
    def admit(self, prompt_ids: list[int], params: SamplingParams) -> Request:
        req = Request(prompt_ids=list(prompt_ids), params=params)
        req.slot = self.state_pool.alloc()
        req.pages = self.kv_pool.alloc(len(prompt_ids))
        return req

    def release(self, req: Request) -> None:
        if req.pages is not None:
            self.kv_pool.free(req.pages)
            req.pages = None
        if req.slot >= 0:
            self.state_pool.free(req.slot)
            req.slot = -1

    # ---- steps ----
    def prefill(self, req: Request) -> int:
        started = time.perf_counter()
        t = len(req.prompt_ids)
        ids = torch.tensor(req.prompt_ids, dtype=torch.int64, device=self.device)
        ctx = ForwardContext(
            mode="prefill",
            kv_pool=self.kv_pool,
            state_pool=self.state_pool,
            positions=torch.arange(t, dtype=torch.int64, device=self.device),
            slot=req.slot,
            pages=req.pages,
            seq_len=t,
        )
        out = self.model.forward(ids, ctx)
        logits = self.model.logits(out["norm"][-1:])
        token = int(self.sampler.sample(logits, [req.params]).item())
        torch.cuda.synchronize()
        req.prefill_seconds = time.perf_counter() - started
        req.push(token)
        return token

    def _plan_decode(self, reqs: list[Request]) -> None:
        counts = [r.context_len for r in reqs]  # pages per row including the new token's page
        indptr = torch.tensor([0] + list(torch.tensor(counts).cumsum(0).tolist()), dtype=torch.int32, device=self.device)
        indices = torch.cat([r.pages for r in reqs])
        last_page_len = torch.ones(len(reqs), dtype=torch.int32, device=self.device)
        self._decode_wrapper.plan(
            indptr,
            indices,
            last_page_len,
            config.NUM_ATTENTION_HEADS,
            config.NUM_KV_HEADS,
            config.HEAD_DIM,
            self.kv_pool.page_size,
            pos_encoding_mode="NONE",
            q_data_type=torch.bfloat16,
            kv_data_type=KV_DTYPE,
            sm_scale=config.HEAD_DIM**-0.5,
        )

    def decode_step(self, reqs: list[Request]) -> list[int]:
        """Process the last generated token of every request and sample the next."""
        assert 0 < len(reqs) <= self.max_batch
        new_pages = self.kv_pool.alloc(len(reqs))
        for i, r in enumerate(reqs):
            r.pages = torch.cat([r.pages, new_pages[i : i + 1]])
        # The token being processed sits at position context_len - 1.
        positions = torch.tensor([r.context_len - 1 for r in reqs], dtype=torch.int64, device=self.device)
        if self.use_graphs and self.graphs is not None:
            logits = self.graphs.run(
                [r.output_ids[-1] for r in reqs],
                [r.context_len - 1 for r in reqs],
                [r.slot for r in reqs],
                new_pages,
                [r.pages for r in reqs],
            )
            next_tokens = self.sampler.sample(logits, [r.params for r in reqs]).tolist()
            for r, tok in zip(reqs, next_tokens):
                r.push(int(tok))
            return next_tokens
        tokens = torch.tensor([r.output_ids[-1] for r in reqs], dtype=torch.int64, device=self.device)
        self._plan_decode(reqs)
        ctx = ForwardContext(
            mode="decode",
            kv_pool=self.kv_pool,
            state_pool=self.state_pool,
            positions=positions,
            slots=torch.tensor([r.slot for r in reqs], dtype=torch.int32, device=self.device),
            out_pages=new_pages,
            decode_wrapper=self._decode_wrapper,
            batch_size=len(reqs),
        )
        out = self.model.forward(tokens, ctx)
        logits = self.model.logits(out["norm"])
        next_tokens = self.sampler.sample(logits, [r.params for r in reqs]).tolist()
        for r, tok in zip(reqs, next_tokens):
            r.push(int(tok))
        return next_tokens

    # ---- convenience ----
    def generate(self, prompt_ids: list[int], params: SamplingParams) -> Request:
        return self.generate_batch([(prompt_ids, params)])[0]

    def generate_batch(self, items: list[tuple[list[int], SamplingParams]]) -> list[Request]:
        reqs = [self.admit(ids, p) for ids, p in items]
        for r in reqs:
            self.prefill(r)
        active = [r for r in reqs if not r.finished]
        torch.cuda.synchronize()
        started = time.perf_counter()
        steps = 0
        while active:
            self.decode_step(active)
            steps += 1
            active = [r for r in active if not r.finished]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        for r in reqs:
            r.decode_seconds = elapsed
            r.decode_steps = steps
        for r in reqs:
            self.release(r)
        return reqs
