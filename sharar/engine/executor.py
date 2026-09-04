"""Single-process executor: ledger-sized pools, radix prefix cache, prefill with
cached prefixes and state checkpoints, batched decode, CUDA graphs, generate."""

from __future__ import annotations

import time

import flashinfer
import torch

from sharar import config
from sharar.engine.context import ForwardContext
from sharar.engine.graphs import DecodeGraphs
from sharar.engine.request import Request, State
from sharar.engine.sampler import Sampler, SamplingParams
from sharar.memory.kv_pool import KV_DTYPE, KvPool
from sharar.memory.ledger import Ledger
from sharar.memory.radix import Match, RadixCache
from sharar.memory.state_pool import StatePool
from sharar.model.qwen38 import Qwen38

CHECKPOINT_INTERVAL = 2048  # GDN snapshots inside long prefills (multiple of 64)


class Executor:
    def __init__(self, model: Qwen38, num_pages: int | None = None, num_slots: int | None = None, max_batch: int = 4, use_cache: bool = True, log=None):
        self.model = model
        self.device = model.device
        self.ledger = Ledger()
        self.ledger.record_cuda_allocated("weights")
        if num_pages is None or num_slots is None:
            plan = self.ledger.plan(max_running=max_batch)
            num_pages = num_pages or plan.kv_tokens
            num_slots = num_slots or plan.state_slots
        self.kv_pool = KvPool(num_pages, self.device)
        self.ledger.record_cuda_allocated("kv_pool")
        self.state_pool = StatePool(num_slots, self.device)
        self.ledger.record_cuda_allocated("state_pool")
        self.cache = RadixCache(self.kv_pool, self.state_pool, self.device) if use_cache else None
        self.sampler = Sampler(self.device)
        self.max_batch = max_batch
        self._ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        # The CUDA-core decode kernel rejects GQA group size 6 (24/4 heads); the
        # tensor-core path is what the reference runs for this model too.
        self._decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(self._ws, "NHD", use_tensor_cores=True)
        self._prefill_ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self._paged_prefill = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self._prefill_ws, "NHD")
        self.ledger.record_cuda_allocated("workspaces")
        self.graphs: DecodeGraphs | None = None
        self.use_graphs = False
        if log:
            log(f"  pools: {num_pages} KV pages ({self.kv_pool.bytes / 2**30:.2f} GiB), {num_slots} state slots ({num_slots * self.state_pool.bytes_per_slot / 2**30:.2f} GiB)")

    def capture_graphs(self, log=print) -> None:
        self.graphs = DecodeGraphs(self)
        self.graphs.capture(log=log)
        self.ledger.record_cuda_allocated("cuda_graphs")
        self.use_graphs = True

    # ---- lifecycle ----
    def admit(self, prompt_ids: list[int], params: SamplingParams, req: Request | None = None) -> Request:
        if req is None:
            req = Request(prompt_ids=list(prompt_ids), params=params)
        n = len(prompt_ids)
        if self.cache is not None:
            # Keep at least one token to prefill so the last logits exist.
            match = self.cache.match(prompt_ids[:-1])
            if match.length > 0:
                self.cache.lock(match.node, +1)
                req.match = match
                req.prefix_len = match.length
        need_pages = n - req.prefix_len
        if self.cache is not None:
            self.cache.evict(need_pages=need_pages, need_slots=1)
        req.slot = self.state_pool.alloc()
        if req.prefix_len:
            self.state_pool.copy_slot(req.match.snapshot_slot, req.slot)
        own = self.kv_pool.alloc(need_pages)
        req.pages = torch.cat([req.match.pages, own]) if req.prefix_len else own
        req.prefill_pos = req.prefix_len
        req.state = State.PREFILL
        req.t_admit = time.perf_counter()
        return req

    def can_admit(self, prompt_len: int, reserve_tokens: int = 0) -> bool:
        """Enough free pages and a free slot for this prompt (after evicting cache)."""
        need = prompt_len + reserve_tokens
        if self.cache is not None:
            self.cache.evict(need_pages=need, need_slots=1)
        return len(self.kv_pool._free) >= need and len(self.state_pool._free) >= 1

    def retract(self, req: Request) -> None:
        """Give a request's resources back and requeue it (memory pressure).
        Its progress is lost except for whatever the cache still holds."""
        if req.match is not None:
            self.cache.lock(req.match.node, -1)
            req.match = None
        if req.pages is not None:
            self.kv_pool.free(req.pages[req.prefix_len :])
        if req.slot >= 0:
            self.state_pool.free(req.slot)
        for slot in req.checkpoints.values():
            self.state_pool.free(slot)
        req.pages = None
        req.slot = -1
        req.checkpoints = {}
        req.prefix_len = 0
        req.prefill_pos = 0
        req.output_ids = []
        req.stream_cursor = 0
        req.state = State.QUEUED

    def release(self, req: Request) -> None:
        """Return a finished request's resources; hand its sequence to the cache."""
        if req.released:
            return
        cached = False
        if self.cache is not None and req.slot >= 0 and req.pages is not None and req.prefill_done:
            cached = self._insert_into_cache(req)
        if req.match is not None:
            self.cache.lock(req.match.node, -1)
            req.match = None
        if not cached:
            if req.pages is not None:
                self.kv_pool.free(req.pages[req.prefix_len :])
            if req.slot >= 0:
                self.state_pool.free(req.slot)
            for slot in req.checkpoints.values():
                self.state_pool.free(slot)
        req.pages = None
        req.slot = -1
        req.checkpoints = {}
        req.released = True

    def _insert_into_cache(self, req: Request) -> bool:
        # The committed slot holds the state after every token that went
        # through the model: the prompt and all outputs but the last sampled one.
        ids = req.prompt_ids + req.output_ids[:-1]
        if req.pages is None or req.pages.shape[0] < len(ids):
            return False
        n = len(ids)
        if n == 0 or n > req.pages.shape[0]:
            return False
        pages = req.pages[:n]
        snapshots = {n: req.slot, **{p: s for p, s in req.checkpoints.items() if p < n}}
        for p, s in req.checkpoints.items():
            if p >= n:
                self.state_pool.free(s)
        new_start = self.cache.insert(ids, pages, snapshots)
        # Own pages that duplicate an already-cached prefix go back to the pool;
        # the cache's own prefix pages (before prefix_len) stay where they are.
        dup_start = req.prefix_len
        if new_start > dup_start:
            self.kv_pool.free(req.pages[dup_start:new_start])
        # pages beyond n (the unsampled last token's page) are unused
        if req.pages.shape[0] > n:
            self.kv_pool.free(req.pages[n:])
        return True

    def flush_cache(self) -> None:
        if self.cache is not None:
            self.cache.flush()

    # ---- steps ----
    def _plan_paged_prefill(self, total_pages: torch.Tensor, new_tokens: int) -> None:
        qo_indptr = torch.tensor([0, new_tokens], dtype=torch.int32, device=self.device)
        kv_indptr = torch.tensor([0, total_pages.shape[0]], dtype=torch.int32, device=self.device)
        last_page_len = torch.ones(1, dtype=torch.int32, device=self.device)
        self._paged_prefill.plan(
            qo_indptr, kv_indptr, total_pages, last_page_len,
            config.NUM_ATTENTION_HEADS, config.NUM_KV_HEADS, config.HEAD_DIM, self.kv_pool.page_size,
            causal=True, q_data_type=torch.bfloat16, kv_data_type=KV_DTYPE, sm_scale=config.HEAD_DIM**-0.5,
        )

    def _checkpoint_positions(self, start: int, end: int, total: int) -> list[int]:
        """Absolute positions (multiples of 64, strictly inside this chunk's new
        tokens (start, end)) where GDN state is snapshotted: every
        CHECKPOINT_INTERVAL for shared long prefixes, plus the last 64-boundary
        before the prompt end so an identical prompt re-sent later recomputes at
        most 64 tokens (the last token is never served from cache: its logits
        are needed)."""
        if self.cache is None:
            return []
        first = (start // CHECKPOINT_INTERVAL + 1) * CHECKPOINT_INTERVAL
        positions = [p for p in range(first, end, CHECKPOINT_INTERVAL) if p > start]
        last = ((total - 1) // 64) * 64
        if start < last < end and last not in positions:
            positions.append(last)
        return sorted(positions)

    def prefill(self, req: Request, max_tokens: int | None = None) -> int | None:
        """Prefill up to `max_tokens` of the remaining prompt (all of it by
        default). Each chunk continues from the state and pages left by the
        previous one, exactly like a cache hit. Only the final chunk samples a
        token; it is returned, otherwise None."""
        started = time.perf_counter()
        n = len(req.prompt_ids)
        start = req.prefill_pos
        end = n if max_tokens is None else min(n, start + max_tokens)
        new_ids = req.prompt_ids[start:end]
        t = len(new_ids)
        assert t > 0, (start, end)
        ids = torch.tensor(new_ids, dtype=torch.int64, device=self.device)
        positions = torch.arange(start, end, dtype=torch.int64, device=self.device)
        ctx = ForwardContext(
            mode="prefill",
            kv_pool=self.kv_pool,
            state_pool=self.state_pool,
            positions=positions,
            slot=req.slot,
            pages=req.pages[start:end],
            seq_len=t,
            prefix_len=start,
            prefix_pages=req.pages[:start] if start else None,
        )
        if start:
            self._plan_paged_prefill(req.pages[:end], t)
            ctx.paged_prefill_wrapper = self._paged_prefill
        for pos in self._checkpoint_positions(start, end, n):
            if not self.state_pool._free:
                self.cache.evict(need_slots=1)
            if not self.state_pool._free:
                break
            slot = self.state_pool.alloc()
            ctx.checkpoint_positions.append(pos)
            ctx.checkpoint_slots.append(slot)
            req.checkpoints[pos] = slot
        out = self.model.forward(ids, ctx)
        req.prefill_pos = end
        token = None
        if end == n:
            logits = self.model.logits(out["norm"][-1:])
            token = int(self.sampler.sample(logits, [req.params]).item())
            if self.cache is not None:
                # Snapshot the state after the whole prompt: the next turn of a
                # conversation extends exactly this sequence.
                if not self.state_pool._free:
                    self.cache.evict(need_slots=1)
                if self.state_pool._free:
                    snap = self.state_pool.alloc()
                    self.state_pool.copy_slot(req.slot, snap)
                    req.checkpoints[n] = snap
        torch.cuda.synchronize()
        req.prefill_seconds += time.perf_counter() - started
        if token is not None:
            req.push(token)
        return token

    def prefill_batch(self, chunks: list[tuple[Request, int]]) -> None:
        """Prefill several requests' next chunks in one forward. `chunks` is a
        list of (request, number_of_new_tokens). Requests whose prompt completes
        get their first token sampled."""
        if len(chunks) == 1:
            self.prefill(chunks[0][0], chunks[0][1])
            return
        started = time.perf_counter()
        ids_list, pos_list, page_list, cu, kv_indptr, kv_indices = [], [], [], [0], [0], []
        seq_slots, prefix_lens, seq_lens = [], [], []
        ck_pos, ck_slots, ck_seqs = [], [], []
        for si, (req, n_new) in enumerate(chunks):
            start = req.prefill_pos
            end = start + n_new
            total = len(req.prompt_ids)
            ids_list += req.prompt_ids[start:end]
            pos_list += list(range(start, end))
            page_list.append(req.pages[start:end])
            cu.append(cu[-1] + n_new)
            kv_indices.append(req.pages[:end])
            kv_indptr.append(kv_indptr[-1] + end)
            seq_slots.append(req.slot)
            prefix_lens.append(start)
            seq_lens.append(n_new)
            for pos in self._checkpoint_positions(start, end, total):
                if not self.state_pool._free:
                    self.cache.evict(need_slots=1)
                if not self.state_pool._free:
                    break
                slot = self.state_pool.alloc()
                ck_pos.append(pos)
                ck_slots.append(slot)
                ck_seqs.append(si)
                req.checkpoints[pos] = slot
        d = self.device
        ctx = ForwardContext(
            mode="prefill", kv_pool=self.kv_pool, state_pool=self.state_pool,
            positions=torch.tensor(pos_list, dtype=torch.int64, device=d),
            pages=torch.cat(page_list), seq_len=cu[-1],
            cu_seqlens=torch.tensor(cu, dtype=torch.int32, device=d),
            seq_slots=torch.tensor(seq_slots, dtype=torch.int32, device=d),
            seq_prefix_lens=prefix_lens, seq_lens=seq_lens,
            checkpoint_positions=ck_pos, checkpoint_slots=ck_slots, checkpoint_seqs=ck_seqs,
        )
        self._paged_prefill.plan(
            ctx.cu_seqlens, torch.tensor(kv_indptr, dtype=torch.int32, device=d), torch.cat(kv_indices),
            torch.ones(len(chunks), dtype=torch.int32, device=d),
            config.NUM_ATTENTION_HEADS, config.NUM_KV_HEADS, config.HEAD_DIM, self.kv_pool.page_size,
            causal=True, q_data_type=torch.bfloat16, kv_data_type=KV_DTYPE, sm_scale=config.HEAD_DIM**-0.5,
        )
        ctx.paged_prefill_wrapper = self._paged_prefill
        ids = torch.tensor(ids_list, dtype=torch.int64, device=d)
        out = self.model.forward(ids, ctx)
        last_rows = [c - 1 for c in cu[1:]]
        finishing = [i for i, (req, n_new) in enumerate(chunks) if req.prefill_pos + n_new == len(req.prompt_ids)]
        tokens = None
        if finishing:
            logits = self.model.logits(out["norm"][[last_rows[i] for i in finishing]])
            tokens = self.sampler.sample(logits, [chunks[i][0].params for i in finishing]).tolist()
        for i, (req, n_new) in enumerate(chunks):
            req.prefill_pos += n_new
        for i, tok in zip(finishing, tokens or []):
            req = chunks[i][0]
            if self.cache is not None:
                if not self.state_pool._free:
                    self.cache.evict(need_slots=1)
                if self.state_pool._free:
                    snap = self.state_pool.alloc()
                    self.state_pool.copy_slot(req.slot, snap)
                    req.checkpoints[len(req.prompt_ids)] = snap
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        for req, _ in chunks:
            req.prefill_seconds += elapsed
        for i, tok in zip(finishing, tokens or []):
            chunks[i][0].push(int(tok))

    def warmup(self, log=None) -> None:
        """Run every prefill path once so first requests do not pay backend
        initialization (b12x module load, cuDNN handles, Triton compiles)."""
        from sharar.parity.prompts import random_ids

        params = SamplingParams(max_new_tokens=1)
        a = self.admit(random_ids(300, 11), params)
        self.prefill(a, max_tokens=256)  # chunk 1
        self.prefill(a)  # chunk 2 with prefix
        b = self.admit(random_ids(40, 12), params)
        c = self.admit(random_ids(70, 13), params)
        self.prefill_batch([(b, 40), (c, 70)])
        for r in (a, b, c):
            self.release(r)
        self.flush_cache()
        torch.cuda.synchronize()
        if log:
            log("  warmup done")

    def _plan_decode(self, reqs: list[Request]) -> None:
        counts = [r.context_len for r in reqs]
        indptr = torch.tensor([0] + list(torch.tensor(counts).cumsum(0).tolist()), dtype=torch.int32, device=self.device)
        indices = torch.cat([r.pages for r in reqs])
        last_page_len = torch.ones(len(reqs), dtype=torch.int32, device=self.device)
        self._decode_wrapper.plan(
            indptr, indices, last_page_len,
            config.NUM_ATTENTION_HEADS, config.NUM_KV_HEADS, config.HEAD_DIM, self.kv_pool.page_size,
            pos_encoding_mode="NONE", q_data_type=torch.bfloat16, kv_data_type=KV_DTYPE, sm_scale=config.HEAD_DIM**-0.5,
        )

    def decode_step(self, reqs: list[Request]) -> list[int]:
        """Process the last generated token of every request and sample the next."""
        assert 0 < len(reqs) <= self.max_batch
        if len(self.kv_pool._free) < len(reqs):
            if self.cache is not None:
                self.cache.evict(need_pages=len(reqs))
            if len(self.kv_pool._free) < len(reqs):
                raise MemoryError("KV pool exhausted during decode")
        new_pages = self.kv_pool.alloc(len(reqs))
        for i, r in enumerate(reqs):
            r.pages = torch.cat([r.pages, new_pages[i : i + 1]])
        if self.use_graphs and self.graphs is not None:
            logits = self.graphs.run(
                [r.output_ids[-1] for r in reqs],
                [r.context_len - 1 for r in reqs],
                [r.slot for r in reqs],
                new_pages,
                [r.pages for r in reqs],
            )
        else:
            positions = torch.tensor([r.context_len - 1 for r in reqs], dtype=torch.int64, device=self.device)
            tokens = torch.tensor([r.output_ids[-1] for r in reqs], dtype=torch.int64, device=self.device)
            self._plan_decode(reqs)
            ctx = ForwardContext(
                mode="decode", kv_pool=self.kv_pool, state_pool=self.state_pool, positions=positions,
                slots=torch.tensor([r.slot for r in reqs], dtype=torch.int32, device=self.device),
                out_pages=new_pages, decode_wrapper=self._decode_wrapper, batch_size=len(reqs),
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
            self.release(r)
        return reqs
