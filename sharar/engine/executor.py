"""Single-process executor: ledger-sized pools, radix prefix cache, prefill with
cached prefixes and state checkpoints, batched decode, CUDA graphs, generate."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from sharar.target import Target
from sharar.engine.context import ForwardContext
from sharar.engine.graphs import DecodeGraphs
from sharar.engine.request import Request, State
from sharar.engine.sampler import InvalidRequest, Sampler, SamplingParams
from sharar.memory.kv_pool import KV_DTYPE, KvPool
from sharar.memory.ledger import Ledger
from sharar.memory.radix import RadixCache
from sharar.memory.request_state import StateManager
from sharar.engine.contracts import ProposerContext, Proposer
from sharar.engine.profile import ServingProfile
from sharar.engine.upload import upload
from sharar.memory.state_pool import StatePool
if TYPE_CHECKING:
    from sharar.model.qwen38 import Qwen38

CHECKPOINT_INTERVAL = 2048  # GDN snapshots inside long prefills (multiple of 64)


@dataclass
class PrefillBatch:
    chunks: list[tuple[Request, int]]
    ids: torch.Tensor
    context: ForwardContext
    started: float


class Executor:
    timing = False  # benchmarks set True: synchronize to attribute prefill wall time (class attribute: test executors skip __init__)
    _ple = None  # the model's prefix-lookup embedding layer when it has one (Flash-Next); class attribute for the same reason

    def __init__(self, model: Qwen38, num_pages: int | None = None, num_slots: int | None = None, max_batch: int = 4, use_cache: bool = True, log=None, max_context: int | None = None, spec_config=None):
        self.target: Target = model.target
        max_context = self.target.context_length if max_context is None else max_context
        self.profile = ServingProfile(max_context=max_context, max_batch=max_batch)
        import flashinfer

        self.max_context = max_context
        self.model = model
        self.device = model.device
        self._ple = getattr(model, "ple", None)  # prefix-lookup embeddings need the tokens before each forward
        self.ledger = Ledger(target=self.target)
        self.ledger.record_cuda_allocated("weights")
        future, extra_page = 2 * 2**30, 0
        if spec_config is not None:
            from sharar.memory.spec_budget import budget
            future, extra_page = budget(spec_config, max_batch, max_context, cache_slots=num_slots or 96, target=self.target)
        if num_pages is None or num_slots is None:
            plan = self.ledger.plan(max_running=max_batch, context_len=max_context, future_bytes=future, extra_page_bytes=extra_page)
            num_pages = num_pages or plan.kv_tokens
            num_slots = num_slots or plan.state_slots
        if type(num_pages) is not int or num_pages < 2 or type(num_slots) is not int or num_slots < 2:
            raise ValueError("pools need at least one usable entry plus the reserved dummy entry")
        from sharar.memory.ledger import check_load_headroom
        check_load_headroom(num_pages * (self.target.kv_bytes_per_token + extra_page) + num_slots * self.target.state_bytes_per_slot + future + 8 * 2**30)
        self.kv_pool = KvPool(num_pages, self.device, target=self.target)
        self.ledger.record_cuda_allocated("kv_pool")
        self.state_pool = StatePool(num_slots, self.device, target=self.target)
        if hasattr(model, "register_state"):
            model.register_state(self.state_pool)  # e.g. the PLE convolution window (StatePool.aux)
        self.ledger.record_cuda_allocated("state_pool")
        cache = RadixCache(self.kv_pool, self.state_pool, self.device) if use_cache else None
        self.state = StateManager(self.kv_pool, self.state_pool, cache)
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
        self.spec = None
        self.feature_request = None  # explicit target features when no proposer is installed (draft training)
        if log:
            log(f"  pools: {num_pages} KV pages ({self.kv_pool.bytes / 2**30:.2f} GiB), {num_slots} state slots ({num_slots * self.state_pool.bytes_per_slot / 2**30:.2f} GiB)")
        if spec_config is not None:
            self.enable_spec(spec_config.depth, log=log, proposer=spec_config.name, state_strategy=spec_config.state_strategy,
                             draft_precision=spec_config.draft_precision, draft_dir=spec_config.draft_dir)

    @property
    def cache(self):
        return self.state.cache

    @cache.setter
    def cache(self, value):
        self.state.cache = value

    def _features_wanted(self):
        """Target features to compute: the proposer's, or an explicit request
        (draft training extracts features without a proposer)."""
        if self.spec is not None:
            return self.spec.proposer.requirements.features
        return getattr(self, "feature_request", None)  # test doubles bypass __init__

    def enable_spec(self, k: int = 3, log=None, *, proposer: str = "mtp", state_strategy: str = "snapshots", draft_precision: str = 'bf16',
                    draft_dir: str | None = None) -> None:
        """Select the adapter and its numerical profile once, before capture."""
        if self.graphs is not None or self.spec is not None:
            raise ValueError("enable speculation once, before graph capture")
        from sharar.memory.spec_budget import SpecConfig, budget
        from sharar.memory.ledger import check_load_headroom
        fixed, per_page = budget(SpecConfig(proposer, k, state_strategy, draft_precision), self.max_batch, self.max_context,
                                 cache_slots=self.state_pool.num_slots, target=self.target)
        check_load_headroom(fixed + per_page * self.kv_pool.num_pages + 8 * 2**30)
        context = ProposerContext(self.device, self.max_batch, self.max_context)
        from sharar.proposers.factory import create
        adapter = create(proposer, k, self.model, context, self.kv_pool.num_pages, draft_precision=draft_precision, draft_dir=draft_dir)
        self.set_proposer(adapter, state_strategy=state_strategy)
        self.ledger.record_cuda_allocated("proposer_and_verify")
        if log:
            log(f"  proposer: {adapter.requirements.name}, persistent {adapter.requirements.persistent_bytes / 2**30:.2f} GiB, verify buffers {self.spec.bytes / 2**30:.2f} GiB, k={k}"
                + (f", draft {draft_dir}" if draft_dir else ""))

    def set_proposer(self, proposer: Proposer, *, state_strategy: str = "snapshots") -> None:
        from sharar.engine.spec import SpecDecoder
        if self.graphs is not None or self.spec is not None:
            raise ValueError("select a proposer once, before graph capture")
        if proposer.context != ProposerContext(self.device, self.max_batch, self.max_context):
            raise ValueError("proposer context does not match executor")
        limit = proposer.requirements.max_depth
        if type(proposer.k) is not int or not 1 <= proposer.k <= limit:
            raise ValueError("unsupported proposal width")
        if any(type(i) is not int or not 0 <= i < self.target.num_layers for i in proposer.requirements.features.layers):
            raise ValueError("invalid target feature layer")
        bind = getattr(proposer, 'bind_checkpoint_cache', None)
        if bind is not None and self.cache is not None:
            bind(self.state_pool)
        self.spec = SpecDecoder(self, proposer, state_strategy)

    def capture_graphs(self, log=print) -> None:
        self.graphs = DecodeGraphs(self)
        self.graphs.capture(log=log)
        if self.spec is not None:
            self.spec.capture_graphs(log=log)
        self.ledger.record_cuda_allocated("cuda_graphs")
        self.use_graphs = True


    # ---- lifecycle ----
    def validate_request(self, prompt_ids: list[int], params: SamplingParams) -> None:
        params.validate()
        if not isinstance(prompt_ids, list) or not prompt_ids:
            raise InvalidRequest("prompt must be a non-empty token list")
        vocab = self.target.vocab_size
        if any(type(t) is not int or not 0 <= t < vocab for t in prompt_ids):
            raise InvalidRequest(f"token IDs must be integers in [0, {vocab})")
        if not isinstance(params.stop_token_ids, (tuple, list)) or any(type(t) is not int or not 0 <= t < vocab for t in params.stop_token_ids):
            raise InvalidRequest("invalid stop token ID")
        if len(prompt_ids) + params.max_new_tokens > self.max_context:
            raise InvalidRequest(f"prompt plus output budget exceeds context limit {self.max_context}")
        # A single request must be able to finish even after all other readers
        # leave. Full verify blocks need temporary pages beyond the output cap.
        padding = self.spec.k if self.spec is not None and (params.greedy or not self.spec.proposer.requirements.greedy_only) else 0
        if len(prompt_ids) + params.max_new_tokens + padding > self.kv_pool.num_pages - 1:
            raise InvalidRequest("request plus speculative scratch exceeds KV pool capacity")

    def admit(self, prompt_ids: list[int], params: SamplingParams, req: Request | None = None) -> Request:
        self.validate_request(prompt_ids, params)
        if req is None:
            req = Request(prompt_ids=list(prompt_ids), params=params)
        if req.generator is None and not params.greedy:
            req.generator = self.sampler.generator(params.seed)
        self.state.admit(req, req.prefill_ids)
        req.state = State.PREFILL
        req.t_admit = time.perf_counter()
        return req

    def can_admit(self, prompt_ids: list[int], reserve_tokens: int = 0) -> bool:
        ledger = getattr(self, "ledger", None)
        if ledger is not None and not ledger.above_floor():
            # Unified memory: pool IDs being free does not mean DRAM is. Hold
            # admission (running requests keep decoding and release memory as
            # they finish) instead of letting the host guard stop the server.
            self.floor_holds = getattr(self, "floor_holds", 0) + 1
            return False
        return self.state.can_admit(prompt_ids, reserve_tokens)

    def admission_reserve(self, req: Request, limit: int) -> int:
        padding = self.spec.k if self.spec is not None and (req.params.greedy or not self.spec.proposer.requirements.greedy_only) else 0
        return min(limit, req.params.max_new_tokens - len(req.output_ids) + padding)

    def cache_stats(self) -> dict | None:
        if self.cache is None:
            return None
        stats = self.cache.stats()
        features = getattr(self.spec.proposer, 'feature_cache', None) if self.spec is not None else None
        if features is not None:
            stats['draft_features'] = {'entries': len(features.entries), 'bytes': features.bytes,
                                       'capacity_bytes': features.capacity,
                                       'hits': self.spec.proposer.cache_hits,
                                       'fallbacks': self.spec.proposer.cache_fallbacks}
        return stats

    def enable_cold_tier(self, budget_gib: float | None = None, log=None) -> None:
        """Park evicted prefix-cache leaves in a unified-memory arena instead of
        dropping them (`memory/cold_tier.py`). Budget: SHARAR_COLD_TIER_GIB, else
        what is available now beyond the host floor, the prefill reserve and a
        4 GiB margin, capped at 24 GiB. Call after the proposer exists so its
        auxiliary state travels with parked sessions."""
        import os
        from sharar.memory.cold_tier import ColdTier
        from sharar.memory.ledger import GIB, mem_available_bytes

        if self.cache is None or self.cache.tier is not None:
            return
        if budget_gib is None:
            env = os.environ.get("SHARAR_COLD_TIER_GIB")
            if env is not None:
                budget_gib = float(env)
            else:
                spare = mem_available_bytes() - self.ledger.floor_bytes - 8 * GIB - 4 * GIB
                budget_gib = min(24.0, spare / GIB)
        if budget_gib < 1.0:
            if log:
                log("  cold tier: disabled (less than 1 GiB spare)")
            return
        try:
            self.cache.tier = ColdTier(self.kv_pool, self.state_pool, int(budget_gib * GIB), self.device, log=log)
        except (RuntimeError, MemoryError, OSError, ImportError) as exc:
            if log:
                log(f"  cold tier: disabled ({exc})")
            return
        proposer = self.spec.proposer if self.spec is not None else None
        codec = getattr(proposer, "auxiliary_codec", None)
        if codec is not None:
            self.cache.tier.register_codec(*codec())
        self.ledger.record_cuda_allocated("cold_tier")

    def retract(self, req: Request) -> None:
        self.state.retract(req)
        if self.spec is not None:
            self.spec.proposer.release(req)
        req.replay_len = len(req.output_ids)
        req.state = State.QUEUED

    def release(self, req: Request) -> None:
        if req.released:
            return
        if self.spec is not None and hasattr(self.spec.proposer, "prepare_release"):
            self.spec.proposer.prepare_release(req)
        self.state.release(req)
        if self.spec is not None:
            self.spec.proposer.release(req)

    def flush_cache(self) -> None:
        self.state.flush()

    def reload_draft(self, draft_dir: str | None) -> str:
        """Copy a checkpoint (None = the pinned publisher draft) into the serving
        draft between rounds; returns the served source. Only parallel drafts
        (DSpark, DFlash2) hold a reloadable model."""
        if self.spec is None or not hasattr(self.spec.proposer, "model") or not hasattr(self.spec.proposer.model, "load_weights"):
            raise ValueError("no reloadable draft is being served")
        from sharar.proposers.factory import draft_snapshot
        snapshot = draft_snapshot(self.spec.proposer.name, draft_dir)
        self.spec.proposer.model.load_weights(snapshot)
        return self.spec.proposer.model.source

    # ---- steps ----
    def _plan_paged_prefill(self, prefix_pages: torch.Tensor, new_tokens: int) -> None:
        # Host index pointers: FlashInfer's plan() reads them on the CPU, so
        # device tensors would cost a synchronization per plan.
        qo_indptr = torch.tensor([0, new_tokens], dtype=torch.int32)
        kv_indptr = torch.tensor([0, prefix_pages.shape[0]], dtype=torch.int32)
        last_page_len = torch.ones(1, dtype=torch.int32)
        self._paged_prefill.plan(
            qo_indptr, kv_indptr, prefix_pages, last_page_len,
            self.target.num_attention_heads, self.target.num_kv_heads, self.target.head_dim, self.kv_pool.page_size,
            causal=False, q_data_type=torch.bfloat16, kv_data_type=KV_DTYPE, sm_scale=self.target.head_dim**-0.5,
        )

    def _checkpoint_positions(self, start: int, end: int, total: int) -> list[int]:
        """Absolute positions aligned to this forward's 64-token GDN chunks.
        Snapshot near each CHECKPOINT_INTERVAL, plus the last local boundary
        before the prompt end so an identical prompt re-sent later recomputes at
        most 64 tokens (the last token is never served from cache: its logits
        are needed)."""
        if self.cache is None:
            return []
        first = (start // CHECKPOINT_INTERVAL + 1) * CHECKPOINT_INTERVAL
        positions = [start + ((p - start + 63) // 64) * 64 for p in range(first, end, CHECKPOINT_INTERVAL)]
        positions = [p for p in positions if start < p < end]
        last = start + ((total - start - 1) // 64) * 64
        if start < last < end and last not in positions:
            positions.append(last)
        return sorted(positions)

    def prefill(self, req: Request, max_tokens: int | None = None) -> int | None:
        batch = self._prepare_single_prefill(req, max_tokens)
        return self._finish_prefill(batch, self.model.forward(batch.ids, batch.context))

    def prefill_batch(self, chunks: list[tuple[Request, int]]) -> None:
        batch = self._prepare_prefill(chunks)
        self._finish_prefill(batch, self.model.forward(batch.ids, batch.context))

    def _prepare_single_prefill(self, req: Request, max_tokens: int | None) -> PrefillBatch:
        """Prefill up to `max_tokens` of the remaining prompt (all of it by
        default). Each chunk continues from the state and pages left by the
        previous one, exactly like a cache hit. Only the final chunk samples a
        token; it is returned, otherwise None."""
        started = time.perf_counter()
        n = req.prefill_len
        start = req.prefill_pos
        end = n if max_tokens is None else min(n, start + max_tokens)
        new_ids = req.prefill_ids[start:end]
        t = len(new_ids)
        assert t > 0, (start, end)
        ids = upload(new_ids, torch.int64, self.device)
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
            self._plan_paged_prefill(req.pages[:start], t)
            ctx.paged_prefill_wrapper = self._paged_prefill
        if self._ple is not None:
            ctx.ple_history = torch.tensor([self._ple_history(req, start)], dtype=torch.int64)
        for _, pos, slot in self.state.prefill_checkpoints([(req, self._checkpoint_positions(start, end, n))]):
            ctx.checkpoint_positions.append(pos)
            ctx.checkpoint_slots.append(slot)
        if self._features_wanted() is not None:
            ctx.feature_request = self._features_wanted()
        return PrefillBatch([(req, t)], ids, ctx, started)

    def _prepare_prefill(self, chunks: list[tuple[Request, int]]) -> PrefillBatch:
        """Prefill several requests' next chunks in one forward. `chunks` is a
        list of (request, number_of_new_tokens). Requests whose prompt completes
        get their first token sampled."""
        if len(chunks) == 1:
            return self._prepare_single_prefill(chunks[0][0], chunks[0][1])
        started = time.perf_counter()
        ids_list, pos_list, page_list, cu, kv_indptr, kv_indices = [], [], [], [0], [0], []
        seq_slots, prefix_lens, seq_lens = [], [], []
        boundaries = []
        for si, (req, n_new) in enumerate(chunks):
            start = req.prefill_pos
            end = start + n_new
            total = req.prefill_len
            ids_list += req.prefill_ids[start:end]
            pos_list += list(range(start, end))
            page_list.append(req.pages[start:end])
            cu.append(cu[-1] + n_new)
            kv_indices.append(req.pages[:start])
            kv_indptr.append(kv_indptr[-1] + start)
            seq_slots.append(req.slot)
            prefix_lens.append(start)
            seq_lens.append(n_new)
            boundaries.append((req, self._checkpoint_positions(start, end, total)))
        checkpoints = self.state.prefill_checkpoints(boundaries)
        d = self.device
        ctx = ForwardContext(
            mode="prefill", kv_pool=self.kv_pool, state_pool=self.state_pool,
            positions=upload(pos_list, torch.int64, d),
            pages=torch.cat(page_list), seq_len=cu[-1],
            cu_seqlens=upload(cu, torch.int32, d),
            seq_slots=upload(seq_slots, torch.int32, d),
            seq_prefix_lens=prefix_lens, seq_lens=seq_lens,
            checkpoint_positions=[p for _, p, _ in checkpoints],
            checkpoint_slots=[s for _, _, s in checkpoints],
            checkpoint_seqs=[i for i, _, _ in checkpoints],
        )
        if any(prefix_lens):
            self._paged_prefill.plan(
                torch.tensor(cu, dtype=torch.int32), torch.tensor(kv_indptr, dtype=torch.int32), torch.cat(kv_indices),
                torch.ones(len(chunks), dtype=torch.int32),
                self.target.num_attention_heads, self.target.num_kv_heads, self.target.head_dim, self.kv_pool.page_size,
                causal=False, q_data_type=torch.bfloat16, kv_data_type=KV_DTYPE, sm_scale=self.target.head_dim**-0.5,
            )
            ctx.paged_prefill_wrapper = self._paged_prefill
        ids = upload(ids_list, torch.int64, d)
        if self._ple is not None:
            ctx.ple_history = torch.tensor([self._ple_history(req, req.prefill_pos) for req, _ in chunks], dtype=torch.int64)
        if self._features_wanted() is not None:
            ctx.feature_request = self._features_wanted()
        return PrefillBatch(list(chunks), ids, ctx, started)

    def _ple_history(self, req: Request, start: int) -> list[int]:
        """The two tokens before absolute position `start` of the request (EOS before the sequence)."""
        full = req.prompt_ids + req.output_ids
        eos = self.model.eos
        return [full[i] if i >= 0 else eos for i in (start - 2, start - 1)]

    def _ple_rows_block(self, reqs: list[Request], drafts: list[list[int]]) -> torch.Tensor:
        """Table rows for each request's verify block (its pending token then the drafted ones) -> device uint8 [B*(k+1), heads, row_bytes]."""
        k = len(drafts[0])
        tokens = torch.tensor([[r.output_ids[-1], *d] for r, d in zip(reqs, drafts)], dtype=torch.int64).reshape(-1)
        histories = torch.tensor([self._ple_history(r, r.context_len - 1) for r in reqs], dtype=torch.int64)
        return self._ple.rows_for(tokens, [(i * (k + 1), k + 1) for i in range(len(reqs))], histories, self.model.eos)

    def _ple_rows(self, reqs: list[Request]) -> torch.Tensor:
        """Table rows for each request's pending decode token (host gather) -> device uint8 [B, heads, row_bytes]."""
        tokens = torch.tensor([r.output_ids[-1] for r in reqs], dtype=torch.int64)
        histories = torch.tensor([self._ple_history(r, r.context_len - 1) for r in reqs], dtype=torch.int64)
        return self._ple.rows_for(tokens, [(i, 1) for i in range(len(reqs))], histories, self.model.eos)

    def _finish_prefill(self, batch: PrefillBatch, out) -> int | None:
        chunks, started = batch.chunks, batch.started
        if len(chunks) == 1:
            return self._finish_single_prefill(batch, out)
        cu = [0]
        for _, n in chunks:
            cu.append(cu[-1] + n)
        last_rows = [c - 1 for c in cu[1:]]
        finishing = [i for i, (req, n_new) in enumerate(chunks) if req.prefill_pos + n_new == req.prefill_len]
        tokens = None
        if finishing:
            logits = self.model.logits(out["norm"][[last_rows[i] for i in finishing]])
            tokens = self.sampler.sample(logits, [chunks[i][0].params for i in finishing], [chunks[i][0].generator for i in finishing]).tolist()
        if self.spec is not None:
            for i, (req, n_new) in enumerate(chunks):
                rows = {"norm": out["norm"][cu[i]:cu[i + 1]]}
                if "features" in out:
                    rows["features"] = {lid: value[cu[i]:cu[i + 1]] for lid, value in out["features"].items()}
                self.spec.ingest_prefill(req, req.prefill_ids[req.prefill_pos:req.prefill_pos + n_new], req.prefill_pos, rows)
        for i, (req, n_new) in enumerate(chunks):
            self.state.commit_prefill(req, req.prefill_pos + n_new)
        for i in finishing:
            req = chunks[i][0]
            self.state.checkpoint(req, req.prefill_len, copy=True)
        if self.timing:  # benchmarks: wall time to completion; serving keeps the GPU queue full
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        for req, _ in chunks:
            req.prefill_seconds += elapsed
        for i, tok in zip(finishing, tokens or []):
            chunks[i][0].push(int(tok))
            if self.spec is not None:
                self.spec.start(chunks[i][0], int(tok))

    def _finish_single_prefill(self, batch, out):
        req, t = batch.chunks[0]
        start, n, started = req.prefill_pos, req.prefill_len, batch.started
        end = start + t
        new_ids = req.prefill_ids[start:end]
        self.state.commit_prefill(req, end)
        if self.spec is not None:
            self.spec.ingest_prefill(req, new_ids, start, out)
        token = None
        if end == n:
            logits = self.model.logits(out["norm"][-1:])
            token = int(self.sampler.sample(logits, [req.params], [req.generator]).item())
            # Allocate the immutable endpoint before the proposer attaches its
            # features; the active committed slot continues advancing in decode.
            self.state.checkpoint(req, n, copy=True)
            if self.spec is not None:
                self.spec.start(req, token)
        if self.timing:
            torch.cuda.synchronize()
        req.prefill_seconds += time.perf_counter() - started
        if token is not None:
            req.push(token)
        return token

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
        if self.spec is not None:
            # Two speculative rounds at batch 1 and 2 so the first real request
            # does not pay the lazy kernel loads of the verify/accept path.
            d = self.admit(random_ids(50, 14), SamplingParams(max_new_tokens=12))
            e = self.admit(random_ids(60, 15), SamplingParams(max_new_tokens=12))
            self.prefill(d)
            self.prefill(e)
            self.decode_step([d])
            self.decode_step([d, e])
            for r in (d, e):
                self.release(r)
            self.spec.reset_stats()
        self.flush_cache()
        torch.cuda.synchronize()
        if log:
            log("  warmup done")

    def _plan_decode(self, reqs: list[Request], pages: list[torch.Tensor]) -> None:
        counts = [r.context_len for r in reqs]
        indptr = torch.tensor([0] + list(torch.tensor(counts).cumsum(0).tolist()), dtype=torch.int32)
        indices = torch.cat(pages)
        last_page_len = torch.ones(len(reqs), dtype=torch.int32)
        self._decode_wrapper.plan(
            indptr, indices, last_page_len,
            self.target.num_attention_heads, self.target.num_kv_heads, self.target.head_dim, self.kv_pool.page_size,
            pos_encoding_mode="NONE", q_data_type=torch.bfloat16, kv_data_type=KV_DTYPE, sm_scale=self.target.head_dim**-0.5,
        )

    def decode_step(self, reqs: list[Request]) -> list[int]:
        """Process the last generated token of every request and sample the next.
        With speculative decoding on, greedy requests go through spec rounds
        and sampled ones through the plain step."""
        assert 0 < len(reqs) <= self.max_batch
        spec_reqs = [r for r in reqs if self.spec is not None and self.spec.ready(r)
                     and (r.params.greedy or not self.spec.proposer.requirements.greedy_only)]
        plain = [r for r in reqs if r not in spec_reqs]
        need = len(plain) + (len(spec_reqs) * self.spec.steps if spec_reqs else 0)
        # Retraction retries the entire batch. Reserve both paths before either
        # advances so memory pressure cannot replay a partly completed round.
        self.state.ensure_pages(need)
        if spec_reqs:
            self.spec.round(spec_reqs)
        if not plain:
            return [r.output_ids[-1] for r in reqs]
        reqs = plain
        new_pages = self.state.reserve(reqs, 1)
        step_pages = [self.state.step_pages(r) for r in reqs]
        if self.use_graphs and self.graphs is not None:
            result = self.graphs.run(
                [r.output_ids[-1] for r in reqs],
                [r.context_len - 1 for r in reqs],
                [r.slot for r in reqs],
                new_pages,
                step_pages,
                return_features=self.spec is not None,
                ple_rows=self._ple_rows(reqs) if self._ple is not None else None,
            )
            if self.spec is not None:
                logits, out = result
            else:
                logits = result
        else:
            positions = upload([r.context_len - 1 for r in reqs], torch.int64, self.device)
            tokens = upload([r.output_ids[-1] for r in reqs], torch.int64, self.device)
            self._plan_decode(reqs, step_pages)
            ctx = ForwardContext(
                mode="decode", kv_pool=self.kv_pool, state_pool=self.state_pool, positions=positions,
                slots=upload([r.slot for r in reqs], torch.int32, self.device),
                out_pages=new_pages, decode_wrapper=self._decode_wrapper, batch_size=len(reqs),
            )
            if self._ple is not None:
                ctx.ple_rows = self._ple_rows(reqs)
            if self._features_wanted() is not None:
                ctx.feature_request = self._features_wanted()
            out = self.model.forward(tokens, ctx)
            logits = self.model.logits(out["norm"])
        next_tokens = self.sampler.sample(logits, [r.params for r in reqs], [r.generator for r in reqs]).tolist()
        for i, (r, tok) in enumerate(zip(reqs, next_tokens)):
            pending = r.output_ids[-1]
            self.state.commit(r, 1)
            if self.spec is not None:
                rows = {"norm": out["norm"][i:i + 1]}
                if "features" in out:
                    rows["features"] = {lid: value[i:i + 1] for lid, value in out["features"].items()}
                self.spec.ingest_prefill(r, [pending], r.committed_len - 1, rows)
                self.spec.start(r, int(tok))
            r.push(int(tok))
        return next_tokens

    # ---- convenience ----
    def generate(self, prompt_ids: list[int], params: SamplingParams) -> Request:
        return self.generate_batch([(prompt_ids, params)])[0]

    def generate_batch(self, items: list[tuple[list[int], SamplingParams]]) -> list[Request]:
        if not 1 <= len(items) <= self.max_batch:
            raise InvalidRequest(f"batch must contain 1 through {self.max_batch} requests")
        for ids, params in items:
            self.validate_request(ids, params)
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
