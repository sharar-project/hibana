"""Single-process executor: ledger-sized pools, radix prefix cache, prefill with
cached prefixes and state checkpoints, batched decode, CUDA graphs, generate."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from sharar.target import Target
from sharar.distributed import all_reduce_min, get_tp_info
from sharar.engine.debug import debug_sync
from sharar.engine.context import ForwardContext
from sharar.engine.graphs import DecodeGraphs
from sharar.engine.request import Request, State
from sharar.engine.grammar import sample_masks
from sharar.engine.sampling_rng import request_seed
from sharar.engine.sampler import InvalidRequest, Sampler, SamplingParams
from sharar.cache.kv_pool import KV_DTYPE, KvPool
from sharar.cache.ledger import Ledger, after_plan_bytes
from sharar.cache.radix import RadixCache
from sharar.cache.snapshot import GRID
from sharar.cache.request_state import StateManager
from sharar.engine.contracts import ProposerContext, Proposer
from sharar.engine.profile import ServingProfile
from sharar.engine.upload import upload
from sharar.cache.state_pool import StatePool
if TYPE_CHECKING:
    from sharar.model.qwen38 import Qwen38

PREFIX_ATTENTION_MIN = 1024  # cached prefix pages from which a 27B prefill's prefix attention runs verify_attn.cu


@dataclass
class PrefillBatch:
    chunks: list[tuple[Request, int]]
    ids: torch.Tensor
    context: ForwardContext
    started: float


class Executor:
    timing = False  # benchmarks set True: synchronize to attribute prefill wall time (class attribute: test executors skip __init__)
    extras = None  # the model's forward side inputs (`forward_extras()`, e.g. Flash-Next's); class attribute for the same reason

    def __init__(self, model: Qwen38, num_pages: int | None = None, num_slots: int | None = None, max_batch: int = 4, use_cache: bool = True, log=None, max_context: int | None = None, spec_config=None):
        self.target: Target = model.target
        max_context = self.target.context_length if max_context is None else max_context
        self.profile = ServingProfile(max_context=max_context, max_batch=max_batch, chunk_size=self.target.prefill_chunk,
                                      solo_chunk_size=self.target.solo_prefill_chunk)
        import flashinfer

        self.max_context = max_context
        self.model = model
        self.device = model.device
        self.extras = model.forward_extras()  # what the model's forwards need beyond the context (None for the 27B)
        self.ledger = Ledger(target=self.target)
        self.ledger.record_cuda_allocated("weights")
        from sharar.cache.ssm_format import SsmFormat
        from sharar.engine.config import current
        self.ssm_format = SsmFormat.from_config(current())  # GDN recurrent state storage (SHARAR_SSM_STATE)
        state_bytes = self.ssm_format.state_bytes_per_slot(self.target)
        cfg = current()

        def after_pools(cache_slots):
            """Bytes allocated after the pools (the drafter's, sized by the slots it may retain features for, and the SSD
            tier's staging) and extra bytes per KV page."""
            future, extra_page = 0, 0
            if spec_config is not None:
                from sharar.cache.spec_budget import budget
                future, extra_page = budget(spec_config, max_batch, cache_slots=cache_slots, target=self.target)
            if cfg.ssd:  # the SSD tier pins its staging buffers in the DRAM the pools come from: plan around them
                from sharar.cache.ssd import SsdTier
                future += SsdTier.memory_bound(self.target, cfg.ssd_block_tokens, extra_page, **cfg.ssd_buffers)
            # and what this executor allocates after the pools: workspaces, CUDA graphs, first-use allocations, activations
            return future + sum(after_plan_bytes(max_batch).values()), extra_page

        # the state slots an SSD restore can lock beyond the running requests' (the startup slot floor counts them)
        restore_slots = 0
        if cfg.ssd and use_cache:
            from sharar.cache.ssd import RESTORE_SLOTS as restore_slots
        # the slots the plan is first sized for: its target of 96, or the slot floor
        guess = num_slots or max(96, 2 * max_batch + 1 + restore_slots)
        future, extra_page = after_pools(guess)
        plan = None
        planned_pages = num_pages is None
        if num_pages is None or num_slots is None:
            plan = self.ledger.plan(max_running=max_batch, context_len=max_context, future_bytes=future, extra_page_bytes=extra_page,
                                    state_bytes_per_slot=state_bytes, restore_slots=restore_slots)
            if num_slots is None and plan.state_slots > guess:
                # more slots than the drafter was budgeted for (DFlash2 keeps a feature window per slot): budget the
                # plan's slots and plan again; the larger budget can only give fewer slots, which it still covers
                future, extra_page = after_pools(plan.state_slots)
                plan = self.ledger.plan(max_running=max_batch, context_len=max_context, future_bytes=future,
                                        extra_page_bytes=extra_page, state_bytes_per_slot=state_bytes,
                                        restore_slots=restore_slots)
            num_pages = num_pages or plan.kv_tokens
            num_slots = num_slots or plan.state_slots
        num_pages, num_slots = all_reduce_min(num_pages, num_slots)  # tensor-parallel ranks admit and evict alike
        if type(num_pages) is not int or num_pages < 2 or type(num_slots) is not int or num_slots < 2:
            raise ValueError("pools need at least one usable entry plus the reserved dummy entry")
        kv_dtype, rounding, nvfp4 = KV_DTYPE, None, False
        kv_token_bytes = self.target.kv_bytes_per_token
        if cfg.kv_format == "nvfp4":
            # 4.5 bits a value (codes and scales) where FP8 takes 8: the memory planned for FP8 pages holds more of
            # them (the 27B 16/9; Flash-Next ~1.4, its BF16 index keys unchanged)
            nvfp4 = True
            kv_token_bytes = self.target.nvfp4_kv_bytes_per_token
            if planned_pages and not cfg.kv_pool_tokens:
                num_pages = num_pages * (self.target.kv_bytes_per_token + extra_page) // (kv_token_bytes + extra_page)
        elif cfg.kv_format != "fp8":
            # An accuracy screen: a BF16 pool whose writes round through the screened format, in the FP8 pool's memory
            from sharar.kernels.kv_fp4 import SIMULATED
            if cfg.kv_format not in SIMULATED:
                raise ValueError(f"SHARAR_KV_FORMAT must be fp8 or one of {', '.join(SIMULATED)}")
            if cfg.ssd:
                raise ValueError("a simulated KV format screens a target without the SSD tier (SHARAR_SSD=0)")
            kv_dtype, rounding, num_pages = torch.bfloat16, SIMULATED[cfg.kv_format], num_pages // 2
            kv_token_bytes *= kv_dtype.itemsize
        from sharar.cache.ledger import check_load_headroom
        # the pools and everything allocated after them must leave the floor: an explicit SHARAR_KV_POOL_TOKENS is held
        # to the same rule as a planned pool
        try:
            check_load_headroom(num_pages * (kv_token_bytes + extra_page) + num_slots * state_bytes + future)
        except MemoryError as exc:
            if plan is not None and not planned_pages and num_pages != plan.kv_tokens:
                raise MemoryError(f"{exc}: SHARAR_KV_POOL_TOKENS={num_pages} leaves less than the plan's headroom; "
                                  f"the plan alone gives {plan.kv_tokens} KV tokens") from None
            raise
        # One write-only page per padded graph row: the widest bucket's padding times the widest verify block any
        # proposer may use (speculation can be enabled after construction).
        from sharar.engine.graphs import capture_sizes
        from sharar.proposers.factory import MAX_DEPTHS
        width = max(MAX_DEPTHS.values()) + 1
        self.kv_pool = KvPool(num_pages, self.device, target=self.target, padding_rows=(max(capture_sizes(max_batch)) - 1) * width,
                              dtype=kv_dtype, rounding=rounding, nvfp4=nvfp4)
        model.register_pages(self.kv_pool)  # per-page rows beyond K/V, e.g. Flash-Next's sparse-attention index keys (KvPool.page_aux)
        self.ledger.record_cuda_allocated("kv_pool")
        self.state_pool = StatePool(num_slots, self.device, target=self.target, ssm_format=self.ssm_format)
        model.register_state(self.state_pool)  # per-slot state beyond the GDN's, e.g. the PLE convolution window (StatePool.aux)
        self.ledger.record_cuda_allocated("state_pool")
        cache = RadixCache(self.kv_pool, self.state_pool, self.device) if use_cache else None
        self.state = StateManager(self.kv_pool, self.state_pool, cache)
        self.sampler = Sampler(self.device)
        self.max_batch = max_batch
        self._ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        # The CUDA-core decode kernel rejects GQA group size 6 (24/4 heads); the
        # tensor-core path is what the reference runs for this model too.
        self._decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(self._ws, "NHD", use_tensor_cores=True)
        wide = max(1, -(-max_batch // 64))  # FlashInfer's scratch grows with the rows: two batches of 64 need twice
        self._prefill_ws = torch.empty(256 * 1024 * 1024 * wide, dtype=torch.uint8, device=self.device)
        self._paged_prefill = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self._prefill_ws, "NHD")
        # Eager verification plans its own wrapper: plan() uploads from a pinned buffer
        # asynchronously, so one wrapper shared with prefill could be re-planned in the same
        # scheduler step before the GPU read the previous plan.
        self._verify_ws = torch.empty(128 * 1024 * 1024 * wide, dtype=torch.uint8, device=self.device)
        self._verify_prefill = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self._verify_ws, "NHD")
        self.ledger.record_cuda_allocated("workspaces")
        self.graphs: DecodeGraphs | None = None
        self.use_graphs = False
        self.spec = None
        if log:
            after = after_plan_bytes(max_batch)
            log("  counted after the pools: " + ", ".join(f"{k.replace('_', ' ')} {v / 2**30:.2f}" for k, v in after.items())
                + f" GiB; floor {self.ledger.floor_bytes / 2**30:.0f} GiB")
            log(f"  pools: {num_pages} KV pages ({self.kv_pool.bytes / 2**30:.2f} GiB), {num_slots} state slots ({num_slots * self.state_pool.bytes_per_slot / 2**30:.2f} GiB, {self.ssm_format.describe()} recurrent state)")
        if spec_config is not None:
            self.enable_spec(spec_config.depth, log=log, proposer=spec_config.name, state_strategy=spec_config.state_strategy,
                             draft_precision=spec_config.draft_precision, draft_dir=spec_config.draft_dir)

    @property
    def attention(self):
        """The planner for every target attention call: prefill, decode and verification, eager and graph-bound."""
        planner = self.__dict__.get("_attention")
        if planner is None:  # built on first use (test executors skip __init__)
            from sharar.kernels.attention_plan import AttentionGeometry, AttentionPlanner
            planner = self._attention = AttentionPlanner(AttentionGeometry.of_target(self.target, self.kv_pool.page_size, getattr(self.kv_pool, "dtype", KV_DTYPE)))
        return planner

    @property
    def cache(self):
        return self.state.cache

    @cache.setter
    def cache(self, value):
        self.state.cache = value

    def _features_wanted(self):
        """Target features to compute: the installed proposer's, else none."""
        spec = getattr(self, "spec", None)  # test doubles bypass __init__
        return spec.proposer.requirements.features if spec is not None else None

    def enable_spec(self, k: int = 3, log=None, *, proposer: str = "mtp", state_strategy: str = "snapshots", draft_precision: str = 'bf16',
                    draft_dir: str | None = None) -> None:
        """Select the adapter and its numerical profile once, before capture."""
        if self.graphs is not None or self.spec is not None:
            raise ValueError("enable speculation once, before graph capture")
        from sharar.cache.spec_budget import SpecConfig, budget
        from sharar.cache.ledger import check_load_headroom
        fixed, per_page = budget(SpecConfig(proposer, k, state_strategy, draft_precision, draft_dir), self.max_batch,
                                 cache_slots=self.state_pool.num_slots, target=self.target)
        after = after_plan_bytes(self.max_batch)
        check_load_headroom(fixed + per_page * self.kv_pool.num_pages + after["graphs"] + after["first_use"] + after["activations"])
        context = ProposerContext(self.device, self.max_batch, self.max_context)
        from sharar.proposers.factory import create
        adapter = create(proposer, k, self.model, context, self.kv_pool.num_pages, draft_precision=draft_precision, draft_dir=draft_dir,
                         padding_rows=self.kv_pool.padding_rows)
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
        from sharar.engine.contracts import check_capabilities
        check_capabilities(proposer)
        caps = proposer.requirements.capabilities
        if caps.checkpoint_cache and self.cache is not None:
            proposer.bind_checkpoint_cache(self.state_pool)
        self.spec = SpecDecoder(self, proposer, state_strategy)

    def variant(self, seqs) -> str:
        """The graph variant a decode or verification batch needs: "dense" unless the model's side inputs say
        otherwise (Flash-Next past its indexer budget: "selecting", or the eager path when not captured)."""
        return self.extras.variant(seqs) if self.extras is not None else "dense"

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
            req.rng_seed = request_seed(params.seed)
        self.state.admit(req, req.prefill_ids)
        req.state = State.PREFILL
        return req

    def can_admit(self, prompt_ids: list[int], reserve_tokens: int = 0) -> bool:
        ledger = getattr(self, "ledger", None)
        if ledger is not None and not all_reduce_min(ledger.above_floor())[0]:
            # Unified memory: pool IDs being free does not mean DRAM is. Hold
            # admission (running requests keep decoding and release memory as
            # they finish) instead of letting the host guard stop the server.
            # A collective with more than one rank: the scheduler calls it in lockstep.
            self.floor_holds = getattr(self, "floor_holds", 0) + 1
            return False
        return self.state.can_admit(prompt_ids, reserve_tokens)

    def prefetch(self, prompt_ids: list[int], reserve_tokens: int = 0) -> bool:
        """The queue head's SSD restore, before `can_admit` (`StateManager.prefetch`): False while it is still reading."""
        return self.state.prefetch(prompt_ids, reserve_tokens)

    def admission_reserve(self, req: Request, limit: int) -> int:
        padding = self.spec.k if self.spec is not None and (req.params.greedy or not self.spec.proposer.requirements.greedy_only) else 0
        return min(limit, req.params.max_new_tokens - len(req.output_ids) + padding)

    def cache_stats(self) -> dict | None:
        if self.cache is None:
            return None
        stats = self.cache.stats()
        proposer = self.spec.proposer if self.spec is not None else None
        if proposer is not None and hasattr(proposer, "cache_hits"):
            # resumes whose drafter found the matched snapshot's metadata (hits) or not (fallbacks: MTP drafts on from a
            # stand-in row, DFlash2 stops drafting)
            stats['draft_resumes'] = {'hits': proposer.cache_hits, 'fallbacks': proposer.cache_fallbacks}
        features = self.spec.proposer.feature_cache if self.spec is not None and self.spec.proposer.requirements.capabilities.checkpoint_cache else None
        if features is not None:
            stats['draft_features'] = {'entries': len(features.entries), 'bytes': features.bytes,
                                       'capacity_bytes': features.capacity,
                                       'hits': self.spec.proposer.cache_hits,
                                       'fallbacks': self.spec.proposer.cache_fallbacks}
        return stats

    def _cache_identity(self) -> dict:
        """What makes stored cache bytes interchangeable between processes: the
        same target revision, decode weights, geometry, recurrent-state format
        and prefill numerics."""
        from sharar.engine.config import current
        t = self.target
        tp = get_tp_info()
        identity = {"target": t.name, "revision": t.model_revision, "w3": getattr(self.model, "w3_identity", None),
                    "kv_bytes_per_token": t.kv_bytes_per_token, "state_bytes_per_slot": self.state_pool.bytes_per_slot,
                    "stable_numerics": "1" if current().stable_numerics else "0",
                    **({"tensor_parallel": f"{tp.rank}/{tp.size}"} if tp.size > 1 else {})}  # a rank's records hold its heads
        if self.state_pool.window is not None:  # FP32 records keep their identity
            identity["ssm_state"] = self.state_pool.ssm_format.describe()
        if getattr(self.kv_pool, "nvfp4", False):  # FP8 records keep their identity
            identity["kv_format"] = "nvfp4"
        return identity

    def enable_ssd_tier(self, log=None) -> None:
        """With SHARAR_SSD=1, put the SSD tier (`cache/ssd.py`) under the prefix
        cache, up to SHARAR_SSD_GIB in SHARAR_SSD_DIR, and list the prefixes it
        already holds. Call after the proposer exists so its auxiliary state
        travels with stored prefixes. The tier runs on every tensor-parallel
        rank or on none; without it, the cache drops evicted prefixes and their
        next requests prefill again."""
        from sharar.cache.ledger import GIB
        from sharar.cache.ssd import SsdTier
        from sharar.cache.ssd_store import SsdStore
        from sharar.engine.config import current

        cfg = current()
        if not cfg.ssd or self.cache is None:
            return
        tier, why = None, "another rank could not open its store"
        proposer = self.spec.proposer if self.spec is not None else None
        extra = proposer.stored_pages() if hasattr(proposer, "stored_pages") else ()
        try:
            store = SsdStore(cfg.ssd_dir, self._cache_identity(), log=log, block=cfg.ssd_block_tokens,
                             writers=cfg.ssd_writers)
            tier = SsdTier(store, self.kv_pool, self.state_pool, self.device, int(cfg.ssd_gib * GIB), trace=cfg.ssd_trace,
                           extra_pages=extra, **cfg.ssd_buffers)
        except (RuntimeError, MemoryError, OSError) as exc:
            why = str(exc)
        if not all_reduce_min(int(tier is not None))[0]:
            if log:
                log(f"  ssd tier: disabled ({why})")
            return
        if proposer is not None and proposer.requirements.capabilities.stored_payload:
            tier.register_codec(*proposer.auxiliary_codec())
        self.cache.ssd = tier
        listed = tier.rebuild(self.cache)
        # never fill the drive: at most what is stored plus the free space beyond a reserve (a tenth of the drive, at
        # most 64 GiB), the same on every rank (each trims its own copy of the same tree)
        fs = os.statvfs(store.base)
        reserve = min(fs.f_blocks * fs.f_frsize // 10, 64 * GIB)
        room = tier.used + max(0, fs.f_bavail * fs.f_frsize - reserve)
        budget = all_reduce_min(min(tier.budget, room) // 2**20)[0] * 2**20  # in MiB
        if budget < tier.budget and log:
            log(f"  ssd tier: budget {tier.budget / GIB:.0f} -> {budget / GIB:.0f} GiB, what the drive has free")
        tier.budget = budget
        if log:
            log(f"  ssd tier: {listed} stored prefixes, {tier.used / GIB:.1f} of {tier.budget / GIB:.0f} GiB, {tier.block}-token blocks")

    def cache_step(self, prefilling: bool = False) -> None:
        """Once per engine loop iteration: the prefix cache's background work
        (finished SSD writes, the disk budget, queued backups; more of them
        while a request is prefilling)."""
        if self.cache is not None:
            self.cache.step(prefilling)

    def persist_prefixes(self, log=None) -> int:
        """At shutdown, with the engine stopped: write every cached prefix to
        the SSD tier so it survives the restart. Returns how many it wrote."""
        if self.cache is None or self.cache.ssd is None:
            return 0
        ssd = self.cache.ssd
        before = ssd.tails
        ssd.abort(self.cache)  # a restore still reading lands nowhere: the engine has stopped
        ssd.patient = True  # the engine has stopped: wait for staging room rather than drop a prefix
        self.cache.evict(need_pages=self.kv_pool.num_pages, need_slots=self.state_pool.num_slots)  # evicts every unlocked leaf
        ssd.drain(self.cache)
        if log:
            log(f"ssd tier: persisted {ssd.tails - before} cached prefixes ({ssd.used / 2**30:.1f} GiB stored, "
                f"{ssd.store.write_errors} write errors)")
        return ssd.tails - before

    def retract(self, req: Request) -> None:
        self.state.retract(req)
        if self.spec is not None:
            self.spec.proposer.release(req)
        req.replay_len = len(req.output_ids)
        req.state = State.QUEUED

    def release(self, req: Request) -> None:
        if req.released:
            return
        # the drafter attaches its metadata to the reply end (the request's own slot) before the tree owns it
        self.state.release(req, attach=self.spec.proposer.prepare_release if self.spec is not None else None)
        if self.spec is not None:
            self.spec.proposer.release(req)

    def flush_cache(self) -> None:
        self.state.flush()

    def await_restore(self, timeout: float = 0.01) -> None:
        """Nothing runs but a restore's reads: wait for them a little rather than spin (a local wait; when the
        restore counts as landed every rank agrees in `cache_step`)."""
        ssd = self.cache.ssd if self.cache is not None else None
        if ssd is not None:
            ssd.wait(timeout)

    def forget(self, req: Request) -> None:
        """A queued request was cancelled: anything held for its admission goes."""
        self.state.forget(req.prefill_ids)

    # ---- steps ----
    def _prefix_attention(self, new_tokens: list[int], prefix_lens: list[int], kv_indices: torch.Tensor) -> tuple | None:
        """The 27B's prefill against an NVFP4 prefix of at least PREFIX_ATTENTION_MIN pages on kernels/csrc/verify_attn.cu
        (2,048 rows: 1.6x FlashInfer's fa2 and merge over an 8K prefix, 1.7x over 20K; below 1K pages fa2 is as fast):
        its plan, or None for FlashInfer."""
        from sharar.engine.config import current
        pool = self.kv_pool
        if (current().verify_attention != "auto" or self.extras is not None or max(prefix_lens) < PREFIX_ATTENTION_MIN
                or not getattr(pool, "nvfp4", False) or (pool.num_kv_heads, pool.head_dim) != (4, 256)
                or self.target.num_attention_heads != 24):
            return None
        from sharar.kernels.verify_attn_cuda import prefix_plan
        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        return prefix_plan(new_tokens, prefix_lens, kv_indices, self.target.num_attention_heads, pool.num_kv_heads, sms)

    def _plan_paged_prefill(self, prefix_pages: torch.Tensor, new_tokens: int) -> None:
        # Host index pointers: FlashInfer's plan() reads them on the CPU, so
        # device tensors would cost a synchronization per plan.
        qo_indptr = torch.tensor([0, new_tokens], dtype=torch.int32)
        kv_indptr = torch.tensor([0, prefix_pages.shape[0]], dtype=torch.int32)
        last_page_len = torch.ones(1, dtype=torch.int32)
        self.attention.paged_prefill(self._paged_prefill, qo_indptr, kv_indptr, prefix_pages, last_page_len, causal=False)

    def _checkpoint_positions(self, req: Request, start: int, end: int) -> list[tuple[int, bool]]:
        """The snapshots the prefill forward [start, end) of req's prompt places, as (position, spare), in the order
        their slots are allocated (`StateManager.prefill_checkpoints`): the message boundary and the last chunk end,
        then the branching point, then a spare chunk end. A position inside the forward sits on the forward's own
        64-token grid (the GDN kernels snapshot at the end of a 64-token chunk of the forward); a chunk's end is a copy
        of the request's slot after the forward (`_chunk_checkpoint`).

        - Chunk end (SGLang's one state per forward, mamba_last_track_seqlen): the forward that ends the prompt
          snapshots its last grid point strictly before the prompt's end, so the same prompt sent again resumes there; an
          earlier chunk snapshots its own end, spare unless it is the prompt's last (the last forward places no grid
          point of its own before the end) or the next forward would have to place the message boundary or the
          branching point at its own start.
        - Message boundary: the start of the prompt's last message (`SamplingParams.message_boundary`, from the chat
          API; 0 for a raw prompt), where the conversation's next turn departs from it.
        - Branching point (`req.branching_seqlen`, SGLang's mamba_branching_seqlen): the KV match's end where it ran a
          grid past the matched snapshot, in the forward that contains it: the next request sharing that prefix resumes
          there.
        Nothing while the cache flushes, or for a request that is not cached."""
        if self.cache is None or self.cache.flushing or not req.cache_enabled or not req.cacheable:
            return []
        n = req.prefill_len
        targets = [(rank, t) for rank, t in ((0, req.params.message_boundary), (1, req.branching_seqlen)) if 0 < t < n]
        points: dict[int, list] = {}  # position -> [spare, allocation rank]

        def place(position: int, spare: bool, rank: int) -> None:
            if position in points:
                points[position] = [points[position][0] and spare, min(points[position][1], rank)]
            else:
                points[position] = [spare, rank]

        for rank, target in targets:
            if start <= target < end:
                p = start + (target - start) // GRID * GRID
                if p > start:
                    place(p, False, rank)
        if end >= n:
            p = start + (n - 1 - start) // GRID * GRID
            if p > start:
                place(p, False, 0)
        else:
            needed = n - end <= GRID or any(end <= t < end + GRID for _, t in targets)  # it stands for a not-spare point
            place(end, not needed, 0 if needed else 2)
        return [(p, spare) for p, (spare, _) in sorted(points.items(), key=lambda item: (item[1][1], -item[0]))]

    def _chunk_checkpoint(self, req: Request, end: int) -> None:
        """Copy the request's slot into the snapshot planned at the end of the chunk it just prefilled, if there is one:
        after the forward, before the proposer ingests the chunk, so draft metadata attaches at `end`."""
        snap = req.private.get(end)
        if snap is not None:
            self.state_pool.copy_slot(req.slot, snap.slot)

    def prefill(self, req: Request, max_tokens: int | None = None) -> int | None:
        try:
            batch = self._prepare_single_prefill(req, max_tokens)
            out = self.model.forward(batch.ids, batch.context)
            debug_sync("after prefill forward", [req])
            return self._finish_prefill(batch, out)
        except BaseException:
            self.state.discard_private(req)  # its snapshots were never written: none outlives the failed step
            raise

    def prefill_batch(self, chunks: list[tuple[Request, int]]) -> None:
        try:
            batch = self._prepare_prefill(chunks)
            out = self.model.forward(batch.ids, batch.context)
            debug_sync("after batched prefill forward", [r for r, _ in chunks])
            self._finish_prefill(batch, out)
        except BaseException:
            for req, _ in chunks:  # every row's, not only the one the scheduler retracts
                self.state.discard_private(req)
            raise

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
            ctx.prefix_attention = self._prefix_attention([t], [start], req.pages[:start])
            if ctx.prefix_attention is None:
                self._plan_paged_prefill(req.pages[:start], t)
                ctx.paged_prefill_wrapper = self._paged_prefill
        for _, pos, slot in self.state.prefill_checkpoints([(req, self._checkpoint_positions(req, start, end))]):
            if pos < end:  # inside the forward: the kernels write it (the chunk's end is copied after the forward)
                ctx.checkpoint_positions.append(pos)
                ctx.checkpoint_slots.append(slot)
        if self._features_wanted() is not None:
            ctx.feature_request = self._features_wanted()
        if self.extras is not None:
            ctx.extras = self.extras.for_prefill([(req, start, t, req.pages[:end])])
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
            ids_list += req.prefill_ids[start:end]
            pos_list += list(range(start, end))
            page_list.append(req.pages[start:end])
            cu.append(cu[-1] + n_new)
            kv_indices.append(req.pages[:start])
            kv_indptr.append(kv_indptr[-1] + start)
            seq_slots.append(req.slot)
            prefix_lens.append(start)
            seq_lens.append(n_new)
            boundaries.append((req, self._checkpoint_positions(req, start, end)))
        # inside each row's forward only: the kernels write those (each chunk's end is copied after the forward)
        checkpoints = [(row, pos, slot) for row, pos, slot in self.state.prefill_checkpoints(boundaries)
                       if pos < prefix_lens[row] + seq_lens[row]]
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
            ctx.prefix_attention = self._prefix_attention(seq_lens, prefix_lens, torch.cat(kv_indices))
            if ctx.prefix_attention is None:
                self.attention.paged_prefill(self._paged_prefill, torch.tensor(cu, dtype=torch.int32), torch.tensor(kv_indptr, dtype=torch.int32),
                                             torch.cat(kv_indices), torch.ones(len(chunks), dtype=torch.int32), causal=False)
                ctx.paged_prefill_wrapper = self._paged_prefill
        ids = upload(ids_list, torch.int64, d)
        if self._features_wanted() is not None:
            ctx.feature_request = self._features_wanted()
        if self.extras is not None:
            ctx.extras = self.extras.for_prefill([(req, req.prefill_pos, n, req.pages[:req.prefill_pos + n]) for req, n in chunks])
        return PrefillBatch(list(chunks), ids, ctx, started)

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
            done = [chunks[i][0] for i in finishing]
            tokens = self.sampler.sample(logits, [r.params for r in done], [r.generator for r in done],
                                         sample_masks(done, logits.shape[-1])).tolist()
        for req, n_new in chunks:
            self._chunk_checkpoint(req, req.prefill_pos + n_new)
        if self.spec is not None:
            for i, (req, n_new) in enumerate(chunks):
                rows = {"norm": out["norm"][cu[i]:cu[i + 1]]}
                if "features" in out:
                    rows["features"] = {lid: value[cu[i]:cu[i + 1]] for lid, value in out["features"].items()}
                self.spec.ingest_prefill(req, req.prefill_ids[req.prefill_pos:req.prefill_pos + n_new], req.prefill_pos, rows)
        for req, n_new in chunks:  # the drafter ingested every row's chunk: each row's snapshots go in, its lock moves
            end = req.prefill_pos + n_new
            self.state.commit_prefill(req, end)
            self.state.checkpoint(req, end)
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
        self._chunk_checkpoint(req, end)
        if self.spec is not None:
            self.spec.ingest_prefill(req, new_ids, start, out)
        self.state.checkpoint(req, end)  # the drafter ingested the chunk: its snapshots go in, the lock moves
        token = None
        if end == n:
            logits = self.model.logits(out["norm"][-1:])
            token = int(self.sampler.sample(logits, [req.params], [req.generator], sample_masks([req], logits.shape[-1])).item())
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
        from sharar.engine.prompts import random_ids

        params = SamplingParams(max_new_tokens=1)
        a = self.admit(random_ids(300, 11), params)
        self.prefill(a, max_tokens=256)  # chunk 1
        self.prefill(a)  # chunk 2 with prefix
        b = self.admit(random_ids(40, 12), params)
        c = self.admit(random_ids(70, 13), params)
        self.prefill_batch([(b, 40), (c, 70)])
        for r in (a, b, c):
            self.release(r)
        if self.max_batch > 4:
            self.flush_cache()  # sixteen live warmup rows need every state slot the plan guarantees
        if self.spec is not None:
            # Two speculative rounds at batch 1 and 2 so the first real request
            # does not pay the lazy kernel loads of the verify/accept path. The
            # random prompts can emit EOS; ignore it so every step keeps four live rows,
            # and budget every round at the full draft width (a depth policy may verify all of it).
            rounds = 2 + sum(1 for size in (4, 8, 16) if size <= self.max_batch)
            spec_params = SamplingParams(max_new_tokens=max(12 if self.max_batch <= 4 else 64, 1 + rounds * self.spec.steps),
                                         ignore_eos=True)
            d = self.admit(random_ids(50, 14), spec_params)
            e = self.admit(random_ids(60, 15), spec_params)
            self.prefill(d)
            self.prefill(e)
            self.decode_step([d])
            self.decode_step([d, e])
            live = [d, e]
            for size in (4, 8, 16):  # every captured size, so no round pays a first-use compile
                if self.max_batch < size:
                    break
                while len(live) < size:
                    r = self.admit(random_ids(55 + 10 * (len(live) - 2), 14 + len(live)), spec_params)
                    self.prefill(r)
                    live.append(r)
                self.decode_step(live)
            for r in live:
                self.release(r)
            # One sampled round per path, so the first sampled request does not compile the sampled-speculation
            # kernels (kernels/sampled_verify.py, the sampled lattice walk).
            sampled = SamplingParams(max_new_tokens=12, ignore_eos=True, temperature=1.0, top_k=20, top_p=0.95, seed=0)
            f = self.admit(random_ids(50, 30), sampled)
            self.prefill(f)
            self.decode_step([f])
            self.release(f)
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
        self.attention.decode(self._decode_wrapper, indptr, indices, last_page_len)

    def decode_step(self, reqs: list[Request]) -> list[int]:
        """Process the last generated token of every request and sample the next.
        With speculative decoding on, greedy requests go through spec rounds
        and sampled ones through the plain step."""
        assert 0 < len(reqs) <= self.max_batch
        spec_reqs = [r for r in reqs if self.spec is not None and self.spec.ready(r)
                     and (r.params.greedy or not self.spec.proposer.requirements.greedy_only)]
        if spec_reqs and self.spec.depth_policy.choose(reqs) == 0:
            spec_reqs = []  # a wide batch at depth 0 (the throughput profile): plain decode; the proposer still follows
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
        seqs = [(r.context_len - 1, 1, pages) for r, pages in zip(reqs, step_pages)]
        variant = self.variant(seqs)
        if self.use_graphs and self.graphs is not None and variant in self.graphs.variants:
            result = self.graphs.run(
                [r.output_ids[-1] for r in reqs],
                [r.context_len - 1 for r in reqs],
                [r.slot for r in reqs],
                new_pages,
                step_pages,
                return_features=self.spec is not None,
                side=(reqs, seqs),
                variant=variant,
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
            if self._features_wanted() is not None:
                ctx.feature_request = self._features_wanted()
            if self.extras is not None:
                ctx.extras = self.extras.for_decode(reqs, seqs)
            out = self.model.forward(tokens, ctx)
            debug_sync("after plain decode forward", reqs)
            logits = self.model.logits(out["norm"])
        next_tokens = self.sampler.sample(logits, [r.params for r in reqs], [r.generator for r in reqs],
                                          sample_masks(reqs, logits.shape[-1])).tolist()
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
