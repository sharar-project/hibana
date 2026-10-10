"""EngineConfig: every engine setting in one frozen value (#44).

The launcher and the CLI pass settings as `SHARAR_*` environment variables;
`EngineConfig.from_env()` reads them once, with the defaults the modules used to
apply at import. `sharar serve` and the benchmark loader install the result
before building the model; modules read `current()` where they construct
things (a model, a linear, a proposer, a graph set), never at import, and the
few per-call checks read one attribute of the installed value. Tests and
benchmarks build their own with `EngineConfig(...)` or `current().replace(...)`
and `install()` it (or use `using()`).
"""

from __future__ import annotations

import dataclasses
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Mapping


def _flag(env: Mapping[str, str], name: str, default: bool = False) -> bool:
    value = env.get(name)
    return default if value is None else value == "1"


def _int(env: Mapping[str, str], name: str, default: int | None) -> int | None:
    value = env.get(name)
    return default if value in (None, "") else int(value)


def _str(env: Mapping[str, str], name: str, default: str | None = None) -> str | None:
    value = env.get(name)
    return default if value in (None, "") else value


@dataclass(frozen=True)
class EngineConfig:
    # --- target and files
    profile: str = "fast"                     # SHARAR_PROFILE: exact / balanced / fast / throughput (resolved by the launcher, reported here)
    target: str | None = None                 # SHARAR_TARGET: qwen38 (default) or flashnext
    hf_cache: str | None = None               # SHARAR_HF_CACHE, else HF_HOME: the 27B's model cache
    flashnext_cache: str | None = None        # SHARAR_FLASHNEXT_CACHE: Flash-Next's model cache
    max_context: int | None = None            # SHARAR_MAX_CONTEXT (0/unset: the target's)
    max_batch: int = 4                        # SHARAR_MAX_BATCH: most requests decoded together, 1-128 (the throughput profile: 16)
    cuda_arch: str = "12.1a"                  # SHARAR_CUDA_ARCH for the JIT-built CUDA kernels
    tp_allreduce: str = "rdma"                # SHARAR_TP_ALLREDUCE: rdma (the one-shot for decode-sized all-reduces between two ranks) or nccl (every one on NCCL, as vLLM)
    # --- weights and numerics
    w3_decode: bool = False                   # SHARAR_W3_DECODE: 3-bit decode copies for calls of <= 64 rows (the 27B's layers and FP8 draft, Flash-Next's trunk)
    w3_checkpoint: str | None = None          # SHARAR_W3_CHECKPOINT: the target's calibrated copies (required on Flash-Next; the 27B rounds to nearest without)
    w3_warning: str | None = None             # SHARAR_W3_WARNING: why the launcher serves without the profile's calibrated copies
    stable_numerics: bool = False             # SHARAR_STABLE_NUMERICS: M-invariant prefill kernels
    gdn_prefill: str = "auto"                 # SHARAR_GDN_PREFILL: auto (CuTe on SM12x), cute, triton
    flashnext_trunk: str = "fp8"              # SHARAR_FLASHNEXT_TRUNK: bf16 | fp8
    flashnext_mixers: str = "fp8"             # SHARAR_FLASHNEXT_MIXERS
    flashnext_mtp_trunk: str | None = None    # SHARAR_FLASHNEXT_MTP_TRUNK: the MTP draft head's precision (unset: the trunk's)
    trunk_keep_bf16: str | None = None        # SHARAR_TRUNK_KEEP_BF16: regex of Flash-Next projections kept BF16
    experts_sim_bits: int = 3                 # SHARAR_EXPERTS_SIM_BITS: 3 | 2, the simulated experts' level set (2: a 2-bit screen)
    flashnext_experts_sim3: int = 0           # SHARAR_FLASHNEXT_EXPERTS_SIM3: routed experts rounded to a 3-bit format with this group (16/32), a quality screen (weights/expert_sim3.py)
    flashnext_experts: str = "w3"             # SHARAR_FLASHNEXT_EXPERTS: w3 (the W3 checkpoint's 3-bit routed experts when it carries them, #73) | nvfp4
    # --- speculation
    spec: str | None = None                   # SHARAR_SPEC as given ("auto"/unset: the proposer's default depth)
    spec_depths: str | None = None            # SHARAR_SPEC_DEPTHS: verify depth per batch bucket, "8:1,16:0" (0: plain decode; unset: SHARAR_SPEC everywhere)
    draft_dir: str | None = None              # SHARAR_DRAFT
    draft_vocab: int = 0                      # SHARAR_DRAFT_VOCAB: reduced MTP draft vocabulary rows (0: full)
    draft_head: str = "fp8"                   # SHARAR_DRAFT_HEAD: fp8 (the target's head) or w3
    draft_head_rescore: int = 8               # SHARAR_DRAFT_HEAD_RESCORE
    device_accept: bool = False               # SHARAR_DEVICE_ACCEPT
    accept_default: str = "2:1.0"             # SHARAR_ACCEPT_DEFAULT (greedy requests that set neither field)
    accept_default_sampled: bool = False      # SHARAR_ACCEPT_DEFAULT_SAMPLED
    accept_default_sampled_threshold: str | None = None  # SHARAR_ACCEPT_DEFAULT_SAMPLED_THRESHOLD: K:M for sampled requests (unset: accept_default's)
    qsa_sparse_graphs: bool = True            # SHARAR_QSA_SPARSE_GRAPHS=0: rows past the indexer budget run eagerly
    # --- cache tiers
    cold_tier_gib: float | None = None        # SHARAR_COLD_TIER_GIB (unset: from free memory, capped at 24)
    session_dir: str | None = None            # SHARAR_SESSION_DIR
    session_gib: float = 200.0                # SHARAR_SESSION_GIB
    session_min_tokens: int = 4096            # SHARAR_SESSION_MIN_TOKENS
    # --- traces
    traces: str | None = None                 # SHARAR_TRACES
    traces_max_mib: int = 32768               # SHARAR_TRACES_MAX_MIB
    trace_confidence: bool = False            # SHARAR_TRACE_CONFIDENCE: the MTP draft's confidence per drafted position in trace rounds (#32, #43; read when the proposer is built)
    # --- diagnostics and benchmark loader
    debug_sync: bool = False                  # SHARAR_DEBUG_SYNC
    commit_profile: bool = False              # SHARAR_COMMIT_PROFILE
    bench_verify_state: str = "snapshots"     # SHARAR_BENCH_VERIFY_STATE
    bench_max_context: int = 0                # SHARAR_BENCH_MAX_CONTEXT

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> EngineConfig:
        e = os.environ if env is None else env
        cold = _str(e, "SHARAR_COLD_TIER_GIB")
        return cls(
            profile=_str(e, "SHARAR_PROFILE", "fast"),
            target=_str(e, "SHARAR_TARGET"),
            hf_cache=_str(e, "SHARAR_HF_CACHE") or _str(e, "HF_HOME"),
            flashnext_cache=_str(e, "SHARAR_FLASHNEXT_CACHE"),
            max_context=_int(e, "SHARAR_MAX_CONTEXT", 0) or None,
            max_batch=_int(e, "SHARAR_MAX_BATCH", 4),
            cuda_arch=_str(e, "SHARAR_CUDA_ARCH", "12.1a"),
            tp_allreduce=_str(e, "SHARAR_TP_ALLREDUCE", "rdma"),
            w3_decode=_flag(e, "SHARAR_W3_DECODE"),
            w3_checkpoint=_str(e, "SHARAR_W3_CHECKPOINT"),
            w3_warning=_str(e, "SHARAR_W3_WARNING"),
            stable_numerics=_flag(e, "SHARAR_STABLE_NUMERICS"),
            gdn_prefill=_str(e, "SHARAR_GDN_PREFILL", "auto"),
            flashnext_trunk=_str(e, "SHARAR_FLASHNEXT_TRUNK", "fp8"),
            flashnext_mixers=_str(e, "SHARAR_FLASHNEXT_MIXERS", "fp8"),
            flashnext_mtp_trunk=_str(e, "SHARAR_FLASHNEXT_MTP_TRUNK"),
            trunk_keep_bf16=_str(e, "SHARAR_TRUNK_KEEP_BF16"),
            flashnext_experts_sim3=_int(e, "SHARAR_FLASHNEXT_EXPERTS_SIM3", 0) or 0,
            experts_sim_bits=_int(e, "SHARAR_EXPERTS_SIM_BITS", 3) or 3,
            flashnext_experts=_str(e, "SHARAR_FLASHNEXT_EXPERTS", "w3"),
            spec=_str(e, "SHARAR_SPEC"),
            spec_depths=_str(e, "SHARAR_SPEC_DEPTHS"),
            draft_dir=_str(e, "SHARAR_DRAFT"),
            draft_vocab=_int(e, "SHARAR_DRAFT_VOCAB", 0),
            draft_head=_str(e, "SHARAR_DRAFT_HEAD", "fp8"),
            draft_head_rescore=_int(e, "SHARAR_DRAFT_HEAD_RESCORE", 8),
            device_accept=_flag(e, "SHARAR_DEVICE_ACCEPT"),
            accept_default=e.get("SHARAR_ACCEPT_DEFAULT", "2:1.0"),
            accept_default_sampled=_flag(e, "SHARAR_ACCEPT_DEFAULT_SAMPLED"),
            accept_default_sampled_threshold=_str(e, "SHARAR_ACCEPT_DEFAULT_SAMPLED_THRESHOLD"),
            qsa_sparse_graphs=e.get("SHARAR_QSA_SPARSE_GRAPHS", "1") != "0",
            cold_tier_gib=float(cold) if cold is not None else None,
            session_dir=_str(e, "SHARAR_SESSION_DIR"),
            session_gib=float(e.get("SHARAR_SESSION_GIB", "200")),
            session_min_tokens=int(e.get("SHARAR_SESSION_MIN_TOKENS", "4096")),
            traces=_str(e, "SHARAR_TRACES"),
            traces_max_mib=_int(e, "SHARAR_TRACES_MAX_MIB", 32768),
            trace_confidence=_flag(e, "SHARAR_TRACE_CONFIDENCE"),
            debug_sync=_flag(e, "SHARAR_DEBUG_SYNC"),
            commit_profile=_flag(e, "SHARAR_COMMIT_PROFILE"),
            bench_verify_state=e.get("SHARAR_BENCH_VERIFY_STATE", "snapshots"),
            bench_max_context=_int(e, "SHARAR_BENCH_MAX_CONTEXT", 0),
        )

    def replace(self, **changes) -> EngineConfig:
        return dataclasses.replace(self, **changes)

    def describe(self) -> dict:
        """Every field, for `/get_server_info` and run records."""
        return dataclasses.asdict(self)

    def model_cache(self, env_name: str | None) -> str | None:
        """The cache a target names by its variable (`Target.cache_env`)."""
        if env_name == "SHARAR_FLASHNEXT_CACHE":
            return self.flashnext_cache
        return None


_installed: EngineConfig | None = None


def current() -> EngineConfig:
    """The installed configuration; the first call without one installs `from_env()`."""
    global _installed
    if _installed is None:
        _installed = EngineConfig.from_env()
    return _installed


def install(config: EngineConfig) -> EngineConfig:
    global _installed
    _installed = config
    return config


def reset() -> None:
    """Forget the installed value (tests): the next `current()` reads the environment again."""
    global _installed
    _installed = None


@contextmanager
def using(config: EngineConfig):
    global _installed
    previous = _installed
    _installed = config
    try:
        yield config
    finally:
        _installed = previous
