"""EngineConfig: every engine setting in one frozen value.

The launcher and the CLI pass settings as `SHARAR_*` environment variables;
`EngineConfig.from_env()` reads them once. The field defaults are the library's
baseline (no lossy option, no SSD tier) that tests and benchmarks build on; a
served engine's defaults are the target's and the profile's
(`sharar/launch/settings.py`), which `sharar engine` resolves beneath the
environment as `sharar serve` does. `sharar serve` and the benchmark loader install the result
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
from pathlib import Path
from typing import Mapping


def _flag(env: Mapping[str, str], name: str, default: bool = False) -> bool:
    value = env.get(name)
    return default if value is None else value == "1"


def _int(env: Mapping[str, str], name: str, default: int | None) -> int | None:
    value = env.get(name)
    return default if value in (None, "") else int(value)


def _budget(env: Mapping[str, str]) -> tuple[int, int]:
    """SHARAR_MOE_BUDGET as (per request, cap): "64" -> (0, 64), "12:64" -> (12, 64), unset -> (0, 0)."""
    value = env.get("SHARAR_MOE_BUDGET") or "0"
    per, sep, cap = value.partition(":")
    if not sep:
        per, cap = "0", per
    if not per.strip().isdigit() or not cap.strip().isdigit():
        raise ValueError(f"SHARAR_MOE_BUDGET must be B or P:B (experts), got {value!r}")
    return int(per), int(cap)


SSD_DIR = str(Path.home() / ".local" / "share" / "sharar" / "ssd")  # the engine's default; the launcher passes $SHARAR_RESULTS/ssd/<target>


def _ssd_block(tokens: int) -> int:
    if tokens < 1 or 2048 % tokens:
        raise ValueError(f"SHARAR_SSD_BLOCK_TOKENS must divide 2048, got {tokens}")
    return tokens


def _positive(env: Mapping[str, str], name: str, default: int) -> int:
    n = _int(env, name, default)
    if n < 1:
        raise ValueError(f"{name} must be at least 1, got {n}")
    return n


def _str(env: Mapping[str, str], name: str, default: str | None = None) -> str | None:
    value = env.get(name)
    return default if value in (None, "") else value


@dataclass(frozen=True)
class EngineConfig:
    # --- target and files
    profile: str = "fast"                     # SHARAR_PROFILE: exact / fast / throughput (resolved by the launcher, reported here)
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
    ssm_state: str = "fp32"                   # SHARAR_SSM_STATE: GDN recurrent state storage, fp32, int8 or int4 (LeapQuant per-window quantization, cache/ssm_format.py; lossy)
    ssm_window: int = 16                      # SHARAR_SSM_WINDOW: committed tokens per requantization (8, 16, 32)
    ssm_compensators: int | None = None       # SHARAR_SSM_COMPENSATORS: FP16 rank-one outlier terms kept beside the quantized state (0-16; unset: 4, or 8 for int4)
    flashnext_trunk: str = "fp8"              # SHARAR_FLASHNEXT_TRUNK: bf16 | fp8
    trunk_keep_bf16: str | None = None        # SHARAR_TRUNK_KEEP_BF16: regex of Flash-Next projections kept BF16
    flashnext_experts: str = "w3"             # SHARAR_FLASHNEXT_EXPERTS: w3 (the W3 checkpoint's 3-bit routed experts when it carries them) | nvfp4
    # --- speculation
    spec: str | None = None                   # SHARAR_SPEC as given ("auto"/unset: the proposer's default depth)
    spec_depths: str | None = None            # SHARAR_SPEC_DEPTHS: verify depth per batch bucket, "8:1,16:0" (0: plain decode; unset: SHARAR_SPEC everywhere)
    spec_step_cost: float = 0.0               # SHARAR_SPEC_STEP_COST: adaptive depth per request (engine/depth.AcceptanceDepth): one more drafted token's cost relative to a depth-1 round; SHARAR_SPEC is the cap (0: the fixed depth)
    draft_dir: str | None = None              # SHARAR_DRAFT
    draft_vocab: int = 0                      # SHARAR_DRAFT_VOCAB: reduced MTP draft vocabulary rows (0: full)
    draft_head: str = "fp8"                   # SHARAR_DRAFT_HEAD: fp8 (the target's head), w3 or w2
    draft_head_rescore: int = 8               # SHARAR_DRAFT_HEAD_RESCORE (default 8, 16 for w2)
    device_accept: bool = False               # SHARAR_DEVICE_ACCEPT
    accept_default: str = "off"               # SHARAR_ACCEPT_DEFAULT: K:M for greedy requests that set neither field (off: exact)
    accept_default_sampled: bool = False      # SHARAR_ACCEPT_DEFAULT_SAMPLED
    accept_default_sampled_threshold: str | None = None  # SHARAR_ACCEPT_DEFAULT_SAMPLED_THRESHOLD: K:M for sampled requests (unset: accept_default's)
    qsa_sparse_graphs: bool = True            # SHARAR_QSA_SPARSE_GRAPHS=0: rows past the indexer budget run eagerly
    qsa_prefill_backend: str = "original"    # SHARAR_QSA_PREFILL_BACKEND: original, auto, optimized (SM121 TP1/TP2, <=32K)
    ple_resident: str = "auto"                 # SHARAR_PLE_RESIDENT: Flash-Next's PLE table in device memory, each tensor-parallel rank its share of the heads; auto: on two Sparks, 0: off
    # --- the SSD tier (cache/ssd.py) under the GPU prefix cache
    ssd: bool = False                         # SHARAR_SSD
    ssd_gib: float = 200.0                    # SHARAR_SSD_GIB: its capacity
    ssd_dir: str = SSD_DIR                    # SHARAR_SSD_DIR: where it stores
    # the SSD tier's tuning
    ssd_block_tokens: int = 256               # SHARAR_SSD_BLOCK_TOKENS: tokens per SSD KV block (divides the 2048-token prefill chunk)
    ssd_trace: bool = False                   # SHARAR_SSD_TRACE: one line per restore in <ssd dir>/v2/restores.jsonl
    ssd_writers: int = 4                      # SHARAR_SSD_WRITERS: the store's writer threads (at least 1)
    ssd_staging_mib: int = 512                # SHARAR_SSD_STAGING_MIB: backup copies in flight to the writers
    ssd_chunk_mib: int = 256                  # SHARAR_SSD_CHUNK_MIB: each half of the restore staging buffer
    kv_pool_tokens: int | None = None         # SHARAR_KV_POOL_TOKENS: a fixed KV pool size, e.g. to compare the SSD tier on and off; unset: planned
    # --- traces
    traces: str | None = None                 # SHARAR_TRACES
    traces_max_mib: int = 32768               # SHARAR_TRACES_MAX_MIB
    trace_confidence: bool = False            # SHARAR_TRACE_CONFIDENCE: the MTP draft's confidence per drafted position in trace rounds (read when the proposer is built)
    # --- diagnostics and benchmark loader
    debug_sync: bool = False                  # SHARAR_DEBUG_SYNC
    bench_verify_state: str = "snapshots"     # SHARAR_BENCH_VERIFY_STATE
    moe_keep: int = 0                         # SHARAR_MOE_KEEP=k: Flash-Next decode and verify forwards, the most top experts a token is guaranteed (alone: each token's top k); 0, or k of 10 and above: the served top-10
    moe_budget: int = 0                       # SHARAR_MOE_BUDGET=[P:]B: with SHARAR_MOE_KEEP, a decode or verify forward reads at most B distinct experts, or min(B, P x its requests) (more only for the tokens' guaranteed experts): those with the most router probability across its tokens (model/flashnext/moe.py, route_budget); 0: off
    moe_budget_per_request: int = 0           # the P of SHARAR_MOE_BUDGET=P:B

    def __post_init__(self) -> None:
        if self.qsa_prefill_backend not in ("original", "auto", "optimized"):
            raise ValueError("SHARAR_QSA_PREFILL_BACKEND must be original, auto, or optimized; "
                             f"got {self.qsa_prefill_backend!r}")

    @property
    def ssd_buffers(self) -> dict[str, int]:
        """The SSD tier's staging sizes in bytes: the pool plan reserves exactly what the tier pins."""
        return {"staging_bytes": self.ssd_staging_mib << 20, "chunk_bytes": self.ssd_chunk_mib << 20}

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> EngineConfig:
        e = os.environ if env is None else env
        budget = _budget(e)  # (per request, cap)
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
            ssm_state=_str(e, "SHARAR_SSM_STATE", "fp32"),
            ssm_window=_int(e, "SHARAR_SSM_WINDOW", 16),
            ssm_compensators=_int(e, "SHARAR_SSM_COMPENSATORS", None),
            flashnext_trunk=_str(e, "SHARAR_FLASHNEXT_TRUNK", "fp8"),
            trunk_keep_bf16=_str(e, "SHARAR_TRUNK_KEEP_BF16"),
            flashnext_experts=_str(e, "SHARAR_FLASHNEXT_EXPERTS", "w3"),
            spec=_str(e, "SHARAR_SPEC"),
            spec_depths=_str(e, "SHARAR_SPEC_DEPTHS"),
            spec_step_cost=float(_str(e, "SHARAR_SPEC_STEP_COST") or 0.0),
            draft_dir=_str(e, "SHARAR_DRAFT"),
            draft_vocab=_int(e, "SHARAR_DRAFT_VOCAB", 0),
            draft_head=_str(e, "SHARAR_DRAFT_HEAD", "fp8"),
            draft_head_rescore=_int(e, "SHARAR_DRAFT_HEAD_RESCORE", 16 if e.get("SHARAR_DRAFT_HEAD") == "w2" else 8),
            device_accept=_flag(e, "SHARAR_DEVICE_ACCEPT"),
            accept_default=e.get("SHARAR_ACCEPT_DEFAULT", "off"),
            accept_default_sampled=_flag(e, "SHARAR_ACCEPT_DEFAULT_SAMPLED"),
            accept_default_sampled_threshold=_str(e, "SHARAR_ACCEPT_DEFAULT_SAMPLED_THRESHOLD"),
            qsa_sparse_graphs=e.get("SHARAR_QSA_SPARSE_GRAPHS", "1") != "0",
            qsa_prefill_backend=_str(e, "SHARAR_QSA_PREFILL_BACKEND", "original"),
            ple_resident=e.get("SHARAR_PLE_RESIDENT", "auto"),
            ssd=_flag(e, "SHARAR_SSD"),
            ssd_gib=float(e.get("SHARAR_SSD_GIB") or "200"),
            ssd_dir=_str(e, "SHARAR_SSD_DIR", SSD_DIR),
            ssd_block_tokens=_ssd_block(_int(e, "SHARAR_SSD_BLOCK_TOKENS", 256)),
            ssd_trace=_flag(e, "SHARAR_SSD_TRACE"),
            ssd_writers=_positive(e, "SHARAR_SSD_WRITERS", 4),
            ssd_staging_mib=_positive(e, "SHARAR_SSD_STAGING_MIB", 512),
            ssd_chunk_mib=_positive(e, "SHARAR_SSD_CHUNK_MIB", 256),
            kv_pool_tokens=_int(e, "SHARAR_KV_POOL_TOKENS", 0) or None,
            traces=_str(e, "SHARAR_TRACES"),
            traces_max_mib=_int(e, "SHARAR_TRACES_MAX_MIB", 32768),
            trace_confidence=_flag(e, "SHARAR_TRACE_CONFIDENCE"),
            debug_sync=_flag(e, "SHARAR_DEBUG_SYNC"),
            bench_verify_state=e.get("SHARAR_BENCH_VERIFY_STATE", "snapshots"),
            moe_keep=_int(e, "SHARAR_MOE_KEEP", 0),
            moe_budget=budget[1],
            moe_budget_per_request=budget[0],
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
