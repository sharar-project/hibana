"""Pinned serving targets: the facts the engine needs about a model, as a value, not module globals.

A `Target` names one validated checkpoint revision and carries the geometry
the engine, pools, graphs and kernels size themselves from (hidden size,
layer kinds, attention and recurrent head shapes, vocabulary, context), the
serving profile, the load budget and the snapshot lookup. One process serves
one target, chosen at launch; nothing here is discovered at run time. The
27B is the first instance; `hibana.config` keeps exporting its constants for
tools and tests that predate this module.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from hibana.distributed import div_even
from hibana.engine.profile import DEFAULT_PROFILE, ServingProfile
from hibana.hardware.gb10 import COMPUTE_CAPABILITY, GPU_NAME, MIN_DRIVER_MAJOR, MIN_TOTAL_MEMORY_KIB


@dataclass(frozen=True)
class Target:
    name: str
    model_id: str
    model_revision: str
    model_cache_key: str
    architecture: str
    hidden_size: int
    intermediate_size: int | None  # dense MLP width; None for a pure-MoE trunk
    num_layers: int
    layer_types: tuple[str, ...]  # "full_attention" | "linear_attention" | "moe" (a block without state), one per layer
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    partial_rotary_factor: float
    rope_theta: float
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel: int
    vocab_size: int
    rms_norm_eps: float
    max_position_embeddings: int
    nvfp4_group_size: int
    weight_load_peak_bytes: int
    profile: ServingProfile = field(default=DEFAULT_PROFILE)
    # Published calibrated 3-bit decode copies (`weights/w3_checkpoint.fetch`); None: calibrate locally.
    w3_repo: str | None = None
    w3_revision: str | None = None
    kv_bytes_per_element: int = 1  # FP8 KV cache
    # Model family: selects the loader in `load_model` and the modules a family adds.
    family: str = "qwen38"
    # Environment variable naming this target's model cache (falls back to HIBANA_HF_CACHE).
    cache_env: str = "HIBANA_HF_CACHE"
    # Gated DeltaNet output gate: "swish" (silu(z), the 27B) or "sigmoid" (Flash-Next).
    output_gate: str = "swish"
    # Most prompt tokens one request prefills per step (`ServingProfile.chunk_size`).
    prefill_chunk: int = 2048
    # The chunk when one request prefills and nothing decodes (`SchedulerConfig.solo_chunk_size`; 0: prefill_chunk).
    solo_prefill_chunk: int = 0
    # Mixture of experts (0 = dense MLP of `intermediate_size`).
    num_experts: int = 0
    experts_per_token: int = 0
    moe_intermediate_size: int = 0
    shared_expert_intermediate_size: int = 0
    # Latent routed experts (0 = experts read the full hidden stream).
    moe_latent_size: int = 0
    routed_scaling_factor: float = 1.0
    # Hyper-connections (0 = plain residual stream).
    hc_count: int = 0
    hc_lowrank: int = 0
    # Prefix-lookup n-gram embedding layers (config numbering, one-based) and their geometry.
    ple_layer_ids: tuple[int, ...] = ()
    ple_ngram: int = 0
    ple_heads_per_ngram: int = 0
    ple_embed_dim: int = 0
    ple_conv_kernel: int = 0
    # Sparse attention indexer (0 = dense attention).
    indexer_budget: int = 0
    indexer_compress_ratio: int = 0
    indexer_n_heads: int = 0
    indexer_head_dim: int = 0
    # In-checkpoint multi-token-prediction layers.
    mtp_layers: int = 0
    # Pinned block drafters trained on this target's features: (proposer, repository, revision).
    draft_pins: tuple[tuple[str, str, str], ...] = ()
    # each pinned draft's acceptance calibration (adaptive verification ranks verify seats by it): name -> the weights
    # of sigmoid(w . [logit p, gap, j / 15, 1]) (proposers/parallel.ParallelProposer.acceptance_features)
    draft_calibration: tuple[tuple[str, tuple[float, ...]], ...] = ()
    # A request's default stop ids (the chat template's end of turn, then end of text).
    stop_token_ids: tuple[int, ...] = (248046, 248044)  # Qwen3.8: <|im_end|>, <|endoftext|>
    # Chat template variables the server renders with unless a request's chat_template_kwargs sets them.
    template_defaults: tuple[tuple[str, object], ...] = ()
    # Puzzle MoE blocks: (checkpoint index, expert count, intermediate width, top-k).
    moe_block_specs: tuple[tuple[int, int, int, int], ...] = ()
    # The recurrent layers' recurrence: "gdn" (Gated DeltaNet) or "mamba2"; it picks the state formats
    # (`launch.settings.SSM_STATES`) and the accepted-state replay.
    recurrence: str = "gdn"
    # The checkpoint's prefix of the in-checkpoint MTP layers' tensors (`cache/spec_budget.py`).
    mtp_prefix: str = "mtp."
    # The first ordinary-text token id: random warm-up prompts draw from it up (below it a vocabulary keeps control ids).
    first_text_token: int = 0
    # Restrict cache formats when a target's attention implements only a subset (None: the shared pool's formats).
    kv_formats: tuple[str, ...] | None = None

    def __post_init__(self):
        if len(self.layer_types) != self.num_layers or any(t not in ("full_attention", "linear_attention", "moe") for t in self.layer_types):
            raise ValueError("layer_types must name every layer as full_attention, linear_attention or moe")

    def shard(self, size: int) -> Target:
        """One tensor-parallel rank's share of `size`: attention and linear-attention heads and the MLP widths
        (dense, routed experts', shared expert's) are per rank (what its pools, kernels and graphs size from); the
        hidden size, the vocabulary, the expert count and the indexer stay whole."""
        if size == 1:
            return self
        split = ("num_attention_heads", "num_kv_heads", "linear_num_key_heads", "linear_num_value_heads",
                 "intermediate_size", "moe_intermediate_size", "shared_expert_intermediate_size")
        return replace(self, **{name: div_even(getattr(self, name), size) for name in split if getattr(self, name)})

    # ---- derived geometry ----
    @property
    def attention_layer_ids(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "full_attention"]

    @property
    def gdn_layer_ids(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "linear_attention"]

    @property
    def context_length(self) -> int:
        return self.profile.max_context

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def kv_bytes_per_token(self) -> int:
        kv = 2 * self.num_kv_heads * self.head_dim * self.kv_bytes_per_element * len(self.attention_layer_ids)
        return kv + self.index_bytes_per_token

    @property
    def nvfp4_kv_bytes_per_token(self) -> int:
        """`kv_bytes_per_token` with K and V as NVFP4 (`HIBANA_KV_FORMAT=nvfp4`): half a byte a value and an E4M3 scale
        per 16, 9/16 of FP8's; the index keys stay BF16."""
        return 2 * self.num_kv_heads * self.head_dim * 9 // 16 * len(self.attention_layer_ids) + self.index_bytes_per_token

    @property
    def index_bytes_per_token(self) -> int:
        """A sparse-attention indexer keeps a raw and a compressed BF16 index key per token and layer."""
        return 2 * self.indexer_head_dim * 2 * len(self.attention_layer_ids) if self.indexer_budget else 0

    @property
    def conv_dim(self) -> int:
        return 2 * self.linear_num_key_heads * self.linear_key_head_dim + self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def conv_state_len(self) -> int:
        return self.linear_conv_kernel - 1

    @property
    def state_bytes_per_layer(self) -> int:
        conv = self.conv_dim * self.conv_state_len * 2  # BF16
        ssm = self.linear_num_value_heads * self.linear_value_head_dim * self.linear_key_head_dim * 4  # FP32
        return conv + ssm

    @property
    def state_bytes_per_slot(self) -> int:
        return self.state_bytes_per_layer * len(self.gdn_layer_ids)

    # ---- snapshot and host validation ----
    def draft_pin(self, proposer: str) -> tuple[str, str]:
        """The pinned (repository, revision) of this target's `proposer` draft."""
        for name, repo, revision in self.draft_pins:
            if name == proposer:
                return repo, revision
        raise ValueError(f"{self.name} has no pinned {proposer} draft")

    def cache_root(self, cache_dir: str | os.PathLike | None = None) -> Path:
        """This target's model cache: `cache_dir`, its own cache setting, or HIBANA_HF_CACHE (HF_HOME)."""
        from hibana.engine.config import current
        cfg = current()
        root = cache_dir or cfg.model_cache(self.cache_env) or cfg.hf_cache
        if not root:
            raise FileNotFoundError("set HIBANA_HF_CACHE (or HF_HOME) to the model cache")
        return Path(root)

    def snapshot_dir(self, cache_dir: str | os.PathLike | None = None) -> Path:
        """Return the pinned snapshot directory or raise if it is incomplete."""
        root = self.cache_root(cache_dir)
        for base in (root, root / "hub"):
            snap = base / self.model_cache_key / "snapshots" / self.model_revision
            if (snap / "config.json").is_file() and (snap / "model.safetensors.index.json").is_file():
                return snap
        raise FileNotFoundError(f"pinned snapshot {self.model_revision} not found under {root}")

    def load_model_config(self, snap: Path) -> dict:
        if self.name == "bolt":
            from hibana.model.bolt.config import check_config
            return check_config(self, snap)
        if self.family == "nemotron":
            from hibana.model.nemotron.config import check_config
            return check_config(self, snap)
        cfg = json.loads((snap / "config.json").read_text())
        text = cfg["text_config"]
        expected = {
            "hidden_size": self.hidden_size, "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_layers, "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_kv_heads, "head_dim": self.head_dim,
            "partial_rotary_factor": self.partial_rotary_factor,
            "linear_num_key_heads": self.linear_num_key_heads, "linear_num_value_heads": self.linear_num_value_heads,
            "linear_key_head_dim": self.linear_key_head_dim, "linear_value_head_dim": self.linear_value_head_dim,
            "linear_conv_kernel_dim": self.linear_conv_kernel, "vocab_size": self.vocab_size,
            "rms_norm_eps": self.rms_norm_eps, "max_position_embeddings": self.max_position_embeddings,
        }
        if self.num_experts:
            expected.update({"num_experts": self.num_experts, "num_experts_per_tok": self.experts_per_token,
                             "moe_intermediate_size": self.moe_intermediate_size,
                             "shared_expert_intermediate_size": self.shared_expert_intermediate_size})
        if self.hc_count:
            expected.update({"hc_count": self.hc_count, "hc_lowrank": self.hc_lowrank})
        if self.ple_layer_ids:
            expected.update({"ple_layer_ids": list(self.ple_layer_ids), "ngram_size": self.ple_ngram, "heads_per_ngram": self.ple_heads_per_ngram,
                             "ple_embed_dim": self.ple_embed_dim, "ple_conv_kernel_size": self.ple_conv_kernel})
        if self.indexer_budget:
            expected.update({"indexer_budget": self.indexer_budget, "indexer_compress_ratio": self.indexer_compress_ratio,
                             "indexer_n_heads": self.indexer_n_heads, "indexer_head_dim": self.indexer_head_dim})
        if self.output_gate != "swish":
            expected["output_gate_type"] = self.output_gate
        wrong = {k: (text.get(k), v) for k, v in expected.items() if text.get(k) != v}
        if cfg["architectures"] != [self.architecture] or wrong:
            raise ValueError(f"unexpected model config: arch={cfg['architectures']} mismatches={wrong}")
        if text["rope_parameters"]["rope_theta"] != self.rope_theta:
            raise ValueError("unexpected rope_theta")
        return cfg

    def layer_types_of(self, cfg: dict) -> list[str]:
        if self.family == "nemotron":  # `check_config` matched the block pattern
            return list(self.layer_types)
        types = cfg["text_config"]["layer_types"]
        if tuple(types) != self.layer_types:
            raise ValueError("layer_types differ from the pinned target")
        return list(types)

    def check(self, require_gpu: bool = True) -> dict:
        """Validate the host and the snapshot. Returns a report dict; raises on failure."""
        import subprocess
        report: dict = {"target": self.name, "model": self.model_id, "revision": self.model_revision}
        if os.uname().machine != "aarch64":
            raise RuntimeError("host architecture must be aarch64")
        mem = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            mem[key] = int(rest.split()[0])
        report["mem_total_gib"] = round(mem["MemTotal"] / 2**20, 1)
        report["mem_available_gib"] = round(mem["MemAvailable"] / 2**20, 1)
        if mem["MemTotal"] < MIN_TOTAL_MEMORY_KIB:
            raise RuntimeError("at least 120 GB host memory is required")
        snap = self.snapshot_dir()
        self.load_model_config(snap)
        report["snapshot"] = str(snap)
        if require_gpu:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            if name != GPU_NAME or cap != COMPUTE_CAPABILITY:
                raise RuntimeError(f"GPU must be {GPU_NAME} SM{COMPUTE_CAPABILITY}, found {name} {cap}")
            report["gpu"] = name
            report["compute_capability"] = f"{cap[0]}.{cap[1]}"
            try:
                drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                     capture_output=True, text=True, timeout=10, check=True).stdout.strip()
                if int(drv.split(".")[0]) < MIN_DRIVER_MAJOR:
                    raise RuntimeError(f"driver {drv} is older than {MIN_DRIVER_MAJOR}")
                report["driver"] = drv
            except FileNotFoundError:
                report["driver"] = "unknown (nvidia-smi missing)"
            report["torch"] = torch.__version__
        return report


def _qwen38() -> Target:
    from hibana.model import qwen38_config as c
    return Target(
        name="qwen38", model_id=c.MODEL_ID, model_revision=c.MODEL_REVISION, model_cache_key=c.MODEL_CACHE_KEY,
        architecture=c.MODEL_ARCHITECTURE, hidden_size=c.HIDDEN_SIZE, intermediate_size=c.INTERMEDIATE_SIZE,
        num_layers=c.NUM_LAYERS,
        layer_types=tuple("full_attention" if (i + 1) % c.FULL_ATTENTION_INTERVAL == 0 else "linear_attention" for i in range(c.NUM_LAYERS)),
        num_attention_heads=c.NUM_ATTENTION_HEADS, num_kv_heads=c.NUM_KV_HEADS, head_dim=c.HEAD_DIM,
        partial_rotary_factor=c.PARTIAL_ROTARY_FACTOR, rope_theta=c.ROPE_THETA,
        linear_num_key_heads=c.LINEAR_NUM_KEY_HEADS, linear_num_value_heads=c.LINEAR_NUM_VALUE_HEADS,
        linear_key_head_dim=c.LINEAR_KEY_HEAD_DIM, linear_value_head_dim=c.LINEAR_VALUE_HEAD_DIM,
        linear_conv_kernel=c.LINEAR_CONV_KERNEL, vocab_size=c.VOCAB_SIZE, rms_norm_eps=c.RMS_NORM_EPS,
        max_position_embeddings=c.MAX_POSITION_EMBEDDINGS, nvfp4_group_size=c.NVFP4_GROUP_SIZE,
        weight_load_peak_bytes=c.WEIGHT_LOAD_PEAK_BYTES, w3_repo=c.W3_REPO, w3_revision=c.W3_REVISION,
        draft_pins=tuple((name, *pin) for name, pin in c.DRAFT_PINS.items()),
        draft_calibration=tuple(c.DRAFT_CALIBRATION.items()),
    )


def _flashnext() -> Target:
    from hibana.model.flashnext import config as c
    return Target(
        name="flashnext", family="flashnext", cache_env="HIBANA_FLASHNEXT_CACHE",
        model_id=c.MODEL_ID, model_revision=c.MODEL_REVISION, model_cache_key=c.MODEL_CACHE_KEY,
        architecture=c.MODEL_ARCHITECTURE, hidden_size=c.HIDDEN_SIZE, intermediate_size=None,
        num_layers=c.NUM_LAYERS, layer_types=c.LAYER_TYPES,
        num_attention_heads=c.NUM_ATTENTION_HEADS, num_kv_heads=c.NUM_KV_HEADS, head_dim=c.HEAD_DIM,
        partial_rotary_factor=c.PARTIAL_ROTARY_FACTOR, rope_theta=c.ROPE_THETA,
        linear_num_key_heads=c.LINEAR_NUM_KEY_HEADS, linear_num_value_heads=c.LINEAR_NUM_VALUE_HEADS,
        linear_key_head_dim=c.LINEAR_KEY_HEAD_DIM, linear_value_head_dim=c.LINEAR_VALUE_HEAD_DIM,
        linear_conv_kernel=c.LINEAR_CONV_KERNEL, vocab_size=c.VOCAB_SIZE, rms_norm_eps=c.RMS_NORM_EPS,
        max_position_embeddings=c.MAX_POSITION_EMBEDDINGS, nvfp4_group_size=c.NVFP4_GROUP_SIZE,
        weight_load_peak_bytes=c.WEIGHT_LOAD_PEAK_BYTES, output_gate="sigmoid",
        # Every 2K chunk streams all 512 experts' weights (47 GB over the 48 layers); 4K chunks halve that per token
        # (with the MoE's pass-per-block kernel, 16K prompts prefill ~4% faster) for 1 GiB more activations and
        # 2 s instead of 1 s that a step holds other requests' decode (the prefill budget is already 4,096 rows).
        prefill_chunk=c.PREFILL_CHUNK, solo_prefill_chunk=c.SOLO_PREFILL_CHUNK,
        num_experts=c.NUM_EXPERTS, experts_per_token=c.EXPERTS_PER_TOKEN, moe_intermediate_size=c.MOE_INTERMEDIATE_SIZE,
        shared_expert_intermediate_size=c.SHARED_EXPERT_INTERMEDIATE_SIZE,
        hc_count=c.HC_COUNT, hc_lowrank=c.HC_LOWRANK,
        ple_layer_ids=c.PLE_LAYER_IDS, ple_ngram=c.PLE_NGRAM, ple_heads_per_ngram=c.PLE_HEADS_PER_NGRAM,
        ple_embed_dim=c.PLE_EMBED_DIM, ple_conv_kernel=c.PLE_CONV_KERNEL,
        indexer_budget=c.INDEXER_BUDGET, indexer_compress_ratio=c.INDEXER_COMPRESS_RATIO,
        indexer_n_heads=c.INDEXER_N_HEADS, indexer_head_dim=c.INDEXER_HEAD_DIM, mtp_layers=c.MTP_LAYERS,
        w3_repo=c.W3_REPO, w3_revision=c.W3_REVISION, draft_pins=tuple((name, *pin) for name, pin in c.DRAFT_PINS.items()),
    )


def _nemotron() -> Target:
    from hibana.model.nemotron import config as c
    # Mamba-2 on the linear-attention fields (model/nemotron/config.py): key heads are the B/C groups, the key head
    # dim the state size, value heads the SSM heads. Attention has no positional encoding (no rotary dims).
    return Target(
        name="nemotron", family="nemotron",
        model_id=c.MODEL_ID, model_revision=c.MODEL_REVISION, model_cache_key=c.MODEL_CACHE_KEY,
        architecture=c.MODEL_ARCHITECTURE, hidden_size=c.HIDDEN_SIZE, intermediate_size=None,
        num_layers=c.NUM_LAYERS, layer_types=c.LAYER_TYPES,
        num_attention_heads=c.NUM_ATTENTION_HEADS, num_kv_heads=c.NUM_KV_HEADS, head_dim=c.HEAD_DIM,
        partial_rotary_factor=0.0, rope_theta=0.0,
        linear_num_key_heads=c.MAMBA_GROUPS, linear_num_value_heads=c.MAMBA_HEADS,
        linear_key_head_dim=c.SSM_STATE, linear_value_head_dim=c.MAMBA_HEAD_DIM,
        linear_conv_kernel=c.CONV_KERNEL, vocab_size=c.VOCAB_SIZE, rms_norm_eps=c.RMS_NORM_EPS,
        max_position_embeddings=c.MAX_POSITION_EMBEDDINGS, nvfp4_group_size=c.NVFP4_GROUP_SIZE,
        weight_load_peak_bytes=c.WEIGHT_LOAD_PEAK_BYTES,
        prefill_chunk=c.PREFILL_CHUNK, solo_prefill_chunk=c.SOLO_PREFILL_CHUNK,
        num_experts=c.NUM_EXPERTS, experts_per_token=c.EXPERTS_PER_TOKEN, moe_intermediate_size=c.MOE_INTERMEDIATE_SIZE,
        shared_expert_intermediate_size=c.SHARED_EXPERT_INTERMEDIATE_SIZE, mtp_layers=c.MTP_LAYERS,
        stop_token_ids=c.STOP_TOKEN_IDS, w3_repo=c.W3_REPO, w3_revision=c.W3_REVISION,
        recurrence="mamba2", mtp_prefix=f"{c.PREFIX}.mtp.", first_text_token=c.FIRST_TEXT_TOKEN,
        moe_latent_size=c.MOE_LATENT_SIZE, routed_scaling_factor=c.ROUTED_SCALING_FACTOR,
        kv_formats=("fp8",),
        # The template drops earlier turns' reasoning by default, so each new user message changed tokens the prefix
        # cache already held: an agent's next task re-prefilled the whole previous one. Kept, a conversation renders
        # append-only, as the Qwen3.8 templates do by default (their preserve_thinking).
        template_defaults=(("truncate_history_thinking", False),),
    )


QWEN38 = _qwen38()
FLASHNEXT = _flashnext()
NEMOTRON = _nemotron()


def _bolt() -> Target:
    from hibana.model.bolt import config as c
    # Same recurrence and attention as Super; Puzzle prunes only the MoE geometry.
    return replace(NEMOTRON, name="bolt", model_id=c.MODEL_ID, model_revision=c.MODEL_REVISION,
                   model_cache_key=c.MODEL_CACHE_KEY, layer_types=c.LAYER_TYPES,
                   max_position_embeddings=c.BOLT.context, num_experts=0, experts_per_token=9,
                   moe_intermediate_size=0, mtp_layers=0, w3_repo=None, w3_revision=None,
                   prefill_chunk=c.PREFILL_CHUNK, solo_prefill_chunk=0,
                   weight_load_peak_bytes=c.WEIGHT_LOAD_PEAK_BYTES,
                   draft_pins=tuple((name, *pin) for name, pin in c.DRAFT_PINS.items()),
                   moe_block_specs=tuple((i, e, width, 9) for i, e, width in c.MOE_SPECS),
                   kv_formats=("fp8", "sim-bf16"))  # sim-bf16: the parity tests' unquantized cache


NEMOTRON_BOLT = _bolt()
TARGETS: dict[str, Target] = {t.name: t for t in (QWEN38, FLASHNEXT, NEMOTRON, NEMOTRON_BOLT)}
DEFAULT_TARGET = QWEN38


def get(name: str | None) -> Target:
    if name is None:
        return DEFAULT_TARGET
    try:
        return TARGETS[name]
    except KeyError:
        raise ValueError(f"unknown target {name!r}; choose from {sorted(TARGETS)}") from None


def load_model(target: Target, snap: Path, **kwargs):
    """Construct the target's model object; the executor reads its `.target`."""
    if target.name == "bolt":
        from hibana.model.bolt.model import Bolt
        return Bolt(snap, target=target, **kwargs)
    if target.family == "qwen38":
        from hibana.model.qwen38 import Qwen38
        return Qwen38(snap, target=target, **kwargs)
    if target.family == "flashnext":
        from hibana.model.flashnext.model import FlashNext
        return FlashNext(snap, target=target, **kwargs)
    if target.family == "nemotron":
        from hibana.model.nemotron.model import Nemotron
        return Nemotron(snap, target=target, **kwargs)
    raise ValueError(f"no loader for target {target.name!r}")
