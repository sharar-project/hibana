"""Pinned serving targets: the facts the engine needs about a model, as a value, not module globals.

A `Target` names one validated checkpoint revision and carries the geometry
the engine, pools, graphs and kernels size themselves from (hidden size,
layer kinds, attention and recurrent head shapes, vocabulary, context), the
serving profile, the load budget and the snapshot lookup. One process serves
one target, chosen at launch; nothing here is discovered at run time. The
27B is the first instance; `sharar.config` keeps exporting its constants for
tools and tests that predate this module.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from sharar.engine.profile import DEFAULT_PROFILE, ServingProfile
from sharar.hardware.gb10 import COMPUTE_CAPABILITY, GPU_NAME, MIN_DRIVER_MAJOR, MIN_TOTAL_MEMORY_KIB


@dataclass(frozen=True)
class Target:
    name: str
    model_id: str
    model_revision: str
    model_cache_key: str
    architecture: str
    hidden_size: int
    intermediate_size: int
    num_layers: int
    layer_types: tuple[str, ...]  # "full_attention" | "linear_attention", one per layer
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
    kv_bytes_per_element: int = 1  # FP8 KV cache

    def __post_init__(self):
        if len(self.layer_types) != self.num_layers or any(t not in ("full_attention", "linear_attention") for t in self.layer_types):
            raise ValueError("layer_types must name every layer as full_attention or linear_attention")

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
        return 2 * self.num_kv_heads * self.head_dim * self.kv_bytes_per_element * len(self.attention_layer_ids)

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
    def snapshot_dir(self, cache_dir: str | os.PathLike | None = None) -> Path:
        """Return the pinned snapshot directory or raise if it is incomplete."""
        root = Path(cache_dir or os.environ.get("SHARAR_HF_CACHE") or os.environ.get("HF_HOME") or "")
        if not str(root):
            raise FileNotFoundError("set SHARAR_HF_CACHE (or HF_HOME) to the model cache")
        for base in (root, root / "hub"):
            snap = base / self.model_cache_key / "snapshots" / self.model_revision
            if (snap / "config.json").is_file() and (snap / "model.safetensors.index.json").is_file():
                return snap
        raise FileNotFoundError(f"pinned snapshot {self.model_revision} not found under {root}")

    def load_model_config(self, snap: Path) -> dict:
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
        wrong = {k: (text.get(k), v) for k, v in expected.items() if text.get(k) != v}
        if cfg["architectures"] != [self.architecture] or wrong:
            raise ValueError(f"unexpected model config: arch={cfg['architectures']} mismatches={wrong}")
        if text["rope_parameters"]["rope_theta"] != self.rope_theta:
            raise ValueError("unexpected rope_theta")
        return cfg

    def layer_types_of(self, cfg: dict) -> list[str]:
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
    from sharar.model import qwen38_config as c
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
        weight_load_peak_bytes=c.WEIGHT_LOAD_PEAK_BYTES,
    )


QWEN38 = _qwen38()
TARGETS: dict[str, Target] = {QWEN38.name: QWEN38}
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
    if target.name == "qwen38":
        from sharar.model.qwen38 import Qwen38
        return Qwen38(snap, target=target, **kwargs)
    raise ValueError(f"no loader for target {target.name!r}")
