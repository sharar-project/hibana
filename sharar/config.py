"""Pinned facts about the one supported target: model, hardware and profile.

Nothing here is user-selectable. `check()` validates the host against these
facts and is the preflight for every command that touches the GPU or the model.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from sharar.model.qwen38_config import (
    MODEL_ID,
    MODEL_REVISION,
    MODEL_CACHE_KEY,
    MODEL_ARCHITECTURE,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    NUM_LAYERS,
    FULL_ATTENTION_INTERVAL,
    NUM_ATTENTION_HEADS,
    NUM_KV_HEADS,
    HEAD_DIM,
    PARTIAL_ROTARY_FACTOR,
    ROPE_THETA,
    LINEAR_NUM_KEY_HEADS,
    LINEAR_NUM_VALUE_HEADS,
    LINEAR_KEY_HEAD_DIM,
    LINEAR_VALUE_HEAD_DIM,
    LINEAR_CONV_KERNEL,
    VOCAB_SIZE,
    RMS_NORM_EPS,
    MAX_POSITION_EMBEDDINGS,
    NVFP4_GROUP_SIZE,
    WEIGHT_LOAD_PEAK_BYTES,
)
from sharar.hardware.gb10 import (
    GPU_NAME,
    COMPUTE_CAPABILITY,
    MIN_DRIVER_MAJOR,
    MIN_TOTAL_MEMORY_KIB,
    HOST_FLOOR_KIB,
    UNIFIED_PHYSICAL_MEMORY,
)
from sharar.engine.profile import DEFAULT_PROFILE

CONTEXT_LENGTH = DEFAULT_PROFILE.max_context


def snapshot_dir(cache_dir: str | os.PathLike | None = None) -> Path:
    """Return the pinned snapshot directory or raise if it is incomplete."""
    root = Path(cache_dir or os.environ.get("SHARAR_HF_CACHE") or os.environ.get("HF_HOME") or "")
    if not str(root):
        raise FileNotFoundError("set SHARAR_HF_CACHE (or HF_HOME) to the model cache")
    candidates = [root, root / "hub"]
    for base in candidates:
        snap = base / MODEL_CACHE_KEY / "snapshots" / MODEL_REVISION
        if (snap / "config.json").is_file() and (snap / "model.safetensors.index.json").is_file():
            return snap
    raise FileNotFoundError(f"pinned snapshot {MODEL_REVISION} not found under {root}")


def load_model_config(snap: Path) -> dict:
    cfg = json.loads((snap / "config.json").read_text())
    text = cfg["text_config"]
    expected = {
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "num_hidden_layers": NUM_LAYERS,
        "full_attention_interval": FULL_ATTENTION_INTERVAL,
        "num_attention_heads": NUM_ATTENTION_HEADS,
        "num_key_value_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "partial_rotary_factor": PARTIAL_ROTARY_FACTOR,
        "linear_num_key_heads": LINEAR_NUM_KEY_HEADS,
        "linear_num_value_heads": LINEAR_NUM_VALUE_HEADS,
        "linear_key_head_dim": LINEAR_KEY_HEAD_DIM,
        "linear_value_head_dim": LINEAR_VALUE_HEAD_DIM,
        "linear_conv_kernel_dim": LINEAR_CONV_KERNEL,
        "vocab_size": VOCAB_SIZE,
        "rms_norm_eps": RMS_NORM_EPS,
        "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
    }
    wrong = {k: (text.get(k), v) for k, v in expected.items() if text.get(k) != v}
    if cfg["architectures"] != [MODEL_ARCHITECTURE] or wrong:
        raise ValueError(f"unexpected model config: arch={cfg['architectures']} mismatches={wrong}")
    if text["rope_parameters"]["rope_theta"] != ROPE_THETA:
        raise ValueError("unexpected rope_theta")
    return cfg


def layer_types(cfg: dict) -> list[str]:
    types = cfg["text_config"]["layer_types"]
    assert len(types) == NUM_LAYERS
    for i, t in enumerate(types):
        want = "full_attention" if (i + 1) % FULL_ATTENTION_INTERVAL == 0 else "linear_attention"
        assert t == want, (i, t)
    return types


def _meminfo() -> dict[str, int]:
    out = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = line.partition(":")
        out[key] = int(rest.split()[0])
    return out


def check(require_gpu: bool = True) -> dict:
    """Validate the host and the snapshot. Returns a report dict; raises on failure."""
    report: dict = {"model": MODEL_ID, "revision": MODEL_REVISION}
    if os.uname().machine != "aarch64":
        raise RuntimeError("host architecture must be aarch64")
    mem = _meminfo()
    report["mem_total_gib"] = round(mem["MemTotal"] / 2**20, 1)
    report["mem_available_gib"] = round(mem["MemAvailable"] / 2**20, 1)
    if mem["MemTotal"] < MIN_TOTAL_MEMORY_KIB:
        raise RuntimeError("at least 120 GB host memory is required")
    snap = snapshot_dir()
    load_model_config(snap)
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
            drv = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout.strip()
            if int(drv.split(".")[0]) < MIN_DRIVER_MAJOR:
                raise RuntimeError(f"driver {drv} is older than {MIN_DRIVER_MAJOR}")
            report["driver"] = drv
        except FileNotFoundError:
            report["driver"] = "unknown (nvidia-smi missing)"
        report["torch"] = torch.__version__
    return report
