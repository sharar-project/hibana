"""Compatibility facade over the default target (`sharar.target.QWEN38`).

The engine reads geometry from `executor.target`; these module constants and
functions remain for kernels, tools, benchmarks and tests written against the
27B, and always describe that target. `check()` is the preflight for every
command that touches the GPU or the model.
"""

from __future__ import annotations

from sharar.model.qwen38_config import (  # noqa: F401  re-exported constants of the default target
    MODEL_ID, MODEL_REVISION, MODEL_CACHE_KEY, MODEL_ARCHITECTURE, HIDDEN_SIZE, INTERMEDIATE_SIZE, NUM_LAYERS,
    FULL_ATTENTION_INTERVAL, NUM_ATTENTION_HEADS, NUM_KV_HEADS, HEAD_DIM, PARTIAL_ROTARY_FACTOR, ROPE_THETA,
    LINEAR_NUM_KEY_HEADS, LINEAR_NUM_VALUE_HEADS, LINEAR_KEY_HEAD_DIM, LINEAR_VALUE_HEAD_DIM, LINEAR_CONV_KERNEL,
    VOCAB_SIZE, RMS_NORM_EPS, MAX_POSITION_EMBEDDINGS, NVFP4_GROUP_SIZE, WEIGHT_LOAD_PEAK_BYTES,
)
from sharar.hardware.gb10 import (  # noqa: F401
    GPU_NAME, COMPUTE_CAPABILITY, MIN_DRIVER_MAJOR, MIN_TOTAL_MEMORY_KIB, HOST_FLOOR_KIB, UNIFIED_PHYSICAL_MEMORY,
)
from sharar.target import DEFAULT_TARGET, QWEN38  # noqa: F401

CONTEXT_LENGTH = DEFAULT_TARGET.context_length


def snapshot_dir(cache_dir=None):
    return DEFAULT_TARGET.snapshot_dir(cache_dir)


def load_model_config(snap):
    return DEFAULT_TARGET.load_model_config(snap)


def layer_types(cfg):
    return DEFAULT_TARGET.layer_types_of(cfg)


def check(require_gpu: bool = True) -> dict:
    return DEFAULT_TARGET.check(require_gpu)
