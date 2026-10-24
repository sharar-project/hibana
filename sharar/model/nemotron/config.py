# SPDX-License-Identifier: Apache-2.0
"""Pinned NVIDIA-Nemotron-3.5-Super checkpoint and geometry (NVIDIA's ModelOpt NVFP4 export; the text model and its
MTP head are served, the C-RADIO vision tower and its projector are never read).

The text model is 88 blocks, each one mixer behind a pre-norm and a residual: 40 Mamba-2, 40 latent MoE and 8
attention. `sharar.target` names every block by its checkpoint index, so pools and features use the checkpoint's
numbering: the Mamba blocks are the target's recurrent ("linear_attention") layers, the attention blocks its KV layers,
and the MoE blocks hold no per-request state.

The Mamba-2 geometry rides on the target's linear-attention fields, which size the state pool: key heads are the
8 B/C groups and the key head dim is the state size 128, value heads are the 128 SSM heads of dim 64, so the pool's
[HV, V, K] FP32 state is Mamba's [heads, head dim, state] and its conv width is x + B + C (8192 + 2 x 1024).
"""

from __future__ import annotations

import json
from pathlib import Path

MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"
MODEL_REVISION = "6aae0d6d0285e561bbb72efa10ba1425d41ebddc"
MODEL_CACHE_KEY = "models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"
MODEL_ARCHITECTURE = "NemotronH_Omni_Reasoning_V3"
PREFIX = "language_model"

STOP_TOKEN_IDS = (11, 2)  # <|im_end|>, </s> (generation_config.json)

HIDDEN_SIZE = 4096
VOCAB_SIZE = 131_072
RMS_NORM_EPS = 1e-5
MAX_POSITION_EMBEDDINGS = 1_048_576
# `layers_block_type` in Nemotron-H's pattern notation (M Mamba-2, E MoE, * attention).
PATTERN = "MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME"
BLOCK_TYPES = tuple({"M": "mamba", "E": "moe", "*": "attention"}[c] for c in PATTERN)
NUM_LAYERS = len(BLOCK_TYPES)
LAYER_TYPES = tuple({"mamba": "linear_attention", "attention": "full_attention", "moe": "moe"}[t] for t in BLOCK_TYPES)

NUM_ATTENTION_HEADS = 32
NUM_KV_HEADS = 2
HEAD_DIM = 128

# Mamba-2
MAMBA_HEADS = 128
MAMBA_HEAD_DIM = 64
MAMBA_GROUPS = 8
SSM_STATE = 128
CONV_KERNEL = 4
MAMBA_CHUNK = 128
MAMBA_INNER = MAMBA_HEADS * MAMBA_HEAD_DIM  # 8192

# Latent MoE
NUM_EXPERTS = 512
EXPERTS_PER_TOKEN = 22
MOE_INTERMEDIATE_SIZE = 2688
MOE_LATENT_SIZE = 1024
SHARED_EXPERT_INTERMEDIATE_SIZE = 5376
ROUTED_SCALING_FACTOR = 5.0
NVFP4_GROUP_SIZE = 16

MTP_LAYERS = 1  # num_nextn_predict_layers: one MTP step, stored as an attention block and an MoE block

# Resident weights: NVFP4 experts ~63 GiB (40 trunk layers), FP8 Mamba/attention/head ~5 GiB, BF16 embeddings, router
# and latent projections ~2 GiB, the MTP head ~1.6 GiB; plus one layer's expert shards staged during the load.
WEIGHT_LOAD_PEAK_BYTES = 76 * 2**30
PREFILL_CHUNK = 4096
SOLO_PREFILL_CHUNK = 8192

# config.json `llm_config` -> the value this target was built for.
EXPECTED = {
    "model_type": "nemotron_h", "hidden_size": HIDDEN_SIZE, "vocab_size": VOCAB_SIZE, "layer_norm_epsilon": RMS_NORM_EPS,
    "max_position_embeddings": MAX_POSITION_EMBEDDINGS, "num_attention_heads": NUM_ATTENTION_HEADS,
    "num_key_value_heads": NUM_KV_HEADS, "head_dim": HEAD_DIM, "mamba_num_heads": MAMBA_HEADS,
    "mamba_head_dim": MAMBA_HEAD_DIM, "n_groups": MAMBA_GROUPS, "ssm_state_size": SSM_STATE, "conv_kernel": CONV_KERNEL,
    "chunk_size": MAMBA_CHUNK, "expand": 2, "n_routed_experts": NUM_EXPERTS, "num_experts_per_tok": EXPERTS_PER_TOKEN,
    "moe_intermediate_size": MOE_INTERMEDIATE_SIZE, "moe_latent_size": MOE_LATENT_SIZE,
    "moe_shared_expert_intermediate_size": SHARED_EXPERT_INTERMEDIATE_SIZE, "n_shared_experts": 1,
    "routed_scaling_factor": ROUTED_SCALING_FACTOR, "n_group": 1, "topk_group": 1, "norm_topk_prob": True,
    "mlp_hidden_act": "relu2", "num_nextn_predict_layers": MTP_LAYERS, "tie_word_embeddings": False,
    "mamba_ssm_cache_dtype": "float32",
}


def check_config(target, snap: Path) -> dict:
    """config.json of the snapshot, after checking its text model against the pinned geometry; raises on a mismatch."""
    cfg = json.loads((snap / "config.json").read_text())
    text = cfg["llm_config"]
    wrong = {k: (text.get(k), v) for k, v in EXPECTED.items() if text.get(k) != v}
    if tuple(text.get("layers_block_type", ())) != BLOCK_TYPES:
        wrong["layers_block_type"] = (text.get("layers_block_type"), list(BLOCK_TYPES))
    if cfg["architectures"] != [target.architecture] or wrong:
        raise ValueError(f"unexpected model config: arch={cfg['architectures']} mismatches={wrong}")
    return cfg
