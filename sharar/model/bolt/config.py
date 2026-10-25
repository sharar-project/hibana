"""The heterogeneous Bolt checkpoint, independent of Qwen's GDN target geometry.

Source: NVIDIA's config.json at MODEL_REVISION. In particular, the global
num_experts=512 is superseded by each MoE block's n_routed_experts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB"
MODEL_REVISION = "cf2effc5c4b794ee9fb8d34eab3d1d65ce64142f"
MODEL_CACHE_KEY = "models--nvidia--NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB"
MODEL_ARCHITECTURE = "NemotronH_Omni_Reasoning_V3"
STOP_TOKEN_IDS = (11, 2)
W3_REPO = W3_REVISION = None
PREFILL_CHUNK = 2048
# Packed tensors plus a staged MoE block, tokenizer/runtime and load temporaries.
WEIGHT_LOAD_PEAK_BYTES = 28 * 2**30

# (residual-block index, routed expert count, expert intermediate width).
MOE_SPECS = (
    (1, 240, 2048), (3, 240, 1536), (5, 240, 2048), (8, 304, 2304),
    (10, 240, 2304), (12, 240, 2048), (14, 240, 2048), (17, 304, 2304),
    (19, 304, 2048), (21, 432, 2560), (23, 432, 2304), (26, 240, 2048),
    (28, 432, 2304), (30, 304, 2560), (32, 304, 2304), (34, 432, 2304),
    (37, 368, 2304), (39, 368, 2048), (41, 368, 2304), (43, 432, 2560),
    (45, 368, 2560), (48, 496, 2560), (50, 432, 2560), (52, 240, 2304),
    (54, 368, 2304), (56, 368, 2560), (59, 304, 2048), (61, 240, 2304),
    (63, 240, 2304), (65, 240, 2048), (67, 240, 2048), (70, 368, 2048),
    (72, 368, 2304), (74, 432, 2304), (76, 432, 2560), (79, 496, 2560),
    (81, 432, 2560), (83, 432, 2560), (85, 496, 2560), (87, 368, 2560),
)
ATTENTION_IDS = (7, 16, 25, 36, 47, 58, 69, 78)


@dataclass(frozen=True)
class Block:
    kind: str
    experts: int = 0
    intermediate: int = 0
    topk: int = 0


@dataclass(frozen=True)
class BoltConfig:
    blocks: tuple[Block, ...]
    hidden: int = 4096
    latent: int = 1024
    shared_intermediate: int = 5376
    attention_heads: int = 32
    kv_heads: int = 2
    attention_head_dim: int = 128
    mamba_heads: int = 128
    mamba_head_dim: int = 64
    mamba_groups: int = 8
    mamba_state_dim: int = 128
    conv_kernel: int = 4
    vocab: int = 131072
    context: int = 262144
    norm_eps: float = 1e-5
    route_scale: float = 5.0

    @property
    def mamba_inner(self) -> int:
        return self.mamba_heads * self.mamba_head_dim

    @property
    def conv_dim(self) -> int:
        return self.mamba_inner + 2 * self.mamba_groups * self.mamba_state_dim

    @property
    def mamba_ids(self) -> tuple[int, ...]:
        return tuple(i for i, b in enumerate(self.blocks) if b.kind == "mamba")

    def state_bytes_per_slot(self, ssm_element_bytes: int = 4) -> int:
        """FP32 SSM by default; convolution history is BF16, excluding the current input."""
        ssm = self.mamba_inner * self.mamba_state_dim * ssm_element_bytes
        conv = self.conv_dim * (self.conv_kernel - 1) * 2
        return len(self.mamba_ids) * (ssm + conv)

    def kv_bytes_per_token(self, element_bytes: int = 2) -> int:
        return sum(b.kind == "attention" for b in self.blocks) * 2 * self.kv_heads * self.attention_head_dim * element_bytes


def _pinned() -> BoltConfig:
    blocks = [Block("mamba") for _ in range(88)]
    for i in ATTENTION_IDS:
        blocks[i] = Block("attention")
    for i, experts, intermediate in MOE_SPECS:
        blocks[i] = Block("moe", experts, intermediate, 9)
    return BoltConfig(tuple(blocks))


BOLT = _pinned()
LAYER_TYPES = tuple({"mamba": "linear_attention", "attention": "full_attention", "moe": "moe"}[b.kind] for b in BOLT.blocks)


def load_config(path: str | Path) -> BoltConfig:
    """Validate the pinned architecture from a config file or snapshot directory.

    This checks geometry and equations, not checkpoint provenance or weight
    completeness. It imports no remote code.
    """
    path = Path(path)
    raw = json.loads((path / "config.json" if path.is_dir() else path).read_text())
    if raw.get("model_type") != "nemotron_h_omni" or raw.get("architectures") != ["NemotronH_Omni_Reasoning_V3"]:
        raise ValueError("expected the Nemotron Bolt omni architecture")
    text = raw.get("llm_config", {})
    expected = {
        "model_type": "nemotron_h_puzzle", "architectures": ["NemotronHPuzzleForCausalLM"],
        "hidden_size": BOLT.hidden, "moe_latent_size": BOLT.latent,
        "moe_shared_expert_intermediate_size": BOLT.shared_intermediate,
        "num_attention_heads": BOLT.attention_heads, "num_key_value_heads": BOLT.kv_heads,
        "head_dim": BOLT.attention_head_dim, "mamba_num_heads": BOLT.mamba_heads,
        "mamba_head_dim": BOLT.mamba_head_dim, "n_groups": BOLT.mamba_groups,
        "ssm_state_size": BOLT.mamba_state_dim, "conv_kernel": BOLT.conv_kernel,
        "vocab_size": BOLT.vocab, "max_position_embeddings": BOLT.context,
        "layer_norm_epsilon": BOLT.norm_eps, "routed_scaling_factor": BOLT.route_scale,
        "mlp_hidden_act": "relu2", "mamba_hidden_act": "silu",
        "scoring_func": "sigmoid", "norm_topk_prob": True, "n_group": 1, "topk_group": 1,
        "n_shared_experts": 1, "use_qk_norm": False, "attention_bias": False,
        "mlp_bias": False, "use_bias": False, "use_conv_bias": True,
        "residual_in_fp32": False, "tie_word_embeddings": False,
        "num_nextn_predict_layers": 0, "mtp_block_configs": [],
        "chunk_size": 128, "mamba_proj_bias": False, "sliding_window": None,
        "time_step_limit": [0.0, {"__float__": "Infinity"}],
    }
    wrong = {k: (text.get(k), v) for k, v in expected.items() if text.get(k) != v}
    if wrong:
        raise ValueError(f"unexpected Bolt config: {wrong}")
    blocks = text.get("block_configs", [])
    if len(blocks) != len(BOLT.blocks):
        raise ValueError("expected 88 Bolt residual blocks")
    for i, (got, want) in enumerate(zip(blocks, BOLT.blocks)):
        spec = {"block_type": want.kind}
        if want.kind == "moe":
            spec.update(n_routed_experts=want.experts, moe_intermediate_size=want.intermediate, num_experts_per_tok=want.topk)
        if got != spec:
            raise ValueError(f"Bolt block {i}: expected {spec}, got {got}")
    if text.get("layers_block_type") != [b.kind for b in BOLT.blocks]:
        raise ValueError("layers_block_type disagrees with Bolt block_configs")
    return BOLT


def check_config(target, snap: Path) -> dict:
    load_config(snap)
    if tuple(target.layer_types) != LAYER_TYPES:
        raise ValueError("Bolt target block ordering differs from the pinned checkpoint")
    if target.moe_block_specs != tuple((i, e, width, 9) for i, e, width in MOE_SPECS):
        raise ValueError("Bolt target expert geometry differs from the pinned Puzzle blocks")
    return json.loads((snap / "config.json").read_text())
