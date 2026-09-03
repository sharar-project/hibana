"""Qwen3.8-27B hybrid model assembled from the layer modules.

`prefill()` runs one full sequence with fresh state and can capture per-layer
outputs in the reference's layout: each decoder layer yields
(hidden_states, residual) where residual is the stream before this layer's MLP
output is added; the next layer's input norm performs that addition.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from sharar import config
from sharar.kernels import ops
from sharar.model.attention import Attention
from sharar.model.gdn import GatedDeltaNet
from sharar.model.mlp import Mlp
from sharar.model.norm import GemmaRMSNorm
from sharar.model.rope import PartialRotary
from sharar.weights.modelopt import load_nvfp4_linear
from sharar.weights.safetensors import SnapshotWeights

PREFIX = "model.language_model"


class DecoderLayer:
    def __init__(self, w: SnapshotWeights, index: int, kind: str, device: torch.device, rope: PartialRotary, ragged: ops.RaggedAttention, fused_silu_quant: bool):
        p = f"{PREFIX}.layers.{index}"
        self.index = index
        self.kind = kind
        self.input_norm = GemmaRMSNorm(w.get(f"{p}.input_layernorm.weight").to(device), config.RMS_NORM_EPS)
        self.post_norm = GemmaRMSNorm(w.get(f"{p}.post_attention_layernorm.weight").to(device), config.RMS_NORM_EPS)
        if kind == "full_attention":
            self.mixer = Attention(w, f"{p}.self_attn", device, rope, ragged)
        else:
            self.mixer = GatedDeltaNet(w, f"{p}.linear_attn", device)
        self.mlp = Mlp(w, f"{p}.mlp", device, fused_silu_quant=fused_silu_quant)

    def prefill(self, hidden: torch.Tensor, residual: torch.Tensor | None, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden
            hidden = self.input_norm(hidden)
        else:
            hidden, residual = self.input_norm.fused_add(hidden, residual)
        if self.kind == "full_attention":
            hidden = self.mixer.prefill(hidden, positions)
        else:
            hidden = self.mixer.prefill(hidden)
        hidden, residual = self.post_norm.fused_add(hidden, residual)
        hidden = self.mlp(hidden)
        return hidden, residual


class Qwen38:
    def __init__(self, snap: Path, device: torch.device | str = "cuda", fused_silu_quant: bool = True, log=print):
        self.device = torch.device(device)
        cfg = config.load_model_config(snap)
        types = config.layer_types(cfg)
        started = time.perf_counter()
        w = SnapshotWeights(snap)
        self.embed = w.get(f"{PREFIX}.embed_tokens.weight").to(self.device)
        self.rope = PartialRotary(
            config.HEAD_DIM,
            int(config.HEAD_DIM * config.PARTIAL_ROTARY_FACTOR),
            config.ROPE_THETA,
            config.MAX_POSITION_EMBEDDINGS,
            self.device,
        )
        self.ragged = ops.RaggedAttention(self.device, config.NUM_ATTENTION_HEADS, config.NUM_KV_HEADS, config.HEAD_DIM)
        self.layers: list[DecoderLayer] = []
        for i, kind in enumerate(types):
            self.layers.append(DecoderLayer(w, i, kind, self.device, self.rope, self.ragged, fused_silu_quant))
            if log and (i + 1) % 8 == 0:
                log(f"  loaded {i + 1}/{len(types)} layers ({time.perf_counter() - started:.0f}s)")
        self.final_norm = GemmaRMSNorm(w.get(f"{PREFIX}.norm.weight").to(self.device), config.RMS_NORM_EPS)
        self.lm_head = load_nvfp4_linear(w, ["lm_head"], self.device)
        w.close()
        torch.cuda.synchronize()
        if log:
            log(f"  weights loaded in {time.perf_counter() - started:.0f}s, "
                f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated")

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor, capture: bool = False) -> dict:
        """Run one sequence. Returns final normalized hidden states and, when
        capture is set, embed/per-layer/norm tensors for parity."""
        t = input_ids.shape[0]
        positions = torch.arange(t, dtype=torch.int64, device=self.device)
        hidden = torch.nn.functional.embedding(input_ids, self.embed)
        out: dict = {"input_ids": input_ids, "positions": positions}
        if capture:
            out["embed"] = hidden
            out["layers"] = []
        residual = None
        for layer in self.layers:
            hidden, residual = layer.prefill(hidden, residual, positions)
            if capture:
                out["layers"].append((hidden, residual))
        normed, _ = self.final_norm.fused_add(hidden, residual)
        out["norm"] = normed
        return out

    @torch.inference_mode()
    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return ops.nvfp4_linear(hidden, self.lm_head)

    @torch.inference_mode()
    def argmax_and_topk(self, hidden: torch.Tensor, k: int = 8, chunk: int = 1024) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-position argmax, top-k ids and top-k logits without holding all logits."""
        arg, top_ids, top_vals = [], [], []
        for start in range(0, hidden.shape[0], chunk):
            lg = self.logits(hidden[start : start + chunk]).float()
            vals, ids = lg.topk(k, dim=-1)
            arg.append(lg.argmax(dim=-1))
            top_ids.append(ids)
            top_vals.append(vals)
        return torch.cat(arg), torch.cat(top_ids), torch.cat(top_vals)
