"""Qwen3.8-27B hybrid model assembled from the layer modules.

`forward()` runs a prefill (one fresh sequence) or a decode step (a batch of
one new token each) according to the ForwardContext. Per-layer capture keeps
the reference's layout: each decoder layer yields (hidden_states, residual)
where residual is the stream before this layer's MLP output is added.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from sharar import config
from sharar.engine.context import ForwardContext, scratch_prefill_context
from sharar.model.gdn_plan import prepare_gdn_prefill
from sharar.engine.contracts import ForwardResult
from sharar.kernels import ops
from sharar.model.attention import Attention
from sharar.model.gdn import GatedDeltaNet
from sharar.model.mlp import Mlp
from sharar.model.norm import GemmaRMSNorm
from sharar.model.rope import PartialRotary
from sharar.weights.modelopt import load_nvfp4_linear
from sharar.weights.safetensors import SnapshotWeights

PREFIX = "model.language_model"
# Retained compatibility switch for prequantized MLP inputs. Both routes now
# use the phase-specific norm and the same BF16-to-NVFP4 boundary.
FUSED_NORM_NVFP4 = __import__("os").environ.get("SHARAR_FUSED_NORM_NVFP4", "0") == "1"


class DecoderLayer:
    def __init__(self, w: SnapshotWeights, index: int, kind: str, device: torch.device, rope: PartialRotary, ragged: ops.RaggedAttention, fused_silu_quant: bool):
        p = f"{PREFIX}.layers.{index}"
        self.index = index
        self.kind = kind
        self.input_norm = GemmaRMSNorm(w.get(f"{p}.input_layernorm.weight").to(device), config.RMS_NORM_EPS)
        self.post_norm = GemmaRMSNorm(w.get(f"{p}.post_attention_layernorm.weight").to(device), config.RMS_NORM_EPS)
        if kind == "full_attention":
            self.mixer = Attention(w, f"{p}.self_attn", index, device, rope, ragged)
        else:
            self.mixer = GatedDeltaNet(w, f"{p}.linear_attn", index, device)
        self.mlp = Mlp(w, f"{p}.mlp", device, fused_silu_quant=fused_silu_quant)

    def forward(self, hidden: torch.Tensor, residual: torch.Tensor | None, ctx: ForwardContext) -> tuple[torch.Tensor, torch.Tensor]:
        inplace = not ctx.capture
        # Input norm also emits the FP8 input of the mixer's first projection.
        hidden, hidden_q, residual = self.input_norm.fused_add_quant(hidden, residual, self.mixer.input_scale, inplace=inplace, verify=ctx.mode == "verify")
        hidden = self.mixer.forward(hidden, ctx, hidden_q)
        if FUSED_NORM_NVFP4:
            # Preserve the same target arithmetic with prequantized MLP inputs.
            pre, residual = self.post_norm.fused_add_nvfp4(hidden, residual, self.mlp.gate_up.input_scale_inv, inplace=inplace, verify=ctx.mode == "verify")
            hidden = self.mlp.forward_prequantized(pre)
        else:
            hidden, residual = self.post_norm.fused_add(hidden, residual, inplace=inplace, verify=ctx.mode == "verify")
            hidden = self.mlp(hidden)
        return hidden, residual

    def prefill(self, hidden: torch.Tensor, residual: torch.Tensor | None, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Stand-alone prefill without pools (parity tools)."""
        return self.forward(hidden, residual, scratch_prefill_context(hidden.shape[0], hidden.device))


class Qwen38:
    def __init__(self, snap: Path, device: torch.device | str = "cuda", fused_silu_quant: bool = True, log=print):
        from sharar.memory.ledger import check_load_headroom
        check_load_headroom(config.WEIGHT_LOAD_PEAK_BYTES)
        self.device = torch.device(device)
        cfg = config.load_model_config(snap)
        types = config.layer_types(cfg)
        started = time.perf_counter()
        w = SnapshotWeights(snap)
        self.embed = w.get(f"{PREFIX}.embed_tokens.weight").to(self.device)
        w.close()
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
            # GB10 counts source mappings and CUDA allocations against the
            # same physical pool. No layer retains CPU checkpoint tensors.
            # Reopen a shard on demand rather than holding every source map
            # until the entire model is materialized.
            w.close()
            if log and (i + 1) % 8 == 0:
                log(f"  loaded {i + 1}/{len(types)} layers ({time.perf_counter() - started:.0f}s)")
        self.final_norm = GemmaRMSNorm(w.get(f"{PREFIX}.norm.weight").to(self.device), config.RMS_NORM_EPS)
        self.lm_head = load_nvfp4_linear(w, ["lm_head"], self.device)
        w.close()
        torch.cuda.synchronize()
        if log:
            log(f"  weights loaded in {time.perf_counter() - started:.0f}s, "
                f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated")
        from sharar.kernels import w3
        if w3.ENABLED:
            w3.attach_model(self, log=log)

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, ctx: ForwardContext, capture: bool = False) -> ForwardResult:
        """Run one prefill sequence or one decode batch. Returns the final
        normalized hidden states and, when capture is set, per-layer tensors."""
        if ctx.mode == "prefill":
            prepare_gdn_prefill(ctx, len(input_ids), input_ids.device)
        hidden = torch.nn.functional.embedding(input_ids, self.embed)
        ctx.capture = capture
        out: dict = {"input_ids": input_ids, "positions": ctx.positions}
        if capture:
            out["embed"] = hidden
            out["layers"] = []
        residual = None
        for layer in self.layers:
            hidden, residual = layer.forward(hidden, residual, ctx)
            if layer.index in ctx.feature_request.layers:
                if "features" not in out:
                    out["features"] = {}
                # New storage: later in-place residual updates cannot overwrite
                # a selected post-block stream. No work for unrequested layers.
                out["features"][layer.index] = hidden + residual
            if capture:
                out["layers"].append((hidden, residual))
        normed, pre_norm = self.final_norm.fused_add(hidden, residual, verify=ctx.mode == "verify")
        out["norm"] = normed
        out["pre_norm"] = pre_norm  # hidden + residual before the final norm
        return out

    def prefill(self, input_ids: torch.Tensor, capture: bool = False) -> dict:
        """Stand-alone prefill without pools (parity tools)."""
        return self.forward(input_ids, scratch_prefill_context(input_ids.shape[0], self.device), capture)

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
