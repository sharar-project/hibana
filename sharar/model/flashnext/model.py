# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-Flash-Next assembled for the engine: hyper-connected decoder layers (GDN or attention,
then routed experts), the PLE before layer 1, the global mixer and a BF16-or-FP8 lm_head.

`forward()` follows the 27B model's contract (`ForwardContext` prefill / decode; `norm` is the
mixer output the head reads). The routed experts keep the checkpoint's NVFP4; every other matrix
is FP8 at load (`SHARAR_FLASHNEXT_TRUNK=bf16` keeps the checkpoint's BF16 for parity runs).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch

from sharar.engine.context import ForwardContext, scratch_prefill_context
from sharar.engine.contracts import ForwardResult
from sharar.kernels import ops
from sharar.model.flashnext.attention import FlashNextAttention
from sharar.model.flashnext.gdn import FlashNextGdn
from sharar.model.flashnext.hyper import HyperConnection
from sharar.model.flashnext.linear import TrunkLinear
from sharar.model.flashnext.moe import MoeLayer
from sharar.model.flashnext.ple import PleLayer
from sharar.model.gdn_plan import prepare_gdn_prefill
from sharar.model.rope import PartialRotary
from sharar.target import FLASHNEXT, Target
from sharar.weights.safetensors import SnapshotWeights

PREFIX = "model.language_model"


class DecoderLayer:
    def __init__(self, w, snap: Path, index: int, kind: str, device: torch.device, rope, ragged, target: Target, precision: str):
        p = f"{PREFIX}.layers.{index}"
        self.index, self.kind = index, kind
        hc, hidden, eps = target.hc_count, target.hidden_size, target.rms_norm_eps
        self.ple = PleLayer(w, snap, f"{p}.ple", device, target, TrunkLinear, precision) if (index + 1) in target.ple_layer_ids else None
        mixers = os.environ.get("SHARAR_FLASHNEXT_MIXERS", "bf16")
        self.attn_hc = HyperConnection.load(w, f"{p}.attn_hyper_connection", hc, hidden, eps, device, precision=mixers)
        self.mlp_hc = HyperConnection.load(w, f"{p}.mlp_hyper_connection", hc, hidden, eps, device, precision=mixers)
        if kind == "full_attention":
            self.mixer = FlashNextAttention(w, f"{p}.self_attn", index, device, rope, ragged, target, precision)
        else:
            self.mixer = FlashNextGdn(w, f"{p}.linear_attn", index, device, target, precision)
        self.moe = MoeLayer(w, f"{p}.mlp", device, target, TrunkLinear, precision)

    @property
    def bytes(self) -> int:
        return (self.attn_hc.bytes + self.mlp_hc.bytes + self.mixer.bytes + self.moe.bytes + (self.ple.bytes if self.ple else 0))

    def forward(self, hyper: torch.Tensor, input_ids: torch.Tensor, ctx: ForwardContext, eos: int) -> torch.Tensor:
        if self.ple is not None:
            hyper = hyper + self.ple.forward(hyper, input_ids, ctx, eos)
        h, res = self.attn_hc.mix(hyper)
        h = self.mixer.forward(h, ctx)
        hyper = self.attn_hc.combine(h, res)
        h, res = self.mlp_hc.mix(hyper)
        h = self.moe(h)
        return self.mlp_hc.combine(h, res)


class FlashNext:
    def __init__(self, snap: Path, device: torch.device | str = "cuda", log=print, target: Target = FLASHNEXT, precision: str | None = None, **_ignored):
        from sharar.memory.ledger import check_load_headroom
        self.target = target
        self.precision = precision or os.environ.get("SHARAR_FLASHNEXT_TRUNK", "fp8")
        check_load_headroom(target.weight_load_peak_bytes)
        self.device = torch.device(device)
        cfg = target.load_model_config(snap)
        types = target.layer_types_of(cfg)
        self.eos = int(cfg["text_config"]["eos_token_id"])
        started = time.perf_counter()
        w = SnapshotWeights(snap)
        self.embed = w.get(f"{PREFIX}.embed_tokens.weight").to(self.device)
        self.rope = PartialRotary(target.head_dim, target.rotary_dim, target.rope_theta, target.max_position_embeddings, self.device)
        self.ragged = ops.RaggedAttention(self.device, target.num_attention_heads, target.num_kv_heads, target.head_dim)
        self.layers: list[DecoderLayer] = []
        for i, kind in enumerate(types):
            self.layers.append(DecoderLayer(w, snap, i, kind, self.device, self.rope, self.ragged, target, self.precision))
            w.close()  # the PLE keeps its own maps; expert shards are reopened per layer
            if log and (i + 1) % 8 == 0:
                log(f"  loaded {i + 1}/{len(types)} layers ({time.perf_counter() - started:.0f}s, {torch.cuda.memory_allocated() / 2**30:.1f} GiB)")
        self.mixer = HyperConnection.load(w, f"{PREFIX}.hyper_connection_mixer", target.hc_count, target.hidden_size, target.rms_norm_eps, self.device,
                                          combine=False, precision=os.environ.get("SHARAR_FLASHNEXT_MIXERS", "bf16"))
        self.lm_head = TrunkLinear.load(w, ["lm_head.weight"], self.device, self.precision)
        w.close()
        torch.cuda.synchronize()
        # Bytes one decode token streams: the trunk, the head, one embedding row and its routed experts.
        moe = self.layers[0].moe
        expert_bytes = (moe.w13.numel() + moe.w2.numel() + (moe.s13.numel() + moe.s2.numel()) * 4) // target.num_experts
        trunk = sum(layer.bytes - layer.moe.bytes for layer in self.layers) + sum(layer.moe.router.numel() * 2 + layer.moe.shared_gate_up.bytes + layer.moe.shared_down.bytes for layer in self.layers)
        self.stream_bytes = trunk + self.mixer.bytes + self.lm_head.bytes + len(self.layers) * target.experts_per_token * expert_bytes
        if log:
            log(f"  weights loaded in {time.perf_counter() - started:.0f}s, {torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated "
                f"({self.precision} trunk, {self.stream_bytes / 1e9:.2f} GB per decode token)")

    def register_state(self, pool) -> None:
        """Allocate this model's per-slot state beyond the GDN's in the executor's state pool."""
        if self.ple is not None:
            self.ple.register(pool)

    @property
    def ple(self) -> PleLayer | None:
        for layer in self.layers:
            if layer.ple is not None:
                return layer.ple
        return None

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, ctx: ForwardContext, capture: bool = False) -> ForwardResult:
        if ctx.feature_request.layers:
            raise NotImplementedError("layer feature capture is not defined for the hyper-connected trunk yet")
        if ctx.mode == "prefill":
            prepare_gdn_prefill(ctx, len(input_ids), input_ids.device)
        x = torch.nn.functional.embedding(input_ids, self.embed)
        hyper = torch.cat([x] * self.target.hc_count, dim=-1)
        ctx.capture = capture
        out: dict = {"input_ids": input_ids, "positions": ctx.positions}
        if capture:
            out["embed"] = x
            out["layers"] = []
        for layer in self.layers:
            hyper = layer.forward(hyper, input_ids, ctx, self.eos)
            if capture:
                out["layers"].append((hyper, None))
        # `norm` is the stream the head reads: the hyper-connected residual before the global
        # mixer (the MTP head consumes the same stream); `logits()` applies the mixer.
        out["norm"] = hyper
        return out

    def prefill(self, input_ids: torch.Tensor, capture: bool = False) -> dict:
        """Stand-alone prefill without pools (parity tools)."""
        return self.forward(input_ids, scratch_prefill_context(input_ids.shape[0], self.device), capture)

    @torch.inference_mode()
    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: the hyper stream [T, hc*H] from `forward()["norm"]` -> logits [T, V]."""
        return self.lm_head(self.mixer.mix(hidden)[0])

    @torch.inference_mode()
    def argmax_and_topk(self, hidden: torch.Tensor, k: int = 8, chunk: int = 1024) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        arg, top_ids, top_vals = [], [], []
        for start in range(0, hidden.shape[0], chunk):
            lg = self.logits(hidden[start : start + chunk]).float()
            vals, ids = lg.topk(k, dim=-1)
            arg.append(lg.argmax(dim=-1))
            top_ids.append(ids)
            top_vals.append(vals)
        return torch.cat(arg), torch.cat(top_ids), torch.cat(top_vals)
