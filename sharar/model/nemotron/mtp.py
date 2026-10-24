# SPDX-License-Identifier: Apache-2.0
"""Nemotron's in-checkpoint MTP step as the speculative draft (the `mtp` proposer's head; `model/mtp_layer.py`
holds what every target's MTP layer shares).

For token x_i and the target's hidden h_{i-1} of the previous position (`forward()["norm"]`),
  u = eh_proj(cat(enorm(embed(x_i)), hnorm(h_{i-1})))
  u = u + attention(norm_0(u))                 # no positional encoding, its own BF16 paged KV
  d = final_layernorm(u + moe(norm_1(u)))     # a latent MoE block like the trunk's
  logits = lm_head(d)                          # the target's FP8 head
which predicts x_{i+1}; chained drafting feeds d back as the next hidden. The reference measured top-1 agreement with
the target's greedy tokens of 93.9% at depth 1 and 79% at depth 3 this way; with d taken before final_layernorm,
depth 2 fell to 8%, and with the target's hidden before norm_f depth 1 fell to 40%.

Formats (draft-only: the target verifies, so these move acceptance, never output). eh_proj and the attention's q|k|v
and o are quantized to FP8 at load (`PRECISION`; `TrunkLinear`'s per-tensor weight scale and per-call activation scale,
as Flash-Next's MTP layer), which halves the 138 MB of BF16 they read per step; the experts stay the checkpoint's
NVFP4 (or the W3 checkpoint's 3-bit ones when the target loaded them, `Nemotron.mtp_experts3`), the router and
latent projections its BF16 (`DraftMoe` can quantize them or route fewer experts; neither is on).
On recorded target rows (`profiling/nemotron_draft_step.py --accept`: 12 greedy sequences, 3.5K drafted positions)
FP8 projections, FP8 latent projections and the 2-bit draft head each left tokens/round at depth 3 within 0.01 of the
BF16 head's 3.42; routing 12 of 22 experts held it (3.43) but 8 lost 2% (3.35). The draft head is the server's choice
(SHARAR_DRAFT_HEAD): the target's FP8 lm_head (537 MB, 62% of a BF16 step's kernel time) or a 2/3-bit copy whose top
candidates its FP8 rows rescore (`set_draft_head`).
"""

from __future__ import annotations

import flashinfer
import torch

from sharar.cache.kv_pool import KvPool
from sharar.kernels import ops
from sharar.model.flashnext.linear import TrunkLinear
from sharar.model.mtp_layer import MtpLayer
from sharar.model.nemotron.config import PREFIX as MODEL_PREFIX
from sharar.model.nemotron.moe import NemotronMoe
from sharar.model.nemotron.norm import RMSNorm

PREFIX = f"{MODEL_PREFIX}.mtp.layers"


class DraftMoe(NemotronMoe):
    """The MTP head's latent MoE with draft-only options (the target never reads it, so they move acceptance, never
    output): `precision="fp8"` keeps FP8 copies of router|fc1 and fc2 (one per-tensor weight scale each, activations
    quantized per call as `TrunkLinear` does; the routing logits are then BF16 values, upcast), and `topk` < 22 routes
    each row to fewer experts (renormalized over them, as SHARAR_MOE_KEEP does for the trunk)."""

    def __init__(self, w, prefix: str, device: torch.device, target, precision: str = "bf16", topk: int | None = None,
                 w3=None):
        super().__init__(w, prefix, device, target, w3=w3)
        if precision not in ("bf16", "fp8"):
            raise ValueError(f"MTP MoE precision must be bf16 or fp8, not {precision!r}")
        self.draft_topk = topk or self.topk
        if not 1 <= self.draft_topk <= self.topk:
            raise ValueError(f"MTP MoE top-k must be 1..{self.topk}, not {self.draft_topk}")
        self.fp8 = precision == "fp8"
        if self.fp8:
            # beside the BF16 matrices, which the small-call kernels' weight list still names (21 MB)
            self.gate_fc1_q = TrunkLinear.from_bf16(self.gate_fc1, "fp8")
            self.fc2_q = TrunkLinear.from_bf16(self.fc2, "fp8")

    @property
    def dense_bytes(self) -> int:
        return super().dense_bytes + (self.gate_fc1_q.bytes + self.fc2_q.bytes if self.fp8 else 0)

    def __call__(self, x: torch.Tensor, ctx=None) -> torch.Tensor:
        from sharar.model.nemotron.moe import route_topk
        x = x.contiguous()
        if not self.fp8:
            if self.draft_topk == self.topk:
                return super().__call__(x, ctx)  # the trunk block's own path, bitwise
            if x.shape[0] <= getattr(self, "SMALL_ROWS", 0):
                return self.small(x, ctx, k=self.draft_topk)
            fused = torch.mm(x, self.gate_fc1.T, out_dtype=torch.float32)  # [M, E + L]
        else:
            fused = self.gate_fc1_q(x).float()
        ids, weights, latent = route_topk(fused, self.bias, self.draft_topk, self.latent)
        routed = self.routed(latent, ids, weights)
        if not self.fp8:
            return torch.addmm(self.shared(x), routed, self.fc2.T)
        return self.fc2_q(routed) + self.shared(x)


class NemotronMtpHead(MtpLayer):
    DRAFT_HEADS = True  # set_draft_head: reduced vocabulary and/or a 2/3-bit shortlist over the FP8 head
    PRECISION = "fp8"  # eh_proj and the attention's q|k|v and o: "fp8" (quantized at load) or "bf16" (the checkpoint's)
    MOE_PRECISION = "bf16"  # the MoE's router|fc1 and fc2 latent projections (`DraftMoe`)
    MOE_TOPK = None  # experts per row in the draft's MoE (None: the checkpoint's 22)

    def __init__(self, w, device: torch.device, model, num_pages: int, precision: str | None = None,
                 moe_precision: str | None = None, padding_rows: int = 0, moe_topk: int | None = None):
        t = self.target = model.target
        self.embed, self.lm_head = model.embed, model.lm_head
        from sharar.proposers.draft_head import TargetHead
        self.draft_head = TargetHead(self.lm_head)  # replaced by set_draft_head
        self.hidden_width = t.hidden_size
        precision = precision or self.PRECISION
        eps = t.rms_norm_eps
        g = lambda name: w.get(f"{PREFIX}.{name}").to(device)  # noqa: E731
        self.enorm = RMSNorm(g("0.enorm.weight"), eps)
        self.hnorm = RMSNorm(g("0.hnorm.weight"), eps)
        self.eh = TrunkLinear.from_bf16(g("0.eh_proj.weight"), precision)  # [H, 2H] over cat(embedding, hidden)
        self.attn_norm = RMSNorm(g("0.norm.weight"), eps)
        self.moe_norm = RMSNorm(g("1.norm.weight"), eps)
        self.final_norm = RMSNorm(g("1.final_layernorm.weight"), eps)
        self.moe = DraftMoe(w, f"{PREFIX}.1.mixer", device, t, moe_precision or self.MOE_PRECISION, moe_topk or self.MOE_TOPK,
                            w3=getattr(model, "mtp_experts3", None))
        self._attention_setup(w, f"{PREFIX}.0.mixer", device, t, None, precision, num_pages, padding_rows)

    def _attention_setup(self, w, prefix: str, device: torch.device, target, rope, precision: str, num_pages: int,
                         padding_rows: int) -> None:
        """The shared layer's draft KV pool and plan around Nemotron's attention (no norm, rotary or gate)."""
        from sharar.kernels.attention_plan import AttentionGeometry, AttentionPlanner
        self.device, self.rope = device, None
        self.num_heads, self.num_kv_heads, self.head_dim = target.num_attention_heads, target.num_kv_heads, target.head_dim
        self.q_size, self.kv_size = self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim
        self.qkv = TrunkLinear.load(w, [f"{prefix}.q_proj.weight", f"{prefix}.k_proj.weight", f"{prefix}.v_proj.weight"], device, precision)
        self.o = TrunkLinear.load(w, [f"{prefix}.o_proj.weight"], device, precision)
        self.ragged = ops.RaggedAttention(device, self.num_heads, self.num_kv_heads, self.head_dim)
        self.kv = KvPool(num_pages, device, layer_ids=[0], dtype=torch.bfloat16, target=target, padding_rows=padding_rows)
        self._ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        self._attn = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self._ws, "NHD")
        self.attention = AttentionPlanner(AttentionGeometry.of_target(target, 1, torch.bfloat16))  # BF16 draft KV, page 1

    @property
    def bytes(self) -> int:
        return (self.eh.bytes + self.qkv.bytes + self.o.bytes + self.moe.bytes + self.kv.bytes
                + getattr(self.draft_head, "bytes", 0))

    def _qkv_write(self, h: torch.Tensor, positions: torch.Tensor, write_pages: torch.Tensor):
        t = h.shape[0]
        qkv = self.qkv(h)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        self.kv.write(0, write_pages, k.reshape(t, self.num_kv_heads, self.head_dim), v.reshape(t, self.num_kv_heads, self.head_dim))
        return q.reshape(t, self.num_heads, self.head_dim).contiguous(), qkv

    def _attention(self, h: torch.Tensor, positions: torch.Tensor, write_pages: torch.Tensor, wrapper) -> torch.Tensor:
        q, _ = self._qkv_write(h, positions, write_pages)
        return self.o(wrapper.run(q, self.kv.cache(0)).view(h.shape[0], self.q_size))

    def _fuse(self, tokens: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        e = self.enorm(torch.nn.functional.embedding(tokens, self.embed))
        return self.eh(torch.cat([e, self.hnorm(hidden)], dim=-1))

    def _attention_input(self, u: torch.Tensor) -> torch.Tensor:
        return self.attn_norm(u)

    def _layer(self, u: torch.Tensor, positions: torch.Tensor, write_pages: torch.Tensor, wrapper) -> torch.Tensor:
        a = self._attention(self._attention_input(u), positions, write_pages, wrapper)
        x, resid = self.moe_norm.fused_add(a, u)
        return self._final(self.moe(x, None), resid)

    def _final(self, m: torch.Tensor, resid: torch.Tensor) -> torch.Tensor:
        return self.final_norm(m + resid)

    @torch.inference_mode()
    def logits(self, d: torch.Tensor) -> torch.Tensor:
        return ops.fp8_linear(d, self.lm_head)

    def head_input(self, d: torch.Tensor) -> torch.Tensor:
        return d

    def set_draft_head(self, ids: torch.Tensor | None = None, fmt: str = "fp8", rescore: int = 0) -> None:
        """Draft through a head of its own (`proposers/draft_head.py`): these token ids only and/or a 2/3-bit copy whose
        top `rescore` candidates the target's FP8 rows rescore."""
        from sharar.proposers.draft_head import ShortlistHead
        self.draft_head = ShortlistHead(self.lm_head, ids, fmt, rescore)
