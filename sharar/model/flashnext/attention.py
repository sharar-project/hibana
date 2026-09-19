# SPDX-License-Identifier: Apache-2.0
"""Flash-Next's attention layers (24 q / 2 kv heads of 256, gated output, quarter rotary) on the
27B's attention module with FP8-at-load projections. Dense over the context: exact while the
context fits the sparse indexer's budget; the indexer is not ported yet."""

from __future__ import annotations

import torch

from sharar.kernels import ops
from sharar.model.attention import Attention
from sharar.model.flashnext.linear import TrunkLinear
from sharar.model.norm import GemmaRMSNorm


class FlashNextAttention(Attention):
    def __init__(self, w, prefix: str, layer_id: int, device: torch.device, rope, ragged: ops.RaggedAttention, target, precision: str):  # noqa: D107
        self.layer_id = layer_id
        self.eps = target.rms_norm_eps
        self.num_heads = target.num_attention_heads
        self.num_kv_heads = target.num_kv_heads
        self.head_dim = target.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # q_proj rows are [head0: q(256) | gate(256), head1: ...], k/v follow.
        self.qkv = TrunkLinear.load(w, [f"{prefix}.q_proj.weight", f"{prefix}.k_proj.weight", f"{prefix}.v_proj.weight"], device, precision)
        self.o = TrunkLinear.load(w, [f"{prefix}.o_proj.weight"], device, precision)
        self.q_norm = GemmaRMSNorm(w.get(f"{prefix}.q_norm.weight").to(device), self.eps)
        self.k_norm = GemmaRMSNorm(w.get(f"{prefix}.k_norm.weight").to(device), self.eps)
        self.rope = rope
        self.ragged = ragged

    @property
    def bytes(self) -> int:
        return self.qkv.bytes + self.o.bytes

    @property
    def input_scale(self):
        return None

    def _qkv(self, x: torch.Tensor, positions: torch.Tensor, x_q: torch.Tensor | None = None):
        return self._split_qkv(self.qkv(x), positions)

    def _finish(self, attn: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        t = attn.shape[0]
        gated = (attn.view(t, self.num_heads, self.head_dim).float() * torch.sigmoid(gate.float())).to(torch.bfloat16)
        return self.o(gated.reshape(t, self.q_size))
