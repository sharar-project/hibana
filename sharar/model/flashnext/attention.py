# SPDX-License-Identifier: Apache-2.0
"""Flash-Next's attention layers (24 q / 2 kv heads of 256, gated output, quarter rotary) on the
27B's attention module with FP8-at-load projections, with the sparse-attention indexer
(`indexer.py`): dense while every row sees at most the 2,048-token budget, otherwise each
row attends over the pages its indexer selected (`kernels/qsa.py`)."""

from __future__ import annotations

import torch

from sharar.kernels import ops
from sharar.model.attention import Attention
from sharar.model.flashnext.indexer import FlashNextIndexer
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
        self.indexer = FlashNextIndexer(w, prefix, layer_id, device, target) if target.indexer_budget else None

    @property
    def bytes(self) -> int:
        return self.qkv.bytes + self.o.bytes + (self.indexer.bytes if self.indexer is not None else 0)

    def forward(self, x: torch.Tensor, ctx, x_q: torch.Tensor | None = None) -> torch.Tensor:
        if self.indexer is None:
            return super().forward(x, ctx, x_q)
        if ctx.qsa_seqs is None:
            if ctx.qsa_members is not None:  # graph replay inside the budget: keys only, dense attention
                self.indexer.maintain(x, ctx, ctx.out_pages if ctx.mode == "decode" else ctx.pages)
            return super().forward(x, ctx, x_q)
        q, k, v, gate = self._qkv(x, ctx.positions)
        token_pages = ctx.out_pages if ctx.mode == "decode" else ctx.pages
        selected = self.indexer.run(x, ctx, token_pages)
        if selected is None:  # every row sees at most the budget: the dense path is exact
            if ctx.mode == "decode":
                ctx.kv_pool.write(self.layer_id, token_pages, k, v)
                attn = ctx.decode_wrapper.run(q, ctx.kv_pool.cache(self.layer_id)).view(x.shape[0], self.q_size)
            else:
                attn = self._prefill_attention(q, k, v, ctx)
            return self._finish(attn, gate)
        from sharar.kernels.qsa import sparse_attention
        ctx.kv_pool.write(self.layer_id, token_pages, k, v)
        k_pool, v_pool = ctx.kv_pool.cache(self.layer_id)
        n = k_pool.shape[0] * k_pool.shape[1]
        attn = sparse_attention(q, k_pool.view(n, self.num_kv_heads, self.head_dim), v_pool.view(n, self.num_kv_heads, self.head_dim),
                                *selected, self.head_dim ** -0.5)
        return self._finish(attn.view(x.shape[0], self.q_size), gate)

    @property
    def input_scale(self):
        return None

    def _qkv(self, x: torch.Tensor, positions: torch.Tensor, x_q: torch.Tensor | None = None):
        return self._split_qkv(self.qkv(x), positions)

    def _finish(self, attn: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        from sharar.kernels.flashnext_fused import sigmoid_gate
        t = attn.shape[0]
        return self.o(sigmoid_gate(attn.view(t, self.num_heads, self.head_dim), gate))
