"""Full attention layer: FP8 QKV/O projections, gated output, q/k Gemma norm, partial RoPE.

Prefill uses FlashInfer ragged causal attention over bf16 K/V (the reference's
no-prefix path) and writes K/V into the FP8 pages. Decode reads the FP8 pages
through FlashInfer paged decode.
"""

from __future__ import annotations

import torch

from sharar import config
from sharar.engine.context import ForwardContext
from sharar.kernels import ops
from sharar.kernels.fp8_quant import gated_quant_fp8
from sharar.kernels.qk_norm_rope import qk_norm_rope
from sharar.model.norm import GemmaRMSNorm
from sharar.model.rope import PartialRotary
from sharar.weights.modelopt import load_fp8_linear
from sharar.weights.safetensors import SnapshotWeights


class Attention:
    def __init__(self, w: SnapshotWeights, prefix: str, layer_id: int, device: torch.device, rope: PartialRotary, ragged: ops.RaggedAttention):
        self.layer_id = layer_id
        self.num_heads = config.NUM_ATTENTION_HEADS
        self.num_kv_heads = config.NUM_KV_HEADS
        self.head_dim = config.HEAD_DIM
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # q_proj rows are [head0: q(256) | gate(256), head1: ...], k/v follow.
        self.qkv = load_fp8_linear(w, [f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"], device)
        self.o = load_fp8_linear(w, [f"{prefix}.o_proj"], device)
        self.q_norm = GemmaRMSNorm(w.get(f"{prefix}.q_norm.weight").to(device), config.RMS_NORM_EPS)
        self.k_norm = GemmaRMSNorm(w.get(f"{prefix}.k_norm.weight").to(device), config.RMS_NORM_EPS)
        self.rope = rope
        self.ragged = ragged

    @property
    def input_scale(self) -> torch.Tensor:
        return self.qkv.input_scale

    def _qkv(self, x: torch.Tensor, positions: torch.Tensor, x_q: torch.Tensor | None = None):
        qkv = ops.fp8_linear(x, self.qkv, x_q=x_q)
        return self._split_qkv(qkv, positions)

    def _split_qkv(self, qkv, positions):
        t = qkv.shape[0]
        gate = qkv[:, : self.q_size * 2].view(t, self.num_heads, 2 * self.head_dim)[..., self.head_dim :]
        q, k = qk_norm_rope(
            qkv, positions, self.rope.cos_sin, self.q_norm.weight, self.k_norm.weight,
            self.num_heads, self.num_kv_heads, self.head_dim, self.rope.rotary_dim, config.RMS_NORM_EPS,
        )
        v = qkv[:, self.q_size * 2 + self.kv_size :].view(t, self.num_kv_heads, self.head_dim).contiguous()
        return q, k, v, gate

    def _finish(self, attn: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return ops.fp8_linear(None, self.o, x_q=self._output_input(attn, gate))

    def _output_input(self, attn, gate):
        t = attn.shape[0]
        # attn * sigmoid(gate), quantized straight into o_proj's FP8 input.
        return gated_quant_fp8(attn.view(t, self.num_heads, self.head_dim), gate, self.o.input_scale)

    def prefill(self, x: torch.Tensor, positions: torch.Tensor, ctx: ForwardContext | None = None, x_q: torch.Tensor | None = None) -> torch.Tensor:
        q, k, v, gate = self._qkv(x, positions, x_q)
        return self._finish(self._prefill_attention(q, k, v, ctx), gate)

    def _prefill_attention(self, q, k, v, ctx):
        t = q.shape[0]
        if ctx is not None and ctx.kv_pool is not None:
            ctx.kv_pool.write(self.layer_id, ctx.pages, k, v)
        if ctx is not None and (ctx.prefix_len > 0 or ctx.batched):
            # Cached prefix or several sequences: attend over each sequence's
            # prefix + new pages (FP8) with the causal mask aligned to its end.
            attn = ctx.paged_prefill_wrapper.run(q, ctx.kv_pool.cache(self.layer_id)).view(t, self.q_size)
        else:
            self.ragged.plan([t], q.device)
            attn = self.ragged.run(q, k, v)
        return attn


    def decode(self, x: torch.Tensor, ctx: ForwardContext, x_q: torch.Tensor | None = None) -> torch.Tensor:
        q, k, v, gate = self._qkv(x, ctx.positions, x_q)
        ctx.kv_pool.write(self.layer_id, ctx.out_pages, k, v)
        attn = ctx.decode_wrapper.run(q, ctx.kv_pool.cache(self.layer_id))
        return self._finish(attn.view(x.shape[0], self.q_size), gate)

    def forward(self, x: torch.Tensor, ctx: ForwardContext, x_q: torch.Tensor | None = None) -> torch.Tensor:
        if ctx.mode in ("prefill", "verify"):
            return self.prefill(x, ctx.positions, ctx, x_q)
        return self.decode(x, ctx, x_q)
