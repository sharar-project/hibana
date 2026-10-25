# SPDX-License-Identifier: Apache-2.0
"""Nemotron-H attention: 32 query and 2 KV heads of 128 with no positional encoding, no q/k norm and no output gate.

The 27B's layer (`model/attention.py`) without its gate, norms and rotary: FP8 q|k|v as one projection (the
checkpoint's per-tensor scales requantized to a shared one), FP8 KV pages, FlashInfer's plans at the default
1/sqrt(head_dim) scale, FP8 o_proj. The MTP head's attention is the same module over BF16 weights.

Decode and verify write K and V from the projection output into the FP8 pages in one launch
(`kernels/nemotron_fused.write_kv`) and hand FlashInfer q as a strided view of it, so no q/k/v copy is made;
prefill keeps the shared path (contiguous q/k/v for the ragged kernel and the prefix merge).
"""

from __future__ import annotations

import torch

from sharar.kernels import ops
from sharar.kernels.fp8_quant import quant_fp8
from sharar.kernels.nemotron_fused import write_kv
from sharar.model.attention import Attention
from sharar.weights.modelopt import load_fp8_linear
from sharar.weights.sharding import ROW_PARALLEL, ColumnParallel


class NemotronAttention(Attention):
    def __init__(self, w, prefix: str, layer_id: int, device: torch.device, ragged: ops.RaggedAttention, target):  # noqa: D107
        self.layer_id = layer_id
        self.num_heads = target.num_attention_heads
        self.num_kv_heads = target.num_kv_heads
        self.head_dim = target.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv = load_fp8_linear(w, [f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"], device,
                                   ColumnParallel((self.q_size, self.kv_size, self.kv_size)))
        self.o = load_fp8_linear(w, [f"{prefix}.o_proj"], device, ROW_PARALLEL)
        self.rope = None
        self.ragged = ragged

    @property
    def bytes(self) -> int:
        return self.qkv.bytes + self.o.bytes

    def _split_qkv(self, qkv, positions):
        t = qkv.shape[0]
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return (q.reshape(t, self.num_heads, self.head_dim).contiguous(), k.reshape(t, self.num_kv_heads, self.head_dim).contiguous(),
                v.reshape(t, self.num_kv_heads, self.head_dim).contiguous(), None)

    def _project_write(self, x, ctx, pages, x_q):
        """q|k|v projection; K and V written to `pages`; q as a [T, heads, head_dim] view of the projection rows."""
        qkv = ops.fp8_linear(x, self.qkv, x_q=x_q)
        k_cache, v_cache = ctx.kv_pool.cache(self.layer_id)
        write_kv(qkv, pages, k_cache, v_cache, self.q_size, self.kv_size)
        return qkv[:, :self.q_size].view(qkv.shape[0], self.num_heads, self.head_dim)

    def decode(self, x, ctx, x_q=None):
        q = self._project_write(x, ctx, ctx.out_pages, x_q)
        attn = ctx.decode_wrapper.run(q, ctx.kv_pool.cache(self.layer_id))
        return self._finish(attn.view(q.shape[0], self.q_size), None)

    def prefill(self, x, positions, ctx=None, x_q=None):
        if ctx is None or ctx.mode == "prefill" or ctx.kv_pool is None or not (ctx.prefix_len > 0 or ctx.batched):
            return super().prefill(x, positions, ctx, x_q)
        # verify: the rows attend over their sequences' FP8 pages, their own K/V written first
        q = self._project_write(x, ctx, ctx.pages, x_q)
        attn = ctx.paged_prefill_wrapper.run(q, ctx.kv_pool.cache(self.layer_id))
        return self._finish(attn.view(q.shape[0], self.q_size), None)

    def _output_input(self, attn, gate):
        return quant_fp8(attn.reshape(attn.shape[0], self.q_size), self.o.input_scale)
