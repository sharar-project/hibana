"""Full attention layer: FP8 QKV/O projections, gated output, q/k Gemma norm, partial RoPE.

Prefill attends to new K/V in BF16 and merges that result with attention over
the FP8 cached prefix. Verification and decode read the FP8 pages.
"""

from __future__ import annotations

import torch

from hibana.target import QWEN38, Target
from hibana.engine.context import ForwardContext
from hibana.kernels import ops
from hibana.kernels.fp8_quant import gated_quant_fp8
from hibana.kernels.qk_norm_rope import qk_norm_rope
from hibana.model.norm import GemmaRMSNorm
from hibana.model.rope import PartialRotary
from hibana.weights.modelopt import load_fp8_linear
from hibana.weights.sharding import ROW_PARALLEL, ColumnParallel
from hibana.weights.safetensors import SnapshotWeights


class Attention:
    def __init__(self, w: SnapshotWeights, prefix: str, layer_id: int, device: torch.device, rope: PartialRotary, ragged: ops.RaggedAttention,
                 target: Target = QWEN38):
        self.layer_id = layer_id
        self.eps = target.rms_norm_eps
        self.num_heads = target.num_attention_heads
        self.num_kv_heads = target.num_kv_heads
        self.head_dim = target.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # q_proj rows are [head0: q(256) | gate(256), head1: ...], k/v follow. Head counts are this rank's
        # (`Target.shard`): q|k|v column-parallel by head, o_proj row-parallel.
        self.qkv = load_fp8_linear(w, [f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"], device,
                                   ColumnParallel((2 * self.q_size, self.kv_size, self.kv_size)))
        self.o = load_fp8_linear(w, [f"{prefix}.o_proj"], device, ROW_PARALLEL)
        self.q_norm = GemmaRMSNorm(w.get(f"{prefix}.q_norm.weight").to(device), self.eps)
        self.k_norm = GemmaRMSNorm(w.get(f"{prefix}.k_norm.weight").to(device), self.eps)
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
            self.num_heads, self.num_kv_heads, self.head_dim, self.rope.rotary_dim, self.eps,
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
        if ctx is None or ctx.mode == "prefill":
            self.ragged.plan(ctx.seq_lens if ctx is not None and ctx.batched else [t], q.device)
            if ctx is not None and ctx.prefix_attention is not None:
                # the cached prefix on kernels/csrc/verify_attn.cu, merged there with the new rows' causal attention
                from hibana.kernels.verify_attn_cuda import prefix_attention
                new_out, new_lse = self.ragged.run_return_lse(q, k, v)
                attn = prefix_attention(q.view(t, self.num_heads, self.head_dim), ctx.kv_pool.cache(self.layer_id),
                                        ctx.kv_pool.scales(self.layer_id),
                                        ctx.prefix_attention, self.head_dim ** -0.5, new_out, new_lse)
            elif ctx is not None and ctx.paged_prefill_wrapper is not None:
                from flashinfer.cascade import merge_state

                new_out, new_lse = self.ragged.run_return_lse(q, k, v)
                prefix_out, prefix_lse = ctx.paged_prefill_wrapper.run(
                    q, ctx.kv_pool.cache(self.layer_id), return_lse=True, kv_cache_sf=ctx.kv_pool.scales(self.layer_id),
                )
                attn, _ = merge_state(new_out, new_lse, prefix_out, prefix_lse)
            else:
                attn = self.ragged.run(q, k, v)
            if ctx is not None and ctx.kv_pool is not None:
                ctx.kv_pool.write(self.layer_id, ctx.pages, k, v)
            return attn.view(t, self.q_size)
        if ctx is not None and ctx.kv_pool is not None:
            ctx.kv_pool.write(self.layer_id, ctx.pages, k, v)
        if ctx is not None and ctx.verify_attention is not None:
            # many rows a sequence over an NVFP4 cache: kernels/csrc/verify_attn.cu
            from hibana.kernels.verify_attn_cuda import verify_attention
            qo, kv_indptr, kv_indices, splits = ctx.verify_attention
            attn = verify_attention(q.view(t, self.num_heads, self.head_dim), ctx.kv_pool.cache(self.layer_id),
                                    ctx.kv_pool.scales(self.layer_id), kv_indptr, kv_indices, qo, self.head_dim ** -0.5,
                                    splits=splits).view(t, self.q_size)
        elif ctx is not None and (ctx.prefix_len > 0 or ctx.batched):
            # Cached prefix or several sequences: attend over each sequence's
            # prefix + new pages (FP8) with the causal mask aligned to its end.
            attn = ctx.paged_prefill_wrapper.run(q, ctx.kv_pool.cache(self.layer_id),
                                                 kv_cache_sf=ctx.kv_pool.scales(self.layer_id)).view(t, self.q_size)
        else:
            self.ragged.plan([t], q.device)
            attn = self.ragged.run(q, k, v)
        return attn


    def decode(self, x: torch.Tensor, ctx: ForwardContext, x_q: torch.Tensor | None = None) -> torch.Tensor:
        q, k, v, gate = self._qkv(x, ctx.positions, x_q)
        ctx.kv_pool.write(self.layer_id, ctx.out_pages, k, v)
        attn = ctx.decode_wrapper.run(q, ctx.kv_pool.cache(self.layer_id), kv_cache_sf=ctx.kv_pool.scales(self.layer_id))
        return self._finish(attn.view(x.shape[0], self.q_size), gate)

    def forward(self, x: torch.Tensor, ctx: ForwardContext, x_q: torch.Tensor | None = None) -> torch.Tensor:
        if ctx.mode in ("prefill", "verify"):
            return self.prefill(x, ctx.positions, ctx, x_q)
        return self.decode(x, ctx, x_q)
