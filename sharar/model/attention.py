"""Full attention layer: FP8 QKV/O projections, gated output, q/k Gemma norm, partial RoPE."""

from __future__ import annotations

import torch

from sharar import config
from sharar.kernels import ops
from sharar.model.norm import GemmaRMSNorm
from sharar.model.rope import PartialRotary
from sharar.weights.modelopt import load_fp8_linear
from sharar.weights.safetensors import SnapshotWeights


class Attention:
    def __init__(self, w: SnapshotWeights, prefix: str, device: torch.device, rope: PartialRotary, ragged: ops.RaggedAttention):
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

    def prefill(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        t = x.shape[0]
        qkv = ops.fp8_linear(x, self.qkv)
        q_gate, k, v = qkv.split([self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
        q_gate = q_gate.view(t, self.num_heads, 2 * self.head_dim)
        q, gate = q_gate[..., : self.head_dim], q_gate[..., self.head_dim :]
        q = self.q_norm(q.reshape(-1, self.head_dim)).view(t, self.num_heads, self.head_dim)
        k = self.k_norm(k.reshape(-1, self.head_dim)).view(t, self.num_kv_heads, self.head_dim)
        q = self.rope(positions, q)
        k = self.rope(positions, k)
        self.ragged.plan([t], x.device)
        attn = self.ragged.run(q.contiguous(), k.contiguous(), v.contiguous())
        attn = attn.view(t, self.num_heads, self.head_dim) * torch.sigmoid(gate)
        return ops.fp8_linear(attn.reshape(t, self.q_size), self.o)
