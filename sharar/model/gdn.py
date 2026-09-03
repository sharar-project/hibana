"""Gated DeltaNet linear-attention layer (48 of the 64 layers).

Prefill path mirrors SGLang's GDNAttnBackend.forward_extend with the Triton
kernels: FP8 qkv/z projection, bf16 b/a projection, causal Conv1D with state
write-back, fused gating, chunked gated delta rule with state write-back, gated
RMSNorm, FP8 output projection.
"""

from __future__ import annotations

import torch

from sharar import config
from sharar.kernels import ops
from sharar.kernels.conv1d import causal_conv1d_fn
from sharar.kernels.fla.chunk import chunk_gated_delta_rule
from sharar.kernels.fla.fused_gdn_gating import fused_gdn_gating
from sharar.kernels.fla.layernorm_gated import RMSNorm as RMSNormGated
from sharar.weights.modelopt import load_fp8_linear
from sharar.weights.safetensors import SnapshotWeights


class GdnState:
    """Conv and SSM state slots for one layer. Slot 0 is used by the parity forward."""

    def __init__(self, num_slots: int, device: torch.device):
        conv_dim = 2 * config.LINEAR_NUM_KEY_HEADS * config.LINEAR_KEY_HEAD_DIM + config.LINEAR_NUM_VALUE_HEADS * config.LINEAR_VALUE_HEAD_DIM
        self.conv = torch.zeros((num_slots, conv_dim, config.LINEAR_CONV_KERNEL - 1), dtype=torch.bfloat16, device=device)
        self.ssm = torch.zeros(
            (num_slots, config.LINEAR_NUM_VALUE_HEADS, config.LINEAR_VALUE_HEAD_DIM, config.LINEAR_KEY_HEAD_DIM),
            dtype=torch.float32,
            device=device,
        )


class GatedDeltaNet:
    def __init__(self, w: SnapshotWeights, prefix: str, device: torch.device, num_state_slots: int = 1):
        self.num_k_heads = config.LINEAR_NUM_KEY_HEADS
        self.num_v_heads = config.LINEAR_NUM_VALUE_HEADS
        self.head_k_dim = config.LINEAR_KEY_HEAD_DIM
        self.head_v_dim = config.LINEAR_VALUE_HEAD_DIM
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        # Fused [q | k | v | z] FP8 projection; qkv and z carry different weight scales.
        self.in_qkvz = load_fp8_linear(w, [f"{prefix}.in_proj_qkv", f"{prefix}.in_proj_z"], device)
        self.in_ba = torch.cat([w.get(f"{prefix}.in_proj_b.weight"), w.get(f"{prefix}.in_proj_a.weight")], dim=0).to(device)
        self.conv_weight = w.get(f"{prefix}.conv1d.weight").to(device).view(self.conv_dim, config.LINEAR_CONV_KERNEL)
        self.A_log = w.get(f"{prefix}.A_log").to(device).float()
        self.dt_bias = w.get(f"{prefix}.dt_bias").to(device)
        self.norm = RMSNormGated(self.head_v_dim, eps=config.RMS_NORM_EPS, group_size=None, norm_before_gate=True, device=device, dtype=torch.bfloat16, activation="swish")
        with torch.no_grad():
            self.norm.weight.copy_(w.get(f"{prefix}.norm.weight").to(device))
        self.out = load_fp8_linear(w, [f"{prefix}.out_proj"], device)
        self.state = GdnState(num_state_slots, device)

    def prefill(self, x: torch.Tensor, slot: int = 0) -> torch.Tensor:
        """Fresh-sequence prefill: the slot's conv and SSM state are cleared first,
        because the chunk kernel reads the initial state from the pool."""
        t = x.shape[0]
        device = x.device
        self.state.conv[slot].zero_()
        self.state.ssm[slot].zero_()
        qkvz = ops.fp8_linear(x, self.in_qkvz)
        ba = x @ self.in_ba.T
        mixed_qkv = qkvz[:, : 2 * self.key_dim + self.value_dim]
        z = qkvz[:, 2 * self.key_dim + self.value_dim :]
        b, a = ba.split([self.num_v_heads, self.num_v_heads], dim=-1)
        b = b.contiguous()
        a = a.contiguous()

        query_start_loc = torch.tensor([0, t], dtype=torch.int32, device=device)
        cache_indices = torch.tensor([slot], dtype=torch.int32, device=device)
        has_initial_state = torch.tensor([False], device=device)
        mixed_qkv = causal_conv1d_fn(
            mixed_qkv.transpose(0, 1),
            self.conv_weight,
            None,
            activation="silu",
            conv_states=self.state.conv,
            has_initial_state=has_initial_state,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            seq_lens_cpu=[t],
        ).transpose(0, 1)[:t]

        q, k, v = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(1, t, self.num_k_heads, self.head_k_dim)
        k = k.reshape(1, t, self.num_k_heads, self.head_k_dim)
        v = v.reshape(1, t, self.num_v_heads, self.head_v_dim)
        g, beta = fused_gdn_gating(self.A_log, a, b, self.dt_bias)
        core, _, _ = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=self.state.ssm,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )
        core = core.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core = self.norm(core, z).reshape(t, self.value_dim)
        return ops.fp8_linear(core, self.out)
