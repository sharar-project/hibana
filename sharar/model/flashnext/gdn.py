# SPDX-License-Identifier: Apache-2.0
"""Flash-Next's Gated DeltaNet: the 27B's kernels and state handling with FP8-at-load projections
and a sigmoid output gate (`output_gate_type`)."""

from __future__ import annotations

import torch

from sharar.kernels.fla.layernorm_gated import rms_norm_gated
from sharar.model.flashnext.linear import TrunkLinear
from sharar.model.gdn import GatedDeltaNet
from sharar.weights.sharding import ROW_PARALLEL, ColumnParallel


class FlashNextGdn(GatedDeltaNet):
    def __init__(self, w, prefix: str, layer_id: int, device: torch.device, target, precision: str):  # noqa: D107
        self.layer_id = layer_id
        self.device = device
        self.eps = target.rms_norm_eps
        self.num_k_heads = target.linear_num_key_heads
        self.num_v_heads = target.linear_num_value_heads
        self.head_k_dim = target.linear_key_head_dim
        self.head_v_dim = target.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.gate_activation = target.output_gate
        # Head counts are this rank's (`Target.shard`): every per-head tensor keeps its rows of q, k, v (and z, b,
        # a), whole heads of each, so a value head stays with its key head; out_proj is row-parallel.
        qkv_heads = ColumnParallel((self.key_dim, self.key_dim, self.value_dim))
        v_heads = ColumnParallel((self.num_v_heads,))
        self.in_qkv = TrunkLinear.load(w, [f"{prefix}.in_proj_qkv.weight"], device, precision, qkv_heads)  # [q | k | v] head-first
        self.in_z = TrunkLinear.load(w, [f"{prefix}.in_proj_z.weight"], device, precision, ColumnParallel((self.value_dim,)))
        self.in_ba = torch.cat([v_heads(w.get(f"{prefix}.in_proj_b.weight")), v_heads(w.get(f"{prefix}.in_proj_a.weight"))], dim=0).to(device)
        self.conv_weight = qkv_heads(w.get(f"{prefix}.conv1d.weight").view(-1, target.linear_conv_kernel)).contiguous().to(device)
        self.A_log = v_heads(w.get(f"{prefix}.A_log")).to(device).float().contiguous()
        self.dt_bias = v_heads(w.get(f"{prefix}.dt_bias")).to(device).contiguous()
        self.norm_weight = w.get(f"{prefix}.norm.weight").to(device).repeat(self.num_v_heads).contiguous()
        self.out = TrunkLinear.load(w, [f"{prefix}.out_proj.weight"], device, precision, ROW_PARALLEL)
        self._scratch_conv = None
        self._scratch_ssm = None

    @property
    def bytes(self) -> int:
        return self.in_qkv.bytes + self.in_z.bytes + self.out.bytes + (self.in_ba.numel() + self.conv_weight.numel()) * 2

    @property
    def input_scale(self):
        return None

    def _project(self, x: torch.Tensor, x_q: torch.Tensor | None = None):
        rows = x.reshape(-1, x.shape[-1]).shape[0]
        if self.in_qkv._uses_w3(rows) and self.in_z._uses_w3(rows):
            # both 3-bit copies read the same input: quantize it once (the bits each call's own quantization gives)
            from sharar.kernels import w3
            flat = x.reshape(-1, x.shape[-1]).contiguous()
            xq, xs = w3.quant_groups(flat)
            mixed_qkv = w3.w3_linear(xq, self.in_qkv.w3, xs).reshape(*x.shape[:-1], self.in_qkv.out_features)
            z = w3.w3_linear(xq, self.in_z.w3, xs).reshape(*x.shape[:-1], self.in_z.out_features)
        else:
            mixed_qkv = self.in_qkv(x)
            z = self.in_z(x)
        b, a = self._project_ba(x)
        return mixed_qkv, z, b, a

    def _window_norm(self, z):
        """The window kernel's epilogue: the gated RMS norm with this model's gate, then the input quantization of the out
        projection's 3-bit copy when this call uses it (E4M3 per 32-group, `w3.quant_groups`), else BF16."""
        z = z.reshape(-1, self.value_dim)
        groups = self.out.scale is not None and self.out._uses_w3(z.shape[0])
        return (z, self.norm_weight, self.eps, "groups" if groups else None, self.gate_activation)

    def _window_out(self, out, z, t, *, verify=False):
        if isinstance(out, tuple):  # the 3-bit copy's quantized input and its group scales
            from sharar.kernels import w3
            x_q, x_scale = out
            return w3.w3_linear(x_q, self.out.w3, x_scale)
        return self.out(out)

    def _finish(self, core: torch.Tensor, z: torch.Tensor, t: int, *, verify: bool = False) -> torch.Tensor:
        core = rms_norm_gated(
            x=core.reshape(t, self.value_dim), weight=self.norm_weight, z=z.reshape(t, self.value_dim),
            eps=self.eps, group_size=self.head_v_dim, activation=self.gate_activation,
        )
        return self.out(core)
