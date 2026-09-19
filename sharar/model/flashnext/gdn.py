# SPDX-License-Identifier: Apache-2.0
"""Flash-Next's Gated DeltaNet: the 27B's kernels and state handling with FP8-at-load projections
and a sigmoid output gate (`output_gate_type`)."""

from __future__ import annotations

import torch

from sharar.kernels.fla.layernorm_gated import rms_norm_gated
from sharar.memory.state_pool import CONV_DIM
from sharar.model.flashnext.linear import TrunkLinear
from sharar.model.gdn import GatedDeltaNet


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
        assert self.conv_dim == CONV_DIM
        self.gate_activation = target.output_gate
        self.in_qkv = TrunkLinear.load(w, [f"{prefix}.in_proj_qkv.weight"], device, precision)  # [q | k | v] head-first
        self.in_z = TrunkLinear.load(w, [f"{prefix}.in_proj_z.weight"], device, precision)
        self.in_ba = torch.cat([w.get(f"{prefix}.in_proj_b.weight"), w.get(f"{prefix}.in_proj_a.weight")], dim=0).to(device)
        self.conv_weight = w.get(f"{prefix}.conv1d.weight").to(device).view(self.conv_dim, target.linear_conv_kernel)
        self.A_log = w.get(f"{prefix}.A_log").to(device).float().contiguous()
        self.dt_bias = w.get(f"{prefix}.dt_bias").to(device).contiguous()
        self.norm_weight = w.get(f"{prefix}.norm.weight").to(device).repeat(self.num_v_heads).contiguous()
        self.out = TrunkLinear.load(w, [f"{prefix}.out_proj.weight"], device, precision)
        self._scratch_conv = None
        self._scratch_ssm = None

    @property
    def bytes(self) -> int:
        return self.in_qkv.bytes + self.in_z.bytes + self.out.bytes + (self.in_ba.numel() + self.conv_weight.numel()) * 2

    @property
    def input_scale(self):
        return None

    def _project(self, x: torch.Tensor, x_q: torch.Tensor | None = None):
        mixed_qkv = self.in_qkv(x)
        z = self.in_z(x)
        b, a = self._project_ba(x)
        return mixed_qkv, z, b, a

    def _finish(self, core: torch.Tensor, z: torch.Tensor, t: int, *, verify: bool = False) -> torch.Tensor:
        core = rms_norm_gated(
            x=core.reshape(t, self.value_dim), weight=self.norm_weight, bias=None, z=z.reshape(t, self.value_dim),
            eps=self.eps, group_size=self.head_v_dim, norm_before_gate=True, is_rms_norm=True,
            activation=self.gate_activation,
        )
        return self.out(core)
