# SPDX-License-Identifier: Apache-2.0
"""Hyper-connections: `hc` residual streams of `hidden` width mixed into one block input and
re-injected after the block (SGLang's GatedResidual with a per-branch grouped Gemma norm).

mix:     normed = grouped_gemma_rms(hyper)                     [T, hc*H], one weight per column
         gate   = sigmoid(up(silu(down(normed) / hc)))         [T, hc*H], rank `lowrank`
         input  = mean over streams of gate * normed           [T, H]
combine: inject = 2 * sigmoid(inject(normed) / hc)             [T, hc]
         hyper' = hyper + block_output * inject                 per stream
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def grouped_gemma_rms(x: torch.Tensor, weight: torch.Tensor, eps: float, group: int) -> torch.Tensor:
    """RMS statistics per `group` columns, one (1 + w) weight per column, FP32 inside."""
    xf = x.float()
    g = xf.reshape(*xf.shape[:-1], -1, group)
    xn = (g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + eps)).flatten(-2)
    return (xn * (1.0 + weight.float())).to(x.dtype)


class HyperConnection:
    """The three mixer matrices are `TrunkLinear`s: the checkpoint's BF16, or FP8 at load with a
    dynamic activation scale (`precision="fp8"`, opt-in via SHARAR_FLASHNEXT_MIXERS)."""

    def __init__(self, norm: torch.Tensor, down: torch.Tensor, up: torch.Tensor, inject: torch.Tensor | None, hc: int, hidden: int, eps: float,
                 precision: str = "bf16"):
        from sharar.model.flashnext.linear import TrunkLinear
        self.hc, self.hidden, self.eps = hc, hidden, eps
        self.norm = norm.contiguous()  # [hc*hidden]
        self.down = TrunkLinear.from_bf16(down, precision)  # [rank, hc*hidden]
        self.up = TrunkLinear.from_bf16(up, precision)  # [hc*hidden, rank]
        self.inject = TrunkLinear.from_bf16(inject, precision) if inject is not None else None  # [hc, hc*hidden]

    @classmethod
    def load(cls, w, prefix: str, hc: int, hidden: int, eps: float, device, combine: bool = True, precision: str = "bf16") -> "HyperConnection":
        g = lambda name: w.get(f"{prefix}.{name}").to(device)  # noqa: E731
        return cls(g("hc_norm.weight"), g("input_mix_weight_down.weight"), g("input_mix_weight_up.weight"),
                   g("block_inject_weight.weight") if combine else None, hc, hidden, eps, precision)

    @property
    def bytes(self) -> int:
        return self.norm.numel() * 2 + self.down.bytes + self.up.bytes + (self.inject.bytes if self.inject is not None else 0)

    def mix(self, hyper: torch.Tensor):
        normed = grouped_gemma_rms(hyper, self.norm, self.eps, self.hidden)
        gate = torch.sigmoid(self.up(F.silu(self.down(normed) / self.hc)))
        mixed = (gate.unflatten(-1, (self.hc, self.hidden)) * normed.unflatten(-1, (self.hc, self.hidden))).mean(-2)
        return mixed.to(hyper.dtype), (hyper, normed)

    def combine(self, block_output: torch.Tensor, residuals) -> torch.Tensor:
        hyper, normed = residuals
        inject = 2 * torch.sigmoid(self.inject(normed) / self.hc)  # [T, hc]
        r = hyper.unflatten(-1, (self.hc, self.hidden))
        return (r + block_output.unsqueeze(-2) * inject.unsqueeze(-1)).flatten(-2).to(hyper.dtype)
