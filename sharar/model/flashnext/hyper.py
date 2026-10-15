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
    dynamic activation scale (`precision="fp8"`, the default; SHARAR_FLASHNEXT_MIXERS=bf16 keeps the checkpoint's)."""

    def __init__(self, norm: torch.Tensor, down: torch.Tensor, up: torch.Tensor, inject: torch.Tensor | None, hc: int, hidden: int, eps: float,
                 precision: str = "bf16", fused: bool = True):
        from sharar.model.flashnext.linear import TrunkLinear
        self.hc, self.hidden, self.eps, self.fused = hc, hidden, eps, fused
        self.norm = norm.contiguous()  # [hc*hidden]
        # The fused path folds the 1/hc of silu(down(x)/hc) and sigmoid(inject(x)/hc) into the matrices
        # (a power of two: exact in BF16).
        scale = 1.0 / hc if fused else 1.0
        self.down = TrunkLinear.from_bf16((down * scale).to(torch.bfloat16), precision)  # [rank, hc*hidden]
        self.up = TrunkLinear.from_bf16(up, precision)  # [hc*hidden, rank]
        # The inject matrix stays BF16: FP8 GEMMs need N % 16 == 0 and it has hc rows (80 KB, nothing to save).
        self.inject = TrunkLinear.from_bf16((inject * scale).to(torch.bfloat16), "bf16") if inject is not None else None  # [hc, hc*hidden]

    @classmethod
    def load(cls, w, prefix: str, hc: int, hidden: int, eps: float, device, combine: bool = True, precision: str = "bf16", fused: bool = True) -> "HyperConnection":
        g = lambda name: w.get(f"{prefix}.{name}").to(device)  # noqa: E731
        return cls(g("hc_norm.weight"), g("input_mix_weight_down.weight"), g("input_mix_weight_up.weight"),
                   g("block_inject_weight.weight") if combine else None, hc, hidden, eps, precision, fused)

    @property
    def bytes(self) -> int:
        return self.norm.numel() * 2 + self.down.bytes + self.up.bytes + (self.inject.bytes if self.inject is not None else 0)

    def mix(self, hyper: torch.Tensor):
        if self.fused:
            from sharar.kernels import flashnext_fused as fk
            # The norm writes its block maxima, so the down projection's FP8 quantization is one parallel launch (many
            # rows) or none (the GEMV's own, few rows): the unfused chain's bits, faster (#9).
            normed, parts = fk.grouped_gemma_rms(hyper, self.norm, self.eps, self.hidden, amax=True)
            mixed = fk.gate_mean(self.up(self.down(normed, amax_parts=parts), act=fk.ACT_SILU), normed, self.hc, self.hidden)
            return mixed, (hyper, normed)
        normed = grouped_gemma_rms(hyper, self.norm, self.eps, self.hidden)
        gate = torch.sigmoid(self.up(F.silu(self.down(normed) / self.hc)))
        mixed = (gate.unflatten(-1, (self.hc, self.hidden)) * normed.unflatten(-1, (self.hc, self.hidden))).mean(-2)
        return mixed.to(hyper.dtype), (hyper, normed)

    def combine(self, block_output: torch.Tensor, residuals) -> torch.Tensor:
        hyper, normed = residuals
        if self.fused:
            from sharar.kernels import flashnext_fused as fk
            return fk.inject_combine(hyper, block_output, self.inject(normed), self.hc, self.hidden)
        inject = 2 * torch.sigmoid(self.inject(normed) / self.hc)  # [T, hc]
        r = hyper.unflatten(-1, (self.hc, self.hidden))
        return (r + block_output.unsqueeze(-2) * inject.unsqueeze(-1)).flatten(-2).to(hyper.dtype)
