# SPDX-License-Identifier: Apache-2.0
"""Hyper-connections: `hc` residual streams of `hidden` width mixed into one block input and
re-injected after the block (SGLang's GatedResidual with a per-branch grouped Gemma norm).

mix:     normed = grouped_gemma_rms(hyper)                     [T, hc*H], one weight per column
         gate   = sigmoid(up(silu(down(normed) / hc)))         [T, hc*H], rank `lowrank`
         input  = mean over streams of gate * normed           [T, H]
combine: inject = 2 * sigmoid(inject(normed) / hc)             [T, hc]
         hyper' = hyper + block_output * inject                 per stream

Prefill-sized calls (`hyper_prefill.ROWS` rows and up) carry `WideNorm` between a combine and the next mix instead
of the BF16 normalized streams: the kernels in `sharar/kernels/hyper_prefill.py` read the streams about half as often.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F


class WideNorm(NamedTuple):
    """A site's normalized streams as `hyper_prefill.combine_norm` leaves them: E4M3 with row scales, 1/rms per stream
    and the site's inject logits (FP32)."""
    nq: torch.Tensor
    xs: torch.Tensor
    rstd: torch.Tensor
    inj: torch.Tensor | None


def grouped_gemma_rms(x: torch.Tensor, weight: torch.Tensor, eps: float, group: int) -> torch.Tensor:
    """RMS statistics per `group` columns, one (1 + w) weight per column, FP32 inside."""
    xf = x.float()
    g = xf.reshape(*xf.shape[:-1], -1, group)
    xn = (g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + eps)).flatten(-2)
    return (xn * (1.0 + weight.float())).to(x.dtype)


class HyperConnection:
    """The three mixer matrices are `TrunkLinear`s: the checkpoint's BF16, or FP8 at load with a
    dynamic activation scale (`precision="fp8"`, the default; "bf16" keeps the checkpoint's)."""

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
    def load(cls, w, prefix: str, hc: int, hidden: int, eps: float, device, combine: bool = True, precision: str = "fp8", fused: bool = True) -> "HyperConnection":
        g = lambda name: w.get(f"{prefix}.{name}").to(device)  # noqa: E731
        return cls(g("hc_norm.weight"), g("input_mix_weight_down.weight"), g("input_mix_weight_up.weight"),
                   g("block_inject_weight.weight") if combine else None, hc, hidden, eps, precision, fused)

    @property
    def bytes(self) -> int:
        return self.norm.numel() * 2 + self.down.bytes + self.up.bytes + (self.inject.bytes if self.inject is not None else 0)

    def _wide(self, rows: int) -> bool:
        from sharar.kernels import hyper_prefill
        return self.fused and rows >= hyper_prefill.ROWS and self.down.scale is not None and self.up.scale is not None

    def _wide_norm(self, hyper: torch.Tensor, block=None, inj=None):
        from sharar.kernels import hyper_prefill
        y, nq, xs, rstd, own = hyper_prefill.combine_norm(hyper, block, inj, self.norm, self.inject.weight if self.inject is not None else None,
                                                          self.hidden, self.eps)
        return y, WideNorm(nq, xs, rstd, own)

    def mix(self, hyper: torch.Tensor, normed=None):
        """`normed`: (normed, block maxima) of `hyper` under this norm, or a `WideNorm`, when the previous `combine`
        computed them."""
        if isinstance(normed, WideNorm) or (normed is None and self._wide(hyper.shape[0])):
            from sharar.kernels import hyper_prefill
            if normed is None:
                _, normed = self._wide_norm(hyper)
            mixed = hyper_prefill.mix(normed.nq, normed.xs, normed.rstd, hyper, self.down, self.up, self.norm, self.hidden)
            return mixed, (hyper, normed)
        if self.fused:
            from sharar.kernels import flashnext_fused as fk
            # The norm writes its block maxima, so the down projection's FP8 quantization is one parallel launch (many
            # rows) or none (the GEMV's own, few rows): the unfused chain's bits, faster.
            normed, parts = normed if normed is not None else fk.grouped_gemma_rms(hyper, self.norm, self.eps, self.hidden, amax=True)
            mixed = fk.gate_mean(self.up(self.down(normed, amax_parts=parts), act=fk.ACT_SILU), normed, self.hc, self.hidden)
            return mixed, (hyper, normed)
        normed = grouped_gemma_rms(hyper, self.norm, self.eps, self.hidden)
        gate = torch.sigmoid(self.up(F.silu(self.down(normed) / self.hc)))
        mixed = (gate.unflatten(-1, (self.hc, self.hidden)) * normed.unflatten(-1, (self.hc, self.hidden))).mean(-2)
        return mixed.to(hyper.dtype), (hyper, normed)

    def combine(self, block_output: torch.Tensor, residuals, then: "HyperConnection | None" = None):
        """hyper'. With `then` (the next hyper-connection, fused), (hyper', (normed, block maxima)) for `then.mix`:
        its norm runs on the combined values in the same pass, so hyper' is not read back."""
        hyper, normed = residuals
        if isinstance(normed, WideNorm):
            if then is not None and then._wide(hyper.shape[0]):
                return then._wide_norm(hyper, block_output, normed.inj)
            from sharar.kernels import flashnext_fused as fk
            out = fk.inject_combine(hyper, block_output, normed.inj, self.hc, self.hidden)
            return (out, None) if then is not None else out
        if self.fused:
            from sharar.kernels import flashnext_fused as fk
            if then is not None and then.fused:
                # the inject GEMV inside the launch where it is cuBLAS's bits (verify rows), else cuBLAS's output
                wi = self.inject.weight
                inj = (normed, wi) if fk.inject_splits(normed.shape[0], wi.shape[1], wi.device) and wi.dtype == torch.bfloat16 else self.inject(normed)
                y, n, parts = fk.inject_combine_norm(hyper, block_output, inj, self.hc, self.hidden, then.norm, then.eps)
                return y, (n, parts)
            out = fk.inject_combine(hyper, block_output, self.inject(normed), self.hc, self.hidden)
            return (out, None) if then is not None else out
        inject = 2 * torch.sigmoid(self.inject(normed) / self.hc)  # [T, hc]
        r = hyper.unflatten(-1, (self.hc, self.hidden))
        out = (r + block_output.unsqueeze(-2) * inject.unsqueeze(-1)).flatten(-2).to(hyper.dtype)
        return (out, None) if then is not None else out
