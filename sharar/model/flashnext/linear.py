# SPDX-License-Identifier: Apache-2.0
"""Trunk projections. The checkpoint ships them in BF16; the engine keeps an FP8 copy with one
weight scale and quantizes activations per call (the draft's dynamic scheme), or the BF16
matrix itself for parity runs. A 3-bit decode copy can be attached later like the draft's.

SHARAR_TRUNK_KEEP_BF16=<regex> keeps the matching projections in BF16 (a diagnostic: which
module's quantization moves the logits, `bench/flashnext_quant.py`). Finer FP8 scales (per
row, per output channel) were measured and do not beat numerical noise (#5)."""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch

FP8_MAX = 448.0
GEMV_MAX_ROWS = 4


@dataclass
class TrunkLinear:
    weight: torch.Tensor  # [N, K] bf16, or float8_e4m3fn with `scale`
    scale: torch.Tensor | None = None  # [1, 1] fp32 weight scale (fp8 only)
    w3: object = None  # optional 3-bit decode copy (sharar.kernels.w3.W3Weight)

    @classmethod
    def from_bf16(cls, weight: torch.Tensor, precision: str) -> "TrunkLinear":
        if weight.ndim != 2 or weight.dtype != torch.bfloat16:
            raise ValueError("trunk projections are BF16 matrices in the checkpoint")
        if precision == "bf16":
            return cls(weight.contiguous())
        if precision != "fp8":
            raise ValueError(f"unknown trunk precision {precision!r}")
        scale = (weight.float().abs().amax() / FP8_MAX).clamp_min(1e-12).reshape(1, 1)
        packed = (weight.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        return cls(packed.contiguous(), scale)

    @classmethod
    def load(cls, w, names: list[str], device, precision: str) -> "TrunkLinear":
        """One projection, or several fused along the output dimension."""
        parts = [w.get(n).to(device) for n in names]
        weight = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        from sharar.engine.config import current
        keep = current().trunk_keep_bf16
        if keep and re.search(keep, names[0]):
            precision = "bf16"
        return cls.from_bf16(weight, precision)

    @property
    def out_features(self) -> int:
        return self.weight.shape[0]

    @property
    def in_features(self) -> int:
        return self.weight.shape[1]

    @property
    def bytes(self) -> int:
        return self.w3.bytes if self.w3 is not None else self.weight.numel() * self.weight.element_size()

    def dequantize(self) -> torch.Tensor:
        return self.weight.float() * self.scale.float() if self.scale is not None else self.weight.float()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1])
        if self.scale is None:
            out = flat @ self.weight.T
        else:
            flat = flat.contiguous()
            if self.w3 is not None:
                from sharar.kernels import w3
                if flat.shape[0] <= w3.MAX_ROWS:
                    return w3.w3_linear(flat, self.w3).reshape(*x.shape[:-1], self.out_features)
            k = flat.shape[1]
            if flat.numel() <= 65536:  # decode rows: one-kernel quantization that also emits alpha
                from sharar.kernels.flashnext_fused import quant_fp8_small
                x_q, x_scale, alpha = quant_fp8_small(flat, self.scale)
            else:
                from sharar.kernels.draft_fp8 import quant_dynamic
                x_q, x_scale = quant_dynamic(flat)
                alpha = None
            if flat.shape[0] <= GEMV_MAX_ROWS and k % 512 == 0 and k <= 8192:  # the CUDA GEMV kernel's shape rule
                from sharar.kernels.fp8_gemv_cuda import fp8_gemv_cuda
                out = fp8_gemv_cuda(x_q, self.weight, alpha if alpha is not None else (x_scale * self.scale).reshape(1))
            elif flat.shape[0] <= GEMV_MAX_ROWS and k % 16 == 0:  # other widths (mixers K=320/10240, shared down K=640)
                from sharar.kernels.flashnext_fused import fp8_gemv_any
                out = fp8_gemv_any(x_q, self.weight, alpha if alpha is not None else (x_scale * self.scale).reshape(1))
            else:
                out = torch._scaled_mm(x_q, self.weight.T, scale_a=x_scale, scale_b=self.scale, out_dtype=torch.bfloat16)
        return out.reshape(*x.shape[:-1], self.out_features)
