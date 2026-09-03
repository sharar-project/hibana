"""Gemma-style RMSNorm: out = x / rms(x) * (1 + weight), computed in fp32."""

from __future__ import annotations

import torch


class GemmaRMSNorm:
    def __init__(self, weight: torch.Tensor, eps: float):
        # (1 + w) in fp32, matching the reference's native path.
        self.weight_plus_one = 1.0 + weight.float()
        self.eps = eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        return (xf * torch.rsqrt(var + self.eps) * self.weight_plus_one).to(x.dtype)

    def fused_add(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """residual <- x + residual (bf16); return (norm(residual), residual)."""
        residual = x + residual
        return self(residual), residual
