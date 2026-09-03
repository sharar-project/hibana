"""Gemma-style RMSNorm: out = x / rms(x) * (1 + weight), computed in fp32.

The fused Triton kernel (sharar.kernels.fused_norm) does the residual add, the
norm and optionally the FP8 static quantization of the output in one pass. The
torch path is kept as the readable reference for tests.
"""

from __future__ import annotations

import torch

from sharar.kernels.fused_norm import gemma_rmsnorm


class GemmaRMSNorm:
    def __init__(self, weight: torch.Tensor, eps: float):
        self.weight = weight.contiguous()
        # (1 + w) in fp32, matching the reference's native path.
        self.weight_plus_one = 1.0 + weight.float()
        self.eps = eps

    def torch_forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        return (xf * torch.rsqrt(var + self.eps) * self.weight_plus_one).to(x.dtype)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            return gemma_rmsnorm(x.reshape(-1, x.shape[-1]), self.weight, self.eps)[0].view(x.shape)
        return gemma_rmsnorm(x, self.weight, self.eps)[0]

    def fused_add(self, x: torch.Tensor, residual: torch.Tensor, inplace: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """residual <- x + residual (bf16); return (norm(residual), residual)."""
        if not inplace:
            residual = residual.clone()
        out, _ = gemma_rmsnorm(x, self.weight, self.eps, residual=residual)
        return out, residual

    def fused_add_quant(self, x: torch.Tensor, residual: torch.Tensor | None, fp8_scale: torch.Tensor, inplace: bool = False):
        """Like fused_add but also returns the FP8-quantized output. With
        residual None (first layer) the residual becomes x itself."""
        if residual is None:
            out, fp8 = gemma_rmsnorm(x, self.weight, self.eps, fp8_scale=fp8_scale)
            return out, fp8, x
        if not inplace:
            residual = residual.clone()
        out, fp8 = gemma_rmsnorm(x, self.weight, self.eps, residual=residual, fp8_scale=fp8_scale)
        return out, fp8, residual
