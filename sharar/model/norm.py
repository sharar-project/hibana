"""Gemma-style RMSNorm: out = x / rms(x) * (1 + weight), computed in fp32.

The native CUDA norm preserves the FP32 residual sum and reduction order used
by this checkpoint's reference runtime. FP8 quantization consumes its BF16
output. Verification preserves the pinned compiled reference's separate
variance/numerator precision in kernels.verify_norm. The torch path is kept
as a readable numerical reference for tests.
"""

from __future__ import annotations

import torch

def gemma_rmsnorm(x, weight, eps, residual=None, fp8_scale=None, *, verify=False, consume_input=False):
    from flashinfer.norm import gemma_rmsnorm as native_norm, gemma_fused_add_rmsnorm
    from sharar.kernels.fp8_quant import quant_fp8

    if residual is None:
        if verify:
            from sharar.kernels.verify_norm import verify_input_norm
            out = verify_input_norm(x, weight, eps)
        else:
            out = native_norm(x, weight, eps)
    elif verify:
        from sharar.kernels.verify_norm import verify_add_norm
        out = verify_add_norm(x, residual, weight, eps)
    else:
        # consume_input: the caller no longer reads x, so the norm overwrites it instead of a copy (prefill: two
        # copies of a [rows, hidden] BF16 tensor per layer). The arithmetic is the same.
        out = x if consume_input and x.is_contiguous() else x.clone()
        gemma_fused_add_rmsnorm(out, residual, weight, eps)
    return out, quant_fp8(out, fp8_scale) if fp8_scale is not None else None


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

    def fused_add(self, x: torch.Tensor, residual: torch.Tensor, inplace: bool = False, *, verify: bool = False,
                  consume_input: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize the FP32 sum, with the compiled numerator when verifying; consume_input: x may be overwritten."""
        if not inplace:
            residual = residual.clone()
        out, _ = gemma_rmsnorm(x, self.weight, self.eps, residual=residual, verify=verify, consume_input=consume_input)
        return out, residual

    def fused_add_quant(self, x: torch.Tensor, residual: torch.Tensor | None, fp8_scale: torch.Tensor, inplace: bool = False, *,
                        verify: bool = False, consume_input: bool = False):
        """Like fused_add but also returns the FP8-quantized output. With
        residual None (first layer) the residual becomes x itself."""
        if residual is None:
            out, fp8 = gemma_rmsnorm(x, self.weight, self.eps, fp8_scale=fp8_scale, verify=verify)
            return out, fp8, x
        if not inplace:
            residual = residual.clone()
        out, fp8 = gemma_rmsnorm(x, self.weight, self.eps, residual=residual, fp8_scale=fp8_scale, verify=verify,
                                 consume_input=consume_input)
        return out, fp8, residual
