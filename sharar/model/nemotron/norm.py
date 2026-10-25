# SPDX-License-Identifier: Apache-2.0
"""Nemotron-H's RMSNorm: out = x / rms(x) * weight (a plain weight, not Gemma's 1 + w), FP32 reduction.

The same interface as `model/norm.GemmaRMSNorm` on FlashInfer's plain kernels: the residual sum stays in the residual
tensor, and the FP8 copy the next projection reads comes from the BF16 output.

The residual forms run as one launch (`kernels/cute/nemotron_norm.py`): FlashInfer's fused add + RMS norm with the
outputs written where the caller wants them, so a call that keeps its input or the old residual makes no clone,
and with the FP8 copy fused in. Bitwise the clone + `fused_add_rmsnorm` + `quant_fp8` chain it replaces, which
serves the shapes the fused kernel does not take (`nemotron_norm.supported`).
"""

from __future__ import annotations

import torch


def _fused_ok(x: torch.Tensor, residual: torch.Tensor) -> bool:
    from sharar.kernels.cute import nemotron_norm
    return nemotron_norm.supported(x, residual)


def rmsnorm(x, weight, eps, residual=None, fp8_scale=None, *, consume_input=False, residual_out=None, placeholders=None):
    """norm(x) (residual None) or norm(x + residual) with the sum in `residual_out` (default: residual itself), and
    the FP8 copy at `fp8_scale` when given. consume_input: x may be overwritten by the output."""
    from flashinfer.norm import fused_add_rmsnorm, rmsnorm as native_norm
    from sharar.kernels.fp8_quant import quant_fp8

    if residual is None:
        out = native_norm(x, weight, eps)
        return out, quant_fp8(out, fp8_scale) if fp8_scale is not None else None
    if residual_out is None:
        residual_out = residual
    if _fused_ok(x, residual):
        from sharar.kernels.cute.nemotron_norm import residual_norm
        out = x if consume_input else torch.empty_like(x)
        q = residual_norm(x, residual, weight, eps, out, residual_out, fp8_scale, placeholders)
        return out, q
    if residual_out is not residual:
        residual_out.copy_(residual)  # the clone the in-place kernel works on
    out = x if consume_input and x.is_contiguous() else x.clone()
    fused_add_rmsnorm(out, residual_out, weight, eps)  # residual += out; out = norm(residual)
    return out, quant_fp8(out, fp8_scale) if fp8_scale is not None else None


class RMSNorm:
    def __init__(self, weight: torch.Tensor, eps: float):
        self.weight = weight.contiguous()
        self.eps = eps
        # the FP8 output and scale arguments of a call without the FP8 copy (never read; made here, not under capture)
        self._placeholders = (torch.empty(1, dtype=torch.float8_e4m3fn, device=weight.device),
                              torch.ones(1, dtype=torch.float32, device=weight.device))

    @property
    def bytes(self) -> int:
        return self.weight.numel() * self.weight.element_size()

    def torch_forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm(x.reshape(-1, x.shape[-1]), self.weight, self.eps)[0].view(x.shape)

    @staticmethod
    def _sum_out(residual: torch.Tensor, inplace: bool) -> torch.Tensor:
        """Where the new residual goes: residual itself, or (not inplace) a new tensor."""
        return residual if inplace else torch.empty_like(residual)

    def fused_add(self, x: torch.Tensor, residual: torch.Tensor, inplace: bool = False, *, verify: bool = False,
                  consume_input: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """norm(x + residual) and the new residual; consume_input: x may be overwritten."""
        out_res = self._sum_out(residual, inplace)
        out, _ = rmsnorm(x, self.weight, self.eps, residual=residual, consume_input=consume_input, residual_out=out_res,
                         placeholders=self._placeholders)
        return out, out_res

    def fused_add_quant(self, x: torch.Tensor, residual: torch.Tensor | None, fp8_scale: torch.Tensor | None, inplace: bool = False, *,
                        verify: bool = False, consume_input: bool = False):
        """Like fused_add, also returning the FP8 copy at `fp8_scale` (None: no copy). With residual None (the first
        block) the residual becomes x itself."""
        if residual is None:
            out, fp8 = rmsnorm(x, self.weight, self.eps, fp8_scale=fp8_scale)
            return out, fp8, x
        out_res = self._sum_out(residual, inplace)
        out, fp8 = rmsnorm(x, self.weight, self.eps, residual=residual, fp8_scale=fp8_scale, consume_input=consume_input,
                           residual_out=out_res, placeholders=self._placeholders)
        return out, fp8, out_res
