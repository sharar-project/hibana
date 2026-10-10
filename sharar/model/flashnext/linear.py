# SPDX-License-Identifier: Apache-2.0
"""Trunk projections. The checkpoint ships them in BF16; the engine keeps an FP8 copy with one
weight scale and quantizes activations per call (the draft's dynamic scheme), or the BF16
matrix itself for parity runs. A 3-bit decode copy can be attached later like the draft's.

SHARAR_TRUNK_KEEP_BF16=<regex> keeps the matching projections in BF16 (a diagnostic: which
module's quantization moves the logits, `profiling/flashnext_quant.py`). Finer FP8 scales (per
row, per output channel) were measured and do not beat numerical noise (#5)."""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch

from sharar.weights.sharding import REPLICATED, Shard

FP8_MAX = 448.0
GEMV_MAX_ROWS = 4


@dataclass
class TrunkLinear:
    weight: torch.Tensor  # [N, K] bf16, or float8_e4m3fn with `scale`
    scale: torch.Tensor | None = None  # [1, 1] fp32 weight scale (fp8 only)
    w3: object = None  # optional 3-bit decode copy (sharar.kernels.w3.W3Weight)
    shard: Shard = REPLICATED  # which part of the checkpoint's matrix this rank holds (its 3-bit copy loads the same)

    @classmethod
    def from_bf16(cls, weight: torch.Tensor, precision: str, shard: Shard = REPLICATED) -> "TrunkLinear":
        """`weight` is the checkpoint's whole matrix; this rank keeps its `shard`, quantized with the whole matrix's
        FP8 scale, so every rank's part is exactly its rows or columns of the single-GPU copy."""
        if weight.ndim != 2 or weight.dtype != torch.bfloat16:
            raise ValueError("trunk projections are BF16 matrices in the checkpoint")
        if precision == "bf16":
            return cls(shard(weight).contiguous(), shard=shard)
        if precision != "fp8":
            raise ValueError(f"unknown trunk precision {precision!r}")
        scale = (weight.float().abs().amax() / FP8_MAX).clamp_min(1e-12).reshape(1, 1)
        packed = (shard(weight).float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        return cls(packed.contiguous(), scale, shard=shard)

    @classmethod
    def load(cls, w, names: list[str], device, precision: str, shard: Shard = REPLICATED) -> "TrunkLinear":
        """One projection, or several fused along the output dimension; `shard` keeps this rank's part."""
        parts = [w.get(n).to(device) for n in names]
        weight = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        from sharar.engine.config import current
        keep = current().trunk_keep_bf16
        if keep and re.search(keep, names[0]):
            precision = "bf16"
        return cls.from_bf16(weight, precision, shard)

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

    def _uses_w3(self, rows: int) -> bool:
        if self.w3 is None:
            return False
        from sharar.kernels import w3
        return rows <= w3.MAX_ROWS

    def __call__(self, x: torch.Tensor, act: int = 0, amax_parts: torch.Tensor | None = None) -> torch.Tensor:
        """y = act(x) @ W^T. `act` (`flashnext_fused.ACT_*`) applies SiLU, or SiLU-mul to a [gate | up] input of
        twice the width, inside the FP8 quantization where it can (the BF16 activation is then never written);
        `amax_parts`: per-block maxima of |x| its producer computed (`grouped_gemma_rms(..., amax=True)`), which
        let the FP8 path quantize many rows in parallel. Either way the result is bitwise the unfused chain's."""
        from sharar.kernels import flashnext_fused as fk
        flat = x.reshape(-1, x.shape[-1])
        rows = flat.shape[0]
        k = self.in_features
        if self.scale is not None and act == fk.ACT_SILU_MUL and self._uses_w3(rows) and k % 32 == 0:
            # SiLU-mul straight into the 3-bit GEMM's grouped E4M3 input (one launch; the bits of silu_mul + quant_groups)
            from sharar.kernels import w3
            x_q, x_scale = fk.silu_mul_quant_groups(flat.contiguous())
            return w3.w3_linear(x_q, self.w3, x_scale).reshape(*x.shape[:-1], self.out_features)
        if self.scale is None or self._uses_w3(rows):
            if act == fk.ACT_SILU:
                flat = torch.nn.functional.silu(flat)
            elif act == fk.ACT_SILU_MUL:
                flat = fk.silu_mul(flat)
            if self.scale is None:
                out = flat @ self.weight.T
            else:
                from sharar.kernels import w3
                out = w3.w3_linear(flat.contiguous(), self.w3)
            return out.reshape(*x.shape[:-1], self.out_features)
        flat = flat.contiguous()
        if rows <= GEMV_MAX_ROWS and k % 512 == 0 and k <= 8192:  # the CUDA GEMV kernel's shape rule
            # decode rows: the GEMV quantizes its input itself (one launch; the bits of quant_fp8_small + fp8_gemv_cuda)
            from sharar.kernels.fp8_gemv_cuda import fp8_gemv_quant_cuda
            if act == fk.ACT_SILU:
                flat = torch.nn.functional.silu(flat)
            elif act == fk.ACT_SILU_MUL:
                flat = fk.silu_mul(flat)
            out = fp8_gemv_quant_cuda(flat, self.weight, self.scale)
        elif rows <= GEMV_MAX_ROWS and k % 16 == 0:  # other widths (mixers K=320/10240, shared down K=640)
            if act == fk.ACT_SILU:  # computed once here: per GEMV program, the precise SiLU costs more than a launch
                flat, act = torch.nn.functional.silu(flat), fk.ACT_NONE
            out = fk.fp8_gemv_quant_any(flat, self.weight, self.scale, act, amax_parts=amax_parts if act == fk.ACT_NONE else None)
        else:
            if act == fk.ACT_SILU:
                flat = torch.nn.functional.silu(flat)
            elif act == fk.ACT_SILU_MUL:
                flat = fk.silu_mul(flat)
            if amax_parts is not None and act == fk.ACT_NONE:  # the producer's block maxima: a parallel quantization
                x_q, x_scale = fk.quant_fp8_parts(flat, amax_parts)
            elif flat.numel() <= 65536:  # decode rows: one-kernel quantization
                x_q, x_scale, _ = fk.quant_fp8_small(flat, self.scale)
            else:
                from sharar.kernels.draft_fp8 import quant_dynamic
                x_q, x_scale = quant_dynamic(flat)
            out = torch._scaled_mm(x_q, self.weight.T, scale_a=x_scale, scale_b=self.scale, out_dtype=torch.bfloat16)
        return out.reshape(*x.shape[:-1], self.out_features)
