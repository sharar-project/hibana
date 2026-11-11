"""CUDA small-M FP8 GEMV (see csrc/fp8_gemv.cu), compiled on first use.

The build is cached by torch under $HOME/.cache/torch_extensions, so it persists across restarts.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parent / "csrc" / "fp8_gemv.cu"


@lru_cache(maxsize=1)
def _module():
    from torch.utils.cpp_extension import load

    from hibana.engine.config import current
    arch = current().cuda_arch
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", arch)
    # One build per source revision (as kernels/w3.py): checkouts of different revisions share the build cache.
    revision = hashlib.sha256(_SRC.read_bytes()).hexdigest()[:16]
    return load(
        name=f"sharar_fp8_gemv_{revision}",
        sources=[str(_SRC)],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )


def fp8_gemv_cuda(x_q: torch.Tensor, w_q: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """x_q: [M<=4, K] fp8, w_q: [N, K] fp8, alpha: [] or [1] fp32 -> [M, N] bf16."""
    return _module().fp8_gemv(x_q.contiguous(), w_q, alpha.reshape(1).contiguous())


def fp8_gemv_quant_cuda(x: torch.Tensor, w_q: torch.Tensor, w_scale: torch.Tensor, scale_out: torch.Tensor | None = None) -> torch.Tensor:
    """x: BF16 [M<=4, K], quantized per tensor inside the GEMV exactly as `flashnext_fused.quant_fp8_small` quantizes
    it (one launch instead of two, the same bits); w_q: [N, K] fp8 with its [1, 1] fp32 scale -> [M, N] bf16.
    `scale_out` ([1] fp32) receives the activation scale when given."""
    return _module().fp8_gemv_quant(x.contiguous(), w_q, w_scale.reshape(1).contiguous(), scale_out)
