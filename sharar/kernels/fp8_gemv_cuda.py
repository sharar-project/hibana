"""CUDA small-M FP8 GEMV (see csrc/fp8_gemv.cu), compiled on first use.

The build is cached by torch under $HOME/.cache/torch_extensions; in the dev
container HOME lives on the results volume, so it persists. Phase 9 replaces
this JIT step with a prebuilt artifact.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parent / "csrc" / "fp8_gemv.cu"


@lru_cache(maxsize=1)
def _module():
    from torch.utils.cpp_extension import load

    arch = os.environ.get("SHARAR_CUDA_ARCH", "12.1a")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", arch)
    return load(
        name="sharar_fp8_gemv",
        sources=[str(_SRC)],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )


def fp8_gemv_cuda(x_q: torch.Tensor, w_q: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """x_q: [M<=4, K] fp8, w_q: [N, K] fp8, alpha: [] or [1] fp32 -> [M, N] bf16."""
    return _module().fp8_gemv(x_q.contiguous(), w_q, alpha.reshape(1).contiguous())
