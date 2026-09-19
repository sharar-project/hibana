"""Device-side wait on a pinned host flag (csrc/wait_flag.cu), compiled on first use like fp8_gemv_cuda."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parent / "csrc" / "wait_flag.cu"


@lru_cache(maxsize=1)
def _module():
    from torch.utils.cpp_extension import load

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", os.environ.get("SHARAR_CUDA_ARCH", "12.1a"))
    return load(name="sharar_wait_flag", sources=[str(_SRC)], extra_cuda_cflags=["-O3"], verbose=False)


def wait_flag(flag: torch.Tensor, value: int, error: torch.Tensor, timeout_ms: float = 2000.0) -> None:
    """Enqueue (or capture) a kernel that blocks the stream until pinned int32 `flag` >= value."""
    _module().wait_flag(flag, int(value), error, int(timeout_ms * 1e6))
