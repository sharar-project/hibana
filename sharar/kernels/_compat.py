"""Stand-ins for the SGLang helpers the vendored kernels call.

Sharar targets one CUDA device, so every platform probe collapses to a constant.
"""

from __future__ import annotations

import contextlib
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def is_arch_support_pdl() -> bool:
    # Programmatic dependent launch exists from Hopper on; GB10 is SM 12.1.
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 9


# SGLang exposes this as a (major, minor) tuple, and the fla utils compare it.
torch_release = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length() if n > 0 else 1


@contextlib.contextmanager
def device_context(device: torch.device):
    with torch.cuda.device(device.index):
        yield
