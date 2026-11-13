"""The linear-layer interface every weight format implements.

Formats: `Fp8Linear` (static activation scale) and `Nvfp4Linear` from the
ModelOpt release (the 27B), `TrunkLinear` (FP8 at load with a dynamic
activation scale, or BF16: Flash-Next), `DraftFp8Linear` (the parallel
draft's dynamic FP8). Each keeps its own row-count dispatch, which sets its
arithmetic; what they share is the surface quantization and calibration work
on, so a 3-bit copy, a calibration pass or a draft-only head is written once:

- `lin(x)` for BF16 rows `x` [..., K] -> [..., N];
- `out_features`, `in_features`, `bytes` (resident weight bytes);
- `dequantize()`: the served weights as FP32 [N, K], what a decode copy is packed from;
- `w3`: the optional 3-bit decode copy (`kernels/w3.py`), used for calls of at most 64 rows.

A model lists its projections with `linears()` -> iterable of (name, linear).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class Linear(Protocol):
    w3: object

    @property
    def out_features(self) -> int: ...

    @property
    def in_features(self) -> int: ...

    @property
    def bytes(self) -> int: ...

    def dequantize(self) -> torch.Tensor: ...

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...
