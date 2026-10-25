# Copyright (c) 2023-2026 The ggml authors
# SPDX-License-Identifier: MIT
"""Reference readers for GGML blocks stored in Bolt's safetensors weights.

Q8_0 is [rows, K/32, 34] bytes; IQ2_XS/XXS use [rows, K/256, 74/66]. These
are GGML encodings even though the enclosing file is not GGUF. Decoding
materializes a matrix: this is for bring-up, not a packed GEMM kernel.
The IQ2 table and layout follow ggml-org/llama.cpp at
d327bf1f0780f87b8c01b367b61853b0c291d7c4 (MIT; THIRD_PARTY_NOTICES.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
from torch.nn import functional as F

from sharar.weights.iq2_grid import GRID
from sharar.weights.iq2_xs_grid import GRID_XS

FORMATS = {"Q8_0": (32, 34), "IQ2_XS": (256, 74), "IQ2_XXS": (256, 66)}


def _validate(packed: torch.Tensor, quant: str) -> int:
    if quant not in FORMATS:
        raise ValueError(f"unsupported GGML encoding {quant!r}")
    block, payload = FORMATS[quant]
    if packed.dtype != torch.uint8 or packed.ndim != 3 or packed.shape[-1] != payload or not packed.shape[1]:
        raise ValueError(f"{quant} needs uint8 [rows, K/{block}, {payload}] blocks")
    return block


@lru_cache(maxsize=None)
def _tables(device: torch.device, quant: str):
    grid = torch.tensor(GRID_XS if quant == "IQ2_XS" else GRID, dtype=torch.float32, device=device)
    sign_codes = [i | ((i.bit_count() & 1) << 7) for i in range(128)]
    signs = torch.tensor([[1 - 2 * ((code >> bit) & 1) for bit in range(8)] for code in sign_codes],
                         dtype=torch.float32, device=device)
    return grid, signs


def dequantize(packed: torch.Tensor, quant: str) -> torch.Tensor:
    """Packed rows -> FP32 [rows,K] on the same device; no reinterpretation as affine INT2."""
    block = _validate(packed, quant)
    rows = packed.shape[0]
    blocks = packed.reshape(-1, packed.shape[-1])
    scales = blocks[:, :2].contiguous().view(torch.float16).float()
    if quant == "Q8_0":
        values = blocks[:, 2:].contiguous().view(torch.int8).float()
        return (scales * values).reshape(rows, packed.shape[1] * block)

    grid, signs = _tables(packed.device, quant)
    if quant == "IQ2_XS":
        descriptors = blocks[:, 2:66].contiguous().view(torch.uint16).long()
        pairs = blocks[:, 66:].long()
        subscales = torch.stack((pairs & 15, pairs >> 4), -1).flatten(-2)
        factors = scales * (subscales + 0.5) * 0.25
        values = grid[descriptors & 511] * signs[descriptors >> 9]
        return (values * factors.repeat_interleave(2, -1)[..., None]).reshape(rows, -1)

    # Each 256-weight block has eight 32-weight subblocks. Their first four
    # bytes index eight-element grids; the last uint32 packs four seven-bit
    # sign indices and a four-bit subscale. The eighth sign is parity.
    payload = blocks[:, 2:].reshape(-1, 8, 8).long()
    control = sum(payload[..., 4 + byte] << (8 * byte) for byte in range(4))
    magnitudes = grid[payload[..., :4]]
    shifts = torch.arange(4, device=packed.device) * 7
    sign_ids = (control[..., None] >> shifts) & 127
    subscales = scales[:, :, None, None] * (0.5 + (control >> 28).float()[..., None, None]) * 0.25
    return (subscales * magnitudes * signs[sign_ids]).reshape(rows, packed.shape[1] * block)


@dataclass
class GgmlLinear:
    """Linear interface for reference arithmetic; only packed weights persist.

    Every call expands the matrix transiently. This does not provide the memory
    or latency behavior of native IQ2/Q8 kernels and does not enable serving.
    """
    weight: torch.Tensor
    quant: str
    w3: object = None

    def __post_init__(self):
        _validate(self.weight, self.quant)

    @property
    def out_features(self) -> int:
        return self.weight.shape[0]

    @property
    def in_features(self) -> int:
        return self.weight.shape[1] * FORMATS[self.quant][0]

    @property
    def bytes(self) -> int:
        return self.weight.numel()

    def dequantize(self) -> torch.Tensor:
        return dequantize(self.weight, self.quant)

    def prepare(self):
        """Return a compact native FP8-compute linear; preparation precedes graph capture."""
        from sharar.kernels.ggml import pack_weight
        return pack_weight(self.weight, self.quant)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.dequantize().to(x.dtype))
