"""How the Gated DeltaNet recurrent state is stored in a `StatePool` slot.

`fp32` (the default) keeps each layer's state as the kernels compute it: [HV, V, K] FP32 per slot, 3 MiB per
layer on both targets. `int8` and `int4` are LeapQuant (arXiv 2609.38166): per-window recurrent-state
quantization to 8 or 4 bits.

A quantized layer state is a boundary state plus the committed tokens since it:

    S = Gamma_{1:n} * (C . dequant(Z) . B + K~ U~^T) + sum_j Gamma_{j+1:n} k_j u_j^T

- `z`, `scale`, `smooth`: the integer residual Z with a per-value-row scale B (amax / 127, or amax / 7 at 4 bits)
  after dividing each key column by its smoothing factor C (the square root of the column's mean magnitude);
  4 bits pack two values per byte, key columns 2i and 2i+1 in the low and high nibble, offset by 8;
- `comp_k`, `comp_u`: `rank` compensator tokens, an FP16 rank-r fit of the boundary state's largest structure
  (power iteration) that keeps the outliers out of the integer range;
- `log_k`, `log_u`, `log_g`: the window's committed tokens as delta-rule records (L2-normalized key, the
  correction u = beta (v - (alpha S)^T k), the log decay), FP16/FP32, at most `window` of them;
- `log_len`: the number of records, per value head.

Verification reads the state through this expansion and never writes it. A commit appends the accepted tokens'
records; when the window fills, it folds them into a new boundary state, refits the compensators and quantizes
once. Quantization error therefore enters once per `window` committed tokens, not once per token.

One byte row per (layer, slot) holds the whole record, so code that clears, copies, stores or restores a slot
needs no format knowledge, and a zeroed row is the zero state. The kernels live in `sharar/kernels/ssm_window.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import torch

ALIGN = 256
BITS = {"int8": 8, "int4": 4}


def _aligned(nbytes: int) -> int:
    return -(-nbytes // ALIGN) * ALIGN


@dataclass(frozen=True)
class WindowLayout:
    """Byte layout of one quantized (layer, slot) row."""

    key_heads: int
    value_heads: int
    k: int
    v: int
    window: int
    rank: int
    bits: int = 8

    @property
    def regions(self) -> tuple[tuple[str, torch.dtype, tuple[int, ...]], ...]:
        h, hv, k, v, p, r = self.key_heads, self.value_heads, self.k, self.v, self.window, self.rank
        return (
            ("z", torch.int8, (hv, v, k)) if self.bits == 8 else ("z", torch.uint8, (hv, v, k // 2)),
            ("scale", torch.float32, (hv, v)),
            ("smooth", torch.float32, (hv, k)),
            ("comp_k", torch.float16, (hv, r, k)),
            ("comp_u", torch.float16, (hv, r, v)),
            ("log_k", torch.float16, (h, p, k)),
            ("log_u", torch.float16, (hv, p, v)),
            ("log_g", torch.float32, (hv, p)),
            ("log_len", torch.int32, (hv,)),
        )

    @property
    def offsets(self) -> dict[str, int]:
        out, cursor = {}, 0
        for name, dtype, shape in self.regions:
            out[name] = cursor
            cursor += _aligned(prod(shape) * dtype.itemsize)
        return out

    @property
    def nbytes(self) -> int:
        return sum(_aligned(prod(shape) * dtype.itemsize) for _, dtype, shape in self.regions)

    def views(self, raw: torch.Tensor) -> dict[str, torch.Tensor]:
        """Typed views of every region over a [..., nbytes] uint8 tensor (leading dimensions kept)."""
        if raw.dtype != torch.uint8 or raw.shape[-1] != self.nbytes or raw.stride(-1) != 1:
            raise ValueError("window views need a [..., nbytes] uint8 tensor with contiguous rows")
        offsets = self.offsets
        out = {}
        for name, dtype, shape in self.regions:
            start = offsets[name] // dtype.itemsize
            out[name] = raw.view(dtype)[..., start:start + prod(shape)].unflatten(-1, shape)
        return out


@dataclass(frozen=True)
class SsmFormat:
    kind: str = "fp32"
    window: int = 16  # committed tokens per fold into the quantized boundary state
    rank: int | None = None  # compensator tokens (None: the paper's 4, or 8 at 4 bits)

    def __post_init__(self):
        if self.kind not in ("fp32", *BITS):
            raise ValueError(f"SHARAR_SSM_STATE must be fp32, int8 or int4, not {self.kind!r}")
        if self.rank is None:
            object.__setattr__(self, "rank", 8 if self.kind == "int4" else 4)
        if self.window not in (8, 16, 32):
            raise ValueError("SHARAR_SSM_WINDOW must be 8, 16 or 32")
        if not 0 <= self.rank <= 16:
            raise ValueError("SHARAR_SSM_COMPENSATORS must be in 0..16")

    @classmethod
    def from_config(cls, cfg) -> SsmFormat:
        return cls(cfg.ssm_state, cfg.ssm_window, cfg.ssm_compensators)

    @property
    def quantized(self) -> bool:
        return self.kind != "fp32"

    def describe(self) -> str:
        return f"{self.kind}:w{self.window}:r{self.rank}" if self.quantized else "fp32"

    def layout(self, target) -> WindowLayout | None:
        if not self.quantized:
            return None
        return WindowLayout(target.linear_num_key_heads, target.linear_num_value_heads,
                            target.linear_key_head_dim, target.linear_value_head_dim, self.window, self.rank,
                            BITS[self.kind])

    def state_bytes_per_slot(self, target) -> int:
        """Conv window plus recurrent state over every GDN layer."""
        layout = self.layout(target)
        if layout is None:
            return target.state_bytes_per_slot
        conv = target.conv_dim * target.conv_state_len * 2
        return (conv + layout.nbytes) * len(target.gdn_layer_ids)


FP32 = SsmFormat()


class WindowPool:
    """Typed views of a quantized pool's [layers, slots, nbytes] rows: `all[name]` [layers, slots, ...] for the commit
    over every layer, `layer[lid][name]` one layer's [slots, ...], and `single[lid][name]` that layer as
    [1, slots, ...] for decode's per-layer commit."""

    def __init__(self, layout: WindowLayout, raw: torch.Tensor, layer_ids: list[int]):
        self.layout = layout
        self.all = layout.views(raw)
        self.layer = {lid: {name: t[j] for name, t in self.all.items()} for j, lid in enumerate(layer_ids)}
        self.single = {lid: {name: t[j:j + 1] for name, t in self.all.items()} for j, lid in enumerate(layer_ids)}
