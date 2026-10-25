"""Native IQ2_XS, IQ2_XXS and Q8_0 linears on W3's SM12x FP8 MMA engine.

Packing reorders encoded bytes without requantization. During GEMM each decoded
integer is represented as the exact sum of two E4M3 values; FP32 scaling follows
each dot product (per K16 for XS, K32 otherwise). BF16 activations use W3's
per-row/K32 quantizer. No expanded weight matrix persists or is built at run time.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import os
from pathlib import Path

import torch

from sharar.kernels import w3
from sharar.weights.ggml import FORMATS, _validate, dequantize
from sharar.weights.iq2_grid import GRID
from sharar.weights.iq2_xs_grid import GRID_XS

_SRC = Path(__file__).resolve().parent / "csrc" / "ggml_gemm.cu"
_CODEC = _SRC.with_name("ggml_codec.cuh")
SMALL_ROWS = (1, 2, 4, 8)
DECODE_ROWS = (1, 2, 4, 8, 16, 32, 64)

# GB10, grouped FP8 inputs; profiling/iq2_w3.py on actual Bolt tensors.
# Entries are (M tile, N tile, split-K). Small rows use the transposed path.
_DECODE_TILES = {
    (2048, 1024): {16: (16, 64, 4), 32: (32, 32, 2), 64: (32, 64, 2)},
    (1024, 2048): {16: (16, 64, 8), 32: (32, 32, 4), 64: (32, 64, 4)},
    (5376, 4096): {16: (16, 64, 4), 32: (32, 128, 4), 64: (32, 128, 4)},
}


def _decode_config(m: int, n: int, k: int, grouped: bool) -> dict:
    """Measured Bolt tiles, with a geometry-based fallback for other shapes."""
    if m in SMALL_ROWS:
        warps_k = 8 if (n, k) == (1024, 2048) and m in (2, 4) else 4
        return dict(tile_m=16, tile_n=16, splits=1, stage_a=False, small_m=True, warps_k=warps_k)
    measured = _DECODE_TILES.get((n, k), {}).get(m)
    if measured is None:
        tile_m, tile_n, splits = (16 if m == 16 else 32), (64 if m == 16 else 128), 1
        blocks = -(-n // tile_n) * -(-m // tile_m)
        while blocks * splits < 192 and splits < 8 and (k // 256) % (2 * splits) == 0:
            splits *= 2
    else:
        tile_m, tile_n, splits = measured
    return dict(tile_m=tile_m, tile_n=tile_n, splits=splits, stage_a=grouped, small_m=False, warps_k=4)


@lru_cache(maxsize=1)
def _module():
    from torch.utils.cpp_extension import load
    from sharar.engine.config import current
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", current().cuda_arch)
    revision = _source_revision()[:16]
    return load(name=f"sharar_ggml_fp8_{revision}", sources=[str(_SRC)],
                extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"], verbose=False)


def _source_revision() -> str:
    return hashlib.sha256(_SRC.read_bytes() + _CODEC.read_bytes()).hexdigest()


@lru_cache(maxsize=None)
def _execution_lut(quant: str, device: torch.device) -> torch.Tensor:
    """IQ2 byte selectors, or two exact E4M3 components per Q8 integer."""
    if quant == "Q8_0":
        values = torch.arange(256).to(torch.uint8).view(torch.int8).float()
    else:
        grid = torch.tensor(GRID_XS if quant == "IQ2_XS" else GRID, dtype=torch.int64)
        codes = (grid == 25).long() + 2 * (grid == 43).long()
        # Eight nibbles select {8,24,44} and {0,1,-1} from register constants.
        # Their sums recover {8,25,43} exactly. One uint32 per eight weights.
        return (codes << (4 * torch.arange(8))).sum(1).to(device=device, dtype=torch.uint32)
    high = values.to(torch.float8_e4m3fn)
    low = (values - high.float()).to(torch.float8_e4m3fn)
    if quant == "Q8_0":
        table = high.view(torch.uint8).int() | (low.view(torch.uint8).int() << 8)
        return table.to(device=device, dtype=torch.uint32)


@dataclass
class GgmlWeight:
    """Owned compressed payload [N/8,K/Block,8,Bytes] and compact metadata.

    The cached execution table is shared by all weights of a format on a device.
    Use ``linear`` with explicit output/workspace buffers or call this object as
    a Linear. Encoded tensors must remain unmodified during execution.
    """
    values: torch.Tensor
    metadata: torch.Tensor
    quant: str
    out_features: int
    in_features: int
    lut: torch.Tensor
    w3: object = None

    @property
    def bytes(self) -> int:
        return self.values.numel() + self.metadata.numel()

    def blocks(self) -> torch.Tensor:
        """Reconstruct native blocks for reference/export, outside execution."""
        block, size = FORMATS[self.quant]
        kb, n = self.in_features // block, self.out_features
        payload = self.values.reshape(n // 8, kb, 8, -1).permute(0, 2, 1, 3).reshape(n, kb, -1)
        meta = self.metadata.permute(1, 0, 2)
        return torch.cat((meta[..., :2], payload, meta[..., 2:]), -1).reshape(n, kb, size)

    def dequantize(self) -> torch.Tensor:
        return dequantize(self.blocks(), self.quant)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self)


def pack_weight(blocks: torch.Tensor, quant: str) -> GgmlWeight:
    """Losslessly retile native uint8 blocks; N must be divisible by eight."""
    block = _validate(blocks, quant)
    n, kb, _ = blocks.shape
    if not n or n % 8:
        raise ValueError("native GGML GEMM requires positive N divisible by eight")
    if blocks.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("GGML weight preparation must precede CUDA graph capture")
    if not torch.isfinite(blocks[..., :2].contiguous().view(torch.float16)).all().item():
        raise ValueError("GGML block bases must be finite")
    payload = 32 if quant == "Q8_0" else 64
    values = blocks[..., 2:2 + payload].reshape(n // 8, 8, kb, payload).permute(0, 2, 1, 3)
    values = values.clone(memory_format=torch.contiguous_format)
    metadata = torch.cat((blocks[..., :2], blocks[..., 2 + payload:]), -1).permute(1, 0, 2).contiguous()
    return GgmlWeight(values, metadata, quant, n, kb * block, _execution_lut(quant, blocks.device))


def linear(x: torch.Tensor, weight: GgmlWeight, x_scale: torch.Tensor | None = None, *,
           splits: int | None = None, tile_n: int | None = None, tile_m: int | None = None, out: torch.Tensor | None = None,
           workspace: torch.Tensor | None = None, stage_a: bool | None = None,
           small_m: bool | None = None, warps_k: int | None = None) -> torch.Tensor:
    """BF16 or scaled E4M3 [...,K] -> BF16 [...,N], for arbitrary row counts.

    E4M3 input scales are FP32: scalar, one per row, or per row/K32 group.
    Split-K scratch is FP32 with at least splits*M*N elements when splits>1.
    tile_n=32/64/128 gives each warp independent output columns.
    stage_a=True double-buffers IQ2_XXS activations and K32 scales in padded
    shared memory; it requires BF16 inputs or grouped E4M3 activation scales.
    small_m=True swaps the MMA operands for IQ2_XXS M=1/2/4/8. warps_k=1/2/4/8
    partitions K within each CTA and reduces locally, with no global split-K.
    With no explicit tuning arguments, IQ2_XXS power-of-two row counts up to
    64 use the decode recipes. Explicit tuning arguments retain manual control.
    Warm the module before capture; caller buffers are private to an execution.
    """
    if not isinstance(weight, GgmlWeight) or x.ndim < 1 or x.shape[-1] != weight.in_features:
        raise ValueError("GGML linear input must end in the packed weight's K")
    if x.device.type != "cuda" or weight.values.device != x.device:
        raise ValueError("GGML linear requires inputs and weights on one CUDA device")
    if torch.cuda.get_device_capability(x.device) not in ((12, 0), (12, 1)):
        raise ValueError("GGML FP8 GEMM requires SM120/SM121")
    if not x.is_contiguous() or x.data_ptr() % 16 or x.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError("GGML linear requires contiguous, 16-byte-aligned BF16 or E4M3 activations")
    k, n = weight.in_features, weight.out_features
    source = x.reshape(-1, k)
    m = source.shape[0]
    automatic = all(value is None for value in (splits, tile_n, tile_m, stage_a, small_m, warps_k))
    if automatic and weight.quant == "IQ2_XXS" and m in DECODE_ROWS:
        grouped = x.dtype == torch.bfloat16 or (x_scale is not None and x_scale.numel() == m * (k // 32))
        config = _decode_config(m, n, k, grouped)
        splits, tile_n, tile_m = config["splits"], config["tile_n"], config["tile_m"]
        stage_a, small_m, warps_k = config["stage_a"], config["small_m"], config["warps_k"]
    splits = 1 if splits is None else splits
    stage_a = False if stage_a is None else stage_a
    small_m = False if small_m is None else small_m
    warps_k = 4 if warps_k is None else warps_k
    if tile_n is None:
        tile_n = (16 if warps_k == 8 else 64 // warps_k) if small_m and warps_k in (1, 2, 4, 8) else 128
    if tile_m is None:
        tile_m = 16 if m <= 16 else 32 if m <= 32 else 64
    if type(tile_m) is not int or tile_m not in (16, 32, 64):
        raise ValueError("GGML row tile must be 16, 32 or 64")
    if type(splits) is not int or splits not in (1, 2, 4, 8) or splits > k // 32:
        raise ValueError("GGML split count must be 1, 2, 4 or 8 and fit K/32")
    if small_m:
        if weight.quant != "IQ2_XXS" or m not in SMALL_ROWS:
            raise ValueError("small-row GEMM requires IQ2_XXS and M=1/2/4/8")
        if splits != 1 or stage_a or warps_k not in (1, 2, 4, 8):
            raise ValueError("small-row GEMM requires splits=1, no staging, and warps_k=1/2/4/8")
        small_tile = 16 if warps_k == 8 else 64 // warps_k
        if type(tile_n) is not int or tile_n not in (small_tile, 2 * small_tile):
            raise ValueError("small-row output tile must match warps_k")
    elif type(tile_n) is not int or tile_n not in (32, 64, 128):
        raise ValueError("GGML output tile must be 32, 64 or 128")
    shape = (*x.shape[:-1], n)
    if out is None:
        out = torch.empty(shape, device=x.device, dtype=torch.bfloat16)
    if out.device != x.device or out.dtype != torch.bfloat16 or out.shape != shape or not out.is_contiguous() or out.data_ptr() % 4:
        raise ValueError("GGML output must be contiguous BF16 with the output shape and device")
    if workspace is None:
        workspace = torch.empty(splits * m * n if splits > 1 else 0, device=x.device, dtype=torch.float32)
    if (workspace.device != x.device or workspace.dtype != torch.float32 or not workspace.is_contiguous() or workspace.data_ptr() % 8
            or (splits > 1 and workspace.numel() < splits * m * n)):
        raise ValueError("GGML workspace must be contiguous FP32 and fit split partials")
    if not m:
        return out
    if x.dtype == torch.bfloat16:
        if x_scale is not None:
            raise ValueError("BF16 inputs are quantized per group and take no activation scale")
        source, x_scale = w3.quant_groups(source)
    elif (x_scale is None or x_scale.dtype != torch.float32 or x_scale.device != x.device
          or not x_scale.is_contiguous() or x_scale.numel() not in (1, m, m * (k // 32))):
        raise ValueError("E4M3 inputs require contiguous FP32 scalar, row or K32-group scales")
    if stage_a and (weight.quant != "IQ2_XXS" or x_scale.numel() != m * (k // 32)):
        raise ValueError("activation staging requires IQ2_XXS and per-row/K32 activation scales")
    reads = (x, weight.values, weight.metadata, weight.lut, x_scale)
    if any(torch._C._overlaps(out, value) or torch._C._overlaps(workspace, value) for value in reads):
        raise ValueError("GGML output/workspace must not overlap inputs")
    if torch._C._overlaps(out, workspace):
        raise ValueError("GGML output and workspace must not overlap")
    if m:
        with torch.cuda.device(x.device):
            _module().ggml_gemm(source, x_scale, weight.values, weight.metadata, weight.lut,
                                out.reshape(m, n), workspace, ("IQ2_XS", "IQ2_XXS", "Q8_0").index(weight.quant),
                                splits, tile_n, tile_m, stage_a, small_m, warps_k)
    return out
