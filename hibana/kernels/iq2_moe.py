"""Grouped native IQ2 experts: up -> BF16 ReLU² -> grouped FP8 -> down.

Four CUDA launches independent of the number of selected experts. Preparation
builds a stable expert/pair worklist and quantizes inputs. The up GEMM fuses
ReLU² and intermediate quantization; down writes FP32 pairs, then combination
sums in router selection order. No expanded weight matrix is constructed.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import os
from pathlib import Path

import torch

from hibana.kernels import ggml
from hibana.weights.ggml import FORMATS, _validate

_SRC = Path(__file__).resolve().parent / "csrc" / "iq2_moe.cu"
_CODEC = ggml._CODEC
FORMATS_IQ2 = ("IQ2_XS", "IQ2_XXS")


def _source_revision():
    return hashlib.sha256(_SRC.read_bytes() + _CODEC.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def _module():
    from torch.utils.cpp_extension import load
    from hibana.engine.config import current
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", current().cuda_arch)
    return load(name=f"sharar_iq2_moe_{_source_revision()[:16]}", sources=[str(_SRC)],
                extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"], verbose=False)


@dataclass
class Iq2Experts:
    up_values: torch.Tensor
    up_metadata: torch.Tensor
    down_values: torch.Tensor
    down_metadata: torch.Tensor
    quant: str
    hidden: int
    intermediate: int
    lut: torch.Tensor
    base: int = 0

    @property
    def experts(self):
        return self.up_values.shape[0]

    @property
    def bytes(self):
        return sum(t.numel() for t in self.tensors())

    def tensors(self):
        return (self.up_values, self.up_metadata, self.down_values, self.down_metadata)

    def matrix(self, expert: int, up: bool) -> ggml.GgmlWeight:
        """One local expert as a dense packed weight, without copying."""
        if not 0 <= expert < self.experts:
            raise IndexError("local expert index out of range")
        values, metadata = (self.up_values, self.up_metadata) if up else (self.down_values, self.down_metadata)
        n, k = (self.intermediate, self.hidden) if up else (self.hidden, self.intermediate)
        return ggml.GgmlWeight(values[expert], metadata[expert], self.quant, n, k, self.lut)

    def to(self, device) -> "Iq2Experts":
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        return Iq2Experts(*(t.to(device) for t in self.tensors()), self.quant, self.hidden,
                          self.intermediate, ggml._execution_lut(self.quant, device), self.base)


def pack_experts(up: torch.Tensor, down: torch.Tensor, quant: str = "IQ2_XXS", *, base: int = 0) -> Iq2Experts:
    """Losslessly pack native [E,I,H/256,Bytes] / [E,H,I/256,Bytes] blocks.

    H and I are multiples of 256. Supports 1..512 local experts and an expert
    parallel base ID. Prepare outside graph capture; CPU packing is supported.
    """
    if quant not in FORMATS_IQ2 or up.ndim != 4 or down.ndim != 4:
        raise ValueError("IQ2 experts require XS/XXS four-dimensional native blocks")
    e, intermediate, blocks, size = up.shape
    hidden = blocks * 256
    if not 1 <= e <= 512 or intermediate < 256 or intermediate % 256 or hidden < 256:
        raise ValueError("IQ2 expert count must be 1..512 and widths multiples of 256")
    if type(base) is not int or base < 0 or base + e > 2**31 - 1:
        raise ValueError("expert base must fit positive int32 IDs")
    if (up.device != down.device or down.shape != (e, hidden, intermediate // 256, size)
            or size != FORMATS[quant][1]):
        raise ValueError("up/down expert geometry or devices do not match")
    if up.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("IQ2 expert preparation must precede CUDA graph capture")
    def pack(raw):
        experts, n, kb, size = raw.shape
        _validate(raw[0], quant)
        if not torch.isfinite(raw[..., :2].contiguous().view(torch.float16)).all().item():
            raise ValueError("IQ2 expert bases must be finite")
        values = raw[..., 2:66].reshape(experts, n // 8, 8, kb, 64).permute(0, 1, 3, 2, 4)
        metadata = torch.cat((raw[..., :2], raw[..., 66:]), -1).permute(0, 2, 1, 3)
        return values.clone(memory_format=torch.contiguous_format), metadata.contiguous()
    return Iq2Experts(*pack(up), *pack(down), quant, hidden, intermediate, ggml._execution_lut(quant, up.device), base)


@dataclass
class MoeWorkspace:
    count: torch.Tensor
    work: torch.Tensor
    order: torch.Tensor
    xq: torch.Tensor
    xs: torch.Tensor
    hq: torch.Tensor
    hs: torch.Tensor
    pair_outputs: torch.Tensor

    def tensors(self):
        return (self.count, self.work, self.order, self.xq, self.xs, self.hq, self.hs, self.pair_outputs)


def make_workspace(experts: Iq2Experts, rows: int, topk: int, *, tile_rows: int = 16) -> MoeWorkspace:
    if (type(rows) is not int or rows < 1 or type(topk) is not int or not 1 <= topk <= 16
            or type(tile_rows) is not int or tile_rows not in (16, 32, 64)):
        raise ValueError("workspace requires positive rows, topk=1..16 and tile_rows=16/32/64")
    pairs = rows * topk
    capacity = min(pairs, experts.experts) + (pairs + tile_rows - 1) // tile_rows
    if pairs > 2**31 - 1 or capacity > 65535:
        raise ValueError("IQ2 route schedule exceeds supported indexing/grid size")
    device = experts.up_values.device
    def alloc(shape, dtype):
        return torch.empty(shape, device=device, dtype=dtype)
    return MoeWorkspace(alloc((1,), torch.int32), alloc((capacity, 3), torch.int32), alloc((pairs,), torch.int32),
                        alloc((rows, experts.hidden), torch.uint8), alloc((rows, experts.hidden // 32), torch.float32),
                        alloc((pairs, experts.intermediate), torch.uint8), alloc((pairs, experts.intermediate // 32), torch.float32),
                        alloc((pairs, experts.hidden), torch.float32))


def _check_tensor(tensor, device, dtype, shape, name, alignment=1):
    if (not isinstance(tensor, torch.Tensor) or tensor.device != device or tensor.dtype != dtype
            or tuple(tensor.shape) != tuple(shape) or not tensor.is_contiguous() or tensor.data_ptr() % alignment):
        raise ValueError(f"invalid {name}: shape, dtype, device, contiguity or alignment")


def moe(experts: Iq2Experts, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, *,
        tile_rows: int = 16, workspace: MoeWorkspace | None = None,
        out: torch.Tensor | None = None, pairs: bool = False) -> torch.Tensor:
    """BF16 [M,H], int32/int64 IDs and FP32 scores [M,topk] -> BF16 [M,H].

    IDs outside this rank's [base,base+E) contribute zero. Duplicate IDs are
    allowed. With pairs=True return unweighted FP32 [M*topk,H], including zeros
    for nonlocal slots, in workspace.pair_outputs (out must be omitted).
    Buffers belong to one execution/stream. Warm once before graph capture.
    This is the routed expert operation; router, latent projections and shared
    expert are supplied by the model, as in W3's grouped MoE interface.
    """
    if not isinstance(experts, Iq2Experts) or experts.quant not in FORMATS_IQ2 or x.ndim != 2:
        raise ValueError("IQ2 MoE requires packed experts and a two-dimensional input")
    m, h = x.shape
    if x.device.type != "cuda" or h != experts.hidden:
        raise ValueError("IQ2 MoE requires CUDA input with the expert hidden width")
    if torch.cuda.get_device_capability(x.device) not in ((12, 0), (12, 1)):
        raise ValueError("IQ2 MoE requires SM120/SM121")
    _check_tensor(x, x.device, torch.bfloat16, (m, h), "input", 16)
    if ids.ndim != 2 or ids.shape[0] != m or ids.dtype not in (torch.int32, torch.int64) or not 1 <= ids.shape[1] <= 16:
        raise ValueError("routing IDs must be int32/int64 [M,topk] with topk=1..16")
    topk, e, inter = ids.shape[1], experts.experts, experts.intermediate
    if type(tile_rows) is not int or tile_rows not in (16, 32, 64):
        raise ValueError("tile_rows must be 16, 32 or 64")
    if (not 1 <= e <= 512 or h < 256 or inter < 256 or h % 256 or inter % 256
            or type(experts.base) is not int or experts.base < 0 or experts.base + e > 2**31 - 1):
        raise ValueError("invalid IQ2 expert geometry/base")
    _check_tensor(ids, x.device, ids.dtype, (m, topk), "routing IDs")
    _check_tensor(weights, x.device, torch.float32, (m, topk), "routing weights")
    meta = 10 if experts.quant == "IQ2_XS" else 2
    shapes = [(e, inter // 8, h // 256, 8, 64), (e, h // 256, inter, meta),
              (e, h // 8, inter // 256, 8, 64), (e, inter // 256, h, meta)]
    for name, tensor, shape in zip(("up payload", "up metadata", "down payload", "down metadata"), experts.tensors(), shapes):
        _check_tensor(tensor, x.device, torch.uint8, shape, name, 16)
    _check_tensor(experts.lut, x.device, torch.uint32, (512 if meta == 10 else 256,), "IQ2 table", 4)
    if pairs and out is not None:
        raise ValueError("pairs=True returns workspace.pair_outputs; omit out")
    output_shape = (m * topk, h) if pairs else (m, h)
    if not m:
        if out is None:
            return torch.empty(output_shape, device=x.device, dtype=torch.float32 if pairs else torch.bfloat16)
        _check_tensor(out, x.device, torch.bfloat16, output_shape, "output")
        return out
    capacity = min(m * topk, e) + (m * topk + tile_rows - 1) // tile_rows
    if m * topk > 2**31 - 1 or capacity > 65535:
        raise ValueError("IQ2 route schedule exceeds supported indexing/grid size")
    if workspace is None:
        workspace = make_workspace(experts, m, topk, tile_rows=tile_rows)
    if not isinstance(workspace, MoeWorkspace):
        raise ValueError("workspace must be an IQ2 MoeWorkspace")
    scratch = workspace.tensors()
    scratch_shapes = [(1,), (capacity, 3), (m * topk,), (m, h), (m, h // 32),
                      (m * topk, inter), (m * topk, inter // 32), (m * topk, h)]
    dtypes = (torch.int32, torch.int32, torch.int32, torch.uint8, torch.float32, torch.uint8, torch.float32, torch.float32)
    for t, shape, dtype in zip(scratch, scratch_shapes, dtypes):
        _check_tensor(t, x.device, dtype, shape, "workspace", 16)
    if pairs:
        out = workspace.pair_outputs
    elif out is None:
        out = torch.empty((m, h), device=x.device, dtype=torch.bfloat16)
    _check_tensor(out, x.device, torch.float32 if pairs else torch.bfloat16, output_shape, "output")
    reads = (x, ids, weights, *experts.tensors(), experts.lut)
    writes = scratch + (() if pairs else (out,))
    for i, tensor in enumerate(writes):
        if any(torch._C._overlaps(tensor, other) for other in reads + writes[:i]):
            raise ValueError("IQ2 MoE outputs/workspace must not overlap inputs or each other")
    with torch.cuda.device(x.device):
        _module().moe(x, ids, weights, list(experts.tensors()), experts.lut, list(scratch), out,
                      FORMATS_IQ2.index(experts.quant), experts.base, tile_rows, pairs)
    return out


@torch.no_grad()
def reference(experts: Iq2Experts, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Independent decoded FP32 reference with the operator's activation rounding."""
    def quant(value):
        groups = value.float().reshape(*value.shape[:-1], -1, 32)
        scale = (groups.abs().amax(-1, keepdim=True) * (1. / 448.)).clamp_min(1e-12)
        return ((groups / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale).reshape(value.shape)
    xd = quant(x)
    result = torch.zeros_like(x, dtype=torch.float32)
    cache = {}
    for row, choices in enumerate(ids.cpu().tolist()):
        for choice, global_expert in enumerate(choices):
            e = global_expert - experts.base
            if not 0 <= e < experts.experts:
                continue
            if e not in cache:
                cache[e] = experts.matrix(e, True).dequantize(), experts.matrix(e, False).dequantize()
            up, down = cache[e]
            projected = (xd[row:row + 1] @ up.T).bfloat16()
            hidden = projected.relu().square()
            result[row] += weights[row, choice] * (quant(hidden) @ down.T)[0]
    return result.bfloat16()
