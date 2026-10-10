"""Three-bit decode weights: an eight-entry E4M3 codebook, an E4M3 scale per 32 weights, and a tensor-core GEMM for up to 64 rows.

Format (`W3Weight`): for W [N, K], groups of 32 along K. Each weight is a
3-bit code into `codebook` (eight E4M3 values, the default the Lloyd-Max
levels of a Gaussian scaled so the largest is 3.5); each group has an E4M3
scale, each row an FP32 multiplier that restores the scale's range. That is
3 bits of code plus 8 bits of scale per 32 weights: 3.25 bits per weight, 72%
of NVFP4's 4.5 and 41% of FP8's 8.

Codes are stored as bit planes in the order the CUDA kernel's mma fragments
consume them (`csrc/w3_gemm.cu`): a warp owns 32 output rows and streams
128-k chunks; lane (g, t) holds, per chunk and per 8-row tile, one 32-bit
word per plane whose bit (4j + b + rot(s)) mod 32 is bit b of the weight at
row 8*tile + g, k = 16*s + 4*t + j. The kernel turns the three words into
E4M3 B-fragment bytes with three rotates, three masked ORs and one PRMT
lookup, so expansion costs under two integer instructions per weight and
the kernel is bound by the weight stream at 16 to 64 activation rows.

Activations are E4M3: the engine's static-scale projection inputs as they
are, BF16 MLP inputs quantized per row and 32-group by `quant_groups` (the
kernel applies the activation group scale together with the weight group
scale; a single per-row scale measurably lost task accuracy). Enabled for
decode-sized calls (at most `MAX_ROWS` rows) when `SHARAR_W3_DECODE=1` and a
linear carries a packed copy (`attach_model`); prefill keeps the NVFP4 and
FP8 weights and kernels. Quantization here is nearest-level from the served
weights; calibrated source weights use `pack_codes` to retain fitted scales
and load through `SHARAR_W3_CHECKPOINT` before graph capture.
"""
from __future__ import annotations

import hashlib
import os
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch



def enabled() -> bool:
    """SHARAR_W3_DECODE: serve calls of at most MAX_ROWS rows from the 3-bit copies."""
    from sharar.engine.config import current
    return current().w3_decode
MAX_ROWS = 64
GROUP = 32
CHUNK = 128
FP8_MAX = 448.0
# Eight levels, largest 3.5. Uniform is the classic (q - 3.5) grid; Lloyd-Max is the MSE-optimal
# Gaussian quantizer's levels (0.245, 0.756, 1.344, 2.152 sigma) scaled to 3.5 and rounded to E4M3.
CODEBOOKS = {
    "uniform": (-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5),
    "lloyd": (-3.5, -2.25, -1.25, -0.40625, 0.40625, 1.25, 2.25, 3.5),
}
DEFAULT_CODEBOOK = "lloyd"  # every served copy; "uniform" remains for the packing tests and studies
_SRC = Path(__file__).resolve().parent / "csrc" / "w3_gemm.cu"


@dataclass
class W3Weight:
    planes: torch.Tensor  # [N/32, K/128, 3, 32, 4] uint32, fragment order (module docstring)
    scales: torch.Tensor  # [N/32, K/128, 4, 32] uint8 E4M3: for lane column pair t, byte (4*tile + group)*2 + i is row 8*tile + 2*t + i
    row_scale: torch.Tensor  # [N] float32
    out_features: int
    in_features: int
    codebook: tuple[float, ...]
    lut: int  # the codebook as eight E4M3 bytes, entry i in byte i (the kernel's two 32-bit PRMT sources)

    @property
    def bytes(self) -> int:
        return self.planes.numel() * 4 + self.scales.numel() + self.row_scale.numel() * 4


def lut_of(codebook: tuple[float, ...]) -> int:
    raw = torch.tensor(codebook, dtype=torch.float32).to(torch.float8_e4m3fn).view(torch.uint8).tolist()
    return struct.unpack("<q", bytes(raw))[0]


def _codebook(codebook) -> tuple[float, ...]:
    codebook = DEFAULT_CODEBOOK if codebook is None else codebook
    levels = CODEBOOKS[codebook] if isinstance(codebook, str) else tuple(float(v) for v in codebook)
    if len(levels) != 8 or list(levels) != sorted(levels):
        raise ValueError("a codebook is eight ascending levels")
    exact = torch.tensor(levels).to(torch.float8_e4m3fn).float().tolist()
    if exact != list(levels):
        raise ValueError("codebook levels must be exactly representable in E4M3")
    return tuple(levels)


def _rotations(device) -> torch.Tensor:
    s = torch.arange(8, device=device)
    return 16 * (s // 4) + (s % 4)  # rot(s)


def pack(weight: torch.Tensor, codebook=None, rows_per_chunk: int = 4096) -> W3Weight:
    """Nearest-level 3-bit quantization of a [N, K] float matrix (N % 128 == 0, K % 128 == 0) in fragment order.

    Rows are processed `rows_per_chunk` at a time so the integer temporaries stay
    a few hundred MB even for the lm_head (the whole-matrix version needed
    several 10 GB tensors and tripped the memory guard)."""
    n, k = weight.shape
    if k % CHUNK or n % 128:
        raise ValueError("N and K must be multiples of 128")
    levels = _codebook(codebook)
    planes, scales, row_scales = [], [], []
    for r0 in range(0, n, rows_per_chunk):
        pl, sc, rs = _pack_rows(weight[r0:r0 + rows_per_chunk], levels)
        planes.append(pl); scales.append(sc); row_scales.append(rs)
    cat = (lambda xs: xs[0] if len(xs) == 1 else torch.cat(xs))
    return W3Weight(cat(planes), cat(scales), cat(row_scales), n, k, levels, lut_of(levels))


def _pack_rows(weight: torch.Tensor, levels: tuple[float, ...]):
    n, k = weight.shape
    dev = weight.device
    w = weight.float().reshape(n, k // GROUP, GROUP)
    amax = w.abs().amax(dim=-1)  # [N, G]
    scale = amax / max(abs(v) for v in levels)
    row_scale = (scale.amax(dim=-1) / FP8_MAX).clamp_min(1e-12)  # [N]
    scale_fp8 = (scale / row_scale[:, None]).clamp(0, FP8_MAX).to(torch.float8_e4m3fn)
    scale_eff = scale_fp8.float() * row_scale[:, None]
    normalized = w / scale_eff.clamp_min(1e-30)[:, :, None]
    del w
    lv = torch.tensor(levels, device=dev)
    mids = (lv[1:] + lv[:-1]) / 2
    q = torch.bucketize(normalized.reshape(n, k), mids, out_int32=True)  # [N, K] codes 0..7
    del normalized
    return _pack_codes_rows(q, scale_fp8, row_scale)


def _pack_codes_rows(q: torch.Tensor, scale_fp8: torch.Tensor, row_scale: torch.Tensor):
    n, k = q.shape
    dev = q.device
    # fragment order: n = 32 nb + 8 nt + g, k = 128 kc + 16 s + 4 t + j
    q = q.reshape(n // 32, 4, 8, k // CHUNK, 8, 4, 4)  # [nb, nt, g, kc, s, t, j]
    rot = _rotations(dev)  # [s]
    j = torch.arange(4, device=dev)
    words = []
    for b in range(3):
        pos = ((4 * j[None, :] + b + rot[:, None]) % 32).to(torch.int64)  # [s, j]
        bit = ((q >> b) & 1).to(torch.int64)
        words.append((bit << pos[None, None, None, None, :, None, :]).sum(dim=(4, 6)))  # [nb, nt, g, kc, t]
        del bit
    words = torch.stack(words, dim=2)  # [nb, nt, b, g, kc, t]
    words = words.permute(0, 4, 3, 5, 1, 2).reshape(n // 32, k // CHUNK, 32, 12)  # [nb, kc, lane, 3 nt + b]
    planes = words.reshape(n // 32, k // CHUNK, 32, 3, 4).permute(0, 1, 3, 2, 4)  # [nb, kc, i, lane, e]
    planes = planes.bitwise_and(0xFFFFFFFF).to(torch.uint32).contiguous()
    # a lane's accumulator columns are n = 8 nt + 2 t + i, so scales are laid out per t: [nb, kc, t, (nt, grp, i)]
    sc = scale_fp8.view(torch.uint8).reshape(n // 32, 4, 4, 2, k // CHUNK, 4)  # [nb, nt, t, i, kc, grp]
    scales = sc.permute(0, 4, 2, 1, 5, 3).reshape(n // 32, k // CHUNK, 4, 32).contiguous()
    return planes, scales, row_scale.contiguous()


def pack_codes(q: torch.Tensor, scale_fp8: torch.Tensor, row_scale: torch.Tensor,
               codebook=None, rows_per_chunk: int = 4096) -> W3Weight:
    """Pack calibrated codes and their exact scales without quantizing again."""
    n, k = q.shape
    if n % 128 or k % CHUNK or rows_per_chunk < 128 or rows_per_chunk % 128:
        raise ValueError("N, K and rows_per_chunk must be multiples of 128")
    if q.dtype not in (torch.uint8, torch.int32, torch.int64) or torch.any((q < 0) | (q > 7)):
        raise ValueError("3-bit codes must be integers between zero and seven")
    if scale_fp8.shape != (n, k // GROUP) or scale_fp8.dtype != torch.float8_e4m3fn or row_scale.shape != (n,):
        raise ValueError("calibrated scales do not match the weight shape")
    if not torch.isfinite(scale_fp8.float()).all() or not torch.isfinite(row_scale).all() or (row_scale <= 0).any() or (scale_fp8.float() < 0).any():
        raise ValueError("calibrated scales must be finite and nonnegative, with positive row scales")
    levels = _codebook(codebook)
    chunks = [_pack_codes_rows(q[r:r + rows_per_chunk].int(), scale_fp8[r:r + rows_per_chunk], row_scale[r:r + rows_per_chunk])
              for r in range(0, n, rows_per_chunk)]
    return W3Weight(*(torch.cat([part[i] for part in chunks]) for i in range(3)), n, k, levels, lut_of(levels))


def codes(w3: W3Weight) -> torch.Tensor:
    """The 3-bit codes of a packed matrix, [N, K] int64."""
    n, k = w3.out_features, w3.in_features
    dev = w3.planes.device
    words = w3.planes.to(torch.int64).permute(0, 1, 3, 2, 4).reshape(n // 32, k // CHUNK, 32, 4, 3)  # [nb, kc, lane, nt, b]
    words = words.reshape(n // 32, k // CHUNK, 8, 4, 4, 3).permute(0, 4, 2, 1, 3, 5)  # [nb, kc, g, t, nt, b] -> [nb, nt, g, kc, t, b]
    rot = _rotations(dev)
    j = torch.arange(4, device=dev)
    q = torch.zeros(n // 32, 4, 8, k // CHUNK, 8, 4, 4, dtype=torch.int64, device=dev)  # [nb, nt, g, kc, s, t, j]
    for b in range(3):
        pos = (4 * j[None, :] + b + rot[:, None]) % 32  # [s, j]
        q |= ((words[..., b][:, :, :, :, None, :, None] >> pos[None, None, None, None, :, None, :]) & 1) << b
    return q.reshape(n, k)


def group_scales(w3: W3Weight) -> torch.Tensor:
    """Effective per-group scales, [N, K/32] float32."""
    n, k = w3.out_features, w3.in_features
    sc = w3.scales.reshape(n // 32, k // CHUNK, 4, 4, 4, 2).permute(0, 3, 2, 5, 1, 4).reshape(n, k // GROUP)  # [nb, kc, t, nt, grp, i] -> [nb, nt, t, i, kc, grp]
    return sc.contiguous().view(torch.float8_e4m3fn).float() * w3.row_scale[:, None]


def dequantize(w3: W3Weight) -> torch.Tensor:
    """FP32 reference of the packed matrix, [N, K]."""
    lv = torch.tensor(w3.codebook, device=w3.planes.device)
    return lv[codes(w3)] * group_scales(w3).repeat_interleave(GROUP, dim=1)


# ------------------------------------------------------------------ the kernel

@lru_cache(maxsize=1)
def _module():
    from torch.utils.cpp_extension import load

    from sharar.engine.config import current
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", current().cuda_arch)
    # Images can restore older source timestamps into a shared build cache.
    # Separate source revisions so Ninja cannot reuse another revision's object.
    revision = hashlib.sha256(_SRC.read_bytes()).hexdigest()[:16]
    return load(name=f"sharar_w3_gemm_{revision}", sources=[str(_SRC)], extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)


_WORKSPACE: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}
# Split-K factor per (N, K, rows): profiling/w3_bench.py on GB10 (2026-09-17). More blocks help the weight stream
# at 16 rows; at 64 rows the partial-sum traffic and the tensor-core work per block argue for fewer splits.
SPLITS: dict[tuple[int, int, int], int] = {  # (N, K, rows) -> splits
    (5120, 17408, 16): 8, (5120, 17408, 32): 2, (5120, 17408, 64): 2,
    (10240, 5120, 16): 1, (10240, 5120, 32): 1, (10240, 5120, 64): 1,
    (6144, 5120, 16): 1, (6144, 5120, 32): 1, (6144, 5120, 64): 2,
    (5120, 6144, 16): 2, (5120, 6144, 32): 2, (5120, 6144, 64): 2,
    (14336, 5120, 16): 1, (14336, 5120, 32): 1, (14336, 5120, 64): 1,
    (4096, 5120, 16): 4, (4096, 5120, 32): 4, (4096, 5120, 64): 2,
    (5120, 4096, 16): 4, (5120, 4096, 32): 1, (5120, 4096, 64): 2,
}
SPLIT_TARGET = {16: 192, 32: 96, 64: 48}  # blocks (4 warps, 128 rows) a launch should reach before K stays whole


def splits_for(n: int, k: int, m: int = 16) -> int:
    rows = 16 if m <= 16 else 32 if m <= 32 else 64
    if (n, k, rows) in SPLITS:
        return SPLITS[(n, k, rows)]
    blocks, chunks, splits = n // 128, k // CHUNK, 1
    while blocks * splits < SPLIT_TARGET[rows] and chunks % (2 * splits) == 0 and splits < 8:
        splits *= 2
    return splits


def _workspace(device: torch.device, floats: int, blocks: int) -> tuple[torch.Tensor, torch.Tensor]:
    ws, counters = _WORKSPACE.get(device, (None, None))
    if ws is None or ws.numel() < floats or counters.numel() < blocks:
        ws = torch.empty(max(floats, 8 << 20), dtype=torch.float32, device=device)
        counters = torch.zeros(max(blocks, 4096), dtype=torch.int32, device=device)
        _WORKSPACE[device] = (ws, counters)
    return ws, counters


def quant_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row E4M3 quantization of a BF16 [M, K] matrix: (x_q, scale[M]). One scale per row loses the small
    channels next to an outlier (GSM8K -1.7 points, HumanEval -4 with the uniform codebook); serving uses `quant_groups`."""
    out, scale = _module().quant_rows(x.contiguous())
    return out, scale


def quant_groups(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-(row, 32-group) E4M3 quantization of a BF16 [M, K] matrix: (x_q, scale[M, K/32]), MX-style block scaling
    that isolates outlier channels to their own group."""
    out, scale = _module().quant_groups(x.contiguous())
    return out, scale


def w3_linear(x: torch.Tensor, w3: W3Weight, x_scale: torch.Tensor | None = None, splits: int | None = None) -> torch.Tensor:
    """y = x @ W^T for x [M, K] in BF16 (quantized per row here) or E4M3 with its static scale; M <= MAX_ROWS. Returns BF16 [M, N]."""
    m, k = x.shape
    n = w3.out_features
    if m > MAX_ROWS or k != w3.in_features:
        raise ValueError("w3_linear: unsupported call")
    if x.dtype == torch.float8_e4m3fn:
        if x_scale is None:
            raise ValueError("FP8 activations need their scale")
        x_q, scale = x, x_scale
    else:
        x_q, scale = quant_groups(x)
    splits = splits or splits_for(n, k, m)
    ws, counters = _workspace(x.device, splits * m * n if splits > 1 else 0, n // 128)
    return _module().w3_gemm(x_q, scale, w3.planes, w3.scales, w3.row_scale, w3.lut, splits, ws, counters)


# ------------------------------------------------------------------ attaching to the served model

def attach(linears, codebook=DEFAULT_CODEBOOK) -> tuple[int, int]:
    """Nearest-level 3-bit copies for (name, linear) pairs that have none (`weights/linear.py` interface);
    returns (matrices packed, bytes)."""
    count = total = 0
    for _, lin in linears:
        if getattr(lin, "w3", None) is None:
            lin.w3 = pack(lin.dequantize(), codebook)
            total += lin.w3.bytes; count += 1
    return count, total


def attach_model(model, log=print, codebook=DEFAULT_CODEBOOK) -> dict:
    """Pack 3-bit decode copies for every projection `model.linears()` lists (calibrated ones from
    SHARAR_W3_CHECKPOINT when set). The lm_head keeps its NVFP4 weights (a 3-bit copy cost about two
    GSM8K points, 2026-09-17)."""
    total = 0
    count = 0
    from sharar.weights.w3_checkpoint import CalibratedWeights
    from sharar.engine.config import current
    directory = current().w3_checkpoint
    calibrated = CalibratedWeights(directory) if directory else None
    linears = list(model.linears())
    if calibrated is not None and set(calibrated.weights) != {name for name, _ in linears}:
        raise ValueError("calibrated W3 checkpoint must contain every target layer projection")
    for name, lin in linears:
        if getattr(lin, "w3", None) is None:
            lin.w3 = calibrated.load(name, lin, lin.shard) if calibrated else pack(lin.dequantize(), codebook)
            total += lin.w3.bytes; count += 1
    torch.cuda.synchronize()
    model.w3_identity = calibrated.identity if calibrated else "nearest:" + str(codebook)
    _module()  # build the kernel before serving starts
    log(f"  w3: packed {count} matrices, {total / 2**30:.2f} GiB of 3-bit decode weights (codebook {codebook})")
    return {"matrices": count, "bytes": total}


def attach_draft(draft, log=print, codebook=DEFAULT_CODEBOOK) -> dict:
    """Pack 3-bit copies of the parallel draft's FP8 matrices (q, o, gate_up, down of every layer)."""
    count, total = attach(draft.fp8_weights.items(), codebook)
    torch.cuda.synchronize()
    log(f"  w3: packed {count} draft matrices, {total / 2**30:.2f} GiB")
    return {"matrices": count, "bytes": total}
