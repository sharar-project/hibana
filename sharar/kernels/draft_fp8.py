"""Capture-safe per-tensor E4M3 scaling and packing in two kernels.

The scale and packed values follow the eager draft-FP8 pilot, including
round-to-nearest division and the 1e-12 scale floor. Only the small partial
maximum array is materialized; no full-size FP32 activation temporary.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _amax(X, Partial, N: tl.constexpr, BLOCK: tl.constexpr):
    p = tl.program_id(0)
    i = p * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, 0).to(tl.float32)
    tl.store(Partial + p, tl.max(tl.abs(x), 0))


@triton.jit
def _pack(X, Partial, Y, Scale, N: tl.constexpr, PARTS: tl.constexpr,
          REDUCE: tl.constexpr, BLOCK: tl.constexpr):
    p = tl.program_id(0)
    j = tl.arange(0, REDUCE)
    peak = tl.max(tl.load(Partial + j, j < PARTS, 0), 0)
    # PyTorch's division by a Python scalar multiplies by its FP32 reciprocal.
    scale = tl.maximum(peak * (1. / 448.), 1.e-12)
    if p == 0:
        tl.store(Scale, scale)
    i = p * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, 0).to(tl.float32)
    q = tl.minimum(tl.maximum(tl.div_rn(x, scale), -448.), 448.)
    tl.store(Y + i, q.to(tl.float8e4nv), i < N)


def quant_dynamic(x):
    if x.ndim != 2 or not x.is_contiguous() or not x.is_cuda or x.numel() == 0:
        raise ValueError('draft quantization requires a nonempty contiguous CUDA matrix')
    n = x.numel()
    parts = triton.cdiv(n, 4096)
    partial = torch.empty(parts, device=x.device, dtype=torch.float32)
    out = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty((1, 1), device=x.device, dtype=torch.float32)
    _amax[(parts,)](x, partial, n, 4096)
    _pack[(triton.cdiv(n, 2048),)](x, partial, out, scale, n, parts,
                                 triton.next_power_of_2(parts), 2048)
    return out, scale
