"""Fuse draft RMS pointwise work while retaining PyTorch's mean reduction.

N is a runtime argument: at batch > 1 the accepted-row feedback changes it every round, and as a constexpr every new
count would compile (or reload) the kernels on the round's critical path."""
import torch
import triton
import triton.language as tl


@triton.jit
def _square(X, Y, N, D: tl.constexpr, H: tl.constexpr,
            X0: tl.constexpr, X1: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = i // D
    x = tl.load(X + row // H * X0 + row % H * X1 + i % D, i < N, 0).to(tl.float32)
    tl.store(Y + i, x * x, i < N)


@triton.jit
def _finish(X, Variance, Weight, Y, N, D: tl.constexpr, H: tl.constexpr,
            X0: tl.constexpr, X1: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = i // D
    x = tl.load(X + row // H * X0 + row % H * X1 + i % D, i < N, 0).to(tl.float32)
    variance = tl.load(Variance + row, i < N, 0)
    weight = tl.load(Weight + i % D, i < N, 0).to(tl.float32)
    norm = (x * tl.rsqrt(variance + EPS)).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + i, norm * weight, i < N)


def rms_bf16(x, weight, eps=1e-6):
    """Preserve the BF16 cast before weight multiplication and FP32 mean order.

    The K half of a fused KV projection has gaps between tokens. Address it
    directly when producing the contiguous squared input to torch.mean.
    """
    if (x.ndim not in (2, 3) or x.dtype != torch.bfloat16 or x.stride(-1) != 1
            or weight.shape != (x.shape[-1],) or weight.dtype != x.dtype or not weight.is_contiguous()):
        raise ValueError('invalid BF16 draft RMS geometry')
    heads = x.shape[1] if x.ndim == 3 else 1
    stride_head = x.stride(1) if x.ndim == 3 else 0
    squares = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(x.numel(), 1024),)
    _square[grid](x, squares, x.numel(), x.shape[-1], heads, x.stride(0), stride_head,
                  1024, enable_fp_fusion=False)
    variance = squares.mean(-1, keepdim=True)
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    _finish[grid](x, variance, weight, out, x.numel(), x.shape[-1], heads,
                  x.stride(0), stride_head, eps, 1024, enable_fp_fusion=False)
    return out
