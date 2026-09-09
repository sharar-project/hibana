"""Fuse the draft's two-tap convolution while preserving BF16 round points."""
import torch
import triton
import triton.language as tl


@triton.jit
def _conv(X, C, Base, Y, N: tl.constexpr, H: tl.constexpr, S: tl.constexpr,
          C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr, C3: tl.constexpr,
          BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    channel = i % H
    t = i // H % S
    b = i // (H * S)
    out = tl.full((BLOCK,), 0, tl.float32)
    for tap in tl.static_range(2):
        x = tl.load(X + i - tap * H, (i < N) & (t >= tap), 0).to(tl.float32)
        base = tl.load(Base + tap * H + channel, i < N, 0).to(tl.float32)
        c = tl.load(C + b * C0 + t * C1 + tap * C2 + channel // 16 * C3,
                    i < N, 0).to(tl.float32)
        product = (base * x).to(tl.bfloat16).to(tl.float32)
        out = (out + product).to(tl.bfloat16).to(tl.float32)
        out = tl.fma(c, x, out).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + i, out, i < N)


def convolve_bf16(x, coefficients, base):
    x, base = x.contiguous(), base.contiguous()
    b, s, h = x.shape
    if h % 16 or coefficients.shape != (b, s, 2, h // 16) or base.numel() != 2 * h:
        raise ValueError('invalid draft convolution geometry')
    out = torch.empty_like(x)
    _conv[(triton.cdiv(x.numel(), 1024),)](x, coefficients, base, out, x.numel(), h, s,
                                          *coefficients.stride(), 1024, enable_fp_fusion=False)
    return out
