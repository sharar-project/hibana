"""Verification convolution: BF16 products, FP32 accumulation and SiLU."""
import torch
import triton
import triton.language as tl


@triton.jit
def _convolve(X, State, Weight, Slots, Windows, Out,
              N: tl.constexpr, D: tl.constexpr, S: tl.constexpr,
              C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr,
              W0: tl.constexpr, W1: tl.constexpr,
              O0: tl.constexpr, O1: tl.constexpr, O2: tl.constexpr, O3: tl.constexpr,
              SILU: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    channel, token, batch = i % D, i // D % S, i // (D * S)
    valid = i < N
    slot = tl.load(Slots + batch, valid, 0)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for tap in tl.static_range(4):
        index = token + tap - 3
        history = tl.load(State + slot * C0 + channel * C1 + (index + 3) * C2,
                          valid & (index < 0), 0)
        current = tl.load(X + (batch * S + index) * D + channel,
                          valid & (index >= 0), 0)
        value = tl.where(index < 0, history, current)
        weight = tl.load(Weight + channel * W0 + tap * W1, valid, 0).to(tl.float32)
        product = (value.to(tl.float32) * weight).to(tl.bfloat16).to(tl.float32)
        if tap == 0:
            acc = product
        else:
            # The shared convolution kernel rounds products to BF16 but
            # accumulates them in FP32. Rounding each partial sum changes the
            # target during verification, despite matching prefill outputs.
            acc = acc + product
        if tap > 0:
            tl.store(Windows + batch * O0 + token * O1 + channel * O2 + (tap - 1) * O3,
                     value, valid)
    if SILU:
        acc = acc / (1.0 + tl.exp(-acc))
    tl.store(Out + i, acc, valid)


def verify_convolve(x, state, weight, slots, windows, steps, *, activation=True):
    """Return convolution/SiLU and write each candidate's three-token state.

    The committed state is read-only. Window strides may retain a larger
    verification capacity than this call's width; unused rows stay untouched.
    """
    if x.ndim != 2 or not x.is_contiguous() or x.dtype != torch.bfloat16:
        raise ValueError('verification convolution requires a contiguous BF16 matrix')
    rows, dim = x.shape
    if (steps < 1 or rows % steps or slots.shape != (rows // steps,) or not slots.is_contiguous()
            or state.shape[1:] != (dim, 3) or weight.shape != (dim, 4)
            or windows.shape[0] < rows // steps or windows.shape[1] < steps
            or windows.shape[2:] != (dim, 3)
            or any(t.dtype != x.dtype for t in (state, weight, windows))):
        raise ValueError('invalid verification convolution or state geometry')
    out = torch.empty_like(x)
    _convolve[(triton.cdiv(x.numel(), 512),)](
        x, state, weight, slots, windows, out, x.numel(), dim, steps,
        *state.stride(), *weight.stride(), *windows.stride(), activation, 512, enable_fp_fusion=False)
    return out
