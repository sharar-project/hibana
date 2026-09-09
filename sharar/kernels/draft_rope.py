"""Draft RoPE and optional KV scatter, preserving each BF16 round point."""
import torch
import triton
import triton.language as tl


@triton.jit
def _rotate(X, V, Pos, Cos, Sin, Pages, Kout, Vout,
            T: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
            X0: tl.constexpr, X1: tl.constexpr,
            V0: tl.constexpr, V1: tl.constexpr,
            C0: tl.constexpr, S0: tl.constexpr, P0: tl.constexpr, Page0: tl.constexpr,
            CACHE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    half = D // 2
    col, head, row = i % half, i // half % H, i // (H * half)
    valid = row < T
    pos = tl.load(Pos + row * P0, valid, 0)
    c = tl.load(Cos + pos * C0 + col, valid, 0).to(tl.bfloat16).to(tl.float32)
    s = tl.load(Sin + pos * S0 + col, valid, 0).to(tl.bfloat16).to(tl.float32)
    offset = row * X0 + head * X1 + col
    a = tl.load(X + offset, valid, 0).to(tl.float32)
    b = tl.load(X + offset + half, valid, 0).to(tl.float32)
    ac = (a * c).to(tl.bfloat16).to(tl.float32)
    bs = (b * s).to(tl.bfloat16).to(tl.float32)
    bc = (b * c).to(tl.bfloat16).to(tl.float32)
    ass = (a * s).to(tl.bfloat16).to(tl.float32)
    dest = tl.load(Pages + row * Page0, valid, 0) if CACHE else row
    output = (dest * H + head) * D + col
    tl.store(Kout + output, ac - bs, valid)
    tl.store(Kout + output + half, bc + ass, valid)
    if CACHE:
        source = row * V0 + head * V1 + col
        tl.store(Vout + output, tl.load(V + source, valid, 0), valid)
        tl.store(Vout + output + half, tl.load(V + source + half, valid, 0), valid)


def rotate_bf16(x, positions, cos, sin, *, values=None, caches=None, pages=None):
    """Rotate [tokens, heads, dim]; optionally write private, unique KV pages.

    The caller owns page bounds and uniqueness, as with the original indexed
    assignment. Values can be the strided half of a fused K/V projection.
    """
    t, h, d = x.shape
    if x.dtype != torch.bfloat16 or x.stride(-1) != 1 or d % 2:
        raise ValueError('draft RoPE requires BF16 and contiguous even head dimensions')
    if positions.shape != (t,):
        raise ValueError('draft RoPE requires one position per token')
    cached = caches is not None
    if cached:
        if (values is None or values.shape != x.shape or values.stride(-1) != 1
                or values.dtype != x.dtype or pages is None or pages.shape != (t,)
                or any(c.shape[1:] != (1, h, d) or not c.is_contiguous() or c.dtype != x.dtype
                       for c in caches)):
            raise ValueError('invalid draft KV scatter geometry')
        out, vout = caches
    else:
        out = torch.empty_like(x, memory_format=torch.contiguous_format)
        values, vout, pages = x, out, positions
    if t:
        _rotate[(triton.cdiv(t * h * (d // 2), 256),)](
            x, values, positions, cos, sin, pages, out, vout,
            t, h, d, *x.stride()[:2], *values.stride()[:2], cos.stride(0), sin.stride(0),
            positions.stride(0), pages.stride(0),
            cached, 256, enable_fp_fusion=False)
    return out
