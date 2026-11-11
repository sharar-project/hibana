"""Relaxed acceptance's statistics of a drafted token, one pass over its verify row's logits: how many logits rank
above it and how far it trails the top one (engine/spec.SpecDecoder._relaxed_matches). The same numbers as
`(row > row[y]).sum()` and `row.max() - row[y]` in FP32, without the [rows, vocabulary] comparison those write."""
import torch
import triton
import triton.language as tl

SKIPPED = 1 << 30  # `above` of a skipped cell: outranked by more than any top-k
_SKIPPED = tl.constexpr(SKIPPED)


@triton.jit
def _rank_gap(Logits, Rows, Tokens, Above, Top, Value, stride, V, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    cell, part = tl.program_id(0), tl.program_id(1)
    row = tl.load(Rows + cell)
    if row < 0:  # a draft the round does not verify
        tl.store(Above + cell * SPLIT + part, tl.where(part == 0, _SKIPPED, 0))
        tl.store(Top + cell * SPLIT + part, float("inf"))
        tl.store(Value + cell, 0.0)
    else:
        base = Logits + row * stride
        value = tl.load(base + tl.load(Tokens + cell)).to(tl.float32)
        span = tl.cdiv(tl.cdiv(V, SPLIT), BLOCK) * BLOCK
        above = tl.zeros((BLOCK,), tl.int32)
        top = tl.full((BLOCK,), float("-inf"), tl.float32)
        for start in range(part * span, (part + 1) * span, BLOCK):
            index = start + tl.arange(0, BLOCK)
            x = tl.load(base + index, index < V, other=float("-inf")).to(tl.float32)
            above += (x > value).to(tl.int32)
            top = tl.maximum(top, x)
        tl.store(Above + cell * SPLIT + part, tl.sum(above, 0))
        tl.store(Top + cell * SPLIT + part, tl.max(top, 0))
        if part == 0:
            tl.store(Value + cell, value)


def rank_gap(logits: torch.Tensor, rows: torch.Tensor, tokens: torch.Tensor, split: int = 8):
    """logits [N, V] (rows contiguous), rows int64 [M] (a row of `logits` per cell, -1: skip the cell), tokens int64
    [M]: (above int32 [M], the logits greater than the token's; gap float32 [M], the top logit minus the token's).
    A skipped cell has above SKIPPED and gap inf."""
    if logits.ndim != 2 or logits.stride(1) != 1 or rows.shape != tokens.shape or rows.ndim != 1:
        raise ValueError("rank_gap: logits [N, V] with contiguous rows, rows and tokens [M]")
    m = rows.numel()
    above = torch.empty((m, split), dtype=torch.int32, device=logits.device)
    top = torch.empty((m, split), dtype=torch.float32, device=logits.device)
    value = torch.empty(m, dtype=torch.float32, device=logits.device)
    if m:
        _rank_gap[(m, split)](logits, rows, tokens, above, top, value, logits.stride(0), logits.shape[1],
                              SPLIT=split, BLOCK=2048, num_warps=4)
    return above.sum(1, dtype=torch.int32), top.amax(1) - value
