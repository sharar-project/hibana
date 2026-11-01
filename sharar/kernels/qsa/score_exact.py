# SPDX-License-Identifier: Apache-2.0
"""The `original` selection's scores and selection with faster kernels and the same result, for eager prefill on SM121.

`csrc/score_exact.cu` computes `selection._score_kernel`'s scores bit for bit (the same FP32 order), 64 query rows
of one sequence per block sharing each compressed key. Its top-k keeps `topk_pages.cu`'s selection for rows of any
length: the 512 largest by score, then ascending group index, which is the order Torch's top-k returns them in for
`select_pages` (a stable sort; checked in tests/gpu/test_qsa_exact.py), and assembles the pages as
`_assemble_kernel` does. The kernels and their design: README.md.
"""
from functools import lru_cache
from pathlib import Path

import torch

from .compression import CUDA_FLAGS

TILE = 64  # query rows per score block


@lru_cache(maxsize=None)
def load_exact_scorer():
    """Compiled for SM121 (`supported`) on the first eager prefill that needs it, never inside a capture."""
    import tvm_ffi.cpp
    return tvm_ffi.cpp.load('qsa_score_exact', sources=str(Path(__file__).parent / 'csrc/score_exact.cu'),
                            extra_cuda_cflags=CUDA_FLAGS)


def supported(q: torch.Tensor, aux: torch.Tensor, ratio: int, budget: int) -> bool:
    return (q.is_cuda and q.dtype == torch.bfloat16 and aux.dtype == torch.bfloat16 and q.ndim == 3
            and q.shape[1:] == (4, 128) and q.stride(2) == 1 and aux.shape[1:] == (2, 128) and aux.is_contiguous()
            and (ratio, budget) == (4, 2048) and torch.cuda.get_device_capability(q.device) == (12, 1)
            and not torch.cuda.is_current_stream_capturing())


def _tiles(segments, a: int, b: int) -> list[int]:
    """[row0, rows] pairs (relative to a) of the rows [a, b), TILE at most and within one segment."""
    out, start = [], 0
    for n in segments:
        lo, hi = max(start, a), min(start + n, b)
        for r in range(lo, hi, TILE):
            out += [r - a, min(TILE, hi - r)]
        start += n
    return out


def select_exact(q: torch.Tensor, aux: torch.Tensor, table: torch.Tensor, base: torch.Tensor, positions: torch.Tensor,
                 gcap: int, scale: float, segments: list[int], chunk: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """`selection.select_pages(q, aux, table, base, positions, 4, 2048, gcap, scale)` with the same result. `segments`:
    the host-known row count of each sequence, in row order (a score block's rows share a sequence)."""
    rows, device = q.shape[0], q.device
    pages = torch.empty((rows, 2051), dtype=torch.int32, device=device)
    counts = torch.empty((rows,), dtype=torch.int32, device=device)
    if not rows:
        return pages, counts
    module = load_exact_scorer()
    table = table.to(torch.int32).contiguous()
    base32 = base.to(torch.int32).contiguous()
    base64, positions = base.to(torch.int64).contiguous(), positions.to(torch.int64).contiguous()
    step = min(rows, chunk)
    scores = torch.empty((step, max(gcap, 1)), dtype=torch.float32, device=device)
    starts = range(0, rows, step)
    tiles = [_tiles(segments, a, min(rows, a + step)) for a in starts]
    offsets = [0]
    for t in tiles:
        offsets.append(offsets[-1] + len(t) // 2)
    tiles = torch.tensor([x for t in tiles for x in t], dtype=torch.int32).view(-1, 2).to(device)  # one upload
    for i, a in enumerate(starts):
        b = min(rows, a + step)
        s = scores[: b - a]
        module.score(q[a:b], aux, table, base32[a:b], positions[a:b], tiles[offsets[i]:offsets[i + 1]], s[:, :gcap],
                     scale, 8)
        module.topk_pages_long(s[:, :gcap], table, base64[a:b], positions[a:b], pages[a:b], counts[a:b])
    return pages, counts
