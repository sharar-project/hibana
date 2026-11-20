# SPDX-License-Identifier: Apache-2.0
"""Flash-Next's interleaved multimodal RoPE positions (Qwen3.5's 3-D MRoPE rule).

The language model's rotary pairs (32 of them: rotary_dim 64, NeoX halves) are split over the
three position axes **interleaved** — pair ``p`` belongs to axis ``p % 3`` (t, h, w), which for
the checkpoint's ``mrope_section`` ``[11, 11, 10]`` gives exactly 11 t-, 11 h- and 10 w-pairs
(pairs 0,3,…,30 / 1,4,…,31 / 2,5,…,29). This is the layout of the deployed vLLM flash-next
build's kernels (the extracted vLLM reference sources: the QSA pre-indexer's
``h_mask = (pairs % 3 == 1) & …`` masks, with cos/sin still read from the same position-major
table), so a text row, whose three components are equal, is plain RoPE — bitwise the existing
``PartialRotary`` table lookup.

`positions` follows vLLM's ``Qwen3VLForConditionalGeneration._get_mrope_input_positions``
(image case, no lumped video frames): text rows get ``(p, p, p)`` where ``p`` continues across
images from the maximum coordinate the image's grid reached; an image's placeholder rows get
``np.indices((1, H, W))`` over its merged ``(H, W)`` patch grid (raster order over the
placeholder span) offset by the text position ``base`` the span starts at — per merged token
at grid ``(i, j)``: ``(t, h, w) = (base, base + i, base + j)`` (the temporal coordinate stays
``base`` for a still image). Text after the image continues from ``base + max(H, W)``, so a
``base + max(H, W)``, so a prompt with an image runs its later text positions behind its token
count by ``H*W - max(H, W)`` (vLLM's negative ``mrope_position_delta``).
"""

from __future__ import annotations

import numpy as np

# (t, h, w) rotary-pair counts. Pair p uses axis p % 3, which reproduces this split exactly;
# the kernel wrapper asserts it.
MROPE_SECTION = (11, 11, 10)
PAIRS = sum(MROPE_SECTION)  # = rotary_dim // 2 = 32


def positions_3d(prompt_len: int, media) -> np.ndarray | None:
    """``[3, prompt_len]`` int64 RoPE positions for a prompt whose ``media`` images
    (``engine/media.Image``, each with its merged ``grid`` ``(H, W)``) sit between text, or
    None for a prompt without images (plain 1-D positions). ValueError if an image's grid does
    not match its placeholder count or a coordinate overflows int32 cache indices."""
    if not media:
        return None
    if MROPE_SECTION != (11, 11, 10):
        raise ValueError("the interleaved layout assumes mrope_section [11, 11, 10]")
    out = np.empty((3, prompt_len), dtype=np.int64)
    at = 0  # rows written; a text run [at, image.start) takes positions [pos, pos + n)
    pos = 0  # the next text position
    for image in sorted(media, key=lambda m: m.start):
        n_text = image.start - at
        if n_text < 0 or image.end > prompt_len:
            raise ValueError("image placeholders outside the prompt")
        out[:, at:image.start] = np.broadcast_to(np.arange(pos, pos + n_text, dtype=np.int64), (3, n_text))
        pos += n_text
        gh, gw = image.grid
        n = len(image.ids)
        if gh * gw != n:
            raise ValueError(f"image grid {(gh, gw)} does not match {n} placeholders")
        grid = np.indices((1, gh, gw), dtype=np.int64).reshape(3, -1) + pos  # (t, h, w) = (0, i, j) + pos, raster
        out[:, image.start:image.end] = grid
        pos += max(gh, gw)  # the grid's widest coordinate; text continues past it
        at = image.end
    if at < prompt_len:
        out[:, at:] = np.broadcast_to(np.arange(pos, pos + (prompt_len - at), dtype=np.int64), (3, prompt_len - at))
    if out.max() >= 2**31:
        raise ValueError("a RoPE position overflows the int32 cache index range")
    return out
