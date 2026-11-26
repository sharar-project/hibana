"""Rope positions with images, for targets that rotate image tokens by 2-D positions (the Qwen checkpoints' interleaved
M-RoPE). Text rotates by one position on all three axes, plain 1-D rope at `index + delta`. A video frame pair
(engine/media.py `Image.pair`) is an image span like any other, between the text of its timestamp: the checkpoints'
`get_rope_index` splits a clip into t = 1 grids, so nothing here is of video. An image of rows x cols
blocks starting at rope position s gives block (r, c) the position (s, s + r, s + c), and the text after it resumes at
s + max(rows, cols). Image rows read scratch rows of the rotary table written for the forward
(`PartialRotary.write_scratch`). Positions follow from the request's own images alone, so a prefix resumed from the
radix cache or the SSD tier gets the same ones.
"""

from __future__ import annotations


class RopeLayout:
    """A request's image spans and `delta`, rope position minus index after the last image (decode rows are text)."""

    def __init__(self, media):
        self.spans: list[tuple[int, int, int, int, int, int]] = []  # start, end, rope start, rows, cols, delta after
        delta = 0
        for image in sorted(media, key=lambda m: m.start):
            rows, cols = image.grid
            if rows * cols != len(image.ids):
                raise ValueError(f"an image of {rows} x {cols} blocks has {len(image.ids)} placeholders")
            s = image.start + delta
            delta += max(rows, cols) - rows * cols
            self.spans.append((image.start, image.end, s, rows, cols, delta))
        self.delta = delta

    def rows(self, start: int, end: int, base: int, scratch: list[tuple[int, int, int]]) -> list[int]:
        """The rotary table row of each index in [start, end): `index + delta` for text; for an image row the scratch
        row `base + len(scratch)`, its (t, h, w) appended to `scratch`."""
        out: list[int] = []
        i, delta = start, 0
        for a, b, s, rows, cols, after in self.spans:
            if b <= start:
                delta = after
                continue
            if a >= end:
                break
            out.extend(range(i + delta, a + delta))
            for j in range(max(i, a), min(b, end)):
                r, c = divmod(j - a, cols)
                out.append(base + len(scratch))
                scratch.append((s, s + r, s + c))
            i, delta = min(b, end), after
        out.extend(range(i + delta, end + delta))
        return out


NO_IMAGES = RopeLayout(())
