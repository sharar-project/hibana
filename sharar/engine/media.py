"""Images in a prompt, for targets with a vision encoder (Nemotron Super and Bolt: model/nemotron/vision.py).

The API expands each image into the encoder's placeholder tokens and gives them ids the vocabulary does not have:
negative int32s drawn from the image's content (`placeholder_ids`). The prompt the engine stores, caches and
traces carries those ids, so the radix prefix cache, the SSD tier and the recurrent-state snapshots key a prefix
on the images in it, as on its text, and never resume one image's state for another. Where ids reach the model they
become the image token (`text_ids`), and the prefill forward replaces the embedding rows of an image's placeholders
with the encoder's output (`ForwardContext.media_rows`, `media_embeds`).
"""

from __future__ import annotations

import hashlib
from array import array
from dataclasses import dataclass

import torch


@dataclass(eq=False)
class Image:
    start: int  # the prompt position of its first placeholder
    ids: tuple[int, ...]  # its placeholder ids (`placeholder_ids`)
    pixels: torch.Tensor  # the encoder's input, on the CPU (kept for a prefill that is retracted and redone)
    embeds: torch.Tensor | None = None  # [len(ids), hidden] while its prefill is under way

    @property
    def end(self) -> int:
        return self.start + len(self.ids)


def placeholder_ids(digest: bytes, n: int) -> tuple[int, ...]:
    """n ids in [-2^31, -1] drawn from an image's digest: one per position, so two images share a cached prefix only
    if they agree on every placeholder in it. int32, which the SSD tier's block hashes pack (cache/ssd_store.py)."""
    words = array("I", hashlib.shake_256(digest).digest(4 * n))
    return tuple(-1 - (v & 0x7FFF_FFFF) for v in words)


def text_ids(ids: list[int], image_token: int) -> list[int]:
    """`ids` as the model reads them: every placeholder the image token."""
    return [t if t >= 0 else image_token for t in ids]


def check(prompt_ids: list[int], media: tuple[Image, ...]) -> list[int]:
    """`prompt_ids` with each image's span checked against its placeholder ids and zeroed (what remains must be
    vocabulary ids); raises ValueError on a span that does not match or overlaps another."""
    out = list(prompt_ids)
    covered = 0
    for image in sorted(media, key=lambda m: m.start):
        if image.start < covered or out[image.start:image.end] != list(image.ids) or not image.ids:
            raise ValueError("image placeholders do not match the prompt")
        out[image.start:image.end] = [0] * len(image.ids)
        covered = image.end
    return out


def rows(media: tuple[Image, ...], start: int, end: int, offset: int, encode) -> tuple[list[int], list[torch.Tensor]]:
    """The forward rows of prompt positions [start, end) placed from row `offset` that hold image placeholders, and their
    embeddings in row order. `encode(pixels)` runs the encoder the first time an image's span is reached; an image whose
    span ends within the chunk drops its embeddings (the slice taken here keeps them alive for this forward)."""
    out_rows: list[int] = []
    pieces: list[torch.Tensor] = []
    for image in media:
        a, b = max(image.start, start), min(image.end, end)
        if a >= b:
            continue
        if image.embeds is None:
            image.embeds = encode(image.pixels)
            if image.embeds.shape[0] != len(image.ids):
                raise ValueError(f"the encoder made {image.embeds.shape[0]} embeddings for {len(image.ids)} placeholders")
        out_rows += range(offset + a - start, offset + b - start)
        pieces.append(image.embeds[a - image.start:b - image.start])
        if b == image.end:
            image.embeds = None
    return out_rows, pieces
