"""Images in a prompt, for targets with a vision encoder (`model.vision`: engine/contracts.VisionEncoder; Nemotron
Super and Bolt: model/nemotron/vision.py).

The API expands each image into the encoder's placeholder tokens and gives them ids the vocabulary does not have:
negative int32s drawn from the image's content (`placeholder_ids`). The prompt the engine stores, caches and
traces carries those ids, so the radix prefix cache, the SSD tier and the recurrent-state snapshots key a prefix
on the images in it, as on its text, and never resume one image's state for another. Where ids reach the model they
become the image token (`text_ids`), and the prefill forward replaces the embedding rows of an image's placeholders
with the encoder's output (`ForwardContext.media_rows`, `media_embeds`).

The API reads only an image's header (`size`: the placeholder count follows from it) and keeps the encoded bytes. The
executor's `Encoder` decodes and preprocesses an image on a worker thread once its request is admitted, and only
when its span is past the cached prefix; runs the encode on a side stream ahead of the prefill chunk that reads it
(the scheduler ends a chunk before an image still encoding while other requests decode, so their rounds go on); and
drops the embeddings once the span is prefilled.
"""

from __future__ import annotations

import hashlib
import io
from array import array
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch

FORMATS = ("PNG", "JPEG", "WEBP", "GIF")
MAX_IMAGE_PIXELS = 64_000_000  # a decoded image (the encoder resizes it to at most 13,312 patches, 3.4 MP)
# A video frame pair's `data`: the magic, then a little-endian (h, w) header, then both frames' raw RGB bytes —
# self-contained bytes a retracted prefill can decode again, as an image's encoded file is (api/media.py writes them).
PAIR_MAGIC = b"\x93hibana-video-pair\x00"  # 20 bytes; no image format starts with 0x93


def pair_bytes(a: np.ndarray, b: np.ndarray) -> bytes:
    """Two RGB frames [H, W, 3] uint8 as one pair's `data` bytes."""
    if a.shape != b.shape or a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("a video pair is two same-size RGB frames")
    return PAIR_MAGIC + np.int64(a.shape[0]).tobytes() + np.int64(a.shape[1]).tobytes() + a.tobytes() + b.tobytes()


def decode_pair(raw: bytes) -> np.ndarray:
    """A pair's `data` back to its two RGB frames [2, H, W, 3] uint8 (writable, for torch)."""
    head = len(PAIR_MAGIC) + 16
    if len(raw) < head or (len(raw) - head) % 2:
        raise ValueError("truncated video pair")
    h, w = np.frombuffer(raw, dtype=np.int64, count=2, offset=len(PAIR_MAGIC))
    frame = h * w * 3
    if len(raw) - head != 2 * frame:
        raise ValueError("truncated video pair")
    return np.frombuffer(raw, dtype=np.uint8, offset=head).reshape(2, h, w, 3).copy()
# EXIF orientations and the transposes that undo them (Pillow's `ImageOps.exif_transpose`)
_TRANSPOSE = {2: "FLIP_LEFT_RIGHT", 3: "ROTATE_180", 4: "FLIP_TOP_BOTTOM", 5: "TRANSPOSE", 6: "ROTATE_270",
              7: "TRANSVERSE", 8: "ROTATE_90"}


@dataclass(eq=False)
class Image:
    start: int  # the prompt position of its first placeholder
    ids: tuple[int, ...]  # its placeholder ids (`placeholder_ids`)
    data: bytes  # the encoded image, kept for a prefill that is retracted and redone
    pixels: Future | None = None  # the encoder's input, being made on the worker thread
    embeds: torch.Tensor | None = None  # [len(ids), hidden], from its encode until its span is prefilled
    encoded: torch.cuda.Event | None = None  # recorded on the side stream after the encode, until a prefill waits on it
    grid: tuple[int, int] | None = None  # the encoder's merged patch grid (rows, cols); multimodal-RoPE targets read it
    pair: bool = False  # a video frame pair: `data` is `pair_bytes` (two frames, one temporal step), not an image file

    @property
    def end(self) -> int:
        return self.start + len(self.ids)


def _open(raw: bytes):
    """The image's header (Pillow reads no pixels) and its EXIF orientation as stored before the pixel data."""
    from PIL import Image as PILImage, UnidentifiedImageError
    try:
        img = PILImage.open(io.BytesIO(raw), formats=FORMATS)
        orientation = img.getexif().get(0x0112, 1) if "exif" in img.info else 1
    except UnidentifiedImageError:
        raise ValueError("not a readable image (PNG, JPEG, WebP or GIF)") from None
    except (OSError, ValueError, SyntaxError, PILImage.DecompressionBombError) as exc:
        raise ValueError(f"not a readable image ({exc})") from None
    if img.width * img.height > MAX_IMAGE_PIXELS:
        img.close()
        raise ValueError(f"the image has more than {MAX_IMAGE_PIXELS:,} pixels")
    return img, orientation


def size(raw: bytes) -> tuple[int, int]:
    """(width, height) of the image `decode` makes of `raw`, from its header; ValueError if it is not one."""
    img, orientation = _open(raw)
    with img:
        return img.size[::-1] if orientation in (5, 6, 7, 8) else img.size


def decode(raw: bytes) -> np.ndarray:
    """The image as RGB [H, W, 3] uint8: upright by its EXIF orientation, the first frame of an animation, alpha dropped
    (as the reference processor's RGB conversion); ValueError if it does not decode."""
    if raw.startswith(PAIR_MAGIC):
        return decode_pair(raw)  # a video pair: [2, H, W, 3] for `preprocess_pair`
    return _decode_image(raw)


def _decode_image(raw: bytes) -> np.ndarray:
    from PIL import Image as PILImage
    img, orientation = _open(raw)
    with img:
        try:
            if orientation in _TRANSPOSE:
                img = img.transpose(getattr(PILImage.Transpose, _TRANSPOSE[orientation]))
            return np.array(img if img.mode == "RGB" else img.convert("RGB"), dtype=np.uint8)  # writable, for torch
        except (OSError, ValueError, SyntaxError) as exc:
            raise ValueError(f"not a readable image ({exc})") from None


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


class Encoder:
    """The executor's side of `vision` (engine/contracts.VisionEncoder): pixels on a worker thread, encodes on a side
    stream, and each prefill chunk's image rows."""

    def __init__(self, vision, device: torch.device):
        self.vision = vision
        self.stream = torch.cuda.Stream(device) if device.type == "cuda" else None
        self._worker = ThreadPoolExecutor(1, thread_name_prefix="hibana-images")

    def prefetch(self, media: tuple[Image, ...], pos: int) -> None:
        """Start making the pixels of every image whose span ends past prompt position `pos` (a request's cached prefix
        at admission): an image inside the prefix is never decoded."""
        for image in media:
            if image.end > pos and image.embeds is None and image.pixels is None:
                if image.pair:  # a video pair: its merged grid says the clip's resize target
                    prep = lambda frames, g=image.grid: self.vision.preprocess_pair_grid(frames, g)
                    image.pixels = self._worker.submit(lambda data, p=prep: p(decode_pair(data)), image.data)
                else:
                    image.pixels = self._worker.submit(lambda data: self.vision.preprocess(_decode_image(data)),
                                                       image.data)

    def ready(self, media: tuple[Image, ...], start: int, end: int, wait: bool) -> int:
        """Where a prefill chunk of prompt positions [start, end) can end: before the first image in it whose encode has
        not finished, or at `end` with `wait` (its forward then waits for the encode on the GPU). Starts the encode of
        each image in the chunk or the next one whose pixels are made (or, with `wait`, waits for them)."""
        stop = end
        for image in media:
            if image.end <= start or image.start >= 2 * end - start:
                continue
            if image.embeds is None:
                self.prefetch((image,), start)
                if not (wait or image.pixels.done()):
                    stop = min(stop, image.start)
                    continue
                self._encode(image)
            if not wait and image.encoded is not None and not image.encoded.query():
                stop = min(stop, image.start)
        return max(stop, start)

    def rows(self, media: tuple[Image, ...], start: int, end: int, offset: int) -> tuple[list[int], list[torch.Tensor]]:
        """The forward rows of prompt positions [start, end) placed from row `offset` that hold image placeholders, and
        their embeddings in row order, encoding an image here if `ready` did not. An image whose span ends within the
        chunk drops its embeddings (the slice taken here keeps them alive for this forward)."""
        out_rows: list[int] = []
        pieces: list[torch.Tensor] = []
        for image in media:
            a, b = max(image.start, start), min(image.end, end)
            if a >= b:
                continue
            if image.embeds is None:
                self.prefetch((image,), start)
                self._encode(image)
            if image.encoded is not None:  # the forward's stream waits for the encode and owns the embeddings after it
                torch.cuda.current_stream().wait_event(image.encoded)
                image.embeds.record_stream(torch.cuda.current_stream())
                image.encoded = None
            out_rows += range(offset + a - start, offset + b - start)
            pieces.append(image.embeds[a - image.start:b - image.start])
            if b == image.end:
                image.embeds = None
        return out_rows, pieces

    def _encode(self, image: Image) -> None:
        pixels, image.pixels = image.pixels.result(), None
        with torch.cuda.stream(self.stream) if self.stream is not None else nullcontext():
            encode = self.vision.encode_pair if image.pair else self.vision.encode
            image.embeds = encode(pixels)
            if self.stream is not None:
                image.encoded = torch.cuda.Event()
                image.encoded.record(self.stream)
        if image.embeds.shape[0] != len(image.ids):
            raise ValueError(f"the encoder made {image.embeds.shape[0]} embeddings for {len(image.ids)} placeholders")
