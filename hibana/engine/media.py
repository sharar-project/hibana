"""Images in a prompt, for targets with a vision encoder (`model.vision`: engine/contracts.VisionEncoder; Nemotron
Super and Bolt: model/nemotron/vision.py).

The API expands each image into the encoder's placeholder tokens and gives them ids the vocabulary does not have:
negative int32s drawn from the image's content (`placeholder_ids`). The prompt the engine stores, caches and
traces carries those ids, so the radix prefix cache, the SSD tier and the recurrent-state snapshots key a prefix
on the images in it, as on its text, and never resume one image's state for another. Where ids reach the model they
become the image token (`text_ids`), and the prefill forward replaces the embedding rows of an image's placeholders
with the encoder's output (`ForwardContext.media_rows`, `media_embeds`).

A video (a target whose encoder has a `video_token`) is a run of such images, one a frame pair (`Image.pair`): the
API has decoded its frames, its `data` is the two frames (`pair_bytes`), and the encoder's `preprocess_pair` and
`encode_pair` take the place of `preprocess` and `encode`. Its placeholders are content-derived as an image's, so
the caches treat a pair as an image.

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

# Decoding follows vLLM's (`vllm/multimodal/image.py`), so a request reads the same pixels on both servers: its Pillow
# allowlist, EXIF orientation and transparency over white.
FORMATS = ("AVIF", "BLP", "BMP", "CUR", "DCX", "DDS", "DIB", "FITS", "FLI", "FTEX", "GBR", "GIF", "ICNS", "ICO", "IM",
           "IMT", "IPTC", "JPEG", "JPEG2000", "MCIDAS", "MSP", "PCD", "PCX", "PIXAR", "PNG", "PPM", "PSD", "QOI", "SGI",
           "SPIDER", "SUN", "TGA", "TIFF", "WEBP", "XBM", "XPM", "XVTHUMB")
# A decoded image (the encoders resize it to at most 16.7 MP); vLLM's 179 MP costs ~1.8 GB of host memory a decode.
MAX_IMAGE_PIXELS = 64_000_000
# EXIF orientations and the transposes that undo them (Pillow's `ImageOps.exif_transpose`)
_TRANSPOSE = {2: "FLIP_LEFT_RIGHT", 3: "ROTATE_180", 4: "FLIP_TOP_BOTTOM", 5: "TRANSPOSE", 6: "ROTATE_270",
              7: "TRANSVERSE", 8: "ROTATE_90"}
BACKGROUND = (255, 255, 255)
# A video frame pair's `data`: the magic, then a little-endian int64 (h, w), then both frames' RGB bytes: self-contained
# bytes, as an image's encoded file is, that a prefill retracted and redone decodes again (api/media.py makes them)
PAIR_MAGIC = b"\x93hibana-video-pair\x00"


@dataclass(eq=False)
class Image:
    start: int  # the prompt position of its first placeholder
    ids: tuple[int, ...]  # its placeholder ids (`placeholder_ids`)
    data: bytes  # the encoded image, kept for a prefill that is retracted and redone
    pixels: Future | None = None  # the encoder's input, being made on the worker thread
    embeds: torch.Tensor | None = None  # [len(ids), hidden], from its encode until its span is prefilled
    encoded: torch.cuda.Event | None = None  # recorded on the side stream after the encode, until a prefill waits on it
    grid: tuple[int, int] | None = None  # (rows, cols) of its embeddings
    pair: bool = False  # a video frame pair: `data` is `pair_bytes`, for the encoder's `preprocess_pair` and `encode_pair`

    @property
    def end(self) -> int:
        return self.start + len(self.ids)


def _open(raw: bytes):
    """The image's header (Pillow reads no pixels) and its EXIF orientation as stored before the pixel data."""
    from PIL import Image as PILImage, UnidentifiedImageError
    try:
        img = PILImage.open(io.BytesIO(raw), formats=FORMATS)
        orientation = _orientation(img)
    except UnidentifiedImageError:
        raise ValueError("not a readable image (an unknown or refused format)") from None
    except Exception as exc:  # noqa: BLE001 - the allowlist's rarer plugins raise struct.error, KeyError and others
        raise ValueError(f"not a readable image ({exc})") from None
    if img.width * img.height > MAX_IMAGE_PIXELS:
        img.close()
        raise ValueError(f"the image has more than {MAX_IMAGE_PIXELS:,} pixels")
    return img, orientation


def _orientation(img) -> int:
    """The EXIF orientation still to apply, stored before the pixel data; 1 when missing or unreadable. Pillow's base
    `getexif`: PNG's override decodes every pixel to look for an eXIf chunk after them (92 ms for a 4000 x 4000 PNG)."""
    from PIL import Image as PILImage
    if img.format == "TIFF":  # Pillow reports a TIFF's upright size and rotates its pixels as it loads them
        return 1
    try:
        return PILImage.Image.getexif(img).get(0x0112, 1)
    except Exception:  # noqa: BLE001 - a truncated file: `decode` refuses its pixels
        return 1


def size(raw: bytes) -> tuple[int, int]:
    """(width, height) of the image `decode` makes of `raw`, from its header; ValueError if it is not one."""
    img, orientation = _open(raw)
    with img:
        return img.size[::-1] if orientation in (5, 6, 7, 8) else img.size


def decode(raw: bytes) -> np.ndarray:
    """The image as RGB [H, W, 3] uint8: upright by its EXIF orientation, the first frame of an animation, transparency
    composited over white (bitwise as vLLM); ValueError if it does not decode."""
    from PIL import Image as PILImage
    img, orientation = _open(raw)
    with img:
        try:
            # the transpose `exif_transpose` makes, by the orientation `size` read, so the two always agree
            if orientation in _TRANSPOSE:
                img = img.transpose(getattr(PILImage.Transpose, _TRANSPOSE[orientation]))
            img.load()
            if img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
                rgba = img if img.mode == "RGBA" else img.convert("RGBA")
                rgb = PILImage.new("RGB", rgba.size, BACKGROUND)
                rgb.paste(rgba, mask=rgba.split()[3])
            else:
                rgb = img if img.mode == "RGB" else img.convert("RGB")
            return np.array(rgb, dtype=np.uint8)  # writable, for torch
        except Exception as exc:  # noqa: BLE001 - as in `_open`
            raise ValueError(f"not a readable image ({exc})") from None


def pair_bytes(a: np.ndarray, b: np.ndarray) -> bytes:
    """Two RGB frames [H, W, 3] uint8, a video's temporal step, as one pair's `data`."""
    if a.ndim != 3 or a.shape[2] != 3 or a.shape != b.shape or a.dtype != np.uint8 or b.dtype != np.uint8:
        raise ValueError("a video pair is two same-size RGB frames")
    return PAIR_MAGIC + np.array(a.shape[:2], dtype=np.int64).tobytes() + a.tobytes() + b.tobytes()


def decode_pair(raw: bytes) -> np.ndarray:
    """A pair's `data` as its two RGB frames [2, H, W, 3] uint8 (writable, for torch); ValueError if truncated."""
    head = len(PAIR_MAGIC) + 16
    if not raw.startswith(PAIR_MAGIC) or len(raw) < head:
        raise ValueError("truncated video pair")
    h, w = np.frombuffer(raw, dtype=np.int64, count=2, offset=len(PAIR_MAGIC))
    if h < 1 or w < 1 or len(raw) - head != 2 * h * w * 3:
        raise ValueError("truncated video pair")
    return np.frombuffer(raw, dtype=np.uint8, offset=head).reshape(2, h, w, 3).copy()


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
                if image.pair:
                    image.pixels = self._worker.submit(lambda g, data: self.vision.preprocess_pair(decode_pair(data), g),
                                                       image.grid, image.data)
                else:
                    image.pixels = self._worker.submit(lambda data: self.vision.preprocess(decode(data)), image.data)

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
            image.embeds = (self.vision.encode_pair if image.pair else self.vision.encode)(pixels)
            if self.stream is not None:
                image.encoded = torch.cuda.Event()
                image.encoded.record(self.stream)
        if image.embeds.shape[0] != len(image.ids):
            raise ValueError(f"the encoder made {image.embeds.shape[0]} embeddings for {len(image.ids)} placeholders")
