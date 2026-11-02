"""Image parts of a chat request, for a server whose target serves a vision encoder (`model.vision`:
model/nemotron/vision.py).

An image arrives as an OpenAI `image_url` (or `input_image`) part holding a `data:` URL, or as an Anthropic `image`
block with a base64 source (or a `data:` URL); the server fetches nothing. Each is decoded (Pillow), converted to RGB
and preprocessed for the encoder (`prepare`), in the order the chat template renders them. The template writes one
`<image>` token per image; `expand` replaces each with `<img>`, the image's placeholder ids (engine/media.py) and
`</img>`, as the checkpoint's processor expands it, and returns the images for the engine.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
from dataclasses import dataclass

import numpy as np
import torch

from sharar.engine.media import Image, placeholder_ids
from sharar.engine.sampler import InvalidRequest

IMAGE_PARTS = ("image_url", "input_image", "image")
MAX_IMAGE_BYTES = 32 * 2**20  # an encoded image
MAX_IMAGE_PIXELS = 64_000_000  # a decoded one (the encoder resizes it to at most 13,312 patches, 3.4 MP)


@dataclass
class Prepared:
    digest: bytes  # SHA-256 of the encoder's identity and the encoded bytes
    pixels: torch.Tensor  # the encoder's input
    tokens: int  # the embeddings it becomes


def image_bytes(part: dict, where: str) -> bytes:
    """The encoded image of an OpenAI or Anthropic image part."""
    kind = part.get("type")
    if kind == "image":  # Anthropic: {"source": {"type": "base64", "media_type": ..., "data": ...}} or a URL source
        source = part.get("source")
        if not isinstance(source, dict):
            raise InvalidRequest(f"{where}: an image block needs a source")
        if source.get("type") == "base64":
            return _b64(source.get("data"), where)
        url = source.get("url")
    else:  # OpenAI: {"image_url": {"url": ...}} (chat) or {"image_url": "..."} (Responses-style input_image)
        url = part.get("image_url")
        url = url.get("url") if isinstance(url, dict) else url
    if not isinstance(url, str) or not url.startswith("data:"):
        raise InvalidRequest(f"{where}: send images inline, as a data: URL or base64; this server does not fetch URLs")
    header, _, data = url.partition(",")
    if not header.endswith(";base64"):
        raise InvalidRequest(f"{where}: a data: URL image must be base64")
    return _b64(data, where)


def _b64(data, where: str) -> bytes:
    if not isinstance(data, str):
        raise InvalidRequest(f"{where}: image data must be a base64 string")
    if len(data) > MAX_IMAGE_BYTES * 4 // 3 + 4:
        raise InvalidRequest(f"{where}: the image is larger than {MAX_IMAGE_BYTES >> 20} MiB")
    try:
        return base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise InvalidRequest(f"{where}: the image data is not valid base64") from None


def decode(raw: bytes, where: str) -> np.ndarray:
    """The image as RGB [H, W, 3] uint8 (the first frame of an animation; alpha dropped, as the reference processor)."""
    from PIL import Image as PILImage, UnidentifiedImageError
    unreadable = (UnidentifiedImageError, OSError, ValueError, PILImage.DecompressionBombError)
    try:
        img = PILImage.open(io.BytesIO(raw))  # reads the header only
    except UnidentifiedImageError:
        raise InvalidRequest(f"{where}: not a readable image (PNG, JPEG, WebP or GIF)") from None
    except unreadable as exc:
        raise InvalidRequest(f"{where}: not a readable image ({exc})") from None
    with img:
        if img.width * img.height > MAX_IMAGE_PIXELS:
            raise InvalidRequest(f"{where}: the image has more than {MAX_IMAGE_PIXELS:,} pixels")
        try:
            return np.array(img if img.mode == "RGB" else img.convert("RGB"), dtype=np.uint8)  # writable, for torch
        except unreadable as exc:
            raise InvalidRequest(f"{where}: not a readable image ({exc})") from None


def prepare(messages: list[dict], vision) -> list[Prepared]:
    """Every image of `messages`, in the order the template renders them, preprocessed for `vision`'s encoder (an image
    sent twice is decoded once)."""
    out: list[Prepared] = []
    seen: dict[bytes, Prepared] = {}
    for i, message in enumerate(messages):
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for j, part in enumerate(content):
            if not isinstance(part, dict) or part.get("type") not in IMAGE_PARTS:
                continue
            raw = image_bytes(part, f"messages[{i}].content[{j}]")
            digest = hashlib.sha256(vision.identity + raw).digest()
            if digest not in seen:
                pixels = vision.preprocess(decode(raw, f"messages[{i}].content[{j}]"))
                seen[digest] = Prepared(digest, pixels, vision.tokens(pixels))
            out.append(seen[digest])
    return out


def expand(ids: list[int], boundary: int, images: list[Prepared], vision) -> tuple[list[int], int, tuple[Image, ...]]:
    """`ids` with the n-th `<image>` token replaced by `<img>`, the n-th image's placeholders and `</img>`; the message
    boundary moved past what was inserted before it (0 stays unknown); and the engine's images."""
    marks = [i for i, t in enumerate(ids) if t == vision.image_token]
    if len(marks) != len(images):
        raise InvalidRequest(f"the prompt has {len(marks)} <image> markers for {len(images)} images")
    out: list[int] = []
    media: list[Image] = []
    moved, last = boundary, 0
    for mark, image in zip(marks, images):
        out += ids[last:mark]
        out.append(vision.image_start)
        media.append(Image(start=len(out), ids=placeholder_ids(image.digest, image.tokens), pixels=image.pixels))
        out += media[-1].ids
        out.append(vision.image_end)
        last = mark + 1
        if boundary and mark < boundary:
            moved += image.tokens + 1
    out += ids[last:]
    return out, moved, tuple(media)
