"""Image parts of a chat request, for a server whose target serves a vision encoder (`model.vision`:
engine/contracts.VisionEncoder, model/nemotron/vision.py).

An image arrives as an OpenAI `image_url` (or `input_image`) part holding a `data:` URL, or as an Anthropic `image`
block with a base64 source (or a `data:` URL); the server fetches nothing. `prepare` reads each image's header, which
sets its placeholder count, and keeps the encoded bytes; the engine decodes an image only when its span is encoded
(engine/media.py), so an image in a cached prefix is never decoded again. A request's new image is decoded once here
as well, so that one that does not decode is a 400 (or, in a tool result, a note) rather than a failed engine step.

The template writes `<image>` for each image part. `escape` keeps the same text in a message from counting as one (the
template would then place no image itself), and `Markers` encodes the render with only the template's `<image>` as the
image token: user text reads as text. `expand` replaces each marker with `<img>`, the image's placeholder ids and
`</img>`, as the checkpoint's processor expands it, and returns the images for the engine.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
from dataclasses import dataclass

import numpy as np

from hibana.api.messages import MEDIA_NOTE
from hibana.engine import media
from hibana.engine.sampler import InvalidRequest

IMAGE_PARTS = ("image_url", "input_image", "image")
VIDEO_PARTS = ("video_url", "input_video", "video")  # OpenAI `video_url` / Responses `input_video` / bare `video`
MAX_IMAGE_BYTES = 32 * 2**20  # an encoded image
MAX_VIDEO_BYTES = 64 * 2**20  # an encoded video
MAX_IMAGES = 100  # in one request (Anthropic's limit); their tokens are bounded by the context
MAX_VIDEOS = 4  # in one request (upstream TF's prior art caps videos at 4; one is many frames of tokens)
MARKER = "<image>"  # what a marker-convention template writes for an image part
_ESCAPED = "image>"  # a "<image>" of the request's own text while it is rendered (a private-use character for "<")
_PRIVATE = ""  # a template's own image tokens in the request's text are escaped the same way, first char swapped


def _protected(vision) -> tuple[tuple[str, str], ...]:
    """(the string, its escaped stand-in) for everything that places an image in a render: the
    marker, and — for a token-convention encoder (`VisionEncoder.template_tokens`: Flash-Next's
    Qwen template writes `<|vision_start|><|image_pad|><|vision_end|>` itself) — the token
    strings, so a request's own text never places an image."""
    pairs = [(MARKER, _ESCAPED)]
    for name in ("image_token_text", "image_start_text", "image_end_text", "video_token_text"):
        text = getattr(vision, name, None)
        if text:
            pairs.append((text, _PRIVATE + text[1:]))
    return tuple(pairs)
_DECODED: dict[bytes, None] = {}  # digests of images that decoded here: a repeat is not decoded again to check it


@dataclass
class Prepared:
    digest: bytes  # SHA-256 of the encoder's identity and the encoded bytes
    tokens: int  # the prompt tokens it becomes (a video's: its pairs' placeholders and timestamp runs)
    data: bytes  # the encoded image
    grid: tuple[int, int] | None = None  # the encoder's merged patch grid, for multimodal-RoPE positions
    video: bool = False  # a video: `pairs` are its frame pairs' bytes, `times` their timestamps
    pairs: tuple[bytes, ...] = ()  # engine/media.pair_bytes a temporal step (api/media.py makes them on decode)
    times: tuple[float, ...] = ()  # one timestamp a pair, seconds; the prompt writes `<t seconds>` before each span


def image_bytes(part: dict) -> bytes:
    """The encoded image of an OpenAI or Anthropic image part; ValueError saying why there is none."""
    if part.get("type") == "image":  # Anthropic: {"source": {"type": "base64", "media_type": ..., "data": ...}} or a URL source
        source = part.get("source")
        if not isinstance(source, dict):
            raise ValueError("an image block needs a source")
        if source.get("type") == "base64":
            return _b64(source.get("data"))
        url = source.get("url")
    else:  # OpenAI: {"image_url": {"url": ...}} (chat) or {"image_url": "..."} (Responses-style input_image)
        url = part.get("image_url")
        url = url.get("url") if isinstance(url, dict) else url
    if not isinstance(url, str) or not url.startswith("data:"):
        raise ValueError("send images inline, as a data: URL or base64; this server does not fetch URLs")
    header, _, data = url.partition(",")
    if not header.endswith(";base64"):
        raise ValueError("a data: URL image must be base64")
    return _b64(data)


def _b64(data) -> bytes:
    if not isinstance(data, str):
        raise ValueError("image data must be a base64 string")
    if len(data) > MAX_IMAGE_BYTES * 4 // 3 + 4:
        raise ValueError(f"the image is larger than {MAX_IMAGE_BYTES >> 20} MiB")
    try:
        return base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("the image data is not valid base64") from None


def video_bytes(part: dict) -> bytes:
    """The encoded video of an OpenAI video part (`video_url` / `input_video`, a `data:` URL); ValueError saying
    why there is none. The server fetches nothing; decode (ffmpeg) happens in `_read_video`."""
    url = part.get("video_url") or part.get("input_video") or part.get("video")
    url = url.get("url") if isinstance(url, dict) else url
    if not isinstance(url, str) or not url.startswith("data:"):
        raise ValueError("send videos inline, as a base64 data: URL; this server does not fetch URLs")
    header, _, data = url.partition(",")
    if not header.endswith(";base64"):
        raise ValueError("a data: URL video must be base64")
    return _b64_bytes(data, MAX_VIDEO_BYTES)


def _b64_bytes(data, limit: int) -> bytes:
    if not isinstance(data, str):
        raise ValueError("the data must be a base64 string")
    if len(data) > limit * 4 // 3 + 4:
        raise ValueError(f"the media is larger than {limit >> 20} MiB")
    try:
        return base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("the data is not valid base64") from None


def _ffmpeg(*args: str, stdin: bytes = b"") -> bytes:
    """`ffmpeg`/`ffprobe` run on `stdin` (the encoded video); output bytes, or ValueError with its tail."""
    import subprocess
    cmd = list(args)
    try:
        p = subprocess.run(cmd, input=stdin, capture_output=True, timeout=300)
    except FileNotFoundError:
        raise ValueError("no ffmpeg on this server; videos are not served") from None
    except subprocess.TimeoutExpired:
        raise ValueError("decoding the video took too long") from None
    if p.returncode != 0:
        raise ValueError("not a decodable video (" + p.stderr.decode("utf-8", "replace").strip().splitlines()[-1][:120] + ")") from None
    return p.stdout


def _video_info(raw: bytes) -> tuple[int, float, int, int]:
    """(total frames, source fps, width, height) of an encoded video, by ffprobe (counting frames exactly: the
    sample count, and so the token count, depends on it); ValueError if ffprobe cannot read it."""
    import json
    out = _ffmpeg("ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                  "-show_entries", "stream=nb_read_frames,r_frame_rate,width,height", "-of", "json", "-i", "-",
                  stdin=raw)
    try:
        stream = json.loads(out)["streams"][0]
        total = int(stream["nb_read_frames"])
        num, _, den = stream["r_frame_rate"].partition("/")
        fps = float(num) / float(den or 1)
        return total, fps, int(stream["width"]), int(stream["height"])
    except (KeyError, IndexError, ValueError, ZeroDivisionError):
        raise ValueError("not a decodable video") from None


def _video_frames(raw: bytes, indices: list[int], width: int, height: int) -> np.ndarray:
    """The video's sampled frames as RGB [n, H, W, 3] uint8, ffmpeg `select` on the original frame numbers (the
    processor's `np.linspace(...).round()` indices, repeats allowed)."""
    expr = "+".join(f"eq(n,{i})" for i in indices)
    out = _ffmpeg("ffmpeg", "-v", "error", "-i", "pipe:0", "-vf", f"select='{expr}'", "-fps_mode", "passthrough",
                  "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1", stdin=raw)
    n = len(indices)
    if len(out) != n * width * height * 3:
        raise ValueError("the video did not decode to the frames it probed as")
    return np.frombuffer(out, dtype=np.uint8).reshape(n, height, width, 3)


def _read_video(part: dict, vision, seen: dict[bytes, Prepared], decode: bool) -> Prepared:
    """A video part as a `Prepared`: fps-2.0 uniform sampling (the processor's default) capped at
    `HIBANA_VIDEO_MAX_FRAMES` (default 32), one `<|video_pad|>` span a frame pair with `<t seconds>` before it
    (the Qwen3-VL prompt the processor and the deployed vLLM build both write). `decode` also decodes the frames
    here (a bad video is a 400, not a failed engine step) and makes each pair's bytes."""
    from hibana.model.flashnext import vision as fv
    raw = video_bytes(part)
    digest = hashlib.sha256(vision.identity + vision.video_identity + raw).digest()
    if digest not in seen:
        total, fps, w, h = _video_info(raw)
        indices = fv.sample_indices(total, fps)
        t, rows, cols = fv.video_grid(len(indices), w, h)
        times = fv.timestamps(indices, fps)
        prep = Prepared(digest, t * rows * cols + t * fv.TS_TOKENS, raw, (rows, cols), video=True, times=times)
        if decode:
            frames = _video_frames(raw, indices, w, h)
            if len(frames) < 2:  # a single frame pairs with itself, as the processor pads
                frames = np.concatenate([frames, frames[-1:]])
            pad = -len(frames) % 2  # the processor repeats the last frame of an odd clip
            if pad:
                frames = np.concatenate([frames, np.repeat(frames[-1:], pad, axis=0)])
            prep.pairs = tuple(media.pair_bytes(frames[i], frames[i + 1]) for i in range(0, len(frames), 2))
        seen[digest] = prep
    return seen[digest]


def _read(part: dict, vision, seen: dict[bytes, Prepared], decode: bool) -> Prepared:
    raw = image_bytes(part)
    digest = hashlib.sha256(vision.identity + raw).digest()
    if digest not in seen:
        tokens = vision.tokens(*media.size(raw))
        if decode and digest not in _DECODED:
            media.decode(raw)
            if len(_DECODED) >= 4096:
                _DECODED.clear()
            _DECODED[digest] = None
        grid = vision.grid(*media.size(raw)) if hasattr(vision, "grid") else None
        seen[digest] = Prepared(digest, tokens, raw, grid)
    return seen[digest]


MEDIA_PARTS = IMAGE_PARTS + VIDEO_PARTS


def prepare(messages: list[dict], vision, decode: bool = True) -> tuple[list[dict], list[Prepared]]:
    """`messages` with each image or video of a tool message that cannot be read replaced by a note saying why (an
    agent's screenshot that failed is not the user's mistake), and every one of them, in the order the template
    renders them (an image sent twice is read once). `decode`: also decode each not decoded before (count_tokens
    reads headers — and, for videos, the frame probe — only). One elsewhere that cannot be read is a 400."""
    videos = getattr(vision, "video_token", None) is not None  # the target serves video pairs
    out: list[dict] = []
    images: list[Prepared] = []
    seen: dict[bytes, Prepared] = {}
    for i, message in enumerate(messages):
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list) or not any(isinstance(p, dict) and p.get("type") in MEDIA_PARTS for p in content):
            out.append(message)
            continue
        parts = []
        for j, part in enumerate(content):
            kind = part.get("type") if isinstance(part, dict) else None
            if isinstance(part, dict) and kind in MEDIA_PARTS:
                try:
                    if kind in VIDEO_PARTS:
                        if not videos:
                            raise ValueError("this model does not read videos")
                        images.append(_read_video(part, vision, seen, decode))
                        part = {"type": "video"}  # what the template's own video branch matches
                    else:
                        images.append(_read(part, vision, seen, decode))
                except ValueError as exc:
                    if message.get("role") != "tool":
                        raise InvalidRequest(f"messages[{i}].content[{j}]: {exc}") from None
                    part = {"type": "text", "text": MEDIA_NOTE.format(kind="video" if kind in VIDEO_PARTS else "image",
                                                                     why=exc)}
            parts.append(part)
        out.append({**message, "content": parts})
    n = sum(1 for m in images if m.video)
    if n > MAX_VIDEOS:
        raise InvalidRequest(f"the request has {n} videos; at most {MAX_VIDEOS} are read")
    if len(images) - n > MAX_IMAGES:
        raise InvalidRequest(f"the request has {len(images) - n} images; at most {MAX_IMAGES} are read")
    return out, images


def escape(value, vision=None):
    """`value` (messages or tools, at any nesting) with every image-placing string of its text
    escaped for the render (`_protected`); `vision` None keeps the marker alone."""
    pairs = _protected(vision) if vision is not None else ((MARKER, _ESCAPED),)
    if isinstance(value, str):
        for plain, escaped in pairs:
            value = value.replace(plain, escaped)
        return value
    if isinstance(value, list):
        return [escape(v, vision) for v in value]
    if isinstance(value, dict):
        return {k: escape(v, vision) for k, v in value.items()}
    return value


class Markers:
    """`tok` for a render of escaped messages: each `<image>` the template wrote is the image token, and the rest,
    the escapes undone, is encoded with the image tokens read as text (`<image>`, `<img>` and `</img>` in a message are
    its text). Offsets are the render's (an escape has the length of what it stands for)."""

    def __init__(self, tok, vision):
        self.image_token = vision.image_token
        # The marker's escape is undone to the marker (encoded as text); a token string's escape
        # stays, so the request's own `<|image_pad|>` never encodes as the token that places an image.
        self.undo = {escaped: (plain if plain == MARKER else escaped) for plain, escaped in _protected(vision)}
        # A marker-convention template writes `<image>` where an image goes and the token strings must
        # stay text; a token-convention template (Flash-Next) writes the tokens themselves.
        self.text = tok if getattr(vision, "template_tokens", False) \
            else tok.without_specials({vision.image_token, vision.image_start, vision.image_end})

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        ids, offsets, at = [], [], 0
        for k, piece in enumerate(text.split(MARKER)):
            if k:
                ids.append(self.image_token)
                offsets.append((at, at + len(MARKER)))
                at += len(MARKER)
            for escaped, plain in self.undo.items():
                piece = piece.replace(escaped, plain)
            piece_ids, piece_offsets = self.text.encode_with_offsets(piece)
            ids += piece_ids
            offsets += [(a + at, b + at) for a, b in piece_offsets]
            at += len(piece)
        return ids, offsets

    def encode(self, text: str) -> list[int]:
        return self.encode_with_offsets(text)[0]


def expand(ids: list[int], boundary: int, images: list[Prepared], vision, tok=None) -> tuple[list[int], int, tuple[media.Image, ...]]:
    """`ids` with each n-th image or video marker replaced: an image's `<|image_pad|>` (or `<image>` marker) by
    `<|vision_start|>`, its placeholders and `<|vision_end|>`; a video's `<|video_pad|>` by one such span a frame
    pair, each preceded by its `<t seconds>` timestamp tokens (the Qwen3-VL video prompt). The message boundary
    moves past what was inserted before it (0 stays unknown); returns the prompt and the engine's images (a video's
    pairs among them, engine/media.py)."""
    kinds = {vision.image_token: "image"}
    if getattr(vision, "video_token", None) is not None:
        kinds[vision.video_token] = "video"
    marks = [(i, kinds[t]) for i, t in enumerate(ids) if t in kinds]
    wanted = ["video" if p.video else "image" for p in images]
    if [k for _, k in marks] != wanted:
        for kind in ("image", "video"):
            got, need = sum(1 for _, k in marks if k == kind), wanted.count(kind)
            if got != need:
                raise InvalidRequest(f"the template placed {got} of the request's {need} {kind}s")
        raise InvalidRequest("the template placed the request's media in another order")
    it = iter(images)
    out: list[int] = []
    engine_images: list[media.Image] = []
    moved, last = boundary, 0
    for mark, kind in marks:
        image = next(it)
        if kind == "video":
            at = mark - 1 if mark and ids[mark - 1] == vision.image_start else mark  # the template's own span
            end = mark + 2 if mark + 1 < len(ids) and ids[mark + 1] == vision.image_end else mark + 1
            out += ids[last:at]
            if boundary and mark < boundary:
                moved += _expand_video(ids, at, end, image, vision, tok, out, engine_images)
            else:
                _expand_video(ids, at, end, image, vision, tok, out, engine_images)
            last = end
            continue
        out += ids[last:mark]
        # A token-convention template (Flash-Next) writes `<|vision_start|><|image_pad|><|vision_end|>`
        # itself: the pad token between the start the render already placed and its end, which is
        # then the next id; a marker-convention template gets both added here.
        opens = bool(out) and out[-1] == vision.image_start
        closes = opens and mark + 1 < len(ids) and ids[mark + 1] == vision.image_end
        if not opens:
            out.append(vision.image_start)
        engine_images.append(media.Image(start=len(out), ids=media.placeholder_ids(image.digest, image.tokens), data=image.data,
                                         grid=image.grid))
        out += engine_images[-1].ids
        if closes:
            out.append(vision.image_end)
            last = mark + 2
        else:
            out.append(vision.image_end)
            last = mark + 1
        if boundary and mark < boundary:
            moved += image.tokens + (1 if not closes else -1)
    out += ids[last:]
    return out, moved, tuple(engine_images)


def _expand_video(ids: list[int], at: int, end: int, video: Prepared, vision, tok, out: list[int],
                  engine_images: list) -> int:
    """Append to `out` the video's replacement for ids[at:end] (its `<|video_pad|>`, with the template's
    `<|vision_start|>` and `<|vision_end|>` when they wrap it): one `<t seconds>` run, `<|vision_start|>`, a frame
    pair's placeholders and `<|vision_end|>` a temporal step (the processor's prompt, per frame pair). Returns how
    much the prompt grows. Timestamps encode one run at a time, as the deployed vLLM build does (no token merging
    across boundaries)."""
    if tok is None:
        raise InvalidRequest("the template placed a video the server cannot render")
    rows, cols = video.grid
    was = len(out)
    for f, (seconds, pair) in enumerate(zip(video.times, video.pairs or (None,) * len(video.times))):
        out += tok.encode(f"<{seconds:.1f} seconds>")
        out.append(vision.image_start)
        engine_images.append(media.Image(start=len(out), ids=media.placeholder_ids(
            video.digest + f.to_bytes(4, "big"), rows * cols), data=pair, grid=video.grid, pair=True))
        out += engine_images[-1].ids
        out.append(vision.image_end)
    return (len(out) - was) - (end - at)


