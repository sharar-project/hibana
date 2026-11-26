"""Image parts of a chat request, for a server whose target serves a vision encoder (`model.vision`:
engine/contracts.VisionEncoder, model/nemotron/vision.py).

An image arrives as an OpenAI `image_url` (or `input_image`) part holding a `data:` URL, or as an Anthropic `image`
block with a base64 source (or a `data:` URL); the server fetches nothing. `prepare` reads each image's header, which
sets its placeholder count, and keeps the encoded bytes; the engine decodes an image only when its span is encoded
(engine/media.py), so an image in a cached prefix is never decoded again. A request's new image is decoded once here
as well, so that one that does not decode is a 400 (or, in a tool result, a note) rather than a failed engine step.

The template writes the encoder's `marker` for each image part. `escape` keeps the same text in a message from counting
as one, and `Markers` encodes the render with each marker as the image token and the rest as text. `expand` replaces each
such token with the image start, its placeholder ids and the image end, as the checkpoints' processors do, and returns
the images (with their grids) for the engine.

A video (OpenAI `video_url`, `input_video` or `video`, a `data:` URL) needs an encoder with a `video_token` (the Qwen
targets). It is decoded here, by `ffmpeg` and `ffprobe` on the server, into the frames the checkpoint's video processor
samples, and `expand` replaces its video token with what that processor writes in place of `<|video_pad|>`: for each
frame pair, the text of its timestamp, then the image start, the pair's placeholder ids and the image end. The
template's `<|vision_start|>` before and `<|vision_end|>` after stay, as they do for an image and as transformers'
processor keeps them (5.10 and later expand the video token alone). Every pair is an image to the engine
(engine/media.py `Image.pair`), whose ids follow the video's content and the pair's place in it, so the caches resume
a video's prefix as an image's. A server without `ffmpeg` serves everything else and answers a video with a 400.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
import subprocess
from dataclasses import dataclass

import numpy as np

from hibana.api.messages import MEDIA_NOTE
from hibana.engine import media
from hibana.engine.sampler import InvalidRequest

IMAGE_PARTS = ("image_url", "input_image", "image")
VIDEO_PARTS = ("video_url", "input_video", "video")
PARTS = IMAGE_PARTS + VIDEO_PARTS
MAX_IMAGE_BYTES = 32 * 2**20  # an encoded image
MAX_IMAGES = 100  # in one request (Anthropic's limit); their tokens are bounded by the context
MAX_VIDEOS = 4  # in one request: each is many frames of tokens
FFMPEG_TIMEOUT = 300  # seconds a video's probe or decode may take
_DECODED: dict[bytes, None] = {}  # digests of images that decoded here: a repeat is not decoded again to check it


@dataclass
class Prepared:
    digest: bytes  # SHA-256 of the encoder's identity and the encoded bytes
    tokens: int  # the embeddings it becomes
    data: bytes  # the encoded image
    grid: tuple[int, int] | None = None  # (rows, cols) of embeddings
    video: bool = False  # a video: `grid` and `tokens` are of each of its frame pairs, `times` their timestamps
    times: tuple[float, ...] = ()  # seconds, one a frame pair: the prompt writes `<t seconds>` before each pair's span
    pairs: tuple[bytes, ...] = ()  # a frame pair's bytes each (engine/media.pair_bytes), made on decode


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


def video_bytes(part: dict) -> bytes:
    """The encoded video of an OpenAI video part (`video_url` or `input_video`, holding a `data:` URL or the URL
    itself); ValueError saying why there is none. The server fetches nothing."""
    url = part.get("video_url") or part.get("input_video") or part.get("video")
    url = url.get("url") if isinstance(url, dict) else url
    if not isinstance(url, str) or not url.startswith("data:"):
        raise ValueError("send videos inline, as a data: URL; this server does not fetch URLs")
    header, _, data = url.partition(",")
    if not header.endswith(";base64"):
        raise ValueError("a data: URL video must be base64")
    from hibana.engine.config import current
    return _b64(data, current().video_max_bytes, "video")


def _b64(data, limit: int = MAX_IMAGE_BYTES, kind: str = "image") -> bytes:
    if not isinstance(data, str):
        raise ValueError(f"{kind} data must be a base64 string")
    if len(data) > limit * 4 // 3 + 4:
        raise ValueError(f"the {kind} is larger than {limit >> 20} MiB")
    try:
        return base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError(f"the {kind} data is not valid base64") from None


def _ffmpeg(*args: str, raw: bytes) -> bytes:
    """`ffmpeg` or `ffprobe` reading `raw` from stdin -> its stdout; ValueError with its last line when it fails."""
    try:
        p = subprocess.run([*args], input=raw, capture_output=True, timeout=FFMPEG_TIMEOUT)
    except FileNotFoundError:
        raise ValueError("this server has no ffmpeg: videos are not served") from None
    except subprocess.TimeoutExpired:
        raise ValueError("decoding the video took too long") from None
    if p.returncode != 0:
        lines = p.stderr.decode("utf-8", "replace").strip().splitlines()
        raise ValueError("not a decodable video" + (f" ({lines[-1][:120]})" if lines else "")) from None
    return p.stdout


def _video_info(raw: bytes) -> tuple[int, float, int, int]:
    """(frames, fps, width, height) of a video by ffprobe, which counts the frames exactly (the sample, and so the
    token count, depends on the count)."""
    import json
    out = _ffmpeg("ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                  "stream=nb_read_frames,r_frame_rate,width,height", "-of", "json", "-i", "-", raw=raw)
    try:
        s = json.loads(out)["streams"][0]
        num, _, den = s["r_frame_rate"].partition("/")
        return int(s["nb_read_frames"]), float(num) / float(den or 1), int(s["width"]), int(s["height"])
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError):
        raise ValueError("not a decodable video") from None


def _video_frames(raw: bytes, indices: list[int], width: int, height: int) -> np.ndarray:
    """The video's frames `indices` as RGB [n, H, W, 3] uint8, by ffmpeg's `select` on the source frame numbers
    (a sampled index may repeat)."""
    expr = "+".join(f"eq(n\\,{i})" for i in indices)
    out = _ffmpeg("ffmpeg", "-v", "error", "-i", "pipe:0", "-vf", f"select='{expr}'", "-fps_mode", "passthrough", "-f",
                  "rawvideo", "-pix_fmt", "rgb24", "pipe:1", raw=raw)
    if len(out) != len(indices) * width * height * 3:
        raise ValueError("the video did not decode to the frames it probed as")
    return np.frombuffer(out, dtype=np.uint8).reshape(len(indices), height, width, 3)


def _read_video(part: dict, vision, seen: dict[bytes, Prepared], decode: bool) -> Prepared:
    """A video part as a `Prepared`: the frames its processor samples (2 a second, `vision.max_frames` at most) and the
    timestamp of each pair; on `decode` also the frames themselves, as each pair's bytes (a video that does not decode
    is a 400, not a failed engine step)."""
    raw = video_bytes(part)
    digest = hashlib.sha256(vision.identity + vision.video_identity + raw).digest()
    if digest not in seen:
        total, fps, width, height = _video_info(raw)
        indices = vision.sample_indices(total, fps, vision.max_frames)
        steps, grid = vision.video_grid(len(indices), width, height)
        times = tuple(vision.timestamps(indices, fps))
        video = Prepared(digest, steps * grid[0] * grid[1], raw, grid, video=True, times=times)
        if decode:
            frames = _video_frames(raw, indices, width, height)
            pad = -len(frames) % 2  # the processor repeats the last frame of an odd sample
            frames = np.concatenate([frames, np.repeat(frames[-1:], pad, axis=0)]) if pad else frames
            video.pairs = tuple(media.pair_bytes(frames[i], frames[i + 1]) for i in range(0, len(frames), 2))
        seen[digest] = video
    return seen[digest]


def _read(part: dict, vision, seen: dict[bytes, Prepared], decode: bool) -> Prepared:
    raw = image_bytes(part)
    digest = hashlib.sha256(vision.identity + raw).digest()
    if digest not in seen:
        width, height = media.size(raw)
        grid, tokens = vision.grid(width, height), vision.tokens(width, height)
        if decode and digest not in _DECODED:
            media.decode(raw)
            if len(_DECODED) >= 4096:
                _DECODED.clear()
            _DECODED[digest] = None
        seen[digest] = Prepared(digest, tokens, raw, grid)
    return seen[digest]


def prepare(messages: list[dict], vision, decode: bool = True) -> tuple[list[dict], list[Prepared]]:
    """`messages` with each image or video of a tool message that cannot be read replaced by a note saying why (an
    agent's screenshot that failed is not the user's mistake), and every image and video, in the order the template
    renders them (one sent twice is read once). `decode`: also decode each not decoded before (count_tokens reads
    headers, and videos' frame counts, only). One elsewhere that cannot be read is a 400. A video part becomes a
    `video` part, what the templates' video branch matches."""
    out: list[dict] = []
    images: list[Prepared] = []
    seen: dict[bytes, Prepared] = {}
    videos = reads_video(vision)
    for i, message in enumerate(messages):
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list) or not any(isinstance(p, dict) and p.get("type") in PARTS for p in content):
            out.append(message)
            continue
        parts = []
        for j, part in enumerate(content):
            kind = part.get("type") if isinstance(part, dict) else None
            if kind in PARTS:
                video = kind in VIDEO_PARTS
                try:
                    if video and not videos:
                        raise ValueError("this model does not read videos")
                    images.append(_read_video(part, vision, seen, decode) if video else _read(part, vision, seen, decode))
                    if video:
                        part = {"type": "video"}
                except ValueError as exc:
                    if message.get("role") != "tool":
                        raise InvalidRequest(f"messages[{i}].content[{j}]: {exc}") from None
                    part = {"type": "text", "text": MEDIA_NOTE.format(kind="video" if video else "image", why=exc)}
            parts.append(part)
        out.append({**message, "content": parts})
    n_videos = sum(p.video for p in images)
    if n_videos > MAX_VIDEOS:
        raise InvalidRequest(f"the request has {n_videos} videos; at most {MAX_VIDEOS} are read")
    if len(images) - n_videos > MAX_IMAGES:
        raise InvalidRequest(f"the request has {len(images) - n_videos} images; at most {MAX_IMAGES} are read")
    return out, images


def readable(vision) -> tuple[str, ...]:
    """The media parts the server reads: images with an encoder, videos when it has a `video_token`."""
    if vision is None:
        return ()
    return IMAGE_PARTS if not reads_video(vision) else PARTS


def reads_video(vision) -> bool:
    """Whether the encoder reads videos (it has a `video_token`)."""
    return getattr(vision, "video_token", None) is not None


def markers(vision) -> tuple[str, ...]:
    """The strings the template writes where an image, or a video, goes."""
    return (vision.marker,) + ((vision.video_marker,) if reads_video(vision) else ())


def _escaped(marker: str) -> str:
    """The marker in the request's own text while rendered: a private-use character for its "<"."""
    return "\ue000" + marker[1:]


def escape(value, marker):
    """`value` (messages or tools, at any nesting) with the image marker, or each of several (`markers`), in its
    strings escaped for the render."""
    markers = (marker,) if isinstance(marker, str) else marker
    if isinstance(value, str):
        for m in markers:
            value = value.replace(m, _escaped(m))
        return value
    if isinstance(value, list):
        return [escape(v, markers) for v in value]
    if isinstance(value, dict):
        return {k: escape(v, markers) for k, v in value.items()}
    return value


class Markers:
    """`tok` for a render of escaped messages: each image marker the template wrote is the image token; each video
    marker, the image start, the video token and the image end (the template's own markers around `<|video_pad|>`,
    which stay in the prompt as transformers' processor keeps them); and the rest, the escapes undone, is encoded with
    those tokens read as text. Offsets are the render's."""

    def __init__(self, tok, vision):
        self.image_token = vision.image_token
        self.video_token = getattr(vision, "video_token", None)
        self.tokens = {self.image_token: vision.marker}  # a token for each string the template writes for a part
        self.ids = {self.image_token: [self.image_token]}  # what each marker's token becomes
        if self.video_token is not None:
            self.tokens[self.video_token] = vision.video_marker
            self.ids[self.video_token] = [vision.image_start, self.video_token, vision.image_end]
        self.split = re.compile("(" + "|".join(re.escape(m) for m in self.tokens.values()) + ")")
        self.text = tok.without_specials({vision.image_token, vision.image_start, vision.image_end}
                                          | ({self.video_token} if self.video_token is not None else set()))

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        ids, offsets, at = [], [], 0
        tokens = {m: t for t, m in self.tokens.items()}
        for k, piece in enumerate(self.split.split(text)):
            if k % 2:  # a marker the template wrote: its token(s), at its span
                ids += self.ids[tokens[piece]]
                offsets += [(at, at + len(piece))] * len(self.ids[tokens[piece]])
            else:
                for marker in self.tokens.values():
                    piece = piece.replace(_escaped(marker), marker)
                piece_ids, piece_offsets = self.text.encode_with_offsets(piece)
                ids += piece_ids
                offsets += [(a + at, b + at) for a, b in piece_offsets]
            at += len(piece)
        return ids, offsets

    def encode(self, text: str) -> list[int]:
        return self.encode_with_offsets(text)[0]


def expand(ids: list[int], boundary: int, images: list[Prepared], vision, tok=None) -> tuple[list[int], int, tuple[media.Image, ...]]:
    """`ids` with the n-th image token replaced by `<img>`, the n-th image's placeholders and `</img>`, and the n-th
    video token by each of the n-th video's frame pairs as its processor writes them in place of `<|video_pad|>`: the
    tokens of its timestamp (`tok` encodes each on its own), `<img>`, the pair's placeholders and `</img>`; the
    `<img>` before and `</img>` after it (`Markers`) stay. The message boundary moves past what
    was inserted before it (0 stays unknown). Returns the prompt and the engine's images, a video's pairs among them."""
    tokens = {vision.image_token: False}
    if getattr(vision, "video_token", None) is not None:
        tokens[vision.video_token] = True
    marks = [(i, tokens[t]) for i, t in enumerate(ids) if t in tokens]
    if [video for _, video in marks] != [p.video for p in images]:
        for video, kind in ((False, "image"), (True, "video")):
            got, want = sum(1 for _, v in marks if v == video), sum(p.video == video for p in images)
            if got != want:
                raise InvalidRequest(f"the template placed {got} of the request's {want} {kind}s")
        raise InvalidRequest("the template placed the request's images and videos in another order")
    out: list[int] = []
    engine_images: list[media.Image] = []
    moved, last = boundary, 0
    for (mark, video), image in zip(marks, images):
        out += ids[last:mark]
        if video:
            grown = _expand_video(image, vision, tok, out, engine_images)
        else:
            out.append(vision.image_start)
            engine_images.append(media.Image(start=len(out), ids=media.placeholder_ids(image.digest, image.tokens), data=image.data,
                                             grid=image.grid))
            out += engine_images[-1].ids
            out.append(vision.image_end)
            grown = image.tokens + 1
        last = mark + 1
        if boundary and mark < boundary:
            moved += grown
    out += ids[last:]
    return out, moved, tuple(engine_images)


def _expand_video(video: Prepared, vision, tok, out: list[int], engine_images: list[media.Image]) -> int:
    """Append to `out` the video's frame pairs, and each to `engine_images` with the bytes of its two frames; returns how
    much the prompt grows: its new tokens less the video token they replace. A pair's ids follow the video's digest and
    its place in it, so two videos share a prefix only while their sampled frames agree."""
    if tok is None:
        raise InvalidRequest("the template placed a video this server cannot render")
    rows, cols = video.grid
    was = len(out)
    for step, seconds in enumerate(video.times):
        out += tok.encode(vision.timestamp_text(seconds))
        out.append(vision.image_start)
        engine_images.append(media.Image(start=len(out), ids=media.placeholder_ids(video.digest + step.to_bytes(4, "big"), rows * cols),
                                         data=video.pairs[step] if video.pairs else b"", grid=video.grid, pair=True))
        out += engine_images[-1].ids
        out.append(vision.image_end)
    return len(out) - was - 1
