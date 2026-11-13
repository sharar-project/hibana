"""Image input: the engine's placeholder spans and encodes (engine/media.py), the Nemotron preprocessing
(model/nemotron/vision.py), the API's reading, escaping and expansion (api/media.py), both chat APIs on a server with
and without a vision encoder, the executor's prefill rows and the scheduler's chunks around an image still encoding."""

import asyncio
import base64
import io
import re
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import numpy as np
import pytest
import torch

from hibana.api import media as api_media
from hibana.api import messages as messages_api
from hibana.engine import media
from hibana.engine.sampler import InvalidRequest, SamplingParams
from hibana.model.nemotron import vision as v


def encoded(w: int, h: int, mode: str = "RGB", fmt: str = "PNG", **save) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    color = {"RGB": (200, 30, 30), "RGBA": (200, 30, 30, 128), "LA": (90, 128), "P": 3, "L": 90, "CMYK": (0, 50, 90, 0)}[mode]
    Image.new(mode, (w, h), color).save(buf, fmt, **save)
    return buf.getvalue()


def b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


# ---- engine/media.py

def test_placeholder_ids_are_negative_int32_and_follow_the_content():
    a, b = media.placeholder_ids(b"a" * 32, 300), media.placeholder_ids(b"b" * 32, 300)
    assert len(a) == 300 and all(-2**31 <= t <= -1 for t in a)
    assert a == media.placeholder_ids(b"a" * 32, 300)  # the same image: the same ids, so its prefix is cached
    assert sum(x == y for x, y in zip(a, b)) <= 1 and len(set(a)) > 290
    assert media.text_ids([5, a[0], 7], 18) == [5, 18, 7]


def test_check_accepts_matching_spans_and_rejects_the_rest():
    ids = media.placeholder_ids(b"x", 4)
    prompt = [1, 19, *ids, 20, 2]
    image = media.Image(start=2, ids=ids, data=b"")
    assert media.check(prompt, (image,)) == [1, 19, 0, 0, 0, 0, 20, 2]
    with pytest.raises(ValueError):
        media.check(prompt, (media.Image(start=1, ids=ids, data=b""),))
    with pytest.raises(ValueError):  # overlapping spans
        media.check(prompt, (image, media.Image(start=3, ids=ids[1:], data=b"")))


class FakeVision:
    """An encoder whose n-th encode returns rows 100 * n + i, and whose input is the image's decoded height."""

    image_token, image_start, image_end, identity, workspace_bytes = 18, 19, 20, b"fake\n", 0

    def __init__(self):
        self.preprocessed, self.encoded = [], []

    def tokens(self, width, height):
        return v.image_tokens(width, height)

    def preprocess(self, rgb):
        self.preprocessed.append(rgb.shape)
        return torch.tensor(rgb.shape[0])

    def encode(self, pixels):
        self.encoded.append(int(pixels))
        n = int(pixels)
        return torch.arange(n)[:, None].float().repeat(1, 2) + 100 * len(self.encoded)


def test_rows_follow_chunks_across_image_boundaries_and_encode_once():
    vision = FakeVision()
    enc = media.Encoder(vision, torch.device("cpu"))
    first = media.Image(start=3, ids=media.placeholder_ids(b"1", 5), data=encoded(4, 5))  # its "pixels" are 5 rows
    second = media.Image(start=10, ids=media.placeholder_ids(b"2", 3), data=encoded(4, 3))
    both = (first, second)
    rows, pieces = enc.rows(both, 0, 6, offset=4)  # chunk [0, 6): the first image's 3 leading rows
    assert rows == [7, 8, 9] and torch.equal(torch.cat(pieces)[:, 0], torch.tensor([100.0, 101, 102]))
    assert first.embeds is not None and vision.encoded == [5] and first.pixels is None
    rows, pieces = enc.rows(both, 6, 12, offset=0)  # the first image's rest, the second's start
    assert rows == [0, 1, 4, 5] and torch.cat(pieces)[:, 0].tolist() == [103, 104, 200, 201]
    assert first.embeds is None and second.embeds is not None and vision.encoded == [5, 3]  # the first dropped at its end
    rows, pieces = enc.rows(both, 12, 20, offset=0)
    assert rows == [0] and second.embeds is None
    rows, pieces = enc.rows(both, 0, 6, offset=0)  # a retracted prefill decodes and encodes again
    assert rows == [3, 4, 5] and vision.encoded == [5, 3, 5]


def test_only_images_past_the_cached_prefix_are_decoded_and_chunks_stop_before_an_encode():
    vision = FakeVision()
    enc = media.Encoder(vision, torch.device("cpu"))
    cached = media.Image(start=2, ids=media.placeholder_ids(b"1", 4), data=encoded(4, 4))
    later = media.Image(start=20, ids=media.placeholder_ids(b"2", 4), data=encoded(4, 6))
    enc.prefetch((cached, later), pos=10)  # admission after a 10-token cache hit
    assert cached.pixels is None and later.pixels is not None
    later.pixels.result()
    assert vision.preprocessed == [(6, 4, 3)]
    assert enc.ready((cached, later), 10, 14, wait=False) == 14 and later.embeds is None  # 20 is beyond the next chunk
    pending = Future()
    later.pixels = pending  # its pixels still being made: a chunk stops before the image unless it waits
    assert enc.ready((cached, later), 10, 30, wait=False) == 20
    assert enc.ready((cached, later), 20, 30, wait=False) == 20
    pending.set_result(torch.tensor(4))
    assert enc.ready((cached, later), 20, 30, wait=False) == 30 and later.embeds is not None and vision.encoded == [4]


# ---- model/nemotron/vision.py

@pytest.mark.parametrize("size,patches", [((256, 256), (32, 32)), ((4000, 3000), (132, 100)), ((1920, 1080), (120, 68)),
                                          ((2000, 40), (208, 6)), ((1024, 768), (64, 48)), ((100000, 50), (4560, 2)),
                                          ((50, 100000), (2, 4560)), ((20000, 16), (1250, 2)), ((1, 1), (32, 32))])
def test_target_patches_match_the_reference_rule(size, patches):
    """Worked by hand from `_compute_target_patches`: the budget, the 1,024-patch floor and even sides."""
    w, h = v.target_patches(*size)
    assert (w, h) == patches and w % 2 == 0 and h % 2 == 0 and w * h <= v.MAX_PATCHES


@pytest.mark.parametrize("size", [(300000, 1), (1, 300000), (220000, 3)])
def test_an_image_too_narrow_for_one_patch_is_refused(size):
    """The reference divides by zero on these: its short side scales to no patch."""
    with pytest.raises(ValueError, match="too narrow"):
        v.target_patches(*size)
    with pytest.raises(ValueError, match="too narrow"):
        v.image_tokens(*size)


@pytest.mark.parametrize("size", [(500, 300), (300, 500), (64, 48), (3000, 40), (40, 3000), (20000, 16), (7, 9)])
def test_preprocess_resizes_and_normalizes(size):
    width, height = size
    px = v.preprocess(np.full((height, width, 3), 255, dtype=np.uint8))
    w, h = v.target_patches(width, height)
    assert px.shape == (3, h * v.PATCH, w * v.PATCH) and px.dtype == torch.bfloat16
    assert torch.allclose(px[:, 0, 0].float(), (1 - torch.tensor(v.NORM_MEAN)) / torch.tensor(v.NORM_STD), atol=2e-2)
    assert v.image_tokens(width, height) == px.shape[1] * px.shape[2] // (v.PATCH**2 * 4)


def test_pixel_shuffle_orders_as_the_reference():
    x = torch.randn(6, 10, 4)
    n, w, h, c = x[None].size()  # the reference's steps and names (ps_version v2, scale 0.5)
    ref = x[None].view(n, w, h // 2, c * 2).permute(0, 2, 1, 3).contiguous()
    ref = ref.view(n, h // 2, w // 2, c * 4).permute(0, 2, 1, 3).contiguous()[0]
    assert torch.equal(v.pixel_shuffle(x), ref)


# ---- reading images: header sizes match decoding

def oriented_jpeg(orientation: int) -> bytes:
    from PIL import Image
    img, exif = Image.new("RGB", (40, 20), (10, 200, 10)), Image.Exif()
    img.paste((255, 255, 255), (0, 0, 8, 8))
    exif[0x0112] = orientation
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif.tobytes(), quality=95)
    return buf.getvalue()


def animated_gif() -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    frames = [Image.new("P", (30, 17), i) for i in range(3)]
    frames[0].save(buf, "GIF", save_all=True, append_images=frames[1:], duration=50)
    return buf.getvalue()


@pytest.mark.parametrize("raw", [encoded(5, 3), encoded(5, 3, "RGBA"), encoded(7, 2, "LA"), encoded(9, 4, "P"),
                                 encoded(3, 8, "L", "JPEG"), encoded(6, 5, "CMYK", "JPEG"), encoded(11, 6, fmt="WEBP"),
                                 animated_gif(), *(oriented_jpeg(o) for o in range(1, 9))],
                         ids=["rgb", "rgba", "la", "palette", "gray-jpeg", "cmyk-jpeg", "webp", "gif"] + [f"exif-{o}" for o in range(1, 9)])
def test_the_header_size_is_the_decoded_size(raw):
    width, height = media.size(raw)
    rgb = media.decode(raw)
    assert rgb.shape == (height, width, 3) and rgb.dtype == np.uint8 and rgb.flags.writeable


def test_exif_orientation_is_applied():
    """Orientation 6: the stored image turned 90 degrees clockwise to view; the white corner ends top right."""
    upright, turned = media.decode(oriented_jpeg(1)), media.decode(oriented_jpeg(6))
    assert upright.shape == (20, 40, 3) and turned.shape == (40, 20, 3)
    assert upright[0, 0].min() > 200 and turned[0, -1].min() > 200 and turned[0, 0].max() < 220


def test_what_is_not_an_image_is_refused(monkeypatch):
    for raw, why in [(b"not an image", "not a readable image"), (encoded(5, 3, fmt="BMP"), "PNG, JPEG, WebP or GIF")]:
        with pytest.raises(ValueError, match=re.escape(why)):
            media.size(raw)
    truncated = encoded(64, 64)[:-30]  # a header that reads, pixels that do not
    assert media.size(truncated) == (64, 64)
    with pytest.raises(ValueError, match="not a readable image"):
        media.decode(truncated)
    monkeypatch.setattr(media, "MAX_IMAGE_PIXELS", 10)
    with pytest.raises(ValueError, match="more than 10 pixels"):
        media.size(encoded(5, 3))


# ---- api/media.py

def test_image_bytes_from_both_apis():
    raw = encoded(4, 4)
    assert api_media.image_bytes({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(raw)}"}}) == raw
    assert api_media.image_bytes({"type": "input_image", "image_url": f"data:image/png;base64,{b64(raw)}"}) == raw
    assert api_media.image_bytes({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64(raw)}}) == raw
    for bad in ({"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                {"type": "image_url", "image_url": {"url": "data:image/png,abc"}},
                {"type": "image", "source": {"type": "base64", "data": "!!"}}, {"type": "image"}):
        with pytest.raises(ValueError):
            api_media.image_bytes(bad)


VISION = SimpleNamespace(image_token=18, image_start=19, image_end=20, identity=v.IDENTITY, tokens=v.image_tokens)


def url_part(raw: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(raw)}"}}


def test_expand_inserts_each_image_and_moves_the_boundary():
    images = [api_media.Prepared(b"a", 3, b"A"), api_media.Prepared(b"b", 2, b"B")]
    ids, at, out = api_media.expand([1, 18, 2, 18, 3, 4], 4, images, VISION)
    assert ids[:2] == [1, 19] and ids[2:5] == list(out[0].ids) and ids[5:7] == [20, 2]
    assert ids[7] == 19 and ids[8:10] == list(out[1].ids) and ids[10:] == [20, 3, 4]
    assert (out[0].start, out[1].start) == (2, 8) and at == 4 + 4 + 3 and out[0].data == b"A"
    assert api_media.expand([1, 18, 2], 0, images[:1], VISION)[1] == 0  # unknown stays unknown
    with pytest.raises(InvalidRequest, match="placed 1 of the request's 2 images"):
        api_media.expand([1, 18, 2], 0, images, VISION)


def test_prepare_reads_headers_in_render_order_and_a_repeat_once(monkeypatch):
    a, b = encoded(40, 30), encoded(30, 41)
    messages = [{"role": "user", "content": [url_part(a), {"type": "text", "text": "and"},
                                             {"type": "image", "source": {"type": "base64", "data": b64(b)}}]},
                {"role": "tool", "content": [url_part(a)]}]
    decoded = []
    monkeypatch.setattr(media, "decode", lambda raw: decoded.append(raw))
    out, images = api_media.prepare(messages, VISION, decode=False)  # count_tokens: headers only
    assert out == messages and not decoded
    assert [p.digest for p in images] == [images[0].digest, images[1].digest, images[0].digest] and images[0] is images[2]
    assert images[0].tokens == v.image_tokens(40, 30) and images[0].data == a
    api_media.prepare(messages, VISION)
    api_media.prepare(messages, VISION)
    assert decoded == [a, b]  # each new image decodes once, to check it, and not again on a later request


def test_an_unreadable_image_in_a_tool_result_becomes_a_note_and_elsewhere_a_400():
    good, bad = url_part(encoded(64, 48)), {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}
    narrow = url_part(encoded(250000, 1))
    tool = {"role": "tool", "content": [{"type": "text", "text": "shot:"}, bad, good, narrow, url_part(encoded(64, 64)[:-30])]}
    out, images = api_media.prepare([tool], VISION)
    notes = [p["text"] for p in out[0]["content"] if p["type"] == "text"][1:]
    assert len(images) == 1 and out[0]["content"][2] == good and len(notes) == 3
    assert "does not fetch URLs" in notes[0] and "too narrow" in notes[1] and "not a readable image" in notes[2]
    assert all(n.startswith("[image omitted: ") for n in notes)
    for part in (bad, narrow):
        with pytest.raises(InvalidRequest, match=r"messages\[0\].content\[0\]"):
            api_media.prepare([{"role": "user", "content": [part]}], VISION)
    with pytest.raises(InvalidRequest, match="101 images; at most 100"):
        api_media.prepare([{"role": "user", "content": [good] * 101}], VISION)


class CharTokenizer:
    """One token per character (its code point + 100), and the vocabulary's special tokens as one each."""

    SPECIALS = {"<image>": 18, "<img>": 19, "</img>": 20, "<|im_start|>": 10}

    def __init__(self, specials=None):
        self.specials = dict(self.SPECIALS if specials is None else specials)

    def without_specials(self, ids):
        return CharTokenizer({s: i for s, i in self.specials.items() if i not in ids})

    def encode_with_offsets(self, text):
        ids, offsets, at = [], [], 0
        pattern = "|".join(map(re.escape, self.specials)) or r"(?!)"
        for m in re.finditer(pattern, text):
            ids += [ord(c) + 100 for c in text[at:m.start()]]
            offsets += [(i, i + 1) for i in range(at, m.start())]
            ids.append(self.specials[m.group()])
            offsets.append(m.span())
            at = m.end()
        return ids + [ord(c) + 100 for c in text[at:]], offsets + [(i, i + 1) for i in range(at, len(text))]

    def encode(self, text):
        return self.encode_with_offsets(text)[0]

    def decode(self, ids, **_):
        return "ok"


def render(messages, **_):
    """The checkpoint template's image rule: `<image>` per image part, none when the message's text has one."""
    out = []
    for m in messages:
        content = m.get("content")
        parts = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
        text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
        out.append("".join(p.get("text", "") if p.get("type") == "text" else "<image>" if "<image>" not in text else ""
                           for p in parts))
    return "<|im_start|>" + "\n".join(out)


def test_markers_read_only_the_templates_image_markers_as_images():
    tok = CharTokenizer()
    messages = api_media.escape([{"role": "user", "content": [{"type": "text", "text": "a <image> b <img>"}, url_part(b"")]}])
    rendered = render(messages)
    ids, offsets = api_media.Markers(tok, VISION).encode_with_offsets(rendered)
    assert ids.count(18) == 1 and ids[-1] == 18 and 19 not in ids  # the user's markers are their characters
    text = "".join(chr(t - 100) for t in ids[1:-1])
    assert text == "a <image> b <img>" and offsets[-1] == (len(rendered) - 7, len(rendered))
    assert api_media.escape({"k": ["<image>", 3]}) == {"k": ["image>", 3]}


# ---- the chat APIs

def server(vision):
    """An app whose template follows the checkpoint's image rule and whose tokenizer reads characters."""
    queue = asyncio.Queue()
    queue.put_nowait(([2], True, "stop"))
    ticket = SimpleNamespace(queue=queue, req=SimpleNamespace(prefix_len=0))
    engine = SimpleNamespace(submit=AsyncMock(return_value=ticket), cancel=Mock(), stats=lambda: {})
    from hibana.api.server import create_app
    return engine, create_app(engine, CharTokenizer(), SimpleNamespace(render=render), {"context_length": 100_000}, vision=vision)


def post(app, path, body):
    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            return await client.post(path, json=body)
    return asyncio.run(go())


IMAGE = url_part(encoded(64, 48))


def test_chat_with_an_image_submits_its_span():
    engine, app = server(VISION)
    r = post(app, "/v1/chat/completions", {"messages": [{"role": "user", "content": [IMAGE, {"type": "text", "text": "what?"}]}]})
    assert r.status_code == 200, r.text
    ids, params, images = engine.submit.await_args.args
    (image,) = images
    assert len(image.ids) == 256 and ids[:2] == [10, 19] and ids[2:258] == list(image.ids)
    assert ids[258] == 20 and image.start == 2 and image.data == base64.b64decode(IMAGE["image_url"]["url"].split(",")[1])
    assert r.json()["usage"]["prompt_tokens"] == len(ids)


def test_literal_image_markers_in_text_stay_text():
    """The template would place the image at the text's `<image>`, and the tokenizer would read it as the image token:
    the request's own `<image>`, `<img>` and `</img>` are characters, and its one image goes where its part is."""
    engine, app = server(VISION)
    text = {"type": "text", "text": "explain the <image> tag and <img>"}
    r = post(app, "/v1/chat/completions", {"messages": [{"role": "user", "content": [text, IMAGE]}]})
    assert r.status_code == 200, r.text
    ids, _, (image,) = engine.submit.await_args.args
    assert ids.count(19) == ids.count(20) == 1 and ids[-258] == 19 and image.start == len(ids) - 257
    engine, app = server(VISION)  # one reply per server
    r = post(app, "/v1/chat/completions", {"messages": [{"role": "user", "content": "just <image> text"}]})
    assert r.status_code == 200, r.text
    assert 18 not in engine.submit.await_args.args[0] and engine.submit.await_args.args[2] == ()


def test_a_text_only_server_still_refuses_images():
    engine, app = server(None)
    r = post(app, "/v1/chat/completions", {"messages": [{"role": "user", "content": [IMAGE, {"type": "text", "text": "what?"}]}]})
    assert r.status_code == 400 and "text-only" in r.json()["error"]["message"]
    engine.submit.assert_not_awaited()


def test_a_vision_server_refuses_audio_unfetched_urls_and_extreme_shapes():
    engine, app = server(VISION)
    r = post(app, "/v1/chat/completions", {"messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {}}]}]})
    assert r.status_code == 400 and "text and images" in r.json()["error"]["message"]
    for part, why in [({"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}, "does not fetch"),
                      (url_part(encoded(250000, 1)), "too narrow")]:
        r = post(app, "/v1/chat/completions", {"messages": [{"role": "user", "content": [part]}]})
        assert r.status_code == 400 and why in r.json()["error"]["message"], r.text
    engine.submit.assert_not_awaited()


def test_openai_tool_message_images_are_read_or_noted():
    engine, app = server(VISION)
    calls = [{"id": "t", "type": "function", "function": {"name": "shot", "arguments": "{}"}}]
    body = {"messages": [{"role": "user", "content": "look"}, {"role": "assistant", "content": "", "tool_calls": calls},
                         {"role": "tool", "tool_call_id": "t", "content": [IMAGE, {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                                                                           {"type": "input_audio", "input_audio": {}}]}]}
    r = post(app, "/v1/chat/completions", body)
    assert r.status_code == 200, r.text
    ids, _, images = engine.submit.await_args.args
    text = "".join(chr(t - 100) for t in ids if t >= 100)
    assert len(images) == 1 and "[image omitted: send images inline" in text
    assert "[input_audio omitted: this model reads text and images only]" in text


def test_messages_api_images_in_user_turns_and_tool_results():
    block = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64(encoded(64, 48))}}
    url_block = {"type": "image", "source": {"type": "url", "url": "https://example.com/x.png"}}
    body = {"messages": [{"role": "user", "content": [block, {"type": "text", "text": "see"}]},
                         {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "shot", "input": {}}]},
                         {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t",
                                                       "content": [block, url_block, {"type": "document", "source": {}}]}]}]}
    req = messages_api.parse_request(body, images=True)
    assert req.messages[0]["content"][0] == {"type": "image", "source": block["source"]}
    assert req.messages[2]["role"] == "tool" and req.messages[2]["content"][0]["type"] == "image"
    assert req.messages[2]["content"][2]["text"] == "[document omitted: this model reads text and images only]"
    text_only = messages_api.parse_request({"messages": body["messages"][1:]})
    assert "reads text only" in text_only.messages[1]["content"][0]["text"]  # a tool's image stays a note without vision
    with pytest.raises(InvalidRequest, match="text-only"):
        messages_api.parse_request(body)
    with pytest.raises(InvalidRequest, match="document content is not supported; this server runs a model that reads text and images"):
        messages_api.parse_request({"messages": [{"role": "user", "content": [{"type": "document", "source": {}}]}]}, images=True)
    engine, app = server(VISION)
    r = post(app, "/v1/messages", {"model": "m", "max_tokens": 10, **body})
    assert r.status_code == 200, r.text
    ids, _, images = engine.submit.await_args.args
    assert len(images) == 2 and "[image omitted: send images inline" in "".join(chr(t - 100) for t in ids if t >= 100)
    r = post(server(VISION)[1], "/v1/messages", {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": [url_block]}]})
    assert r.status_code == 400 and "does not fetch" in r.text


def test_count_tokens_reads_image_headers_only(monkeypatch):
    engine, app = server(VISION)
    block = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64(encoded(1920, 1080))}}
    monkeypatch.setattr(media, "decode", Mock(side_effect=AssertionError("decoded")))
    r = post(app, "/v1/messages/count_tokens", {"model": "m", "messages": [{"role": "user", "content": [block]}]})
    assert r.status_code == 200, r.text
    assert r.json()["input_tokens"] == 1 + v.image_tokens(1920, 1080) + 2  # <|im_start|>, <img>, </img>


# ---- the executor and the scheduler

def executor(vision, vocab=1000):
    from hibana.engine.executor import Executor
    ex = Executor.__new__(Executor)
    ex.vision = vision
    ex.images = media.Encoder(vision, torch.device("cpu")) if vision is not None else None
    ex.device = torch.device("cpu")
    ex.target = SimpleNamespace(vocab_size=vocab, stop_token_ids=(2,))
    ex.max_context, ex.spec = 10_000, None
    ex.kv_pool = SimpleNamespace(num_pages=10_000)
    return ex


def test_validation_accepts_an_images_span_and_nothing_else():
    ids = media.placeholder_ids(b"img", 4)
    image = media.Image(start=2, ids=ids, data=b"")
    prompt = [1, 19, *ids, 20]
    params = SamplingParams(max_new_tokens=4)
    executor(VISION).validate_request(prompt, params, (image,))
    with pytest.raises(InvalidRequest, match="token IDs"):  # placeholders without their image
        executor(VISION).validate_request(prompt, params)
    with pytest.raises(InvalidRequest, match="without a vision encoder"):
        executor(None).validate_request(prompt, params, (image,))


def test_prefill_rows_and_model_ids():
    from hibana.engine.context import ForwardContext
    vision = FakeVision()
    ex = executor(vision)
    ids = media.placeholder_ids(b"img", 4)
    req = SimpleNamespace(media=(media.Image(start=2, ids=ids, data=encoded(4, 4)),), prefill_pos=0)
    assert ex._model_ids(req, [1, 19, *ids, 20]) == [1, 19, 18, 18, 18, 18, 20]
    assert ex._model_ids(SimpleNamespace(media=()), [1, 2]) == [1, 2]
    ctx = ForwardContext(mode="prefill", kv_pool=None, state_pool=None, positions=torch.zeros(1))
    plain = SimpleNamespace(media=())
    ex._image_rows(ctx, [(plain, 0, 3, 0), (req, 0, 4, 3)])  # a batch: the image request's rows start at 3
    assert ctx.media_rows.tolist() == [5, 6] and ctx.media_embeds.shape == (2, 2) and vision.encoded == [4]
    ctx = ForwardContext(mode="prefill", kv_pool=None, state_pool=None, positions=torch.zeros(1))
    ex._image_rows(ctx, [(plain, 0, 3, 0)])
    assert ctx.media_rows is None
    assert ex.prefill_ready(plain, 7, wait=False) == 7


def test_the_scheduler_ends_a_chunk_before_an_image_still_encoding_while_others_decode():
    from hibana.engine.scheduler import Scheduler, SchedulerConfig
    from tests.unit.test_scheduler import FakeExecutor

    class ImageExecutor(FakeExecutor):
        def __init__(self):
            super().__init__()
            self.encoding, self.asked = True, []

        def prefill_ready(self, req, n, wait):
            self.asked.append(wait)
            return n if wait or not self.encoding else min(n, max(0, 100 - req.prefill_pos))  # an image at 100

    ex = ImageExecutor()
    sched = Scheduler(ex, SchedulerConfig(chunk_size=2048, prefill_budget=2048))
    ids = media.placeholder_ids(b"img", 50)
    image = media.Image(start=100, ids=ids, data=b"")
    alone = sched.submit([1] * 100 + list(ids) + [1] * 10, SamplingParams(max_new_tokens=2, ignore_eos=True), (image,))
    assert alone.media[0] is not image and alone.media[0].ids == ids  # each request owns its images' encode state
    while not alone.finished:
        sched.step()
    assert ex.asked[0] is True  # nothing decoding: the chunk runs and waits for the encode
    talker = sched.submit([1] * 10, SamplingParams(max_new_tokens=50, ignore_eos=True))
    sched.step()
    req = sched.submit([1] * 100 + list(ids) + [1] * 10, SamplingParams(max_new_tokens=2, ignore_eos=True), (image,))
    for _ in range(3):
        sched.step()
    assert req.prefill_pos == 100 and not req.prefill_done and len(talker.output_ids) > 3  # decode goes on
    ex.encoding = False
    sched.step()
    assert req.prefill_done
