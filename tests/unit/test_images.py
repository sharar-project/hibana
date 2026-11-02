"""Image input: the engine's placeholder spans (engine/media.py), the Nemotron preprocessing (model/nemotron/vision.py),
the API's decoding and expansion (api/media.py), both chat APIs on a server with and without a vision encoder, and the
executor's prefill rows."""

import asyncio
import base64
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import numpy as np
import pytest
import torch

from sharar.api import media as api_media
from sharar.api import messages as messages_api
from sharar.engine import media
from sharar.engine.sampler import InvalidRequest
from sharar.model.nemotron import vision as v


def png(w: int, h: int, mode: str = "RGB", color=(200, 30, 30)) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new(mode, (w, h), color if mode == "RGB" else (*color, 128)).save(buf, "PNG")
    return buf.getvalue()


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
    image = media.Image(start=2, ids=ids, pixels=torch.zeros(1))
    assert media.check(prompt, (image,)) == [1, 19, 0, 0, 0, 0, 20, 2]
    with pytest.raises(ValueError):
        media.check(prompt, (media.Image(start=1, ids=ids, pixels=torch.zeros(1)),))
    with pytest.raises(ValueError):  # overlapping spans
        media.check(prompt, (image, media.Image(start=3, ids=ids[1:], pixels=torch.zeros(1))))


def test_rows_follow_chunks_across_image_boundaries_and_encode_once():
    encoded = []

    def encode(pixels):
        encoded.append(pixels)
        return torch.arange(len(pixels))[:, None].float().repeat(1, 2) + 100 * len(encoded)

    first = media.Image(start=3, ids=media.placeholder_ids(b"1", 5), pixels=torch.zeros(5))
    second = media.Image(start=10, ids=media.placeholder_ids(b"2", 3), pixels=torch.zeros(3))
    both = (first, second)
    rows, pieces = media.rows(both, 0, 6, offset=4, encode=encode)  # chunk [0, 6): the first image's 3 leading rows
    assert rows == [7, 8, 9] and torch.equal(torch.cat(pieces)[:, 0], torch.tensor([100.0, 101, 102]))
    assert first.embeds is not None and len(encoded) == 1
    rows, pieces = media.rows(both, 6, 12, offset=0, encode=encode)  # the first image's rest, the second's start
    assert rows == [0, 1, 4, 5] and torch.cat(pieces)[:, 0].tolist() == [103, 104, 200, 201]
    assert first.embeds is None and second.embeds is not None and len(encoded) == 2  # the first dropped at its end
    rows, pieces = media.rows(both, 12, 20, offset=0, encode=encode)
    assert rows == [0] and second.embeds is None
    rows, pieces = media.rows(both, 0, 6, offset=0, encode=encode)  # a retracted prefill encodes again
    assert rows == [3, 4, 5] and len(encoded) == 3


# ---- model/nemotron/vision.py

@pytest.mark.parametrize("size,patches", [((256, 256), (32, 32)), ((4000, 3000), (132, 100)), ((1920, 1080), (120, 68)),
                                          ((2000, 40), (208, 6)), ((1024, 768), (64, 48))])
def test_target_patches_match_the_reference_rule(size, patches):
    """Worked by hand from `_compute_target_patches`: the budget, the 1,024-patch floor and even sides."""
    w, h = v.target_patches(*size)
    assert (w, h) == patches and w % 2 == 0 and h % 2 == 0 and w * h <= v.MAX_PATCHES


def test_preprocess_resizes_and_normalizes():
    rgb = np.full((300, 500, 3), 255, dtype=np.uint8)
    px = v.preprocess(rgb)
    w, h = v.target_patches(500, 300)
    assert px.shape == (3, h * v.PATCH, w * v.PATCH) and px.dtype == torch.bfloat16
    assert torch.allclose(px[:, 0, 0].float(), (1 - torch.tensor(v.NORM_MEAN)) / torch.tensor(v.NORM_STD), atol=2e-2)
    assert v.image_tokens(px) == w * h // 4


def test_pixel_shuffle_orders_as_the_reference():
    x = torch.randn(6, 10, 4)
    n, w, h, c = x[None].size()  # the reference's steps and names (ps_version v2, scale 0.5)
    ref = x[None].view(n, w, h // 2, c * 2).permute(0, 2, 1, 3).contiguous()
    ref = ref.view(n, h // 2, w // 2, c * 4).permute(0, 2, 1, 3).contiguous()[0]
    assert torch.equal(v.pixel_shuffle(x), ref)


# ---- api/media.py

def test_image_bytes_from_both_apis():
    raw = png(4, 4)
    data = base64.b64encode(raw).decode()
    assert api_media.image_bytes({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}, "m") == raw
    assert api_media.image_bytes({"type": "input_image", "image_url": f"data:image/png;base64,{data}"}, "m") == raw
    assert api_media.image_bytes({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}, "m") == raw
    for bad in ({"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                {"type": "image_url", "image_url": {"url": "data:image/png,abc"}},
                {"type": "image", "source": {"type": "base64", "data": "!!"}}, {"type": "image"}):
        with pytest.raises(InvalidRequest):
            api_media.image_bytes(bad, "m")


def test_decode_drops_alpha_and_refuses_what_is_not_an_image(monkeypatch):
    assert api_media.decode(png(5, 3, "RGBA"), "m").shape == (3, 5, 3)
    with pytest.raises(InvalidRequest, match="not a readable image"):
        api_media.decode(b"not an image", "m")
    monkeypatch.setattr(api_media, "MAX_IMAGE_PIXELS", 10)
    with pytest.raises(InvalidRequest, match="more than 10 pixels"):
        api_media.decode(png(5, 3), "m")


VISION = SimpleNamespace(image_token=18, image_start=19, image_end=20, identity=v.IDENTITY, preprocess=v.preprocess,
                         tokens=v.image_tokens)


def test_expand_inserts_each_image_and_moves_the_boundary():
    images = [api_media.Prepared(b"a", torch.zeros(1), 3), api_media.Prepared(b"b", torch.zeros(1), 2)]
    ids, at, out = api_media.expand([1, 18, 2, 18, 3, 4], 4, images, VISION)
    assert ids[:2] == [1, 19] and ids[2:5] == list(out[0].ids) and ids[5:7] == [20, 2]
    assert ids[7] == 19 and ids[8:10] == list(out[1].ids) and ids[10:] == [20, 3, 4]
    assert (out[0].start, out[1].start) == (2, 8) and at == 4 + 4 + 3  # both images were before the boundary
    assert api_media.expand([1, 18, 2], 0, images[:1], VISION)[1] == 0  # unknown stays unknown
    with pytest.raises(InvalidRequest, match="1 <image> markers for 2 images"):
        api_media.expand([1, 18, 2], 0, images, VISION)


def test_prepare_follows_render_order_and_decodes_a_repeat_once():
    a, b = base64.b64encode(png(40, 30)).decode(), base64.b64encode(png(30, 40, color=(0, 0, 255))).decode()
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{a}"}},
                                             {"type": "text", "text": "and"},
                                             {"type": "image", "source": {"type": "base64", "data": b}}]},
                {"role": "tool", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{a}"}}]}]
    out = api_media.prepare(messages, VISION)
    assert [p.digest for p in out] == [out[0].digest, out[1].digest, out[0].digest] and out[0] is out[2]
    assert out[0].pixels.shape[1:] != out[1].pixels.shape[1:]  # landscape and portrait tiles


# ---- the chat APIs

def server(vision):
    """An app whose template writes <image> per image part and whose tokenizer reads it as id 18."""
    def render(messages, **_):
        out = []
        for m in messages:
            content = m.get("content")
            for p in content if isinstance(content, list) else [{"type": "text", "text": content or ""}]:
                out.append("<image>" if p.get("type") in api_media.IMAGE_PARTS else p.get("text", ""))
        return "|".join(out)

    def encode(text):
        return [18 if part == "<image>" else 5 for part in text.split("|") if part]

    queue = asyncio.Queue()
    queue.put_nowait(([2], True, "stop"))
    ticket = SimpleNamespace(queue=queue, req=SimpleNamespace(prefix_len=0))
    engine = SimpleNamespace(submit=AsyncMock(return_value=ticket), cancel=Mock(), stats=lambda: {})
    tok = SimpleNamespace(encode=encode, decode=lambda ids, **kw: "ok")
    from sharar.api.server import create_app
    return engine, create_app(engine, tok, SimpleNamespace(render=render), {"context_length": 100_000}, vision=vision)


def post(app, path, body):
    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            return await client.post(path, json=body)
    return asyncio.run(go())


IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png(64, 48)).decode()}}


def test_chat_with_an_image_submits_its_span():
    engine, app = server(VISION)
    r = post(app, "/v1/chat/completions", {"messages": [{"role": "user", "content": [IMAGE, {"type": "text", "text": "what?"}]}]})
    assert r.status_code == 200, r.text
    ids, params, images = engine.submit.await_args.args
    (image,) = images
    n = v.image_tokens(image.pixels)
    assert n == 256 and ids == [19, *image.ids, 20, 5] and image.start == 1
    assert r.json()["usage"]["prompt_tokens"] == n + 3


def test_a_text_only_server_still_refuses_images():
    engine, app = server(None)
    r = post(app, "/v1/chat/completions", {"messages": [{"role": "user", "content": [IMAGE, {"type": "text", "text": "what?"}]}]})
    assert r.status_code == 400 and "text-only" in r.json()["error"]["message"]
    engine.submit.assert_not_awaited()


def test_a_vision_server_refuses_audio_and_unfetched_urls():
    engine, app = server(VISION)
    r = post(app, "/v1/chat/completions", {"messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {}}]}]})
    assert r.status_code == 400 and "text and images" in r.json()["error"]["message"]
    r = post(app, "/v1/chat/completions", {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}]}]})
    assert r.status_code == 400 and "does not fetch" in r.json()["error"]["message"]
    engine.submit.assert_not_awaited()


def test_messages_api_images_in_user_turns_and_tool_results():
    data = base64.b64encode(png(64, 48)).decode()
    block = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}
    body = {"messages": [{"role": "user", "content": [block, {"type": "text", "text": "see"}]},
                         {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "shot", "input": {}}]},
                         {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": [block]}]}]}
    req = messages_api.parse_request(body, images=True)
    assert req.messages[0]["content"][0] == {"type": "image", "source": block["source"]}
    assert req.messages[2]["role"] == "tool" and req.messages[2]["content"][0]["type"] == "image"
    text_only = messages_api.parse_request({"messages": body["messages"][1:]})
    assert "omitted" in text_only.messages[1]["content"][0]["text"]  # a tool's image stays a note without vision
    with pytest.raises(InvalidRequest):
        messages_api.parse_request(body)
    engine, app = server(VISION)
    r = post(app, "/v1/messages", {"model": "m", "max_tokens": 10, **body})
    assert r.status_code == 200, r.text
    assert len(engine.submit.await_args.args[2]) == 2


# ---- the executor

def executor(vision, vocab=1000):
    from sharar.engine.executor import Executor
    ex = Executor.__new__(Executor)
    ex.model = SimpleNamespace(vision=vision)
    ex.device = torch.device("cpu")
    ex.target = SimpleNamespace(vocab_size=vocab, stop_token_ids=(2,))
    ex.max_context, ex.spec = 10_000, None
    ex.kv_pool = SimpleNamespace(num_pages=10_000)
    return ex


def test_validation_accepts_an_images_span_and_nothing_else():
    from sharar.engine.sampler import SamplingParams
    ids = media.placeholder_ids(b"img", 4)
    image = media.Image(start=2, ids=ids, pixels=torch.zeros(1))
    prompt = [1, 19, *ids, 20]
    params = SamplingParams(max_new_tokens=4)
    executor(VISION).validate_request(prompt, params, (image,))
    with pytest.raises(InvalidRequest, match="token IDs"):  # placeholders without their image
        executor(VISION).validate_request(prompt, params)
    with pytest.raises(InvalidRequest, match="without a vision encoder"):
        executor(None).validate_request(prompt, params, (image,))


def test_prefill_rows_and_model_ids():
    from sharar.engine.context import ForwardContext
    calls = []
    vision = SimpleNamespace(image_token=18, encode=lambda px: calls.append(px) or torch.ones(4, 3))
    ex = executor(vision)
    ids = media.placeholder_ids(b"img", 4)
    req = SimpleNamespace(media=(media.Image(start=2, ids=ids, pixels=torch.zeros(1)),))
    assert ex._model_ids(req, [1, 19, *ids, 20]) == [1, 19, 18, 18, 18, 18, 20]
    assert ex._model_ids(SimpleNamespace(media=()), [1, 2]) == [1, 2]
    ctx = ForwardContext(mode="prefill", kv_pool=None, state_pool=None, positions=torch.zeros(1))
    plain = SimpleNamespace(media=())
    ex._image_rows(ctx, [(plain, 0, 3, 0), (req, 0, 4, 3)])  # a batch: the image request's rows start at 3
    assert ctx.media_rows.tolist() == [5, 6] and ctx.media_embeds.shape == (2, 3) and len(calls) == 1
    ctx = ForwardContext(mode="prefill", kv_pool=None, state_pool=None, positions=torch.zeros(1))
    ex._image_rows(ctx, [(plain, 0, 3, 0)])
    assert ctx.media_rows is None
