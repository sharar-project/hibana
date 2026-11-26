"""Image tokens on the Qwen targets: M-RoPE rows (engine/mrope.py), the rotary table's scratch rows (model/rope.py), the
Qwen preprocessing's sizes, the API's marker and grids, and `without_specials` keeping the other added tokens' ids."""
import json
from types import SimpleNamespace

import pytest
import torch

from hibana.api import media as api_media
from hibana.engine import media
from hibana.engine.mrope import RopeLayout
from hibana.engine.sampler import InvalidRequest
from hibana.model import qwen_vision as qv
from hibana.model.rope import PartialRotary


def image(start: int, rows: int, cols: int):
    return media.Image(start=start, ids=media.placeholder_ids(bytes([start]), rows * cols), data=b"", grid=(rows, cols))


def position(lay: RopeLayout, i: int) -> tuple[int, int, int]:
    scratch: list = []
    row = lay.rows(i, i + 1, -1, scratch)[0]
    return scratch[0] if scratch else (row, row, row)


def test_positions_follow_the_reference_rule():
    """Text is (p, p, p); the block at (r, c) of an image whose first placeholder has rope start s is (s, s + r, s + c);
    the text after it resumes at s + max(rows, cols); a second image starts from the shifted text."""
    a, b = image(3, 2, 3), image(11, 4, 1)  # 6 placeholders at 3..8, then text 9, 10, then 4 placeholders at 11..14
    lay = RopeLayout((b, a))
    assert [position(lay, i) for i in range(3)] == [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
    assert [position(lay, i) for i in range(3, 9)] == [(3, 3, 3), (3, 3, 4), (3, 3, 5), (3, 4, 3), (3, 4, 4), (3, 4, 5)]
    assert position(lay, 9) == (6, 6, 6) and position(lay, 10) == (7, 7, 7)  # 3 + max(2, 3)
    assert [position(lay, i) for i in range(11, 15)] == [(8, 8, 8), (8, 9, 8), (8, 10, 8), (8, 11, 8)]
    assert position(lay, 15) == (12, 12, 12)
    assert lay.delta == (3 - 6) + (4 - 4) and position(lay, 40) == (40 + lay.delta,) * 3


def test_chunked_rows_are_the_whole_prompts_rows():
    lay = RopeLayout((image(2, 3, 4), image(20, 5, 2), image(31, 1, 1)))
    base, whole = 1000, []
    s_whole: list = []
    whole = lay.rows(0, 40, base, s_whole)
    for cut in range(1, 40):
        for a, b in ((0, cut), (cut, 40)):
            scratch: list = []
            rows = lay.rows(a, b, base, scratch)
            assert [scratch[r - base] if r >= base else r for r in rows] == \
                   [s_whole[r - base] if r >= base else r for r in whole[a:b]]


def test_scratch_rows_rotate_each_frequency_by_its_axis():
    rope = PartialRotary(256, 64, 1e7, 64, torch.device("cpu"), mrope_section=(11, 11, 10))
    rope.write_scratch([(5, 5, 5), (7, 20, 30)])
    assert torch.equal(rope.cos_sin[64], rope.cos_sin[5])  # a text position on every axis is the 1-D row
    axis = [0 if j % 3 == 0 or j >= 33 else 1 if j % 3 == 1 else 2 for j in range(32)]  # T, H, W interleaved
    pos = torch.tensor([(7, 20, 30)[a] for a in axis], dtype=torch.float32)
    angles = pos * rope.inv_freq
    assert torch.equal(rope.cos_sin[65], torch.cat((angles.cos(), angles.sin())))
    with pytest.raises(ValueError):
        rope.write_scratch([(0, 0, 0)] * (65_536 + 1))


@pytest.mark.parametrize("size,grid", [((200, 200), (8, 8)), ((16, 16), (8, 8)), ((48, 80), (7, 11)),
                                       ((720, 1280), (22, 40)), ((1080, 1920), (34, 60)), ((4000, 6000), (104, 156)),
                                       ((8192, 8192), (128, 128))])
def test_qwen_sizes_match_the_reference_processor(size, grid):
    """`smart_resize` cases from the reference notes (height, width) -> merged (rows, cols)."""
    h, w = qv.smart_resize(*size)
    assert (h // qv.FACTOR, w // qv.FACTOR) == grid


def test_qwen_preprocess_lays_patches_out_in_2x2_blocks():
    """Patch (row 1, col 0) is the third of block (0, 0); patch (row 0, col 2) the first of block (0, 1): sequence
    indices 2 and 4 on a 16 x 16-patch image (256 x 256 pixels, not resized)."""
    import numpy as np
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    rgb[16:32, 0:16] = 255
    rgb[0:16, 32:48] = 255
    patches, gh, gw = qv.preprocess(rgb)
    assert patches.shape == (256, 3 * 16 * 16) and (gh, gw) == (16, 16)
    assert (patches.float().mean(1) > 0).nonzero().flatten().tolist() == [2, 4]
    with pytest.raises(ValueError):
        qv.smart_resize(1, 500)


def test_qwen_video_resize_follows_the_reference_rule():
    """The video processor's `smart_resize`: sides to 32 pixels, every frame's pixels within the clip's budget, a short
    side scaled up to the factor first; and this module's cap of FRAME_MAX_PIXELS a frame."""
    assert qv.video_resize(4, 1280, 720) == (704, 1280)  # 720 rounds down to 22 factors, as the reference's `round`
    assert qv.video_resize(16, 1280, 720) == (704, 1280)  # 14.4 MP together: within the budget untouched
    assert qv.video_resize(4, 40, 24) == (32, 64)  # a short side up to 32 first, the frame count's budget kept
    assert qv.video_resize(2, 10, 10) == (64, 64)  # 2,048 pixels: grown to the 4,096 minimum
    h, w = qv.video_resize(32, 1280, 720)  # 28.9 MP: shrunk to 32-pixel sides within 25,165,824
    assert (h, w) == (640, 1152) and 32 * h * w <= qv.VIDEO_MAX_PIXELS
    h, w = qv.video_resize(2, 4000, 3000)  # 24 MP together fits the budget, but not 4,194,304 a frame
    assert (h, w) == (1760, 2336) and 2 * h * w <= 2 * qv.FRAME_MAX_PIXELS
    for frames, size in [(2, (300, 4000)), (7, (641, 479)), (100, (301, 299)), (5, (33, 1000))]:
        h, w = qv.video_resize(frames, *size)
        assert h % qv.FACTOR == w % qv.FACTOR == 0
        assert round(frames / 2) * 2 * h * w <= min(qv.VIDEO_MAX_PIXELS, qv.FRAME_MAX_PIXELS * round(frames / 2) * 2)
        assert frames * h * w >= qv.VIDEO_MIN_PIXELS
    with pytest.raises(ValueError):
        qv.video_resize(1, 100, 100)  # fewer frames than a pair
    with pytest.raises(ValueError):
        qv.video_resize(4, 6100, 30)  # aspect beyond 200
    assert qv.video_grid(5, 448, 320) == (3, (10, 14))  # an odd frame count: its last repeats, 3 pairs


def test_qwen_video_sampling_and_timestamps_follow_the_processor():
    assert qv.sample_indices(48, 24.0) == [0, 16, 31, 47]  # int(48 / 24 * 2) frames, spread by linspace
    assert qv.sample_indices(3, 24.0) == [0, 1, 2]  # the 4-frame floor gives no more than the clip has
    assert qv.sample_indices(30, 24.0) == [0, 10, 19, 29]  # int(30 / 24 * 2) is 2: the floor
    assert len(qv.sample_indices(10000, 24.0)) == qv.DEFAULT_MAX_FRAMES  # the cap
    assert len(qv.sample_indices(10000, 24.0, max_frames=7)) == 7
    assert qv.sample_indices(100, 0) == qv.sample_indices(100, 24.0)  # no frame rate: read at 24
    assert qv.timestamps([0, 16, 31, 47], 24.0) == pytest.approx([8 / 24, 39 / 24])  # each pair's mean frame time
    assert qv.timestamps([0, 5, 9], 10.0) == pytest.approx([0.25, 0.9])  # an odd list repeats its last index
    assert qv.timestamp_text(1 / 3) == "<0.3 seconds>"
    with pytest.raises(ValueError):
        qv.sample_indices(0, 24.0)


def test_qwen_preprocess_pair_lays_each_frame_on_its_temporal_slice():
    import numpy as np
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    frame[:32] = 255  # the top-left 2 x 2-block row of a 16 x 16-patch image (not resized)
    pair, gh, gw = qv.preprocess_pair(np.stack([frame, 255 - frame]), (8, 8))
    single, _, _ = qv.preprocess(frame)
    assert pair.shape == (256, 3 * 2 * 16 * 16) and (gh, gw) == (16, 16) and pair.dtype == torch.bfloat16
    # slice 0 is the first frame's (bright) top-left block's first patch, slice 1 the second frame's (dark) same patch
    assert pair[0, :256].float().mean() > 0.9 and pair[0, 256:512].float().mean() < -0.9
    assert torch.equal(pair.view(256, 3, 2, 256)[:, :, 0].reshape(-1), single.reshape(-1))  # a frame's: `preprocess`'s
    same, _, _ = qv.preprocess_pair(np.stack([frame, frame]), (8, 8))
    assert torch.equal(same.view(256, 3, 2, 256)[:, :, 0], same.view(256, 3, 2, 256)[:, :, 1])  # a frame with itself: `encode`'s input


def test_a_videos_pairs_are_image_spans():
    """Three pairs of a 2 x 3 grid, a timestamp's two tokens before each: every pair starts from the text position after
    the previous pair's larger side, as `get_rope_index` makes of a clip split into t = 1 grids."""
    lay = RopeLayout((image(2, 2, 3), image(10, 2, 3), image(18, 2, 3)))
    assert [position(lay, i) for i in (0, 1)] == [(0, 0, 0), (1, 1, 1)]
    assert position(lay, 2) == (2, 2, 2) and position(lay, 7) == (2, 3, 4)
    assert [position(lay, i) for i in (8, 9)] == [(5, 5, 5), (6, 6, 6)]  # 2 + max(2, 3)
    assert position(lay, 10) == (7, 7, 7) and position(lay, 15) == (7, 8, 9)
    assert [position(lay, i) for i in (16, 17)] == [(10, 10, 10), (11, 11, 11)]
    assert position(lay, 18) == (12, 12, 12) and position(lay, 23) == (12, 13, 14)
    assert position(lay, 24) == (15, 15, 15) and lay.delta == 15 - 24
    scratch: list = []
    whole = lay.rows(0, 25, 1000, scratch)
    for cut in range(1, 25):  # chunked, the rows are the whole prompt's
        for a, b in ((0, cut), (cut, 25)):
            s: list = []
            assert [s[r - 1000] if r >= 1000 else r for r in lay.rows(a, b, 1000, s)] == \
                   [scratch[r - 1000] if r >= 1000 else r for r in whole[a:b]]


def test_the_encoder_serves_what_the_video_path_calls():
    """Without weights (the GPU suite loads them): the class has every member api/media.py and media.Encoder call on a
    video, bound as the image path's are."""
    for name in ("preprocess_pair", "encode_pair", "video_grid", "sample_indices", "timestamps", "timestamp_text"):
        assert callable(getattr(qv.QwenVision, name)), name
    assert qv.QwenVision.video_token == qv.VIDEO_TOKEN and qv.QwenVision.video_marker == qv.VIDEO_MARKER
    assert qv.QwenVision.video_grid(5, 448, 320) == (3, (10, 14))  # bound like `preprocess`: no instance needed


def test_the_model_loaders_compile():
    """The two loaders pass the frame cap to the tower; they import their kernels, so on a CPU host they are only
    compiled here (a wrap without parentheses was a SyntaxError the unit suite could not see)."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2] / "hibana" / "model"
    for name in ("qwen38.py", "flashnext/model.py"):
        source = (root / name).read_text()
        compile(source, name, "exec")
        assert "current().video_max_frames" in source


class VideoLike:
    """An encoder reading video, as QwenVision is without its weights."""
    image_token, image_start, image_end, marker = qv.IMAGE_TOKEN, qv.IMAGE_START, qv.IMAGE_END, qv.MARKER
    video_token, video_marker, identity, video_identity = qv.VIDEO_TOKEN, qv.VIDEO_MARKER, b"test", b"video"
    max_frames = qv.DEFAULT_MAX_FRAMES
    sample_indices, timestamps, timestamp_text = staticmethod(qv.sample_indices), staticmethod(qv.timestamps), staticmethod(qv.timestamp_text)


def test_the_qwen_video_marker_escapes_and_expands():
    from tests.unit.test_images import CharTokenizer
    tok = CharTokenizer({"<|vision_start|>": qv.IMAGE_START, "<|vision_end|>": qv.IMAGE_END, "<|image_pad|>": qv.IMAGE_TOKEN,
                         "<|video_pad|>": qv.VIDEO_TOKEN})
    markers = api_media.markers(VideoLike)
    assert markers == (qv.MARKER, qv.VIDEO_MARKER) and api_media.markers(QwenLike()) == (qv.MARKER,)
    text = "a <|vision_start|><|video_pad|><|vision_end|> b <|vision_start|><|image_pad|><|vision_end|>"
    escaped = api_media.escape(text, markers)
    assert "\ue000|vision_start|><|video_pad|>" in escaped and "\ue000|vision_start|><|image_pad|>" in escaped
    rendered = (f"q <|vision_start|><|video_pad|><|vision_end|> {api_media._escaped(qv.VIDEO_MARKER)} "
                f"<|vision_start|><|image_pad|><|vision_end|> {api_media._escaped(qv.MARKER)} w")
    ids, offsets = api_media.Markers(tok, VideoLike).encode_with_offsets(rendered)
    at = [i for i, t in enumerate(ids) if t in (qv.IMAGE_TOKEN, qv.VIDEO_TOKEN)]  # the marks' token indices
    assert ids.count(qv.VIDEO_TOKEN) == 1 and ids.count(qv.IMAGE_TOKEN) == 1 and ids[at[0]] == qv.VIDEO_TOKEN
    assert [rendered[a:b] for a, b in (offsets[i] for i in at)] == [qv.VIDEO_MARKER, qv.MARKER]  # the template's own
    text_ids = "".join(chr(t - 100) for t in ids if t not in (qv.IMAGE_TOKEN, qv.VIDEO_TOKEN))
    assert text_ids == "q  " + qv.VIDEO_MARKER + "  " + qv.MARKER + " w"  # the escapes undone, the markers read as text
    video = api_media.Prepared(b"v" * 32, 12, b"V", (2, 2), video=True, times=(0.0, 12.34, 120.0), pairs=(b"p0", b"p1", b"p2"))
    picture = api_media.Prepared(b"a" * 32, 4, b"A", (2, 2))
    out, moved, engine_images = api_media.expand(ids, at[1], [video, picture], VideoLike, tok)
    k, growth = at[0], 0
    for step, (seconds, data) in enumerate(zip(video.times, video.pairs)):
        ts = tok.encode(qv.timestamp_text(seconds))
        assert out[k:k + len(ts)] == ts and out[k + len(ts)] == qv.IMAGE_START
        image = engine_images[step]
        assert image.pair and image.grid == (2, 2) and image.data == data and len(image.ids) == 4
        assert image.start == k + len(ts) + 1 and out[image.start:image.end] == list(image.ids)
        assert out[image.end] == qv.IMAGE_END
        k = image.end + 1
        growth += len(ts) + 2 + len(image.ids)
    growth -= 1  # less the video token all of them replace
    assert len(set(i for e in engine_images[:3] for i in e.ids)) == 12  # each pair has its own ids
    assert [e.pair for e in engine_images] == [True] * 3 + [False] and engine_images[3].data == b"A"
    assert out[k:k + at[1] - at[0] - 1] == ids[at[0] + 1:at[1]]  # the text between, as it was
    k += at[1] - at[0] - 1
    assert out[k] == qv.IMAGE_START and out[k + 1:k + 5] == list(engine_images[3].ids) and out[k + 5] == qv.IMAGE_END
    assert out[k + 6:] == ids[at[1] + 1:] and len(out) == len(ids) + growth + picture.tokens + 1
    assert moved == at[1] + growth  # the picture's mark, moved by what the video added before it
    with pytest.raises(InvalidRequest, match="placed 1 of the request's 0 images"):
        api_media.expand(ids, 0, [video], VideoLike, tok)
    with pytest.raises(InvalidRequest, match="another order"):
        api_media.expand(ids, 0, [picture, video], VideoLike, tok)
    with pytest.raises(InvalidRequest, match="cannot render"):
        api_media.expand(ids, 0, [video, picture], VideoLike)  # no tokenizer for the timestamps


class QwenLike:
    image_token, image_start, image_end, marker = qv.IMAGE_TOKEN, qv.IMAGE_START, qv.IMAGE_END, qv.MARKER
    identity = b"test"

    def grid(self, width, height):
        return (height // 32, width // 32)


def test_the_qwen_marker_expands_with_grids():
    ids = [1, qv.IMAGE_TOKEN, 2]
    images = [api_media.Prepared(b"a" * 32, 6, b"A", (2, 3))]
    out, moved, engine_images = api_media.expand(ids, 3, images, QwenLike())
    assert out[:2] == [1, qv.IMAGE_START] and out[-2:] == [qv.IMAGE_END, 2] and len(out) == 3 + 6 + 1
    assert engine_images[0].grid == (2, 3) and engine_images[0].start == 2 and moved == 3 + 7
    assert api_media.escape("say <|vision_start|><|image_pad|><|vision_end|>", qv.MARKER) != \
        "say <|vision_start|><|image_pad|><|vision_end|>"


def test_without_specials_keeps_every_other_id():
    """Image tokens that sit past the model vocabulary (as the Qwen checkpoints' do) read as text, and the added
    tokens after them keep their ids."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from hibana.tokenizer import SnapshotTokenizer
    vocab = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz<>|_ ")}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="_"))
    tok.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    tok.add_special_tokens(["<img>", "<think>"])
    t = SnapshotTokenizer.__new__(SnapshotTokenizer)
    t._tok = Tokenizer.from_str(tok.to_str())
    img, think = t.encode("<img>")[0], t.encode("<think>")[0]
    out = t.without_specials({img})
    assert out.encode("<think>") == [think]
    assert img not in out.encode("<img>") and len(out.encode("<img>")) == 5
    assert json.loads(out._tok.to_str())["added_tokens"][-1]["id"] == think


def test_prefill_rows_and_group_firsts_from_one_pass():
    """A batch of a text-only span and an image span starting mid-sequence: each row's rotary row, each row's
    group-first row (index - 3, for Flash-Next's indexer), and every image row written to the scratch rows once."""
    from hibana.engine.context import ForwardContext
    from hibana.engine.executor import Executor
    ex = Executor.__new__(Executor)
    ex.mrope, ex.device = True, torch.device("cpu")
    ex.model = SimpleNamespace(rope=PartialRotary(256, 64, 1e7, 64, torch.device("cpu"), mrope_section=(11, 11, 10)))
    ex.target = SimpleNamespace(indexer_budget=2048, indexer_compress_ratio=4)
    with_image = SimpleNamespace(media=(), rope=RopeLayout((image(2, 2, 3),)))  # placeholders at 2..7, delta -3 after
    ctx = ForwardContext(mode="prefill", kv_pool=None, state_pool=None, positions=torch.zeros(1))
    ex._image_rows(ctx, [(SimpleNamespace(media=(), rope=None), 5, 7, 0), (with_image, 1, 10, 2)])
    scratch = list(range(64, 70))
    assert ctx.rope_positions.tolist() == [5, 6, 1, *scratch, 5, 6]
    assert ctx.rope_first.tolist() == [2, 3, 0, 0, 0, 1, *scratch[:5]]
    assert torch.equal(ex.model.rope.cos_sin[64], ex.model.rope.cos_sin[2])  # (2, 2, 2): the image's first block
