"""The Qwen vision tower (`hibana/model/qwen_vision.py`) and its fused kernels against transformers'
`Qwen3_5VisionModel` (Flash-Next's tower is the same code) in FP32 with the same weights, on the 27B and Flash-Next.
Needs the pinned snapshots and transformers (the eval group)."""

import gc
import json

import pytest
import torch
import torch.nn.functional as F

transformers = pytest.importorskip("transformers")
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionConfig, Qwen3_5VisionModel  # noqa: E402

from hibana.model import qwen_vision as v  # noqa: E402
from tests.gpu.test_nemotron_vision import cos, image  # noqa: E402

EMBED_COS = 0.999


@pytest.fixture(scope="module", params=["qwen38", "flashnext"])
def pair(request):
    """(Hibana's QwenVision, Transformers' vision model with the same weights) for one checkpoint."""
    from hibana.target import TARGETS
    from hibana.weights.safetensors import SnapshotWeights
    try:
        snap = TARGETS[request.param].snapshot_dir()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    cfg = json.loads((snap / "config.json").read_text())
    w = SnapshotWeights(snap)
    ours = v.load(w, torch.device("cuda"), cfg, v.MAX_PIXELS, log=None)
    vc = {k: val for k, val in cfg["vision_config"].items() if k != "model_type"}
    ref = Qwen3_5VisionModel(Qwen3_5VisionConfig(**vc))
    names = [n for n in w.names() if n.startswith(f"{v.PREFIX}.")]
    missing, unexpected = ref.load_state_dict({n[len(v.PREFIX) + 1:]: w.get(n) for n in names}, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    w.close()
    ref = ref.to("cuda", torch.float32).eval()
    yield request.param, ours, ref
    del ours, ref
    gc.collect()
    torch.cuda.empty_cache()


def reference(ref, pixels) -> torch.Tensor:
    """The reference tower on `preprocess`'s patches: both temporal slots filled, as its processor does."""
    patches, gh, gw = pixels
    n = patches.shape[0]
    full = patches.view(n, 3, 1, v.PATCH * v.PATCH).expand(n, 3, v.TEMPORAL, v.PATCH * v.PATCH).reshape(n, -1)
    return ref(full.cuda().float(), grid_thw=torch.tensor([[1, gh, gw]], device="cuda")).pooler_output


def reference_pair(ref, pixels) -> torch.Tensor:
    """The reference tower on `preprocess_pair`'s patches: both temporal slots are the pair's frames (a t = 1 grid, as the
    video reference's `cu_seqlens` make of a clip's frame pairs)."""
    patches, gh, gw = pixels
    return ref(patches.cuda().float(), grid_thw=torch.tensor([[1, gh, gw]], device="cuda")).pooler_output


@pytest.mark.parametrize("size", [(512, 512), (1080, 1920), (1600, 400)])
def test_pair_encode_is_the_image_path(pair, size):
    """`encode_pair` of a frame with itself is bitwise `encode` of it, and of two frames the reference tower on the same
    patches."""
    import numpy as np
    name, ours, ref = pair
    a, b = image(*size, seed=1), image(*size, seed=2)
    grid = ours.grid(size[1], size[0])
    with torch.inference_mode():
        one = ours.encode(ours.preprocess(a))
        same = ours.encode_pair(ours.preprocess_pair(np.stack([a, a]), grid))
        pixels = ours.preprocess_pair(np.stack([a, b]), grid)
        got, expected = ours.encode_pair(pixels), reference_pair(ref, pixels)
    assert torch.equal(one, same)
    assert got.shape == one.shape
    c = cos(got, expected)
    print(f"{name} {size}: pair of a {grid[0]} x {grid[1]} grid, embeddings cos {c:.6f}")
    assert c >= EMBED_COS, (name, size, c)


@pytest.mark.parametrize("size", [(512, 512), (1080, 1920), (1600, 400), (20, 30)])
def test_encoder_matches_the_reference(pair, size):
    name, ours, ref = pair
    pixels = ours.preprocess(image(*size, seed=size[0]))
    with torch.inference_mode():
        expected = reference(ref, pixels)
        got = ours.encode(pixels)  # the fused CUDA path
    assert got.shape == expected.shape == (ours.tokens(size[1], size[0]), ours.hidden)
    c = cos(got, expected)
    print(f"{name} {size}: grid {ours.grid(size[1], size[0])}, embeddings cos {c:.6f}")
    assert c >= EMBED_COS, (name, size, c)


def test_fused_kernels_against_torch():
    """`kernels/vision.py`: the rotary bitwise the torch arithmetic it replaces; the residual add bitwise and its
    LayerNorm within a BF16 rounding of torch's."""
    from hibana.kernels.vision import add_layer_norm, rope_qk
    g = torch.Generator(device="cuda").manual_seed(0)
    for n in (1, 37, 4096):
        qkv = torch.randn(n, 3, v.HEADS, v.HEAD_DIM, device="cuda", generator=g).to(torch.bfloat16)
        ang = torch.rand(n, v.HEAD_DIM // 2, device="cuda", generator=g) * 50
        cos, sin = torch.cat([ang.cos()] * 2, -1), torch.cat([ang.sin()] * 2, -1)
        half = v.HEAD_DIM // 2
        for j, got in enumerate(rope_qk(qkv, cos, sin)):
            f = qkv[:, j].float()
            rot = torch.cat([-f[..., half:], f[..., :half]], -1)
            assert torch.equal(got, (f * cos[:, None] + rot * sin[:, None]).to(torch.bfloat16)), (n, j)
        h, y = (torch.randn(n, v.WIDTH, device="cuda", generator=g).to(torch.bfloat16) for _ in range(2))
        w_, b_ = torch.randn(v.WIDTH, device="cuda", generator=g).to(torch.bfloat16), torch.randn(v.WIDTH, device="cuda", generator=g).to(torch.bfloat16)
        h2, x2 = add_layer_norm(h, y, w_, b_, v.LN_EPS)
        assert torch.equal(h2, h + y)
        ref = F.layer_norm(h + y, (v.WIDTH,), w_, b_, v.LN_EPS)
        assert (x2.float() - ref.float()).abs().max() <= ref.float().abs().max() * 2**-7, n


def test_preprocess_matches_the_processor():
    """`preprocess` against Transformers' `Qwen2VLImageProcessor` (needs torchvision): bitwise, resized or not."""
    pytest.importorskip("torchvision")
    from PIL import Image
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    proc = Qwen2VLImageProcessor(size={"shortest_edge": v.MIN_PIXELS, "longest_edge": v.MAX_PIXELS}, patch_size=16,
                                 temporal_patch_size=2, merge_size=2, image_mean=[0.5] * 3, image_std=[0.5] * 3)
    for size in [(256, 256), (300, 457), (20, 30), (96, 1024)]:
        rgb = image(*size, seed=7)
        out = proc(images=Image.fromarray(rgb), return_tensors="pt")
        patches, gh, gw = v.preprocess(rgb)
        assert out["image_grid_thw"][0].tolist() == [1, gh, gw]
        mine = patches.view(-1, 3, 1, 256).expand(-1, 3, 2, 256).reshape(-1, 1536)
        assert torch.equal(mine, out["pixel_values"].to(torch.bfloat16)), size


@pytest.mark.parametrize("tokens", [1024, 4096, 16384])
def test_encode_cost(pair, tokens):
    """Time and device memory of one encode at 1k, 4k and the cap's 16k embeddings (printed; the memory must stay
    under `workspace_bytes`, which the executor's plan reserves)."""
    import time
    name, ours, _ = pair
    side = int((tokens ** 0.5) * v.FACTOR)
    pixels = ours.preprocess(image(side, side, seed=tokens))
    ours.encode(pixels)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    started = time.perf_counter()
    out = ours.encode(pixels)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() - base - out.numel() * out.element_size()
    print(f"{name}: {out.shape[0]} embeddings ({pixels[0].shape[0]} patches) in {seconds * 1e3:.0f} ms, "
          f"workspace {peak / 2**20:.0f} MiB (reserved {ours.workspace_bytes / 2**20:.0f} MiB)")
    assert out.shape[0] == tokens
    assert peak <= ours.workspace_bytes


def _test_clip(frames: int = 16, size: tuple[int, int] = (48, 64)) -> bytes:
    """A tiny H.264 clip of `frames` solid frames, by ffmpeg on the host (None without it)."""
    import subprocess
    try:
        p = subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[1]}x{size[0]}", "-r", "24",
                            "-i", "pipe:0", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "frag_keyframe+empty_moov", "-f", "mp4", "pipe:1"],
                           input=b"".join(bytes([i * 16 % 256] * 3) * (size[0] * size[1]) for i in range(frames)), capture_output=True, timeout=60)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return p.stdout if p.returncode == 0 and p.stdout else None


def test_video_prompt_is_the_processors():
    """A video request's prompt is `Qwen3VLProcessor`'s, token for token (Hibana's placeholders read as the video
    token), and its M-RoPE positions `get_rope_index`'s, on four shapes: a video; text before the video; two videos; a
    system prompt and a second turn with the same video. The template's `<|vision_start|>` and `<|vision_end|>` stay
    around what replaces `<|video_pad|>` (transformers 5.10 and later). Needs the snapshot, transformers and torchvision
    (the processor's video preprocessing), and ffmpeg on the host."""
    pytest.importorskip("torchvision")
    from transformers import AutoProcessor
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    from hibana.api import media as api_media
    from hibana.api.chat_template import ChatTemplate
    from hibana.engine.mrope import RopeLayout
    from hibana.target import TARGETS
    from hibana.tokenizer import SnapshotTokenizer
    from hibana.weights.safetensors import SnapshotWeights
    try:
        snap = TARGETS["qwen38"].snapshot_dir()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    clip = _test_clip()
    if clip is None:
        pytest.skip("no ffmpeg on this host")
    w = SnapshotWeights(snap)
    vision = v.load(w, torch.device("cuda"), json.loads((snap / "config.json").read_text()), v.MAX_PIXELS, log=None)
    w.close()
    tok, tmpl = SnapshotTokenizer(snap), ChatTemplate(snap)
    url = "data:video/mp4;base64," + __import__("base64").b64encode(clip).decode()

    def part():
        return {"type": "video_url", "video_url": {"url": url}}

    shapes = [
        [{"role": "user", "content": [part(), {"type": "text", "text": "In what order do the colours appear?"}]}],
        [{"role": "user", "content": [{"type": "text", "text": "Watch this, then answer."}, part(),
                                      {"type": "text", "text": "Which way does it move?"}]}],
        [{"role": "user", "content": [part(), part(), {"type": "text", "text": "Are these the same?"}]}],
        [{"role": "system", "content": "You are terse."},
         {"role": "user", "content": [{"type": "text", "text": "What is here?"}, part()]},
         {"role": "assistant", "content": "A clip."},
         {"role": "user", "content": [part(), {"type": "text", "text": "And now?"}]}],
    ]
    proc = AutoProcessor.from_pretrained(snap)
    total, fps, width, height = api_media._video_info(clip)
    frames = api_media._video_frames(clip, list(range(total)), width, height)  # every frame: the processor samples
    holder = Qwen3VLModel.__new__(Qwen3VLModel)
    holder.config = type("C", (), {"vision_config": type("V", (), {"spatial_merge_size": 2})()})()
    for n, messages in enumerate(shapes):
        folded, images = api_media.prepare([dict(m) for m in messages], vision, decode=False)
        rendered = tmpl.render(api_media.escape(folded, api_media.markers(vision)), enable_thinking=False, tools=None)
        ids, _, engine_images = api_media.expand(api_media.Markers(tok, vision).encode(rendered), 0, images, vision, tok)
        ids = [v.VIDEO_TOKEN if i < 0 else i for i in ids]
        def as_hf(m):
            content = m.get("content")
            if isinstance(content, list):
                content = [{"type": "video", "video": "clip.mp4"} if p.get("type") in api_media.VIDEO_PARTS else p for p in content]
            return {**m, "content": content}
        text = proc.apply_chat_template([as_hf(m) for m in messages], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        out = proc(text=[text], videos=[frames] * len(images), video_metadata=[{"fps": fps, "total_num_frames": total}] * len(images), return_tensors="pt")
        hf = out["input_ids"][0].tolist()
        assert ids == hf, f"shape {n}: Hibana {len(ids)} tokens, the processor {len(hf)}"
        with torch.inference_mode():
            positions, delta = holder.get_rope_index(out["input_ids"], out["mm_token_type_ids"], video_grid_thw=out["video_grid_thw"])
        layout, scratch = RopeLayout(engine_images), []
        base = 10**9
        ours = []
        k = 0
        for r in layout.rows(0, len(ids), base, scratch):
            if r >= base:
                ours.append(scratch[k]); k += 1
            else:
                ours.append((r, r, r))
        ours = torch.tensor(ours, dtype=positions.dtype).t()
        assert torch.equal(ours, positions[:, 0, :]) and layout.delta == int(delta[0, 0]), f"shape {n}"
        print(f"video prompt shape {n}: {len(ids)} tokens identical to Qwen3VLProcessor, positions identical ({int((out['mm_token_type_ids'] == 2).sum())} video tokens)")
