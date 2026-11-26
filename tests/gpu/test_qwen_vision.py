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
