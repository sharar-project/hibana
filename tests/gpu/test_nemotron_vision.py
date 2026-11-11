"""The Nemotron image path (`hibana/model/nemotron/vision.py`) against Transformers' `RadioModel` carrying the same
checkpoint weights, followed by the checkpoint's own projector steps written as its remote code writes them
(`modeling_nemotron_h_omni.py`: final LayerNorm when present, `pixel_shuffle` v2 at ratio 0.5, `mlp1` as
RMSNorm -> Linear -> ReLU^2 -> Linear), on Super (which has the final LayerNorm) and Bolt (which has none). Tiles are
square, wide and tall, so a transposed patch grid or pixel shuffle shows. Measured: the tower's features cos 0.99998,
the embeddings 0.9996 to 1.0 (BF16 both sides).

Needs the pinned snapshots (each skipped otherwise) and Transformers (the eval group); holds two copies of the tower
(about 3.2 GiB)."""

import gc

import numpy as np
import pytest
import torch
import torch.nn.functional as F

transformers = pytest.importorskip("transformers")
from transformers.models.radio import RadioConfig, RadioModel  # noqa: E402

from hibana.model.nemotron import vision as v  # noqa: E402

TILE_COS = 0.9995  # the tower's output, BF16 both sides
EMBED_COS = 0.999  # the projected embeddings


def cos(a, b) -> float:
    return float(F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0))


def _snapshot(name):
    from hibana.target import TARGETS
    try:
        return TARGETS[name].snapshot_dir()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="module", params=["nemotron", "bolt"])
def pair(request):
    """(Hibana's RadioVision, Transformers' RadioModel with the same weights, the projector tensors) for one checkpoint."""
    import json
    from hibana.weights.safetensors import SnapshotWeights
    snap = _snapshot(request.param)
    w = SnapshotWeights(snap)
    ours = v.RadioVision(w, torch.device("cuda"))
    cfg = json.loads((snap / "config.json").read_text())["vision_config"]
    fields = RadioConfig.__dataclass_fields__ if hasattr(RadioConfig, "__dataclass_fields__") else RadioConfig().to_dict()
    ref = RadioModel(RadioConfig(**{k: val for k, val in cfg.items() if k in fields}))
    m = f"{v.TOWER}.model"
    state = {"embeddings.patch_projection.weight": w.get(f"{m}.patch_generator.embedder.weight"),
             "embeddings.position_embedding": w.get(f"{m}.patch_generator.pos_embed"),
             "embeddings.cls_register_token": w.get(f"{m}.patch_generator.cls_token.token")}
    for i in range(v.LAYERS):
        b, o = f"{m}.blocks.{i}", f"encoder.layer.{i}"
        for n in ("norm1", "norm2", "mlp.fc1", "mlp.fc2"):
            for p in ("weight", "bias"):
                state[f"{o}.{n}.{p}"] = w.get(f"{b}.{n}.{p}")
        for p in ("weight", "bias"):
            state[f"{o}.attention.output.dense.{p}"] = w.get(f"{b}.attn.proj.{p}")
            for name, part in zip(("query", "key", "value"), w.get(f"{b}.attn.qkv.{p}").chunk(3, dim=0)):
                state[f"{o}.attention.attention.{name}.{p}"] = part
    missing, unexpected = ref.load_state_dict(state, strict=False)
    assert not unexpected and all("layer_scale" in k or "input_conditioner" in k or "summary_idxs" in k for k in missing), missing
    ref.make_preprocessor_external()
    ref = ref.to("cuda", torch.bfloat16).eval()
    final = "vision_projector.vision_final_layernorm"
    head = {"final": (w.get(f"{final}.weight"), w.get(f"{final}.bias")) if f"{final}.weight" in w else None,
            "rms": w.get("mlp1.0.weight"), "fc1": w.get("mlp1.1.weight"), "fc2": w.get("mlp1.3.weight")}
    w.close()
    yield request.param, ours, ref, {k: (tuple(t.cuda() for t in val) if isinstance(val, tuple) else val.cuda()) if val is not None else None
                                     for k, val in head.items()}
    del ours, ref
    gc.collect()
    torch.cuda.empty_cache()


def reference_projector(features: torch.Tensor, rows: int, cols: int, head: dict) -> torch.Tensor:
    """`NemotronH_Omni_Reasoning_V3VisionProjector._project` for one tile, step by step."""
    x = features
    if head["final"] is not None:
        x = F.layer_norm(x, (x.shape[-1],), *head["final"], 1e-6)
    x = x.reshape(1, rows, cols, -1)
    n, w_, h_, c = x.size()  # the reference's names
    x = x.view(n, w_, int(h_ * 0.5), int(c / 0.5)).permute(0, 2, 1, 3).contiguous()
    x = x.view(n, int(h_ * 0.5), int(w_ * 0.5), int(c / 0.25)).permute(0, 2, 1, 3).contiguous()
    x = x.reshape(1, -1, x.shape[-1])[0]
    f = x.float()
    x = (head["rms"].float() * (f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + 1e-5))).to(torch.bfloat16)
    return F.linear(F.relu(F.linear(x, head["fc1"])).pow(2), head["fc2"])


def image(h: int, w: int, seed: int) -> np.ndarray:
    """Smooth gradients, blocks and noise: structure the tower responds to, not a constant."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    img = np.stack([x * 255 // max(w - 1, 1), y * 255 // max(h - 1, 1), ((x // 37 + y // 23) % 2) * 255], -1)
    return np.clip(img + rng.normal(0, 20, img.shape), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("size", [(512, 512), (1080, 1920), (1600, 400)])
def test_encoder_matches_the_reference(pair, size):
    name, ours, ref, head = pair
    pixels = v.preprocess(image(*size, seed=size[0]))
    rows, cols = pixels.shape[1] // v.PATCH, pixels.shape[2] // v.PATCH
    with torch.inference_mode():
        features = ref(pixels.unsqueeze(0).cuda()).features[0]
        expected = reference_projector(features, rows, cols, head)
        got = ours.encode(pixels)
    assert got.shape == expected.shape == (v.image_tokens(size[1], size[0]), ours.hidden)
    c = cos(got, expected)
    print(f"{name} {size}: {rows}x{cols} patches, embeddings cos {c:.6f}")
    assert c >= EMBED_COS, (name, size, c)


def test_tower_matches_the_reference(pair):
    """The tower alone, before the projector: Hibana's blocks against Transformers' layers."""
    name, ours, ref, head = pair
    pixels = v.preprocess(image(700, 900, seed=3))
    with torch.inference_mode():
        c = cos(ours.tower(pixels), ref(pixels.unsqueeze(0).cuda()).features[0])
    print(f"{name}: tower cos {c:.6f}")
    assert c >= TILE_COS, (name, c)


@pytest.mark.parametrize("rows,cols", [(32, 32), (68, 120), (120, 68), (104, 128), (130, 102), (6, 208), (8, 250), (2, 2)])
def test_positions_are_the_cropped_square_bitwise(pair, rows, cols):
    """`_positions` computes only the tile's crop of the position table resized to the larger side's square; on CUDA
    its values are the square's."""
    _, ours, _, _ = pair
    side = max(rows, cols)
    square = F.interpolate(ours.pos, size=(side, side), mode="bilinear", align_corners=False).to(torch.bfloat16)
    assert torch.equal(ours._positions(rows, cols), square[0, :, :rows, :cols].flatten(1).t())


def test_an_extreme_tile_encodes_in_bounded_memory(pair):
    """A 100,000 x 50 image: a 2 x 4,560-patch tile, whose square position table would take 106 GB."""
    _, ours, _, _ = pair
    pixels = v.preprocess(image(50, 100_000, seed=5))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = ours.encode(pixels)
    torch.cuda.synchronize()
    assert out.shape == (v.image_tokens(100_000, 50), ours.hidden) == (2280, ours.hidden)
    assert torch.cuda.max_memory_allocated() - base <= v.ENCODE_BYTES
