# SPDX-License-Identifier: Apache-2.0
"""Phase 3: prove the Flash-Next *video* path matches the references, as `tools/vision_parity.py` proved the image
tower (Phase 1).

Three checks, all on synthetic data (a bright square drifting across a dark field; nothing downloaded):

1. `vision.video_size` (the clip-level smart resize) against transformers 5.15.1's
   `qwen3_vl.video_processing_qwen3_vl.smart_resize` (factor 32, temporal_factor 2, the checkpoint's
   4,096 / 25,165,824 pixel bounds) over randomized (frames, height, width) triples — flagging only cases where our
   extra per-frame cap (FRAME_MAX_PIXELS) does not engage, which is a deliberate deviation.
2. The tower: a 6-frame clip's per-pair `encode_pair` outputs, concatenated, against the transformers
   `Qwen3VLVisionModel` fed the same preprocessed frames as one [[t, h, w]] clip (the Conv3d pairing frame pairs,
   per-frame-pair attention segments, per-frame position table and axial rotary repeated over t). Bit-parity is the
   goal, as Phase 1's was; `encode_pair(f, f)` == `encode(f)` is asserted as well.
3. `mrope.positions_3d` on the video prompt the API builds (`<t seconds>` text runs between per-pair
   <|video_pad|> spans) against `Qwen3VLForConditionalGeneration.get_rope_index` (transformers 5.15.1) on the same
   ids, which splits a video into per-frame (t=1) grids and advances by max(H, W) // 2 a frame — the same stacking
   `positions_3d` gives consecutive image spans.

Usage (GB10, the fork's venv): uv run python tools/video_parity.py [/path/to/snapshot]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hibana.model.flashnext.vision import (FRAME_MAX_PIXELS, MERGE, PATCH, QwenVision, TEMPORAL, VIDEO_MAX_PIXELS,
                                           VIDEO_MIN_PIXELS, video_size)  # noqa: E402
from hibana.weights.safetensors import SnapshotWeights  # noqa: E402

SNAPSHOT = Path(sys.argv[1] if len(sys.argv) > 1 else
                "~/.cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/"
                "7b719225242aacd3dbd3f9407468c2ee9a9d2594")


def resize_parity() -> bool:
    """`video_size` vs the transformers video smart_resize on randomized inputs (skip triples our per-frame cap
    re-budgets: that path is a documented deviation, not a transcription of the reference)."""
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize as ref
    rng = np.random.default_rng(11)
    checked = capped = mismatches = 0
    for _ in range(2000):
        frames = int(rng.integers(1, 129))
        h, w = int(rng.integers(24, 2200)), int(rng.integers(24, 2200))
        try:
            mine = video_size(frames, w, h)
        except ValueError:
            continue
        t_bar = round(frames / TEMPORAL) * TEMPORAL
        if VIDEO_MAX_PIXELS > FRAME_MAX_PIXELS * t_bar:  # our cap would shrink what the reference leaves alone
            capped += 1
            continue
        theirs = ref(frames, h, w, temporal_factor=TEMPORAL, factor=PATCH * MERGE,
                     min_pixels=VIDEO_MIN_PIXELS, max_pixels=VIDEO_MAX_PIXELS)[::-1]  # theirs is (h, w)
        checked += 1
        if mine != theirs:
            mismatches += 1
            if mismatches < 5:
                print(f"  MISMATCH frames={frames} {w}x{h}: mine {mine}, reference {theirs}")
    print(f"[resize] {checked} sizes checked ({capped} skipped: per-frame cap engages), {mismatches} mismatches")
    return mismatches == 0 and checked > 1000


def synthetic_clip(frames: int = 6, w: int = 448, h: int = 320) -> np.ndarray:
    """A bright square drifting left to right across a dark gradient field, [frames, H, W, 3] uint8."""
    rng = np.random.default_rng(3)
    out = np.zeros((frames, h, w, 3), np.uint8)
    for f in range(frames):
        out[f, :, :, 0] = (np.arange(w) / w * 40).astype(np.uint8)[None, :]
        x0 = int((w - 80) * f / max(frames - 1, 1))
        out[f, 60:220, x0:x0 + 80] = (235, 200, 60)
        for _ in range(3):  # small distractors so attention has something to mix
            dx, dy = int(rng.integers(0, w - 40)), int(rng.integers(0, h - 40))
            out[f, dy:dy + 20, dx:dx + 20, 2] = 120
    return out


def reference_clip_pixels(pairs: list[torch.Tensor], rows: int, cols: int) -> torch.Tensor:
    """The pairs' preprocessed frames [2, 3, H, W] as the reference processor's `pixel_values_videos`: one row a
    patch, (C, T=2, 16, 16) flattened with slice 0 from the pair's first frame and slice 1 from its second, patches
    in spatial-merge-block order, pairs in clip order."""
    out = []
    for pair in pairs:
        x = pair.float()  # [2, 3, H, W]
        per = [x[i].reshape(3, rows // MERGE, MERGE, PATCH, cols // MERGE, MERGE, PATCH)
               .permute(1, 4, 2, 5, 0, 3, 6).reshape(-1, 3, PATCH * PATCH) for i in (0, 1)]
        out.append(torch.stack(per, dim=2).reshape(-1, 3 * TEMPORAL * PATCH * PATCH))
    return torch.cat(out)


def build_reference(device: torch.device):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    cfg = Qwen3VLVisionConfig(depth=27, hidden_size=1152, intermediate_size=4304, num_heads=16, in_channels=3,
                              patch_size=16, temporal_patch_size=2, spatial_merge_size=2, out_hidden_size=2560,
                              num_position_embeddings=2304, hidden_act="gelu_pytorch_tanh",
                              deepstack_visual_indexes=[], attn_implementation="sdpa")
    return Qwen3VLVisionModel(cfg).to(device=device, dtype=torch.bfloat16).eval()


def mrope_parity(device) -> bool:
    """`positions_3d` on a video prompt vs transformers' `get_rope_index` on the same ids. The prompt mirrors what
    api/media.expand builds: `<0.0 seconds>`-style text runs between per-pair <|vision_start|>/<|video_pad|>/
    <|vision_end|> spans. The reference needs a model for the method; only its config is read."""
    from hibana.model.flashnext import mrope
    from hibana.engine import media
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig, Qwen3VLTextConfig, \
        Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    rows, cols, t = 4, 6, 3  # a 3-pair clip of a 4x6 merged grid
    vs, ve, pad = 248053, 248054, 248057
    ts_text = ["<0.0 seconds>", "<0.5 seconds>", "<1.0 seconds>"]
    from hibana.tokenizer import SnapshotTokenizer
    tok = SnapshotTokenizer(SNAPSHOT)
    ids: list[int] = tok.encode("describe this clip:")  # leading text
    spans = []
    for f in range(t):
        ids += tok.encode(ts_text[f]) + [vs] + [pad] * (rows * cols) + [ve]
        spans.append((len(ids) - rows * cols - 1, rows, cols))  # the first pad (past the [vs] before it)
    ids += tok.encode(" what moves?")
    # our side: engine Images (one a pair), placeholders any distinct negative ids
    images = []
    for start, r, c in spans:
        images.append(media.Image(start=start, ids=tuple(range(-r * c - start - 1, -start - 1)), data=b"", grid=(r, c)))
    ours = mrope.positions_3d(len(ids), tuple(images))
    assert ours is not None

    # the reference: same ids but video pads as placeholders, mm_token_type_ids marking text(0)/video(2) runs
    video_ids = []
    types = []
    at = 0
    for start, r, c in spans:
        video_ids += ids[at:start - 1]  # text and the timestamp up to the [vs]; the span is appended by hand
        types += [0] * (start - 1 - at)
        video_ids += [vs] + [pad] * (r * c) + [ve]
        types += [0] + [2] * (r * c) + [0]  # only the pads are the video run; the markers read as text (the
        # processor's mm_token_type_ids), and get_rope_index groups runs by type
        at = start + r * c + 1  # past the [ve] after the pads ([vs] sits at start - 1)
    video_ids += ids[at:]
    types += [0] * len(ids[at:])
    text_cfg = Qwen3VLTextConfig(vocab_size=248_064 + 1024, hidden_size=2560, num_hidden_layers=1)
    vision_cfg = Qwen3VLVisionConfig(spatial_merge_size=2)
    cfg = Qwen3VLConfig(text_config=text_cfg, vision_config=vision_cfg)
    model = Qwen3VLModel._from_config(cfg).eval()  # config-only; get_rope_index reads config
    with torch.inference_mode():
        ref, delta = model.get_rope_index(
            torch.tensor([video_ids]), torch.tensor([types]), video_grid_thw=torch.tensor([[t, rows * MERGE, cols * MERGE]]),
            attention_mask=torch.ones(1, len(video_ids), dtype=torch.bool))
    ref = ref.numpy().reshape(3, -1)
    same = np.array_equal(ours, ref)
    print(f"[mrope] positions_3d == get_rope_index: {same} (delta ref {int(delta)}, ours "
          f"{ours.max() + 1 - len(ids)}); last-text pos {ours[0, -1]}")
    if not same:
        d = np.nonzero((ours != ref).any(0))[0]
        print(f"  first differing rows: {d[:10]}, ours[:, {d[0]}]={ours[:, d[0]]}, ref={ref[:, d[0]]}")
    return same


def main() -> None:
    avail = [l for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemAvailable")]
    print(f"[mem] {avail[0]}")
    assert int(avail[0].split()[1]) >= 20 * 2**20, "MemAvailable below 20 GiB: stop (GB10 wedge rules)"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ok = resize_parity()

    snap = SnapshotWeights(SNAPSHOT)
    mine = QwenVision(snap, device)
    ref = build_reference(device)
    names = sorted(n for n in snap.names() if n.startswith("model.visual."))
    ref.load_state_dict({n[len("model.visual."):]: snap.get(n) for n in names}, strict=True)

    clip = synthetic_clip()
    w_px, h_px = video_size(len(clip), clip.shape[2], clip.shape[1])
    rows, cols = h_px // PATCH, w_px // PATCH
    t = (len(clip) + 1) // 2
    pairs = [mine.preprocess_pair(clip[i:i + 2].copy(), w_px, h_px) for i in range(0, len(clip), 2)]
    ours = torch.cat([mine.encode_pair(p) for p in pairs])
    assert tuple(ours.shape) == (t * rows * cols // MERGE**2, 2560), ours.shape

    # encode_pair(f, f) is encode(f): the image path feeds the same patches twice
    img = mine.encode(mine.preprocess(clip[0]))
    dup = mine.encode_pair(pairs[0][:1].expand(2, -1, -1, -1).contiguous())
    same_dup = torch.equal(img, dup)
    print(f"[identity] encode_pair(f, f) == encode(f): {same_dup}")

    with torch.inference_mode():
        pixels = reference_clip_pixels([p.cpu() for p in pairs], rows, cols).to(device)
        r = ref(pixels, torch.tensor([[t, rows, cols]], device=device))
        ref_out = r.pooler_output
    diff = (ours.float() - ref_out.float()).abs()
    cos = torch.nn.functional.cosine_similarity(ours.float(), ref_out.float(), dim=-1)
    passed = diff.max().item() < 1e-2 and cos.min().item() > 0.999 and same_dup
    ok &= passed
    print(f"[clip] 6 frames -> {t} pairs, grid {rows}x{cols}, {ours.shape[0]} tokens | "
          f"max abs diff {diff.max().item():.5f} | cosine min {cos.min().item():.6f} mean {cos.mean().item():.6f} | "
          f"{'PASS' if passed else 'FAIL'}")

    ok &= mrope_parity(device)
    print("PARITY:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
