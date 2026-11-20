# SPDX-License-Identifier: Apache-2.0
"""Phase 1 spike: prove `hibana/model/flashnext/vision.py` matches the transformers reference on the Flash-Next
checkpoint's vision weights.

Reference: `transformers` 5.15.1's `Qwen3VLVisionModel` (which upstream `Qwen4ExpVisionModel` copies and vLLM's
flash-next model inherits) fed the `model.visual.*` tensors of the RadixArk NVFP4 checkpoint — all BF16, unquantized
(the quantizer excludes the tower). Its forward consumes patches in spatial-merge-block order with the frame repeated
along the temporal axis; the spike's tower runs raster order — full per-frame attention is permutation-equivariant, so
both read each patch with its own (h, w) coordinates and agree.

For each synthetic image (colored rectangles with text-like stripes; nothing downloaded): feed the SAME preprocessed
pixels to both paths (the spike's `preprocess`, then re-patchified in the reference's layout), and compare the merged
embeddings [N, 2560]: max abs diff, cosine per image, and the token count against the Qwen patch-grid math.

Usage (GB10, the fork's venv): uv run python tools/vision_parity.py [/path/to/snapshot]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hibana.model.flashnext.vision import MERGE, PATCH, QwenVision, target_size  # noqa: E402
from hibana.weights.safetensors import SnapshotWeights  # noqa: E402

SNAPSHOT = Path(sys.argv[1] if len(sys.argv) > 1 else
                "~/.cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/"
                "7b719225242aacd3dbd3f9407468c2ee9a9d2594")


def synthetic_images() -> list[tuple[str, np.ndarray]]:
    """Colored rectangles with text-like high-frequency stripes, a gradient, and odd sizes."""
    rng = np.random.default_rng(7)
    out = []
    imgs = {
        "rects-640x480": (640, 480),
        "stripes-512x512": (512, 512),
        "odd-817x353": (817, 353),    # uneven rounding of both sides
        "tiny-90x70": (90, 70),       # min-pixels clamp: 2880 px rounds up to 65536+
        "wide-1520x400": (1520, 400), # extreme aspect, many blocks a row
    }
    for name, (w, h) in imgs.items():
        img = np.zeros((h, w, 3), np.uint8)
        for _ in range(8):
            x0, y0 = rng.integers(0, w - 60), rng.integers(0, h - 60)
            cw, ch = rng.integers(40, 120, 2)
            img[y0:y0 + ch, x0:x0 + cw] = rng.integers(30, 255, 3)
        xs = np.arange(w, dtype=np.float32)
        img[:, :, 0] = (xs / w * 255).astype(np.uint8)[None, :]  # horizontal gradient
        if "stripes" in name:  # text-like 3-pixel stripes
            stripe = (np.arange(h) % 6 < 3)[:, None] * 255
            img[:, :, 2] = np.maximum(img[:, :, 2], stripe.astype(np.uint8))
        out.append((name, img))
    return out


def reference_pixels(pixels: torch.Tensor) -> torch.Tensor:
    """The spike's preprocessed image [3, H, W] as the reference processor's `pixel_values`: patches in
    spatial-merge-block order, each row the frame twice along the temporal axis, [P, 3 * 2 * 16 * 16] FP32."""
    c, h, w = pixels.shape
    ph, pw = h // PATCH, w // PATCH
    x = pixels.float().reshape(c, ph // MERGE, MERGE, PATCH, pw // MERGE, MERGE, PATCH)
    x = x.permute(1, 4, 2, 5, 0, 3, 6)  # (blk_r, blk_c, mh, mw, C, ph, pw) — the reference patch order
    x = x.reshape(ph * pw, c, PATCH * PATCH)
    # temporal_patch_size 2: each patch row is (C, T=2, 16, 16) flattened, the frame in both temporal slices
    x = torch.stack([x, x], dim=2).reshape(ph * pw, c * 2 * PATCH * PATCH)
    return x


def build_reference(device: torch.device) -> "torch.nn.Module":
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    cfg = Qwen3VLVisionConfig(depth=27, hidden_size=1152, intermediate_size=4304, num_heads=16, in_channels=3,
                              patch_size=16, temporal_patch_size=2, spatial_merge_size=2, out_hidden_size=2560,
                              num_position_embeddings=2304, hidden_act="gelu_pytorch_tanh",
                              deepstack_visual_indexes=[], attn_implementation="sdpa")
    model = Qwen3VLVisionModel(cfg)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    return model


def main() -> None:
    avail = [l for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemAvailable")]
    print(f"[mem] {avail[0]}")
    free_kib = int(avail[0].split()[1])
    assert free_kib >= 20 * 2**20, "MemAvailable below 20 GiB: stop (GB10 wedge rules)"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[dev] {device}")

    snap = SnapshotWeights(SNAPSHOT)
    visual_names = sorted(n for n in snap.names() if n.startswith("model.visual."))
    assert len(visual_names) == 333, f"expected 333 visual tensors, found {len(visual_names)}"
    print(f"[weights] {len(visual_names)} model.visual.* tensors, "
          f"dtype probe: {snap.get('model.visual.pos_embed.weight').dtype}")

    mine = QwenVision(snap, device)
    print(f"[mine] loaded, {mine.bytes / 2**20:.0f} MiB of tensors")

    ref = build_reference(device)
    missing, unexpected = ref.load_state_dict(
        {n[len("model.visual."):]: snap.get(n) for n in visual_names}, strict=False)
    assert not missing and not unexpected, f"state dict mismatch: {missing[:5]} / {unexpected[:5]}"
    print("[ref] transformers Qwen3VLVisionModel loaded with the same tensors")

    ok = True
    for name, rgb in synthetic_images():
        pixels = mine.preprocess(rgb)
        w_px, h_px = target_size(rgb.shape[1], rgb.shape[0])
        rows, cols = h_px // PATCH, w_px // PATCH
        expect_tokens = rows * cols // MERGE**2
        out = mine.encode(pixels)
        assert tuple(out.shape) == (expect_tokens, 2560), f"{name}: shape {tuple(out.shape)}"

        with torch.inference_mode():
            r = ref(reference_pixels(pixels.cpu()).to(device), torch.tensor([[1, rows, cols]], device=device))
            ref_out = r.pooler_output
        diff = (out.float() - ref_out.float()).abs()
        cos = torch.nn.functional.cosine_similarity(out.float(), ref_out.float(), dim=-1)
        passed = diff.max().item() < 1e-2 and cos.min().item() > 0.999
        ok &= passed
        print(f"[{name}] grid {rows}x{cols} -> {expect_tokens} tokens | "
              f"max abs diff {diff.max().item():.5f} | mean abs {diff.mean().item():.6f} | "
              f"cosine min {cos.min().item():.6f} mean {cos.mean().item():.6f} | {'PASS' if passed else 'FAIL'}")

    print("PARITY:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
