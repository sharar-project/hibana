# SPDX-License-Identifier: Apache-2.0
"""The Nemotron 3.5 checkpoints' image path, shared by Super and Bolt: C-RADIO (a ViT-H/16, 32 blocks of width 1280 behind
3 class and 7 register tokens), pixel shuffle and the `mlp1` projector into the language model's embedding space.

`preprocess` sizes each image to one tile of 16-pixel patches between 1,024 and 13,312 patches, keeping its aspect ratio
and even patch counts, then normalizes it (`image_processing_nemotron_h_omni.py` at the pinned revisions, which both
checkpoints ship identically). `RadioVision.encode` runs the tower on the tile, drops the 10 prefix tokens, applies the
tower's final LayerNorm when the checkpoint has one (Super does, Bolt does not), merges each 2 x 2 patch block into one
5120-wide token (pixel shuffle, `ps_version` v2) and projects it with RMSNorm -> 20480 -> ReLU^2 -> hidden. An image of
(h x w) patches becomes h * w / 4 embeddings, which replace the `<image>` placeholders the API inserts
(`<img>` + that many `<image>` + `</img>`, as the checkpoint's processor does). The Nemotron attention layers have no
positional encoding, so an image needs nothing beyond its embeddings. Weights are the checkpoints' BF16 tensors
(about 1.7 GiB); the arithmetic follows the reference modules in BF16 with FP32 norms.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

TOWER = "vision_model.radio_model"
PATCH = 16
MIN_PATCHES, MAX_PATCHES = 1024, 13_312  # preprocessor_config.json: each image's patch budget
MERGE = 2  # pixel shuffle's downsample ratio 0.5: 2 x 2 patches a token
NORM_MEAN = (0.48145466, 0.4578275, 0.40821073)
NORM_STD = (0.26862954, 0.26130258, 0.27577711)
WIDTH, HEADS, LAYERS = 1280, 16, 32
PREFIX_TOKENS = 10  # 3 class + 7 register tokens, prepended to the patches and dropped after the last block
POS_GRID = 128  # the position table covers 2048 / 16 patches a side
LN_EPS, RMS_EPS = 1e-6, 1e-5
VISION_BYTES = 1_850_000_000  # the tower and projector (BF16) and the FP32 position table as loaded: the load's headroom
# One encode's device memory beyond its output, which runs on the executor's side stream: at most 470 MiB measured
# (a full 13,312-patch tile; 302 MiB at 2 x 4,560 patches), with room for the allocator's blocks
ENCODE_BYTES = 600 * 2**20

# tokenizer.json of both checkpoints: <image> (config.json img_context_token_id), <img>, </img>
IMAGE_TOKEN, IMAGE_START, IMAGE_END = 18, 19, 20
# Part of every image's digest (api/media.py), so a cached or stored prefix is never resumed for pixels prepared
# another way; change it with `preprocess`.
IDENTITY = b"nemotron c-radio: one 16-pixel tile of 1024-13312 patches, bicubic antialiased, CLIP normalized\n"


def target_patches(width: int, height: int, budget: int = MAX_PATCHES) -> tuple[int, int]:
    """(w, h) in patches for a `width` x `height` image: scaled down to `budget` patches, up to MIN_PATCHES, each side
    even (the reference's `_compute_target_patches`, including its rounding). ValueError for an image so narrow that its
    short side scales to no patch (the reference divides by zero there)."""
    patch_h = round(height / PATCH + 0.5)
    patch_w = round(width / PATCH + 0.5)
    factor = min(math.sqrt(budget / (patch_h * patch_w)), 1.0)
    h, w = math.floor(factor * patch_h), math.floor(factor * patch_w)
    if not h or not w:
        raise ValueError(f"a {width} x {height} image is too narrow: in at most {MAX_PATCHES:,} patches of {PATCH} pixels "
                         "its short side gets none")
    if budget > MIN_PATCHES and h * w < MIN_PATCHES:
        up = math.sqrt(MIN_PATCHES / (h * w))
        h, w = math.ceil(up * h), math.ceil(up * w)
    if h % MERGE:
        h = h + MERGE - h % MERGE if (h + MERGE - h % MERGE) * w <= budget else max(MERGE, h - h % MERGE)
    if w % MERGE:
        w = w + MERGE - w % MERGE if h * (w + MERGE - w % MERGE) <= budget else max(MERGE, w - w % MERGE)
    return w, h


def image_tokens(width: int, height: int) -> int:
    """The embeddings `encode` makes of a `width` x `height` image (`target_patches`; ValueError as there)."""
    w, h = target_patches(width, height)
    return w * h // MERGE**2


def preprocess(rgb: np.ndarray) -> torch.Tensor:
    """An RGB image [H, W, 3] uint8 as the tower's input: resized (bicubic, antialiased) to its tile and normalized,
    [3, h, w] BF16 on the CPU."""
    w, h = target_patches(rgb.shape[1], rgb.shape[0])
    t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    if t.shape[-2:] != (h * PATCH, w * PATCH):
        t = F.interpolate(t, size=(h * PATCH, w * PATCH), mode="bicubic", align_corners=False, antialias=True)
    mean, std = torch.tensor(NORM_MEAN).view(1, 3, 1, 1), torch.tensor(NORM_STD).view(1, 3, 1, 1)
    return ((t / 255.0 - mean) / std)[0].to(torch.bfloat16)


def pixel_shuffle(x: torch.Tensor) -> torch.Tensor:
    """[rows, cols, C] -> [rows / 2, cols / 2, 4C], as the reference's `pixel_shuffle` (scale 0.5, v2) orders it."""
    rows, cols, c = x.shape
    x = x.reshape(rows, cols // MERGE, c * MERGE).transpose(0, 1)
    return x.reshape(cols // MERGE, rows // MERGE, c * MERGE**2).transpose(0, 1)


class RadioVision:
    """The tower and projector on the device (engine/contracts.VisionEncoder); `encode` turns one preprocessed tile into
    embeddings."""

    image_token, image_start, image_end, identity = IMAGE_TOKEN, IMAGE_START, IMAGE_END, IDENTITY
    workspace_bytes = ENCODE_BYTES
    preprocess = staticmethod(preprocess)
    tokens = staticmethod(image_tokens)

    def __init__(self, w, device: torch.device):
        def get(name: str) -> torch.Tensor:
            return w.get(name).to(device)

        m = f"{TOWER}.model"
        self.device = device
        self.patch = get(f"{m}.patch_generator.embedder.weight")  # [1280, 3 * 16 * 16]
        pos = get(f"{m}.patch_generator.pos_embed")  # [1, 128 * 128, 1280]
        self.pos = pos.reshape(POS_GRID, POS_GRID, WIDTH).permute(2, 0, 1).unsqueeze(0).float()
        self.prefix = get(f"{m}.patch_generator.cls_token.token")  # [10, 1280]
        names = ("norm1.weight", "norm1.bias", "attn.qkv.weight", "attn.qkv.bias", "attn.proj.weight", "attn.proj.bias",
                 "norm2.weight", "norm2.bias", "mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias")
        self.blocks = [[get(f"{m}.blocks.{i}.{n}") for n in names] for i in range(LAYERS)]
        final = "vision_projector.vision_final_layernorm"  # Super's (its config infers it from the MTP head); Bolt has none
        self.final_norm = (get(f"{final}.weight"), get(f"{final}.bias")) if f"{final}.weight" in w else None
        self.rms, self.fc1, self.fc2 = get("mlp1.0.weight").float(), get("mlp1.1.weight"), get("mlp1.3.weight")
        if self.patch.shape != (WIDTH, 3 * PATCH * PATCH) or self.prefix.shape != (PREFIX_TOKENS, WIDTH) \
                or self.fc1.shape[1] != WIDTH * MERGE**2:
            raise ValueError("unexpected vision tower shapes")
        self.hidden = self.fc2.shape[0]

    @property
    def bytes(self) -> int:
        tensors = [self.patch, self.pos, self.prefix, self.rms, self.fc1, self.fc2, *(t for b in self.blocks for t in b),
                   *(self.final_norm or ())]
        return sum(t.numel() * t.element_size() for t in tensors)

    def _positions(self, rows: int, cols: int) -> torch.Tensor:
        """The position table resized to a square of the larger side and cropped to the tile (the reference's CPE). Only
        the crop is computed: the bilinear kernel at the square's scale writes the same values (bitwise on CUDA) and
        rows * cols of them, where the square of a 2 x 4564-patch tile would be 106 GB."""
        scale = max(rows, cols) / POS_GRID
        pos = torch.ops.aten.upsample_bilinear2d(self.pos, [rows, cols], False, scale, scale).to(torch.bfloat16)
        return pos[0].flatten(1).t()

    @torch.inference_mode()
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """A tile [3, H, W] (`preprocess`) -> [H * W / 1024, hidden] BF16 embeddings."""
        return self.project(self.tower(pixels), pixels.shape[1] // PATCH, pixels.shape[2] // PATCH)

    @torch.inference_mode()
    def tower(self, pixels: torch.Tensor) -> torch.Tensor:
        """The tile's patch features after the last block, [rows * cols, 1280] in row-major patch order."""
        x = pixels.to(self.device, torch.bfloat16)
        rows, cols = x.shape[1] // PATCH, x.shape[2] // PATCH
        patches = x.reshape(3, rows, PATCH, cols, PATCH).permute(1, 3, 0, 2, 4).reshape(rows * cols, -1)
        h = torch.cat([self.prefix, F.linear(patches, self.patch) + self._positions(rows, cols)])
        t, d = h.shape[0], WIDTH // HEADS
        for n1w, n1b, qkv_w, qkv_b, proj_w, proj_b, n2w, n2b, fc1_w, fc1_b, fc2_w, fc2_b in self.blocks:
            qkv = F.linear(F.layer_norm(h, (WIDTH,), n1w, n1b, LN_EPS), qkv_w, qkv_b).view(t, 3, HEADS, d).permute(1, 2, 0, 3)
            a = F.scaled_dot_product_attention(qkv[0:1], qkv[1:2], qkv[2:3])  # [1, heads, t, d]
            h = h + F.linear(a[0].transpose(0, 1).reshape(t, WIDTH), proj_w, proj_b)
            h = h + F.linear(F.gelu(F.linear(F.layer_norm(h, (WIDTH,), n2w, n2b, LN_EPS), fc1_w, fc1_b)), fc2_w, fc2_b)
        return h[PREFIX_TOKENS:]

    @torch.inference_mode()
    def project(self, features: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        """Tower features of a rows x cols grid -> the language model's embeddings, one per 2 x 2 patches."""
        h = features
        if self.final_norm is not None:
            h = F.layer_norm(h, (WIDTH,), *self.final_norm, LN_EPS)
        h = pixel_shuffle(h.view(rows, cols, WIDTH)).reshape(-1, WIDTH * MERGE**2)
        f = h.float()
        h = (self.rms * (f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + RMS_EPS))).to(torch.bfloat16)
        return F.linear(torch.relu(F.linear(h, self.fc1)).square(), self.fc2)
