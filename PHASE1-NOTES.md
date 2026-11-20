# Phase 1 spike — Flash-Next (Qwen3.8) vision tower for Hibana

**Verdict: PARITY PROVEN — bit-identical to the transformers reference (max abs diff 0.0, cosine 1.000000 on all
test images).** `tools/vision_parity.py`, run on GB10 (hibana fork venv, CUDA, coexisting with the tensorfold
production container at ~7 GiB free GPU).

## What was built

- `hibana/model/flashnext/vision.py` (~250 LOC): `QwenVision` implementing the engine's `VisionEncoder` protocol
  (`image_token 248056`, `image_start 248053`, `image_end 248054`, `identity`, `workspace_bytes`, `tokens(w, h)`,
  `preprocess(rgb)`, `encode(pixels)`), plus Qwen smart-resize preprocessing (factor 32, 65,536–16,777,216 px,
  bicubic antialiased, mean/std 0.5). Single-image; video deferred (the tower is frame-wise, so video = repeat).
- `tools/vision_parity.py` (~130 LOC): loads only the 333 `model.visual.*` tensors (all in
  `model-bf16-00001.safetensors`, BF16, quantizer excludes the tower), builds the spike tower + the transformers
  reference on the same tensors, runs 5 synthetic images (rectangles/gradients/text-like stripes; nothing
  downloaded), compares merged embeddings [N, 2560]. Also verifies the patch-grid token math per image.
- Branch `flashnext-vision` in the fork checkout.

## Architecture found (Qwen4Exp vision = Qwen3-VL vision tower)

- 27 blocks, hidden 1152, 16 heads, head_dim 72, MLP 4304 (gelu_tanh), LayerNorm eps 1e-6, qkv+proj with bias.
- **No window attention** — full bidirectional attention per frame (cu_seqlens per frame; nothing like
  Qwen2.5-VL's windows here).
- Patch embed: Conv3d(3→1152, kernel (2,16,16), stride same); an image's frame is fed twice along the temporal
  axis (temporal_patch_size 2).
- Learned position table `pos_embed.weight` [2304, 1152] = 48×48 grid, resampled per patch bilinear
  **align_corners=True** (4 taps summed in FP32).
- Axial 2-D RoPE, theta 10,000 (hardcoded default — `Qwen3VLVisionRotaryEmbedding(head_dim// 2)`; the checkpoint
  sets no rope_parameters): 18 freqs per axis; per token angles = [h·f(18), w·f(18)] concatenated with itself over
  the 72-dim head; rotate_half over the whole head; FP32 rotary products; **frequencies and angles in BF16**
  (the module's `to(bf16)` casts the inv_freq buffer — computing angles in FP32 shifts cos by up to ~0.13 at
  large grids).
- **Patch sequence order is spatial-merge-block order** — (block_row, block_col, row_in_block, col_in_block) —
  confirmed in three independent places: the processor's patchify (transpose (1,4,2,5,0,3,6)), HF's
  `get_vision_position_ids`, and vLLM's triton pos-embed kernel ("the spatial-merge reorder is baked into the
  index math"). The merger is then a plain `view(-1, 4608)`.
- Merger: LayerNorm(1152) per patch → reshape [N/4, 4608] → fc1 4608→4608 → **erf** GELU → fc2 → 2560.
  `deepstack_visual_indexes: []` — no deepstack mergers needed.
- out_hidden_size 2560 = the flashnext LM hidden (matches `qwen38_config`).

## Parity numbers (GB10, bf16, CUDA)

| image | grid | tokens | max abs diff | cosine min | cosine mean |
|---|---|---|---|---|---|
| rects-640x480 | 30x40 | 300 | 0.00000 | 1.000000 | 1.000000 |
| stripes-512x512 | 32x32 | 256 | 0.00000 | 1.000000 | 1.000000 |
| odd-817x353 | 22x52 | 286 | 0.00000 | 1.000000 | 1.000000 |
| tiny-90x70 (min-px clamp) | 16x20 | 80 | 0.00000 | 1.000000 | 1.000000 |
| wide-1520x400 | 24x96 | 576 | 0.00000 | 1.000000 | 1.000000 |

`smart_resize` also cross-checked against the installed transformers implementation: 0 mismatches on 2,000 random
sizes (incl. clamp paths). Vision tensors: 861 MiB as loaded.

### What it took to reach bit-identity (the interesting bits)

1. Block-major patch order everywhere (raster order is equivalent in exact math but diverges in bf16 rounding).
2. Same Conv3d op for the patch embed (folding the temporal kernel into a Linear works in fp32 but rounds
   differently in bf16).
3. inv_freq kept in **BF16** (the reference casts its buffer), angles computed as int64 × BF16.
4. The pos-embed interpolation weights need a **true tensor division** — `tensor / python_int` multiplies by the
   FP32 reciprocal and rounds twice; that single ulp in a tap weight was amplified by 27 attention layers into
   max-abs ~0.04 / cosine ~0.97 output drift.

## Honest caveats

- The reference is transformers **5.15.1's `qwen3_vl` vision model** (the hibana venv). Upstream main has
  `qwen4_exp` whose vision code is a line-for-line copy of it (verified against the main-branch source), and
  vLLM's flash-next model config subclasses `Qwen3VLVisionConfig` — so this is the right lineage, but the gated
  `Qwen/Qwen4-Exp` repo config could not be fetched (401) to confirm vision rope theta ≠ 10,000. Both runnable
  references (HF 5.15.1, vLLM image on GB10) use 10,000.
- The **resize kernel** is not part of parity: the spike resizes with torch bicubic-antialias; the fast processor
  uses PIL bicubic. Same target size (smart_resize matches exactly), slightly different pixel values at edges.
  Phase 2 should either keep torch resize (like `nemotron/vision.py` does) or A/B the choice against served
  quality. No effect on the tower-math parity proven here (both paths were fed identical pixels).
- Single image per encode, CPU-side preprocess, no batching — matches the nemotron pattern and the engine's
  one-image-at-a-time media path.
- `workspace_bytes` (800 MiB) is an engineering estimate (worst case 16,384-patch image: qkv temporaries + rope
  FP32 copies + allocator slack), not measured — Phase 2 should measure like nemotron did (470–600 MiB measured).

## Phase 2 needs (MRoPE + engine integration)

1. **MRoPE**: flashnext text rope is sections [11,11,10]-interleaved over 32 freq slots (Qwen3.5 3-D rule; text
   collapses to 1-D). `hibana/model/rope.py` currently implements plain partial rotary — needs an mrope variant
   fed per-token (t, h, w) positions; vision tokens take the image's grid coords (per merged token: block center
   convention — check vLLM's `get_mrope_input_positions` in the copied vLLM reference sources).
2. Engine wiring: `VisionEncoder` protocol attrs are all present; media path replaces `image_token` rows with
   `encode(preprocess(rgb))` between start/end tokens (existing generic media machinery).
3. Model loading: the visual tensors all sit in one shard (`model-bf16-00001.safetensors`) — the spike's
   `SnapshotWeights`-based lazy get works; check where the flashnext model loader stashes the tower (like
   nemotron's `model.vision` attribute).
4. Decide the resize kernel (torch vs PIL) and whether `identity` string needs changing with it.
5. Measure `workspace_bytes`, add vision tensors to the load's headroom accounting (~0.9 GiB).
6. Optionally reuse the vLLM flash-next + qwen3_vl modeling sources (extracted from the
   docker image) for the MRoPE position layout.

## LOC

- `vision.py` 206 lines (incl. module docstring); `vision_parity.py` 138; both committed on branch
  `flashnext-vision` (the fork checkout).
  Nothing pushed anywhere.
