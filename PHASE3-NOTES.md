# Phase 3 — Flash-Next VIDEO input in Hibana: frame-pair pipeline + e2e VERIFIED

**Verdict: video input works end-to-end on GB10 with the real checkpoint, bit-identical to the transformers
reference.** A synthetic 16-frame clip (a red ball drifting left→right, 448×448 mp4, nothing downloaded) answered
"which direction is the ball moving?" with **`right`** (a frame-reversed copy answered **`left`** — not chance) and
"how many frames show the ball?" with **`8`** (the model counts the 8 timestamped temporal steps the Qwen3-VL prompt
gives it — 16 source frames, 8 frame pairs — verbatim answer below). Text and image regressions pass unchanged,
production TensorFold stopped and restored cleanly (200 OK, 262144 ctx).

## The semantics found (the load-bearing discovery)

**Qwen3-VL's video prompt is not one big pad span.** The processor (`replace_video_token`, transformers 5.15.1) and
the deployed vLLM flash-next build (`get_video_repl`, `vllm_qwen3_vl.py` (in the vLLM reference sources)) both write, per
*temporal step* (frame pair, `temporal_patch_size` 2): a text timestamp `<12.4 seconds>`, then
`<|vision_start|>` + that pair's `rows*cols` `<|video_pad|>` tokens + `<|vision_end|>`. And `get_rope_index` splits a
video's grid into per-frame (t=1) grids — **each pair's span gets exactly image positions**, text (the timestamps)
runs between them, `current_pos += max(H, W) // 2` per frame. Consequences Phase 3 exploits:

- **`mrope.positions_3d` needed zero changes**: a video is just consecutive `Image` spans (one per pair, grid
  (rows, cols)) with timestamp text between — the Phase 2 loop already stacks exactly that. Verified
  `np.array_equal` against `get_rope_index` (`tools/video_parity.py`), deltas matching.
- **The tower needed only a pair encode**: per-frame position table and axial rotary repeat over t
  (`get_vision_position_ids` with `include_temporal=False`), attention is segmented per frame pair
  (`get_vision_cu_seqlens`), the Conv3d patch kernel is the only cross-frame op. `encode_pair(f0, f1)` = the image
  path with the two temporal slices fed different frames; `encode_pair(f, f) == encode(f)` **bitwise** (asserted).
- Timestamps: uniform 2-fps sampling (`sample_frames` default), indices `np.linspace(0, total-1, n).round()`,
  padded to even by repeating the last; a pair's timestamp = mean of its two frames' `idx / fps`.

## What was built (branch `flashnext-vision`, on top of Phase 2)

- `model/flashnext/vision.py`: `video_size` (clip-level `smart_resize` against the checkpoint video config's
  4,096 / 25,165,824-**pixel-total** budget, factor 32, every frame counted — cross-checked against the transformers
  video `smart_resize` on 1,925 randomized (frames, h, w) triples, 0 mismatches), `video_grid`/`video_tokens`,
  `sample_indices`, `timestamps`, `preprocess_pair`/`preprocess_pair_grid`, `encode_pair`, and the caps:
  `HIBANA_VIDEO_MAX_FRAMES` (default **32**, ≥ the 4-frame floor), a per-frame 4,194,304-px cap (see deviations).
- `engine/media.py`: video pairs as self-contained bytes (`pair_bytes`/`decode_pair`, a magic + raw RGB), `Image.pair`
  routing them to `preprocess_pair_grid`/`encode_pair`; text/image paths byte-identical to before.
- `api/media.py`: OpenAI `video_url` / `input_video` / `video` parts (base64 `data:` URL, ≤ 64 MiB, ≤ 4 videos a
  request — upstream TF's prior art), ffmpeg/ffprobe decode (`select` on exact frame numbers, no new Python deps),
  sampling/resize/timestamps per the processor, and `expand` growing the template's single `<|video_pad|>` into the
  per-pair `<t seconds>`+span structure with distinct placeholder ids per pair; `<|video_pad|>` escaped in user text;
  parts normalized to `{"type": "video"}` for the checkpoint template's own video branch. `Prepared.tokens` counts
  the pairs' placeholders plus a 12-token bound a timestamp.
- `api/server.py`: video parts admitted when the encoder has `video_token`; `tok` threaded into `expand`.
- `tools/video_parity.py` + `tests/unit/test_video.py` (4 tests); docs (`api.md` video section, `flashnext.md`).
- **`ENCODE_BYTES` re-measured and fixed: 800 MiB estimate → 2 GiB** (`torch.cuda.max_memory_allocated` around the
  encode: 1,546 MiB image / 1,618 MiB worst-case pair at 16,384 patches — the Phase 2 estimate was low for *images*
  too; the executor now reserves honestly).

## Parity verdict (tools/video_parity.py, GB10, real checkpoint weights)

| check | result |
|---|---|
| clip smart resize vs transformers (1,925 random sizes) | **0 mismatches** (75 skipped where our per-frame cap engages) |
| 6-frame synthetic clip, per-pair `encode_pair` concat vs `Qwen3VLVisionModel` one `[[t,h,w]]` clip | **max abs diff 0.00000, cosine 1.000000** (bit-identical) |
| `encode_pair(f, f)` vs `encode(f)` | `torch.equal` **True** |
| `positions_3d` on the timestamped video prompt vs `get_rope_index` | **exactly equal** (delta −54 both) |

## E2E (GB10, `hibana serve flashnext --detach`, :30004, MTP k=8; TF stopped per procedure)

- **Text**: "capital of Japan" → `Tokyo` (0.21–0.31 s) — unchanged.
- **Image regression** (512², 6 red circles): `6` (0.39–0.51 s) — Phase 2 behavior intact.
- **Video** (448², 16 frames @ 2 fps → 8 pairs × 196 tokens = 1,568 video tokens, prompt 1,664):
  - "Which direction is the ball moving: left or right?" → **`right`** (4.18 s cold incl. 8 pair encodes; 0.85–2.1 s
    warm). Frame-reversed copy → **`left`** (0.85 s).
  - "How many frames of this video show the red ball?" → **`8`** (1.59 s) — the model reads the 8 timestamped steps;
    a source-frame answer would be 16, a pair-count answer 8. Prompt includes timestamps, so 8 is the model's honest
    frame count under the Qwen3-VL representation.
  - "…roughly how far across the frame does it travel?" → *"The ball is moving from left to right, traveling about
    two-thirds of the width of the frame."* (2.10 s) — correct (centre travels 368/448 ≈ 82% of width; edge-to-edge).
  - Over-context (30K-token text + video): clean **400** "prompt plus output budget exceeds context limit".
- **Memory**: load MemAvailable 116 GiB → steady **11.2–12.5 GiB** during video testing (same envelope as Phase 2's
  text-only serve; vision adds ~0.9 GiB, video pairs add only CPU-side bytes ~4–9 MiB a pair); swap untouched
  (SwapFree 213/226 GiB). Guard trips at 3 GiB.
- **Decode sanity**: 32.0–32.9 tok/s (no-think EN story, streaming) — **A/B'd against the Phase 2 commit on the same
  prompt: 32.3–32.7 tok/s**, i.e. no Phase 3 regression. (Phase 2's 80.7 figure was a different, more
  draft-friendly prompt; acceptance is content-dependent.)
- Suites: `tests/unit` **1203 passed** (4 new), GPU qsa/qsa_exact/flashnext_fused **235 passed**.

## Caps and limits chosen

- `HIBANA_VIDEO_MAX_FRAMES=32` default (env override; HF processor floor 4, max 768 upstream — we do not take 768).
- ≤ 4 videos / request, ≤ 64 MiB encoded, ≤ 200 s ffmpeg decode timeout, per-frame ≤ 4,194,304 px
  (16,384 patches — the image worst case, so `ENCODE_BYTES` bounds every encode), total-budget 25,165,824 px as the
  checkpoint config sets. A 32-frame 640×480 clip stays native (4,800 tokens); a 1080p clip at 32 frames shrinks.

## Remaining deviations (updated Phase 2 list)

1. QSA still skipped for media prompts (dense exact attention) — now covering video spans too.
2. Media + DFlash2 drafter still rejected; MTP (the default) fine.
3. Resize kernel still torch-bicubic (both image and per-frame video); ffmpeg decodes instead of decord/torchvision
   (pixel-level decoder variance, same class of deviation).
4. **New**: the per-frame 4,194,304-px cap is tighter than HF for 2-frame clips (HF would allow up to 6.29 Mpx a
   frame there); keeps every encode inside the measured workspace. Also `ffprobe -count_frames` decodes the whole
   container to count frames exactly (correct over fast).
5. Timestamps are computed from the *container's* fps and sampled indices exactly as the processor does, but no
   `fps`/`num_frames` request overrides are exposed yet.

## Upstream-PR gap list (delta)

- All Phase 2 items stand; new separable pieces: the video preprocessor (`video_size`/sampling/timestamps +
  `encode_pair`), the per-pair media plumbing (`Image.pair`), and the `ENCODE_BYTES` measurement fix (worth its own
  line in the nemotron-style load accounting). The timestamp-text discovery (video = timestamped per-pair spans, not
  one pad block) is the thing any port to another engine gets wrong first.

Nothing pushed anywhere. TF (:8000) restored and verified 200 (`Qwen3.8-Flash-Next`, 262144 ctx) after the tests.
