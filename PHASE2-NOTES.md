# Phase 2 — Flash-Next vision in Hibana: interleaved MRoPE + engine wiring — e2e VERIFIED

**Verdict: image input works end-to-end on GB10 with the real checkpoint.** Counting ground truth
correct (6/6 red circles), scene description exactly right (5 red circles / 2 blue squares /
1 green triangle / both text strings read), text decode unchanged (80.7 tok/s no-think), production
TensorFold stopped and restored cleanly. One **latent upstream hibana bug found and fixed** on the
way (`tokenizer.without_specials` shifted added-token ids). Two honest deviations from the deployed
vLLM semantics, both documented below.

## What was built (branch `flashnext-vision`, on top of Phase 1)

- `model/flashnext/mrope.py`: `[3, L]` position table. **Verified equal to vLLM's
  `Qwen3VLForConditionalGeneration._get_mrope_input_positions` over 50 randomized prompts** (text
  `(p,p,p)` continuing across images from `base + max(H, W)`; image rows `(base, base+i, base+j)`
  by merged-grid raster order; `pos_delta = H·W − max(H,W)` behind the token count).
- `kernels/qk_norm_rope.py`: multimodal Triton kernel — rotary pair `p` reads the position of axis
  `p % 3` (t/h/w), the interleaved layout of the deployed flash-next build's own kernels
  (`qsa_pre_indexer.py` (in the vLLM reference sources): `h_mask = (pairs % 3 == 1) & (pairs <=
  3·MROPE_H)`, sections [11, 11, 10] over 32 pairs). **Text rows through it are bitwise the
  original kernel** (`tools/mrope_check.py`: `torch.equal` on q and k); image rows match a torch
  reference of the layout (k bitwise, q within one BF16 ulp at large angles).
- Engine: `ForwardContext.mrope` ([3, T] for image forwards, None keeps the plain 1-D path),
  `Request.row_pos/cur_pos` (the t component; `pos_delta` after the prompt), and every rope-reading
  call site now uses t-positions — prefill chunks (single and batched), decode/verify positions,
  QSA seq starts, MTP draft-KV writes (`ingest_prefill`, `_accepted`, feedback path).
- Vision wiring: `FlashNext.vision = QwenVision(...)` under `HIBANA_VISION` (flashnext preset now
  sets `=1`), `VISION_BYTES` in the load's headroom, media rows `index_copy_`d into the trunk's
  embeddings (nemotron pattern), `mrope_positions` exposed to the executor.
- Media layer: `Image.grid` (merged patch grid) → `vision.grid()` → MRoPE table; token-convention
  template support (Qwen writes `<|vision_start|><|image_pad|><|vision_end|>` itself): the token
  strings are escaped in request text, the encoder's markers kept, `expand` replaces the pad token
  in place.
- Cache hygiene: image requests are `cacheable = False` (their positions run behind their token
  count), and `_cache_identity` gains the encoder's `identity`.

## The upstream bug found (worth its own PR)

`SnapshotTokenizer.without_specials` rebuilt the tokenizer spec **without** the disabled added
tokens; `tokenizers` re-registers added tokens by position, so every special after them shifted id
by the number removed. On flashnext (vision tokens mid-table) every chat render broke: `<think>`
(248068) encoded as `<|file_sep|>` (248065) — chats answered one empty token with total confidence.
Bisection: identical 22-token prompt worked via `/v1/completions`, died via `/v1/chat/completions`;
the submit log showed the id shift. Fix: keep the entries, rewrite their content to an unmatchable
private-use string. (Nemotron never hit it — its vision tokens sit at the tail.)

Second find: the flashnext template never places a `<image>` marker, so the original marker
machinery rejected every image request ("the template placed 0 of the request's 1 images") — hence
the token-convention path above.

## E2E results (GB10, `uv run hibana serve flashnext --detach`, :30004, MTP drafter)

- **Load**: weights 40 s, 53.9 GiB allocated (+ ~0.86 GiB vision tower), serve-to-ready ~100 s
  (W3 check + graph capture). Steady **MemAvailable ≈ 10 Gi** (guard trips at 3 Gi) — same
  coexistence envelope as the text-only taste test; vision adds ~0.9 GiB.
- **Text sanity**: "capital of Japan" → `The capital of Japan is Tokyo.` (8 tok) — identical
  string and count as the pre-change baseline build (A/B'd on commit 90253bf during bisection).
- **Scene (640×480, 5 red circles / 2 blue squares / 1 green triangle, "HELLO 7", "vision test")**,
  thinking off, streamed — verbatim:
  > "This image contains several geometric shapes and text elements on a light gray background.
  > **Text:** … 'HELLO7' … 'vision test' … **Red Circles:** There are 5 red circles. Three are
  > arranged in a horizontal row across the top. Two are arranged in a horizontal row below …
  > **Blue Squares:** There are 2 blue squares."
  — every count, both colors, both strings, and the spatial layout correct. TTFT 0.24–0.27 s.
- **Counting ground truth (512×512: 6 red circles + 4 blue distractor circles + 2 yellow squares)**:
  `"How many red circles…?"` → **`6`** (0.3 s, 2 tokens). Repeated across restarts.
- **Text decode benchmark** (no-think, streaming): **80.7 tok/s** (209 tok / 2.71 s, TTFT 0.07–0.12 s
  warm); zh prose 58.9 tok/s — no vision overhead regression (tower is resident but idle).
- Streaming image answers work (`6` at TTFT 0.87 s cold).

## Honest deviations from the deployed vLLM semantics

1. **Image requests never run QSA selection** (dense exact attention for the request's life; keys
   are still maintained). Reason: the reference's indexer groups and rotates compressed keys by
   *logical t-positions*, which inside an image span are non-injective (a whole grid row shares
   `t`); replicating the deployed plugin's grouping exactly was out of scope for a taste test.
   Dense is exact attention, so quality can only differ from trained-with-sparsity behavior on
   image prompts past the 2048 budget. Decode/verify batches containing an image request are
   forced to the dense graph variant.
2. **Media + the DFlash2 drafter is rejected** (`images serve with the MTP drafter`); the flashnext
   default is MTP. The parallel drafter's ring buffers index by raw positions.
3. Resize kernel still torch-bicubic (Phase 1 caveat); `workspace_bytes` still the 800 MiB
   engineering estimate. Neither blocked correct answers.

## Static checks

- `tools/mrope_check.py`: text rows bitwise identical old/new kernel; mrope rows match the torch
  interleaved reference; `positions_3d` == vLLM algorithm (50 random prompts).
- `pytest tests/unit`: **1199 passed**, 18 skipped. GPU: qsa, qsa_exact, flashnext_fused
  **235 passed**; verify/packed-verify/draft-rope/gdn-plan/hyper-prefill 65 passed (packed-verify's
  7 errors are environmental: no HF cache env for weight-loading tests).
- Text-path non-regression: every changed code path is a no-op when `mrope is None` (all text
  requests), confirmed by the suites above and the served A/B.

## For a clean upstream PR

- Decide the resize kernel (torch vs PIL bicubic) and A/B served quality; measure
  `workspace_bytes` like nemotron did (its 470–600 MiB measured vs our 800 MiB estimate).
- Revisit QSA-under-images against the reference plugin (the deviation above) — or land vision
  with it documented, as here.
- The `without_specials` id-shift fix and the token-convention template support are separable,
  generally useful PRs on their own.
- Docs (`docs/api.md`, `docs/operations.md` still say Flash-Next has no image input), and the
  `Prepared`/`Image` contract additions want a line in `engine/contracts.py`'s `VisionEncoder`.
- Video: deferred (tower is frame-wise; `mrope.positions_3d` already handles multi-frame lumping
  in the reference — needs `grid` per frame).

Nothing pushed anywhere. Production TF (:8000) stopped per procedure (container removed by its own
`stop.sh`) and **restored + verified 200** (`Qwen3.8-Flash-Next`, 262144 ctx) after the tests.
