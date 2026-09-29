# Qwen3.8-Flash-Next in Sharar

The second serving target (`--target flashnext`, `sharar/model/flashnext/`).
Checkpoint: `RadixArk/Qwen3.8-Flash-Next-NVFP4` (ModelOpt NVFP4 W4A4 routed
experts, everything else BF16; 135 GB, 206 shards). Quality in the SGLang
reference (v0.5.20, GB10): GSM8K-300 95.3%, HumanEval 94.5% (27B: 95.0 / 90.9).

## Architecture as served

48 hyper-connected decoder layers (4 residual streams of 2,560, rank-320 mixers):
36 Gated DeltaNet layers (16 K / 48 V heads of 128, the 27B's geometry, sigmoid
output gate) and 12 attention layers (24 q / 2 kv heads of 256, quarter rotary,
gated output, sparse indexer with a 2,048-token budget); every layer routes to
10 of 512 NVFP4 experts (intermediate 640) plus a sigmoid-gated shared expert; a
prefix-lookup n-gram embedding (PLE) before layer 1 reads 16 rows of a 320M-row
FP8 table per token (bigram and trigram hashes, 8 heads each) and adds a gated,
causally convolved projection to the streams; a global mixer feeds the BF16 head.
The checkpoint also ships a one-layer MTP head (BF16 experts) and a vision
tower (not loaded).

## Package

| File | Role |
|---|---|
| `config.py` | pinned revision and geometry; `sharar/target.py` builds `FLASHNEXT` from it |
| `reference.py` | plain-torch prefill, exact to the SGLang server at the noise floor (parity tool) |
| `hyper.py` | hyper-connection mix/combine and the grouped Gemma norm (shared with the reference) |
| `linear.py` | `TrunkLinear`: FP8 at load with a dynamic activation scale (`SHARAR_FLASHNEXT_TRUNK=bf16` for parity) |
| `gdn.py`, `attention.py` | the 27B modules with these projections and the sigmoid gate; the attention layers run the indexer |
| `indexer.py` | sparse-attention indexer: index keys in page aux rows, group compression, top-512 selection |
| `moe.py` | FlashInfer `cutlass_fused_moe` on the checkpoint's packing (fc1 rows are `[up; gate]`) |
| `ple.py` | table as numpy memory maps of the 128 shards; hash, gather, gate, convolution; window state in `StatePool.aux` |
| `model.py` | `FlashNext`: the executor's model contract; `norm` is the hyper stream, `logits()` applies the mixer |
| `mtp.py` | `FlashNextMtpHead`: the checkpoint's MTP layer on the `MtpHead` surface for the `mtp` proposer |

Engine extensions: `Target.family`/`cache_env` and optional family fields;
`StatePool.register_aux` (per-slot state cleared, copied and parked with the
GDN states; verify windows committed like the conv windows);
`FlashNext.forward_extras()` (`extras.py`: the PLE history and table rows and
the indexer bookkeeping on `ForwardContext.extras`, built for eager forwards
and filled into graph buffers before a replay; the engine only asks it which
graph variant a batch needs); `register_pages` / `register_state`.

## Conventions that cost a day

- The GDN output gate is `sigmoid(z)` (`output_gate_type`), not `silu(z)`.
- The fused MoE kernel wants the expert matrix as `[up rows; gate rows]`.
- Activations are quantized once per layer with the loudest expert's calibrated
  scale (the reference must model it that way to match the server).
- `norm.weight` of the GDN is a plain weight (~1); every other norm is Gemma (1 + w).
- `in_proj_qkv` is `[q | k | v]` head-first; `q_proj` is per-head `[q | gate]`.
- The PLE history before a sequence is EOS; a trigram falls back to EOS when an
  EOS lies inside its window. The convolution (kernel 4, dilation 3) needs the
  nine previous gated rows per sequence.

## Numbers (2026-09-19, GB10, FP8 trunk unless noted)

| Measurement | Value |
|---|---|
| Reference vs SGLang, 5 prompts (18 to 512 tokens) | argmax 91 to 100%, confident positions 97 to 100%, next token 5/5 |
| Engine vs SGLang, BF16 trunk | argmax 88 to 100%, confident 99 to 100%, next token 5/5 |
| Engine vs SGLang, FP8 trunk | argmax 82 to 100%, confident 94 to 100%, next token 5/5 |
| Load | 115 s, 69.4 GiB (76.8 with pools and graphs) |
| Bytes per decode token | 6.33 GB (26 ms at 239 GB/s) |
| Plain decode, batch 1 / 4 | 20.7 tok/s (48.3 ms/step) / 80.3 aggregate (49.8 ms/step); SGLang 16.7 at batch 1 |
| GPU time of one batch-1 decode step (graph replay) | 38.1 ms; about 3,600 kernels per step, so ~12 ms of tiny kernels over the 26 ms byte floor and ~10 ms of graph launch and syncs per step |
| Phase 3 fused kernels + FP8 mixers (2026-09-19 19:00) | batch-1 step 48.0 -> 42.1 to 43.2 ms (23.2 to 23.8 tok/s), batch 4 49.9 -> 42.9 to 43.2 ms (93 tok/s); parity unchanged |
| + parallel PLE row reads (20:15) | batch-1 step 40.1 ms (24.9 tok/s), batch 4 43.7 ms (91.6 tok/s). The serial memory-map gather paid up to sixteen NVMe latencies per token (4.7 ms cold); 32 threads of `pread` take 1.4 ms. GPU time per step is 32.5 ms, graph launch 0.05 ms |
| MTP k=3, batch 1 (`bench-decode --spec 3`, reconstruct strategy) | JSON prompt 22.3 -> 44.4 tok/s (1.99x, 79% accepted, 3.32 tok/round), needle prompt 1.79x; per round draft 11.6 ms + verify 59 ms + accept 5 to 40 ms (the accept commit phase varies by prompt and run: 2 to 36 ms, 376 ms once; unexplained, profile next) |
| MTP k=3, batch 4 | JSON 1.97x (172 tok/s), needle 1.54x (136 tok/s); k=2 is not better at either batch size |
| `SHARAR_FLASHNEXT_TRUNK=w3` (nearest-rounded 3.25-bit decode copies over FP8, mixers stay FP8) | batch-1 step 31.0 ms (32.3 tok/s), batch 4 35.8 ms (112 tok/s); parity: long code prompts unchanged, the two short text prompts drop (top-1 71 to 89%, confident 83 to 100%) as nearest rounding did on the 27B; served accuracy gate FAILED: GSM8K 92.0% (FP8 96.0%), HumanEval 89.6% (94.5%), and no faster at concurrency 4 (the 16-row verify is where the 3-bit kernel is compute-bound). Off by default; calibrated (GPTQ) copies are the known fix |
| Served through the launcher (`config/flashnext.env`, MTP k=3, exact greedy, concurrency 4), 2026-09-19 | GSM8K-300 96.0% in 1,002 s (SGLang 95.3% in 1,889 s; 27B 95.0%), HumanEval 94.5% in 390 s (SGLang 94.5% in 714 s; 27B 90.9%) |

Serving: `SHARAR_CONFIG_FILE=config/flashnext.env ./run-sharar.sh` (image
`sharar:main-20260919`, port 30004, MTP k=3, reconstruct; the 27B and Flash-Next
do not fit together, so stop production first). Evidence and launchers:
`~/sharar-results/reviews/2026-09-18-flash-next/`
(`parity-*.json`, `components.log`, `decode-*.log`, `run_*.sh`).

## Sparse attention past 2,048 tokens (2026-09-23)

`indexer.py` ports SGLang's Qwen Sparse Attention: every token's raw index key
and each completed group's compressed key live in its KV page (`KvPool.page_aux`,
so they follow pages through the prefix cache, the cold tier and the session
store); a row past the budget scores the visible groups, keeps the top 512 plus
its pending tail and attends over those pages through SGLang's sparse kernel
(`kernels/qsa.py`, vendored). Below the budget the dense path runs unchanged.
Evidence `$SHARAR_RESULTS/reviews/2026-09-24-qsa/`:

| Check | Result |
|---|---|
| Selection vs SGLang's torch operators (CPU, sparse and dense cases) | identical |
| Engine indexer vs reference selection (GPU: chunked prefill split inside a group, decode steps, scrambled pages) | identical on every row |
| Reference vs SGLang past 2,051 tokens (3K / 6K / 10K prompts) | argmax 91.5 / 87.5 / 94.3%, confident 95.6 / 93.6 / 97.3%, next token 3/3 |
| Serving path (executor), BF16 trunk, same region | confident 89.6 / 95.1 / 95.0% |
| Serving path, FP8 trunk, same region | confident 94.2 / 83.4 / 89.7% (one prompt each; against the BF16 trunk on 8 prompts the FP8 gap is mostly rounding noise, see below) |
| Served needle retrieval, depths 0.1/0.5/0.9 at 4K, 8K, 16K, 32K | 12/12 |
| Served decode at 4K / 8K / 16K / 32K (MTP k=3, eager) | 26.5 / 29.2 / 31.2 / 28.5 tok/s exact, flat with context |
| Served GSM8K 150 at concurrency 4 | 97.3% |

Running eagerly exposed a FlashInfer re-planning race (`kernels/plan_guard.py`).
Cold prefill: 32K in 21.7 s.

## CUDA graphs with the indexer (2026-09-24)

Decode and verification replay graphs at every context length. Inside the
budget the dense graphs run with graph-safe key maintenance
(`FlashNextIndexer.maintain`: raw keys and group compression from static rows);
once any row is past it, a second set of graphs selects on the device:
`kernels/qsa_select.py` scores every visible group through the plan's own page
table in one Triton kernel (relu-sum over the four index heads), takes the top
512 of a fixed-width buffer (the context capacity / 4) and assembles the pages
as before, so no shape depends on the context. The eager path uses the same
kernel (3.6x faster than the torch scoring at 262K for one row).
`SHARAR_QSA_SPARSE_GRAPHS=0` keeps rows past the budget eager. Evidence
`$SHARAR_RESULTS/reviews/2026-09-24-qsa-graphs/` and `2026-09-24-qsa-select/`:

| Check | Result |
|---|---|
| Kernel vs the torch selection (1 to 256 rows, 3K to 70K tokens, padded capacity) | identical sets |
| Engine indexer vs reference selection (GPU) | identical except one float tie at the top-k boundary |
| Selecting graph replay vs eager selection (two rows, 40 steps) | identical |
| Served decode, short context, graphs (greedy exact / relaxed) | 50.6 / 54.7 tok/s (eager 26 to 31) |
| Served decode at 4K / 16K / 64K, selecting graphs (exact / relaxed) | 39.8 / 45.9, 36.6 / 44.6, 36.8 / 42.6 tok/s |
| Same, eager selection (`SHARAR_QSA_SPARSE_GRAPHS=0`) | 29.7 / 34.1, 27.3 / 33.2, 28.2 / 32.9 tok/s, identical spec rounds and acceptances |
| Four concurrent requests at 5K to 12K (needles; 256-token walkthroughs) | 4/4 correct both ways; 20.1 s vs 24.6 s wall |
| Served needles 4K / 16K / 64K, depths 0.1 / 0.5 / 0.9 | 9/9 |
| Served GSM8K 150 at concurrency 4 | 96.7% (dense graphs only: 96.0%) |

## FP8 trunk scales (2026-09-24, #5)

`bench/flashnext_quant.py` scores a trunk precision against the BF16 trunk on
the serving path (8 prompts of 8,000 tokens), with BF16 at another prefill
chunk size as the noise floor: argmax 85.7 / 90.6% and KL 0.36 / 0.27 nats
(dense / sparse rows) from rounding alone. The per-tensor FP8 trunk sits at
82.6 / 87.4% and 0.49 / 0.36; per-row activation scales, per-channel weight
scales, both, or keeping attention, GDN, shared experts or head and PLE in
BF16 all land within the spread between variants. The trunk stays per-tensor
FP8. `SHARAR_TRUNK_KEEP_BF16=<regex>` localizes quantization error for the
calibrated FP4 trunk. Evidence `$SHARAR_RESULTS/reviews/2026-09-24-fp8-scales/`.

## 3-bit trunk sensitivity (2026-09-28, #61)

`bench/flashnext_quant.py score --chunk 64` (the decode path: 64-row calls read the 3-bit copies), 8 prompts of
8,000 tokens, KL(BF16 || trunk) over BF16's top 32. `SHARAR_FLASHNEXT_W3_KEEP_FP8=<regex>` leaves the matching
`linears()` names without their calibrated copy. One module kind at a time in 3-bit (the rest FP8) costs no KL:
every kind lands at or below the FP8 trunk, so no kind is the sensitive one. All of them together double it.

| 3-bit set | GB saved per token | KL dense / sparse | argmax dense / sparse |
|---|---:|---|---|
| none (FP8 trunk) | 0 | 0.507 / 0.437 | 80.7 / 85.1% |
| any single kind (GDN in_qkv / in_z / out, attention qkv / o, shared gate_up / down, PLE, lm_head) | 0.02 to 0.56 | 0.48 to 0.57 / 0.35 to 0.42 | 80.2 to 83.1 / 86.3 to 87.9% |
| all (the `fast` default) | 2.12 | 0.956 / 0.777 | 73.3 / 84.3% |
| all, row gain corrected (`--unbias weight`) | 2.12 | 0.840 / 0.521 | 74.7 / 87.2% |
| all, row gain corrected, layers 40-47 FP8 | 1.83 | 0.578 / 0.368 | 79.9 / 89.1% |

The cause is a coherent bias, not one module: the MSE-optimal codebook and clipped scales shrink each 3-bit row
by about its relative error squared (weight-space gain 0.90 to 0.97, FP8 0.999). Per matrix that is optimal; through
48 residual layers every sublayer's update shrinks the same way while the noise adds incoherently.
`bench/flashnext_w3_calibrate.py --unbias weight` scales each row back through its FP32 row scale (codes, format
and bytes unchanged). The covariance-weighted variant (`--unbias output`) corrects less (1.5% against 4%) and
removed less KL (0.70 overall against 0.60): the calibration covariance under-states what decode inputs see.
Keeping depth bands in FP8 on the corrected set helps only for the last layers (0-11, 12-23, 24-35 FP8: no
change; 36-47 or 40-47: below the FP8 trunk overall; 44-47: half the gain).

Served, image `sharar:main-20260927b`, `fast` profile, batch one; only the checkpoint varies. Speed: 22 decode
prompts paired, h2h geometric mean against FP8. Accuracy: full sets, exact greedy, paired against FP8 on the same image:

| trunk | greedy | t = 0.7 | round (verify) ms | HumanEval | GSM8K |
|---|---|---|---|---|---|
| FP8 | 60.2 tok/s median | 59.5 | 60.6 (50.2) | 93.9%; again 94.5% (2 / 1) | 94.4% (#6's FP8 run 94.5%: 18 / 17) |
| 3-bit, current | x1.193 [1.173, 1.216] | x1.156 [1.135, 1.180] | 52.1 (41.3) | 94.5% (#6) | 94.2% (#6; 19 / 21) |
| 3-bit, gain corrected | x1.174 [1.156, 1.191] | x1.144 [1.131, 1.158] | 53.3 (41.8) | 94.5% (2 / 1) | 93.3% (20 / 34, p 0.08) |
| 3-bit, gain corrected, layers 40-47 FP8 | x1.139 [1.123, 1.156] | x1.136 [1.113, 1.161] | 53.7 (42.7) | 95.7% (3 / 0) | 93.5% (21 / 33, p 0.13) |

The FP8 trunk itself moves by up to 5 HumanEval items between configurations (97.0% in #6, 93.9 and 94.5%
here), and the items the 3-bit trunk lost in #6 are items this FP8 run also fails: with exact acceptance the
draft changes the verify row counts and so the target's arithmetic. The #6 HumanEval cost is within that spread.
Lower decode-path KL did not move GSM8K: every 3-bit variant lands 93.3 to 94.2% against FP8's 94.4 to 94.5%,
none significantly. Neither new variant keeps +15% at t = 0.7, so the `fast` default keeps the current copies;
the gain correction and `SHARAR_FLASHNEXT_W3_KEEP_FP8` stay available as options.
Evidence `$SHARAR_RESULTS/reviews/2026-09-27-trunk-sensitivity/`.

## Draft-only lm_head (2026-09-25)

The three MTP draft steps each read the target's FP8 lm_head (0.63 GB), about 8 of the draft's
10 ms per round (#35). The draft only proposes, so its head can be cheaper without changing what
exact acceptance emits. `SHARAR_DRAFT_HEAD=w3` gives the draft a 3.25-bit copy of the head
(0.24 GiB more memory); its top 8 candidates are rescored with the target's own FP8 rows, so the
draft picks what the full head would pick unless the two disagree on a near-tie
(`SHARAR_DRAFT_HEAD_RESCORE`, default 8; 0 takes the 3-bit argmax). It combines with the reduced
vocabulary (`SHARAR_DRAFT_VOCAB`). Evidence `$SHARAR_RESULTS/reviews/2026-09-25-draft-head/`:

| Check (`bench/draft_head.py`: hidden rows recorded during generation, 1,257 draft steps over 5 prompts) | FP8 head | 3-bit | 3-bit, top 8 rescored | FP8, 65K rows |
|---|---:|---:|---:|---:|
| ms per draft step, 1 / 4 rows (graph replay) | 2.74 / 2.72 | 1.03 / 1.07 | 1.09 / 1.13 | 0.67 / 0.67 |
| Picks equal to the FP8 head's (all / multilingual) | 1 / 1 | 0.939 / 0.886 | **0.991 / 0.980** | 0.966 / 0.886 |

A shortlist of 4, 8 or 16 gives the same 0.9905: the remainder are near-ties the served head's
FP8 activation rounding breaks the other way, not tokens missing from the shortlist.

Served, image `sharar:draft-head`, `config/flashnext.env`, `evals.h2h` (22 prompts per mode, `sharar-evals`),
arms interleaved FP8 / 3-bit / FP8 / 3-bit:

| Arm | probes | greedy median | t=0.7 median | tokens per round |
|---|---:|---:|---:|---:|
| FP8 head | 53.9 / 53.2 | 50.9 / 50.1 | 45.9 / 46.3 | 3.01 / 3.02 |
| 3-bit, rescored | 56.2 / 56.4 | 54.8 / 55.2 | 49.7 / 50.7 | 3.02 / 3.02 |

Paired per prompt: greedy x1.087 [1.075, 1.100], t=0.7 x1.077 [1.065, 1.090], faster on all 22
prompts in both modes and in every category (multilingual x1.06 greedy, x1.08 sampled). The launcher
default in `config/flashnext.env` is `w3`; the code default stays the target's head.

## Draft depth (2026-09-25)

With the cheaper draft head, deeper MTP drafts pay on average (the cap is now 8). Same harness,
arms k = 3, 5, 4, 3, 5; paired per prompt against k = 3:

| k | greedy median | t=0.7 median | greedy paired | t=0.7 paired | by category (greedy) | four streams |
|---|---:|---:|---:|---:|---|---|
| 3 | 54.9 / 54.3 | 47.4 / 49.5 | | | | 96 / 93 |
| 4 | 56.5 | 52.4 | x1.040 [1.010, 1.071] | x1.032 [1.003, 1.062] | code 1.09, math 1.10, prose 0.94 | 97 |
| 5 | 58.3 / 58.9 | 53.9 / 52.1 | x1.053 [1.005, 1.103] | x1.034 [0.986, 1.085] | code 1.16, math 1.14, prose 0.92, multilingual 0.98 | 98 / 95 |

Deeper drafts win where acceptance is high and lose where it is low; `config/flashnext.env` uses
k = 4. Adaptive depth (stop drafting when the draft is unsure) is the follow-up.

## Round idle gaps (2026-09-28)

The GPU waited for the host for about 7.7 ms of each 55.7 ms MTP round (fast profile, batch one, #36;
`bench/flashnext_round.py --gap-profile --kernel-gaps`). Inside the verify graph the node bubbles are only
0.3 ms, so the "9 ms outside kernels" of #25 was this host time between replays. Draft->verify: the PLE rows
waited for the whole draft block, then went through the 32-thread pool even when the page cache held them
(1.1 ms for 80 cached rows), then the plan and indexer rows ran. Now the plan and indexer rows come first, cached
rows are read without the threads (`RWF_NOWAIT`; only misses go to the pool), and the MTP proposer copies each
drafted column to the host behind its step, so a token's rows are read while the next step runs
(`DraftColumns`). Verify->draft: host acceptance and per-request draft KV launches; `SHARAR_DEVICE_ACCEPT=1`
(on in the `fast` profile) now covers MTP too (`feedback_device` projects the draft KV over the whole bucket inside the accept graph).
Evidence `$SHARAR_RESULTS/reviews/2026-09-27-idle-gaps/`:

| fast profile | round ms b1 | GPU idle b1 | served batch one | served four streams |
|---|---:|---:|---|---|
| before | 55.7 | 7.7 | 73.1 tok/s median | 119.2 tok/s aggregate |
| pieces 1+2 (default) | 51.3 | 3.5 | x1.097 [1.090, 1.104], 22/22 | x1.079 [1.068, 1.090], 22/22 |
| + device acceptance (on in `fast`) | 49.8 | 0.45 | x1.159 [1.137, 1.186], 22/22 | x1.174 [1.162, 1.187], 22/22 |

Exact profile: pieces 1+2 x1.067 at batch one with greedy identity 44/44 against the previous engine; device
acceptance adds x1.051 but is not bitwise (identity 2/44; the draft KV's FP8 activation scale spans the whole
bucket, so proposals and round boundaries drift), with acceptance unchanged. The launcher turns it on only in the
`fast` profile (`tools/serving-profile.sh`); `exact` and `balanced`, and the 27B, keep it off.

## Not done

Speculation for sampled requests (MTP is greedy-only, #22), layer feature
capture (DFlash2-style drafters), W3 trunk and experts, fused hyper-connection
and small-GEMV kernels, family-gate inputs,
accuracy gate, production image. Determinism: cold prefills and eager decode
are bit-identical to each other; a prefix-cache-hit continuation diverges from
the cold one (token 9 on the 512-token code prompt), consistent with the FP8
cached-prefix attention the 27B also has, which this model's routing amplifies
into a token flip. Profile (12 eager steps, graphs 50 ms/step): fused-MoE GEMMs
15.2 ms per step with the default tactic hint (the kernel screen got 8.1 ms
with the hint set to the token count, now passed per call), FP8 trunk GEMVs
14.1 ms (at bandwidth), BF16 hyper-connection mixers 5.7 ms, elementwise and
norm kernels 5 to 8 ms, dynamic quantization 0.8 ms, GDN recurrence 0.6 ms.
