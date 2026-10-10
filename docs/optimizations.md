# Optimization inventory

What makes Sharar fast on the GB10 today, in one place: each optimization,
where it lives, whether it is on by default, and what it measured. Update
this page when an optimization lands, changes default or is retired; the
dated numbers behind every row are in the linked studies. Last updated
2026-09-28.

The earlier native batch-one profile streamed roughly 22 GB of weights per
round at about 239 GB/s. Weight traffic remains a major cost; W3's larger
row counts also expose register pressure and memory-load waits. See the
[MLP counter study](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/w3-mlp-profile.md) for the distinction
between effective weight bandwidth and measured hardware counters. The
optimizations below reduce traffic, increase tokens per round, or reduce
time spent waiting.

## Serving profiles

`SHARAR_PROFILE` (or `sharar-ctl serve --exact | --balanced | --fast | --throughput`) sets
the default of every speed/fidelity knob the launcher resolves; a knob set
explicitly in the environment or the config file wins. `/get_server_info`
reports the profile under `config`. **The default is `fast`.**

| Knob | exact | balanced | fast (default) | throughput |
|---|---|---|---|---|
| Relaxed acceptance, greedy requests (`SHARAR_ACCEPT_DEFAULT`) | off | top-2 within 1.0 logit | top-3 within 2.0 logits | as fast |
| Relaxed acceptance, sampled requests (`SHARAR_ACCEPT_DEFAULT_SAMPLED`; threshold `SHARAR_ACCEPT_DEFAULT_SAMPLED_THRESHOLD`, unset: the greedy one) | off | off | on: Flash-Next at the greedy threshold; the 27B at top-2 within 1.0 logit, because at 3:2.0 its sampled thinking looped to `max_tokens` in Claude Code sessions (#11) | as fast |
| Calibrated 3-bit decode copies (`SHARAR_W3_DECODE`; from `SHARAR_W3_CHECKPOINT`, default `$SHARAR_RESULTS/w3-checkpoints/<target>-current`) | off | on for the 27B (layers and draft); off for Flash-Next | on (27B layers and draft, Flash-Next trunk) | as fast |
| Device-resident acceptance (`SHARAR_DEVICE_ACCEPT`) | off | off | on for Flash-Next; off for the 27B | as fast |
| Requests decoded together (`SHARAR_MAX_BATCH`, `sharar serve --max-batch`) | 4 | 4 | 4 | 16 |

The throughput profile (#59) is for evaluations and several agents at once:
it batches up to 16 requests (CUDA graphs for 1, 2, 4, 8 and 16 rows), and a
single request runs exactly as in fast. The 3-bit copies serve calls of at
most 64 rows; a wider verification runs the original NVFP4/FP8 weights.
Flash-Next verifies three drafts at 9-16 requests (`SHARAR_SPEC_DEPTHS=16:3`,
64 rows), so it serves fast's weights at every load for 1.4% of wall time.
The 27B keeps its 15 drafts at every batch: its batches of five or more
verify on the original weights, for GSM8K x1.74 instead of x1.29 with
`8:7,16:3`, the depths that would keep fast's weights (below). Any profile
takes `SHARAR_MAX_BATCH` and `SHARAR_SPEC_DEPTHS` (verify depth per batch
bucket, 0 = plain decode).

Lossless in every profile: speculative decoding itself, the 3-bit draft-only
lm_head of the Flash-Next MTP head, FP8/NVFP4 served weights, graphs.

Measured cost of each lossy option (paired, exact greedy unless noted):

| Option | Speed | Quality |
|---|---|---|
| Relaxed top-2 / 1.0, greedy (27B) | +9.8% batch one, +5.7% batch four | GSM8K 94.7 vs 95.0, HumanEval 91.5 vs 90.9; judged 13 to 14 with 26 ties |
| Relaxed top-3 / 2.0, greedy (27B) | +21.6% batch one vs exact; +3.7% h2h median over top-2 | judged 11 to 15 (not significant); 7% of emitted tokens not the argmax |
| Relaxed, sampled requests (27B, t = 0.6, 2026-09-23, when top-2 / 1.0 was the default) | +4% | GSM8K-300 96.0 to 94.0, HumanEval 94.5 to 91.5 (pooled 4 better / 15 worse, p about 0.02) |
| Relaxed top-2 / 1.0, sampled (27B, t = 0.7, #11, 2026-10-06) | h2h per prompt x1.10 over exact (18 of 22 faster); top-3 / 2.0 x1.13 | the t = 0.6 row above; in Claude Code at temperature 1 its thinking did not loop (a looped request replayed 0/6 vs 4/6 at top-3 / 2.0; a 5-turn session 30/30 tests, no request at `max_tokens`) |
| Relaxed top-3 / 2.0, sampled (Flash-Next, t = 0.7, #22) | median 45.6 to 49.1 tok/s (+8%) | GSM8K-200 at t = 0.7 96.5 to 94.0 (SE about 1.5) |
| 27B calibrated 3-bit copies | decode x1.49 over NVFP4/FP8 (nearest-rounded build), calibrated +8.6% over nearest | full GSM8K 94.5 vs 94.0 exact; HumanEval within noise (layers and draft; the lm_head copy stays off) |
| Flash-Next device-resident acceptance (#36, 2026-09-28; greedy rounds) | h2h over the default without it (fast): batch one x1.057 [1.035, 1.082], four streams x1.088 [1.078, 1.098]; over the previous engine x1.159 / x1.174 | acceptance unchanged (6.14 to 6.18 emitted per round); outputs not identical to the host path (exact greedy identity 2/44: the draft KV is projected over the whole bucket, so round boundaries move); no accuracy eval yet |
| Flash-Next calibrated 3-bit trunk (#6, 2026-09-27) | h2h greedy median 56.1 to 69.6 tok/s (+24%), t = 0.7 49.4 to 62.0 (+26%) | full GSM8K 94.5 to 94.2 (20 better / 23 worse, p 0.76); HumanEval 97.0 to 94.5 (0 / 4, p 0.12), within the FP8 trunk's own spread between configurations (93.9 to 97.0%, #61); decode-path KL to BF16 0.51 to 0.96 |

New lossy options are judged by the lossy-change gate, `tools/lossy-gate.py`
([rules](../CONTRIBUTING.md#judge-a-lossy-option)). Its first reading, the Flash-Next `fast`
profile against `exact` (image `sharar:main-20260927b`, 2026-09-28): decode x1.347 [1.309, 1.389]
at batch one; layer 1 KL 0.081 against a 0.013 floor (argmax 91.0% against 96.4%; prose and
multilingual 85-88%, code and math 96%); tokens per round x1.165; screen pooled -0.8 points
[-2.5, +0.9] (17 better / 23 worse, p 0.43; HumanEval 2/6); verdict `fast`, the full sets needed
for `balanced`. Evidence `$SHARAR_RESULTS/reviews/2026-09-27-lossy-gate-fast/`.

Evidence: [relaxed acceptance](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/relaxed-acceptance.md),
the sampled-path study in the roadmap history, #22, [W3 calibration](w3-calibration.md),
`$SHARAR_RESULTS/reviews/2026-09-27-fn-trunk/` and `2026-09-27-trunk-sensitivity/` (#61) for the Flash-Next trunk,
`$SHARAR_RESULTS/reviews/2026-09-27-idle-gaps/` for Flash-Next device-resident acceptance.

### Throughput profile (#59, 2026-09-28)

In-process scheduler cells (`bench/throughput_batch.py`, at [the archive tag](https://github.com/abdelfattah-lab/sharar/tree/archive/bench-pre-split-2026-10-04/bench)): 64 GSM8K problems
(the accuracy eval's prompt, thinking off, greedy, exact acceptance, 640
tokens) submitted at once, run until idle, cold prefix cache; aggregate
output tok/s including prefill and the tail where fewer than BATCH requests
remain. Image `sharar:main-20260927b` with the branch mounted, fast-profile
weights, 262K context.

| Batch : verify depth (verify rows) | Flash-Next (MTP k=4) | 27B (DFlash2 k=15) |
|---|---:|---:|
| 4 : k, the fast profile (20 / 64) | 123.8, 125.0 | 276.3 |
| 8 : k (40 / 128) | 168.0 | **359.7** |
| 8 : 7 (64) | | 357.3 |
| 8 : 3 / 2 / 1 / 0 | 163.7 / 148.2 / 124.7 / 88.8 | 256.7 / - / - / - |
| 16 : k (80 / 256, original weights; the 27B's throughput profile) | 220.3, 221.8 | **423.2** |
| 16 : 3 (64; Flash-Next's throughput profile) | **212.9** | 345.1 |
| 16 : 7 / 2 / 1 | - / 191.4 / 159.1 | 429.4 / - / 226.8 |
| 16 : 0 (plain decode) | 110.0 | 120.8 |

Two cells per Flash-Next row are two runs. Speculation stays worth it at 16
rows on both targets; plain decode at batch 16 is slower than batch four with
speculation (on Flash-Next 15% of the plain step is the proposer following
every row: 110.0 against 128.6 tok/s with the proposer detached). In the
27B's steady state (16 requests decoding, no prefill) a 16:15 step is 194 ms
for 130 tokens (668 tok/s) and 16:3 129 ms for 57 tokens (440 tok/s), 97-98%
of each GPU kernels. On Flash-Next 16:3 against 16:4 costs 4% tok/s but only
1.4% wall time: the 3-bit trunk writes shorter answers than the FP8 trunk
the 80-row verification falls back to (GSM8K 318 against 335 tokens per
item, as in #6). The 27B's 16:15 is 23% faster than 16:3, and 8:15 matches
8:7 (359.7 against 357.3), so its throughput profile keeps the full depth.
GSM8K correct counts in every cell were 59 to 64 of 64.

Served (HTTP, `sharar-eval` accuracy eval, exact acceptance, greedy,
image `sharar:throughput`, requests in flight = the profile's batch):

| Target, arm | GSM8K-1319 wall | accuracy (paired vs batch four) | HumanEval wall, pass@1 |
|---|---:|---|---|
| Flash-Next fast, 4 in flight | 3338.9 s | 94.0% | 290 s, 94.5% (#6 run) |
| Flash-Next throughput before #36, 16 | 1943.2 s (x1.72) | 94.2% (20 / 17, McNemar p 0.74) | 192.0 s, 93.3% (1 / 3, p 0.62) |
| **Flash-Next throughput with the round idle gaps and device acceptance (#36), 16** | **1814.9 s (x1.84)** | 93.9% (13 / 15, p 0.85) | 176.1 s, 93.9% (1 / 2 vs the #6 batch-four run, p 1.00) |
| Flash-Next 16:4 (80 rows, FP8 trunk) | 1962.0 s (x1.70) | 93.9% (22 / 23, p 1.00); 335 tokens per item against 318 | 177.4 s, 96.3% |
| 27B fast, 4 in flight | 1442.1 s | 94.2% | 88.0 s, 93.3% |
| **27B throughput (full depth; wide verifies on the original weights), 16** | **827.7 s (x1.74)** | 93.6% (20 / 28, p 0.31); 350 against 328 tokens per item | |
| 27B with fast's weights (`SHARAR_SPEC_DEPTHS=8:7,16:3`), 16 | 1120.2 s (x1.29) | 94.4% (14 / 11, p 0.69) | 90.7 s, 93.3% |
Both targets reach the #59 gate (x1.7 on the full GSM8K with accuracy within
noise). Flash-Next does it with fast's weights at every load; the 27B with
its full depth, which verifies batches of five or more on the original
NVFP4/FP8 weights (longer answers, as the exact weights write). With fast's
weights the 27B's batch-four rounds already verify 64 rows, so a wide batch
only trades depth for rows (x1.29): a paired evaluation of a lossy 27B weight
candidate (such as #42) should set `SHARAR_SPEC_DEPTHS=8:7,16:3` so that its
arms serve the candidate's weights at every load. Batch-one latency in the
fast profile is unchanged (h2h probes / greedy median): Flash-Next 70.8 / 72.8
tok/s against 71.1 / 72.9 at #52, the 27B 128.3 / 93.9 against 126.6 / 92.5.

Memory at batch 16 with the 262K context: the pool guarantees one request
at 262K or the batch-four total (4 x 32K tokens) split between the running
requests, and requests share the rest; admission waits when the pool is
full. Flash-Next plans about 277K-279K KV tokens and 20-21 state slots (1
dummy, 16 live, the rest prompt snapshots): 16 requests of about 17K tokens
each at once. The 27B plans 517K KV tokens and 96 slots: 16 requests of
32K. Above four requests, prefill snapshots never take the slots the next
admissions need, so a full batch degrades the prefix cache instead of
stalling admission.

Evidence: `$SHARAR_RESULTS/reviews/2026-09-27-throughput/` (`SUMMARY.md`).

## Weights and kernels

| Optimization | Where | Default | Measured | Evidence |
|---|---|---|---|---|
| NVFP4 MLPs and lm_head, FP8 projections from the ModelOpt release | `sharar/weights/modelopt.py`, `sharar/kernels/ops.py` | On | The served baseline (4.5 and 8 bits per weight) | [operations](operations.md) |
| Small-M FP8 GEMV for the projections (weights stream once for up to four rows) | `sharar/kernels/csrc/fp8_gemv.cu` | On for calls of at most 4 rows | Above cuBLAS's 60% of bandwidth at small M | [BS1](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/gb10-bs1-speedups.md) |
| Exact BS1 QKV/MLP verify bundle (cutlass FP4 MLP tiles and the QKV path for 16 rows) | `sharar/kernels/verify_gemm.py` | On | +1.43% family decode, outputs preserved | [BS1](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/gb10-bs1-speedups.md) |
| Fused verify norm and convolution, GDN prefill planning, QK-norm plus RoPE | `sharar/kernels/verify_norm.py`, `verify_conv.py`, `qk_norm_rope.py`, `sharar/model/gdn_plan.py` | On | Part of the exact bundle and the DFlash2 default | [DFlash2 default](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/default-dflash2.md) |
| **Three-bit decode weights** (3.25 bits per weight plus row multipliers, eight-level E4M3 codebook, tensor-core kernel for up to 64 rows; original NVFP4/FP8 weights retained for larger calls) | `sharar/kernels/w3.py`, `csrc/w3_gemm.cu` | **On** for the layers and the draft in the launcher since 2026-09-17 (`SHARAR_W3_DECODE=1`, by profile), lm_head off (the option was retired 2026-09-30, #72); benches and gates leave it off unless asked | The original all-three-copies measurement gave round cost x0.649 and decode x1.488 [1.444, 1.532] at batch one, round x0.76 at batch four. Nearest rounding loses task accuracy; the lm_head copy stays off. Calibrated target copies now pass their own quality and speed gates | [coverage and dispatch](#w3-coverage-and-dispatch), [kernel study](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/w3-cuda.md), [first study](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/w3-decode.md) |
| 3-bit routed experts for Flash-Next (W3 format per expert, the only resident expert copy; grouped kernel for any row count) | `sharar/kernels/w3_moe.py`, `csrc/w3_gemm.cu`, `sharar/model/flashnext/moe.py`, `sharar/weights/flashnext_w3_experts.py` | Off: used only when the W3 checkpoint carries `layers.N.experts.*` and the 3-bit copies are on; no promoted checkpoint carries them (the full-stage lossy gate rejected the 2026-09-30 calibration, pooled -0.8 pp, p 0.044) | Single-stream x1.19 greedy / x1.18 t=0.7 over NVFP4 experts; -17.2 GiB resident; kernel x1.54 at the 5-row verify; prefill neutral at one request, 4,096-row steps 30% slower in the MoE | [Flash-Next](flashnext.md#3-bit-routed-experts-2026-09-30-73) |
| GPTQ-calibrated W3 target copies from the pinned FP8 release | `sharar/weights/w3_calibration.py`, `w3_checkpoint.py`, `w3_calibrate.py` | **On** in `fast` via `SHARAR_W3_CHECKPOINT` (default `$SHARAR_RESULTS/w3-checkpoints/qwen38-current`, fetched from `yemara/Qwen3.8-27B-NVFP4-sharar-w3@72997f2`, recalibrated 2026-10-01 after the 2026-09-17 artifact was lost) | 2026-09-17 artifact: GSM8K 92.04% to 94.47%, family decode x1.0862 [1.0513, 1.1230] against nearest W3. 2026-10-01 artifact, `fast` against `exact` (lossy gate, full stage): rejected on MBPP -3.2 pp (p 0.029), HumanEval -3.7, GSM8K +0.2, MGSM +0.3, pooled -0.3 [-1.0, +0.5]; speed x1.83; adopted over nearest rounding, cause in #81 | [calibration and adaptation](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/w3-calibration-adaptation.md), [artifact guide](w3-calibration.md) |
| W3 QKV at 64 rows split into two 32-row tiles | `sharar/kernels/csrc/w3_gemm.cu` | Enabled in source for static/per-row scales, N=14336, K=5120, split=1; awaiting deployment | Cached 10K batch-four family decode +0.66% [0.61%, 0.71%], exact outputs/work; no general HTTP gain established | [paired qualification](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/w3-qkv-tiling.md) |
| Shape-stable prefill numerics (M-invariant b/a projection, FP8 window, gated norm) | `sharar/kernels/stable.py`, `ba_proj.py` | Opt-in `SHARAR_STABLE_NUMERICS=1` | Prefix bit-exact across prefill lengths for under 1% prefill cost; research baseline, not a speedup | [numerics](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/target-numerics.md) |

### W3 coverage and dispatch

The current local profile has **304 calibrated target projection matrices
across all 64 target layers**, plus **20 W3 matrices in the five-layer
DFlash2 drafter**. A projection is a learned matrix that transforms the
layer's inputs. Counts below refer to packed matrices: combined Q/K/V and
gate/up each count as one matrix.

| Component | Layers covered | Matrices with W3 copies in each layer | Packed matrices |
|---|---:|---|---:|
| Target feed-forward network (MLP) | All 64 | Combined `gate_up`, plus `down` | 128 |
| Target Gated DeltaNet (recurrent/linear attention) | 48 | `in_qkv`, `in_z`, `out` | 144 |
| Target full attention | 16 | Combined `qkv`, plus output `o` | 32 |
| DFlash2 drafter | All 5 | `q_proj`, `o_proj`, combined `gate_up`, `down_proj` | 20 |

The target's 304 matrices are fitted offline from the pinned FP8 source
using GPTQ, then loaded from the calibrated artifact. The drafter is
separately trained to match that target; its 20 eligible FP8 matrices are
packed into W3 by nearest rounding at construction or reload. They are not
part of the target's GPTQ artifact. Weight quantization is not repeated for
each generated token.

In this serving profile, W3 is selected for eligible matrix calls with
**at most 64 activation rows**. It serves ordinary decode and speculative
verification; it is not gated solely on the request's prefill/decode phase.
Short prefill calls can also qualify. With 15 draft proposals, the target
verifies 16 rows per request, so batches of one, two and four commonly
produce 16, 32 and 64 rows. Larger prompt-processing calls use the original
NVFP4/FP8 weights. Both representations remain resident: W3 adds compact
copies to reduce decode weight traffic, rather than replacing all resident
weights with a smaller model.

The following retain their existing formats: the target's NVFP4 vocabulary
head (`lm_head`, W3 disabled), embeddings, normalization weights, small
Gated DeltaNet b/a control projections and convolution weights; the
drafter's K/V context projections and other unlisted components; and KV
caches and recurrent states. Attention score computation and softmax are
not converted to 3-bit operations.

Three bits describe each stored weight code. Including group scales and
row multipliers, the tensor payload is `3.25 + 32/K` bits per weight for
input width `K`. Inside the W3 kernel, codes expand to FP8 values, multiply
FP8 activations, accumulate in FP32 and produce BF16 outputs. Activations
are not 3-bit. See [3-bit quantization explained](3bit-quantization-explained.md)
for examples, calibration and prior work.

The executable coverage and dispatch rules live in
[target projection enumeration](../sharar/weights/w3_checkpoint.py),
[target dispatch](../sharar/kernels/ops.py),
[drafter matrix selection](../sharar/model/parallel_draft.py) and
[drafter dispatch](../sharar/model/draft_linear.py).

## Speculative decoding

| Optimization | Where | Default | Measured | Evidence |
|---|---|---|---|---|
| DFlash2 parallel drafter, 15 proposals, 16 verify rows per request | `sharar/proposers/parallel.py`, `sharar/model/parallel_draft.py` | On (`SHARAR_SPEC=15`) | 2 to 2.6x over plain decode; k=15 beats 7 and 3 at batch four | [BS4](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/gb10-bs4-engine.md), [DFlash2 default](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/default-dflash2.md) |
| FP8 draft matrices with dynamic activation scaling | `sharar/model/draft_linear.py`, `sharar/kernels/draft_fp8.py` | On since 2026-09-17 (`SHARAR_DRAFT_PRECISION`, bf16 for the publisher's precision) | Draft bytes 2.5 GB instead of 3.7 GB | [performance](performance.md#dflash2-configuration) |
| Device-resident DFlash2 candidate selection | `sharar/kernels/draft_selection.py` | On | Removes the host round trip from the lattice traversal | [selection](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/draft-selection.md) |
| Accepted-state reconstruction instead of state snapshots | `sharar/engine/spec.py`, `SHARAR_VERIFY_STATE` | On | Direct reconstruction of GDN state for the accepted prefix | [direct state](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/direct-state.md) |
| Batched accepted-feature feedback to the drafter | `sharar/proposers/parallel.py` | On | One feature gather per round for the batch | [accepted batch](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/accepted-batch.md) |
| Trained drafter on recorded serving traffic, hot reload of the promoted checkpoint | `sharar/train/`, `sharar-ctl train` | On when a checkpoint is promoted (`draft-runs/current`) | +12.9% batch-one throughput on a coding proxy, +12.4% at concurrency four after a 1,000-request loop | [draft training](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/gb10-draft-training.md), [draft adaptation](draft-adaptation.md) |
| Drafter adapted to the calibrated W3 target | Existing training and reload paths; step-200 artifact | Active on the local GB10 since 2026-09-17 through the promoted checkpoint | Source-file-disjoint coding proxy: served throughput +3.60% at concurrency one and +4.43% at concurrency four; positive paired acceptance and latency intervals | [calibration and adaptation](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/w3-calibration-adaptation.md) |
| Flash-Next MTP draft head: 3-bit copy of the lm_head for the draft steps only, top 8 rescored with the target's FP8 rows | `sharar/proposers/draft_vocab.py`, `SHARAR_DRAFT_HEAD` | On for Flash-Next (`config/targets/flashnext.env`) since 2026-09-25 | Draft step 2.74 to 1.09 ms; picks equal the FP8 head's on 99.05% of recorded rows; served h2h x1.087 greedy, x1.077 t=0.7 (22/22 prompts), tokens per round unchanged | [Flash-Next](flashnext.md#draft-only-lm_head-2026-09-25) |
| Flash-Next round without host gaps before verification: verify plan and indexer rows before the PLE read, PLE rows from the page cache without the thread pool (`RWF_NOWAIT`), each drafted token's PLE rows read while the next draft step runs | `sharar/engine/spec_graphs.py`, `sharar/model/flashnext/ple.py`, `extras.py`, `sharar/proposers/mtp.py` | On since 2026-09-28 | GPU idle draft->verify 3.8 to 0.1 ms per round; round 55.7 to 51.3 ms (fast, batch one); served h2h fast x1.097 batch one, x1.079 at four streams (22/22 prompts); exact x1.067, identity 44/44 | [Flash-Next](flashnext.md#round-idle-gaps-2026-09-28) |
| **Relaxed acceptance** (accept a drafted token within the target's top-k and a logit margin) | `sharar/engine/spec.py`, `sharar/api/validation.py` | On for greedy requests that do not choose: top-2 within 1.0 logit (`SHARAR_ACCEPT_DEFAULT`); sampled requests exact | +9.8% batch one, +5.7% batch four, +14.8% ShareGPT; 53 judged prompts 13 to 14 with 26 ties; GSM8K and HumanEval unchanged | [relaxed acceptance](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/relaxed-acceptance.md), [accuracy](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/accuracy-benchmarks.md) |

## Graphs, scheduling and memory

| Optimization | Where | Default | Measured | Evidence |
|---|---|---|---|---|
| CUDA graphs for decode and the verify/draft rounds | `sharar/engine/graphs.py`, `spec_graphs.py`, `sharar/proposers/parallel_graphs.py` | On | Batch-one GPU busy 98.4% of the round | [round profile](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/gb10-round-profile.md) |
| Cached trace storage accounting and HTTP-side latency percentiles | `sharar/engine/traces.py`, `loop.py`, `scheduler.py` | On | Removes per-step file walks and latency sorting; 5,000-feature-file stats call 23.77 ms to 0.208 µs in a CPU microbenchmark; rotation includes feature bytes | [runtime and calibration](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/w3-calibration-adaptation.md) |
| 2,048-token prefill chunks, 4,096-token prefill budget per step | `sharar/engine/scheduler.py` | On | Bounded time to first token beside decoding requests | [BS4](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/gb10-bs4-engine.md) |
| Prefix caching and managed-memory cold storage for KV pages | `sharar/cache/`, `sharar/engine/context.py` | On | Warm resumes skip the prefill | [cold tier](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/cold-tier.md), [agent context](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/agent-context.md) |
| NVMe session store: parked sessions written through and restored after arena drops or restarts; multi-slab parking for long contexts | `sharar/cache/session_disk.py`, `sharar/cache/cold_tier.py` | On (`SHARAR_SESSION_DIR`) | Disk 9.7 GB/s read, 7.0 GB/s write: ~1 s restore for a 262K session instead of a 351 s prefill | [operations](operations.md) |
| Checkpoint-owned draft features and state retention for resumed sessions | `sharar/engine/context.py` | On | Resumed batch-one completion TPS 11.3 to 36.7 (3.24x); four conversation sessions 28.8 to 78.5 aggregate TPS | [retention](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/gb10-agent-retention.md) |
| Host memory guard (16 GiB `MemAvailable` floor) | `tools/memory-guard.sh`, launcher | On | Keeps unified memory from paging the weights | [operations](operations.md) |
| CPU staging of mapped calibrated W3 tensors before CUDA loading | `sharar/weights/w3_checkpoint.py` | Enabled in source for calibrated checkpoints; awaiting deployment | Warm-cache target construction 63.23 to 11.99 seconds (5.27×); all 304 matrices byte-identical; excludes drafter and graph capture | [W3 runtime profile](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/w3-runtime-profile.md) |
| W3 compiled module identity includes its CUDA source digest | `sharar/kernels/w3.py` | Enabled in source; awaiting deployment | Prevents stale objects across source revisions with older timestamps; rollback regression and final GPU checks pass | [build identity](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/w3-qkv-tiling.md#build-identity-and-final-verification) |

## Measured, not integrated

| Candidate | Result | Status |
|---|---|---|
| FP8 GEMM at 16 rows through the GEMV path (`fp8_m16`) | Round cost x0.985 on the 101-family gate | Moot in the served profile: with `SHARAR_W3_DECODE=1` every FP8 projection at 64 rows or fewer takes the 3-bit copy before the FP8 dispatch; only the exact NVFP4/FP8 path (`SHARAR_W3_DECODE=0`) would use it ([family gate](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/family-gate.md)) |
| Device-resident acceptance (`SHARAR_DEVICE_ACCEPT=1`: accept, reconstruct, conv commit and draft feedback in one graph per bucket, one host wait) | Engine-loop round cost x0.944 at batch four / 10K, x0.883 at 1K, x0.983 at batch one; HTTP +2% at concurrency one, unchanged at four (engine thread per-step 117 ms on both paths) | Opt-in, exact on the 27B; serving gain needs lazy round completion ([study](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/device-accept.md)). Flash-Next MTP (2026-09-28): GPU idle per round 3.5 to 0.45 ms, served h2h fast x1.057 batch one and x1.088 at four streams over the default, tokens per round unchanged; not bitwise (the draft KV is projected over the whole bucket: identity 2/44 on exact); on in the `fast` profile for Flash-Next since 2026-09-28 ([serving profiles](#serving-profiles), [Flash-Next](flashnext.md#round-idle-gaps-2026-09-28)) |
| KV page size 16 | Decode attention unchanged; verify attention 2.52 to 2.08 ms (batch one) and 8.49 to 7.42 ms (batch four) per round at 10K context, under 1% of the round | Parked; `bench/kv_page_size.py` (2026-09-18, at the tag [`archive/bench-pre-split-2026-10-04`](https://github.com/abdelfattah-lab/sharar/tree/archive/bench-pre-split-2026-10-04/bench)) |
| NVFP4 draft body | Round cost x0.971; acceptance not resolved | Needs the trained-draft comparison |
| Prompt-lookup second verify branch | Offline oracle +14 to 50% tokens per round, served +1.4% | Parked on `feat/lookup-branch` ([study](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/lookup-branch.md)) |

Rejected candidates (narrower verification, mixed forwarding, numerical FP8
replacements, the short-resume scheduler, two-path trees, narrow NVFP4
tiles) are in the [decision index](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/README.md).
The W3 MLP cache-policy screen also rejected bypass-L1 and normal-cache
bit-plane loads: both take 43–59% longer than the existing streaming loads
in the six isolated cases ([hardware-counter study](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/w3-mlp-profile.md)).

## Combined effect on the standard cell

10K input, 1K output, batch one, identical prompt IDs, mean over the code,
design and reasoning workloads ([comparison](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/sglang-comparison-2026-09-16.md)):
SGLang DFlash2 30.4 end-to-end tokens per second; Sharar exact 36.5 (1.19x),
the relaxed default 44.2 (1.44x), first 3-bit build 47.4 (1.59x), both 52.3
(1.75x). Scaling those measured arms by the CUDA kernel's gate ratio
(decode x1.488 against the first build's x1.174, same 10K/1K batch-one
families, TTFT 3.96 s unchanged) projects the current tree at about 57
end-to-end (1.9x) with every 3-bit copy on and about 63 (2.1x) with the
relaxed default as well; decode-only about 74 (2.2x) and 84 (2.4x). Measured
on the code workload against the restarted production (2026-09-17, layers
and draft copies, FP8 draft): exact acceptance pinned 43.8 end-to-end and
53.0 decode (SGLang 27.6 and 31.1: 1.59x and 1.70x); the served relaxed
default 53.1 and 67.2 (1.93x and 2.16x). A rerun of the full three-workload
comparison replaces the projections.
