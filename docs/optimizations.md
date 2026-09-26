# Optimization inventory

What makes Sharar fast on the GB10 today, in one place: each optimization,
where it lives, whether it is on by default, and what it measured. Update
this page when an optimization lands, changes default or is retired; the
dated numbers behind every row are in the linked studies. Last updated
2026-09-18.

The earlier native batch-one profile streamed roughly 22 GB of weights per
round at about 239 GB/s. Weight traffic remains a major cost; W3's larger
row counts also expose register pressure and memory-load waits. See the
[MLP counter study](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/w3-mlp-profile.md) for the distinction
between effective weight bandwidth and measured hardware counters. The
optimizations below reduce traffic, increase tokens per round, or reduce
time spent waiting.

## Weights and kernels

| Optimization | Where | Default | Measured | Evidence |
|---|---|---|---|---|
| NVFP4 MLPs and lm_head, FP8 projections from the ModelOpt release | `sharar/weights/modelopt.py`, `sharar/kernels/ops.py` | On | The served baseline (4.5 and 8 bits per weight) | [operations](operations.md) |
| Small-M FP8 GEMV for the projections (weights stream once for up to four rows) | `sharar/kernels/csrc/fp8_gemv.cu` | On for calls of at most 4 rows | Above cuBLAS's 60% of bandwidth at small M | [BS1](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/gb10-bs1-speedups.md) |
| Exact BS1 QKV/MLP verify bundle (cutlass FP4 MLP tiles and the QKV path for 16 rows) | `sharar/kernels/verify_gemm.py` | On | +1.43% family decode, outputs preserved | [BS1](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/gb10-bs1-speedups.md) |
| Fused verify norm and convolution, GDN prefill planning, QK-norm plus RoPE | `sharar/kernels/verify_norm.py`, `verify_conv.py`, `qk_norm_rope.py`, `sharar/model/gdn_plan.py` | On | Part of the exact bundle and the DFlash2 default | [DFlash2 default](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/default-dflash2.md) |
| **Three-bit decode weights** (3.25 bits per weight plus row multipliers, eight-level E4M3 codebook, tensor-core kernel for up to 64 rows; original NVFP4/FP8 weights retained for larger calls) | `sharar/kernels/w3.py`, `csrc/w3_gemm.cu` | **On** for the layers and the draft in the launcher since 2026-09-17 (`SHARAR_W3_DECODE=1`, `SHARAR_W3_DRAFT=1`), lm_head off (`SHARAR_W3_HEAD`); benches and gates leave it off unless asked | The original all-three-copies measurement gave round cost x0.649 and decode x1.488 [1.444, 1.532] at batch one, round x0.76 at batch four. Nearest rounding loses task accuracy; the lm_head copy stays off. Calibrated target copies now pass their own quality and speed gates | [coverage and dispatch](#w3-coverage-and-dispatch), [kernel study](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/w3-cuda.md), [first study](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/w3-decode.md) |
| GPTQ-calibrated W3 target copies from the pinned FP8 release | `sharar/weights/w3_calibration.py`, `w3_checkpoint.py`, `bench/w3_calibrate.py` | Opt-in `SHARAR_W3_CHECKPOINT`; qualified artifact active on the local GB10 since 2026-09-17 | GSM8K 92.04% to 94.47%; family decode x1.0862 [1.0513, 1.1230] against nearest W3, with unchanged kernel and weight size | [calibration and adaptation](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/w3-calibration-adaptation.md), [artifact guide](w3-calibration.md) |
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
| Flash-Next MTP draft head: 3-bit copy of the lm_head for the draft steps only, top 8 rescored with the target's FP8 rows | `sharar/proposers/draft_vocab.py`, `SHARAR_DRAFT_HEAD` | On in `config/flashnext.env` since 2026-09-25 | Draft step 2.74 to 1.09 ms; picks equal the FP8 head's on 99.05% of recorded rows; served h2h x1.087 greedy, x1.077 t=0.7 (22/22 prompts), tokens per round unchanged | [Flash-Next](flashnext.md#draft-only-lm_head-2026-09-25) |
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
| Device-resident acceptance (`SHARAR_DEVICE_ACCEPT=1`: accept, reconstruct, conv commit and draft feedback in one graph per bucket, one host wait) | Engine-loop round cost x0.944 at batch four / 10K, x0.883 at 1K, x0.983 at batch one; HTTP +2% at concurrency one, unchanged at four (engine thread per-step 117 ms on both paths) | Opt-in, exact; serving gain needs lazy round completion ([study](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/device-accept.md)) |
| KV page size 16 | Decode attention unchanged; verify attention 2.52 to 2.08 ms (batch one) and 8.49 to 7.42 ms (batch four) per round at 10K context, under 1% of the round | Parked; `bench/kv_page_size.py` (2026-09-18) |
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
