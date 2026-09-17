# Optimization inventory

What makes Sharar fast on the GB10 today, in one place: each optimization,
where it lives, whether it is on by default, and what it measured. Update
this page when an optimization lands, changes default or is retired; the
dated numbers behind every row are in the linked studies. Last updated
2026-09-17.

Decode on this machine is bound by weight bandwidth (about 239 GB/s
measured, 22 GB per native batch-one round), so every row below either
moves fewer bytes per emitted token, emits more tokens per round, or keeps
the GPU streaming instead of waiting.

## Weights and kernels

| Optimization | Where | Default | Measured | Evidence |
|---|---|---|---|---|
| NVFP4 MLPs and lm_head, FP8 projections from the ModelOpt release | `sharar/weights/modelopt.py`, `sharar/kernels/ops.py` | On | The served baseline (4.5 and 8 bits per weight) | [operations](operations.md) |
| Small-M FP8 GEMV for the projections (weights stream once for up to four rows) | `sharar/kernels/csrc/fp8_gemv.cu` | On for calls of at most 4 rows | Above cuBLAS's 60% of bandwidth at small M | [BS1](archive/studies/gb10-bs1-speedups.md) |
| Exact BS1 QKV/MLP verify bundle (cutlass FP4 MLP tiles and the QKV path for 16 rows) | `sharar/kernels/verify_gemm.py` | On | +1.43% family decode, outputs preserved | [BS1](archive/studies/gb10-bs1-speedups.md) |
| Fused verify norm and convolution, GDN prefill planning, QK-norm plus RoPE | `sharar/kernels/verify_norm.py`, `verify_conv.py`, `qk_norm_rope.py`, `sharar/model/gdn_plan.py` | On | Part of the exact bundle and the DFlash2 default | [DFlash2 default](archive/studies/default-dflash2.md) |
| **Three-bit decode weights** (3.25 bits per weight, eight-level E4M3 codebook, tensor-core kernel for 16 to 64 rows; NVFP4/FP8 kept for prefill) | `sharar/kernels/w3.py`, `csrc/w3_gemm.cu` | **On** for the layers and the draft in the launcher since 2026-09-17 (`SHARAR_W3_DECODE=1`, `SHARAR_W3_DRAFT=1`), lm_head off (`SHARAR_W3_HEAD`); benches and gates leave it off unless asked | The original all-three-copies measurement gave round cost x0.649 and decode x1.488 [1.444, 1.532] at batch one, round x0.76 at batch four. Nearest rounding loses task accuracy; the lm_head copy stays off. Calibrated target copies now pass their own quality and speed gates | [kernel study](archive/studies/w3-cuda.md), [first study](archive/studies/w3-decode.md) |
| GPTQ-calibrated W3 target copies from the pinned FP8 release | `sharar/weights/w3_calibration.py`, `w3_checkpoint.py`, `bench/w3_calibrate.py` | Opt-in `SHARAR_W3_CHECKPOINT`; qualified artifact active on the local GB10 since 2026-09-17 | GSM8K 92.04% to 94.47%; family decode x1.0862 [1.0513, 1.1230] against nearest W3, with unchanged kernel and weight size | [calibration and adaptation](archive/studies/w3-calibration-adaptation.md), [artifact guide](w3-calibration.md) |
| Shape-stable prefill numerics (M-invariant b/a projection, FP8 window, gated norm) | `sharar/kernels/stable.py`, `ba_proj.py` | Opt-in `SHARAR_STABLE_NUMERICS=1` | Prefix bit-exact across prefill lengths for under 1% prefill cost; research baseline, not a speedup | [numerics](archive/studies/target-numerics.md) |

## Speculative decoding

| Optimization | Where | Default | Measured | Evidence |
|---|---|---|---|---|
| DFlash2 parallel drafter, 15 proposals, 16 verify rows per request | `sharar/proposers/parallel.py`, `sharar/model/parallel_draft.py` | On (`SHARAR_SPEC=15`) | 2 to 2.6x over plain decode; k=15 beats 7 and 3 at batch four | [BS4](archive/studies/gb10-bs4-engine.md), [DFlash2 default](archive/studies/default-dflash2.md) |
| FP8 draft matrices with dynamic activation scaling | `sharar/model/draft_linear.py`, `sharar/kernels/draft_fp8.py` | On since 2026-09-17 (`SHARAR_DRAFT_PRECISION`, bf16 for the publisher's precision) | Draft bytes 2.5 GB instead of 3.7 GB | [performance](performance.md#dflash2-configuration) |
| Device-resident DFlash2 candidate selection | `sharar/kernels/draft_selection.py` | On | Removes the host round trip from the lattice traversal | [selection](archive/studies/draft-selection.md) |
| Accepted-state reconstruction instead of state snapshots | `sharar/engine/spec.py`, `SHARAR_VERIFY_STATE` | On | Direct reconstruction of GDN state for the accepted prefix | [direct state](archive/studies/direct-state.md) |
| Batched accepted-feature feedback to the drafter | `sharar/proposers/parallel.py` | On | One feature gather per round for the batch | [accepted batch](archive/studies/accepted-batch.md) |
| Trained drafter on recorded serving traffic, hot reload of the promoted checkpoint | `sharar/train/`, `sharar-ctl train` | On when a checkpoint is promoted (`draft-runs/current`) | +12.9% batch-one throughput on a coding proxy, +12.4% at concurrency four after a 1,000-request loop | [draft training](archive/studies/gb10-draft-training.md), [draft adaptation](draft-adaptation.md) |
| Drafter adapted to the calibrated W3 target | Existing training and reload paths; step-200 artifact | Active on the local GB10 since 2026-09-17 through the promoted checkpoint | Source-file-disjoint coding proxy: served throughput +3.60% at concurrency one and +4.43% at concurrency four; positive paired acceptance and latency intervals | [calibration and adaptation](archive/studies/w3-calibration-adaptation.md) |
| **Relaxed acceptance** (accept a drafted token within the target's top-k and a logit margin) | `sharar/engine/spec.py`, `sharar/api/validation.py` | On for greedy requests that do not choose: top-2 within 1.0 logit (`SHARAR_ACCEPT_DEFAULT`); sampled requests exact | +9.8% batch one, +5.7% batch four, +14.8% ShareGPT; 53 judged prompts 13 to 14 with 26 ties; GSM8K and HumanEval unchanged | [relaxed acceptance](archive/studies/relaxed-acceptance.md), [accuracy](archive/studies/accuracy-benchmarks.md) |

## Graphs, scheduling and memory

| Optimization | Where | Default | Measured | Evidence |
|---|---|---|---|---|
| CUDA graphs for decode and the verify/draft rounds | `sharar/engine/graphs.py`, `spec_graphs.py`, `sharar/proposers/parallel_graphs.py` | On | Batch-one GPU busy 98.4% of the round | [round profile](archive/studies/gb10-round-profile.md) |
| Cached trace storage accounting and HTTP-side latency percentiles | `sharar/engine/traces.py`, `loop.py`, `scheduler.py` | On | Removes per-step file walks and latency sorting; 5,000-feature-file stats call 23.77 ms to 0.208 µs in a CPU microbenchmark; rotation includes feature bytes | [runtime and calibration](archive/studies/w3-calibration-adaptation.md) |
| 2,048-token prefill chunks, 4,096-token prefill budget per step | `sharar/engine/scheduler.py` | On | Bounded time to first token beside decoding requests | [BS4](archive/studies/gb10-bs4-engine.md) |
| Prefix caching and managed-memory cold storage for KV pages | `sharar/memory/`, `sharar/engine/context.py` | On | Warm resumes skip the prefill | [cold tier](archive/studies/cold-tier.md), [agent context](archive/studies/agent-context.md) |
| Checkpoint-owned draft features and state retention for resumed sessions | `sharar/engine/context.py` | On | Resumed batch-one completion TPS 11.3 to 36.7 (3.24x); four conversation sessions 28.8 to 78.5 aggregate TPS | [retention](archive/studies/gb10-agent-retention.md) |
| Host memory guard (16 GiB `MemAvailable` floor) | `tools/memory-guard.sh`, launcher | On | Keeps unified memory from paging the weights | [operations](operations.md) |

## Measured, not integrated

| Candidate | Result | Status |
|---|---|---|
| FP8 GEMM at 16 rows through the GEMV path (`fp8_m16`) | Round cost x0.985 on the 101-family gate | Clean win awaiting integration ([family gate](archive/studies/family-gate.md)) |
| NVFP4 draft body | Round cost x0.971; acceptance not resolved | Needs the trained-draft comparison |
| Prompt-lookup second verify branch | Offline oracle +14 to 50% tokens per round, served +1.4% | Parked on `feat/lookup-branch` ([study](archive/studies/lookup-branch.md)) |

Rejected candidates (narrower verification, mixed forwarding, numerical FP8
replacements, the short-resume scheduler, two-path trees, narrow NVFP4
tiles) are in the [decision index](archive/README.md).

## Combined effect on the standard cell

10K input, 1K output, batch one, identical prompt IDs, mean over the code,
design and reasoning workloads ([comparison](archive/studies/sglang-comparison-2026-09-16.md)):
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
