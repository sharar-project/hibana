# Runtime cleanup, W3 calibration and drafter adaptation

2026-09-17. Evidence root:
`${SHARAR_RESULTS}/reviews/2026-09-17-w3-calibration/`.
Source starts at `bf1be81`; the [evidence manifest](w3-calibration-adaptation-results.json)
links the full results, source archive, per-file hashes, container commands,
server descriptions and memory traces.

## Runtime changes

The engine previously walked the trace directory, including every feature
file, whenever it published scheduler statistics. The trace writer now owns
the file inventory and byte accounting; the engine reads an immutable pair
of counters. Startup scans existing files once, and write failures reconcile
the current file on the writer thread. Rotation now includes feature and
top-k bytes; counting only JSON previously allowed the current file's blobs
to grow past the retention budget. The retention regression test reproduced
125,239 bytes retained under a 40,000-byte cap before this fix.

Time-to-first-token percentiles are now computed on the HTTP side from an
immutable snapshot copied when requests finish. The engine publishes plain
counters without sorting its latency history on every step.

With one synthetic trace file and 5,000 empty feature files, the previous
directory scan took a median 23.77 ms; cached trace stats took 0.208 µs.
For 2,048 latency samples, the full summary took 73.68 µs and the engine's
counter-only summary took 0.416 µs. These are CPU call timings, **not measured
end-to-end throughput gains**. `runtime_microbench.py` and
`runtime-stats-samples.json` preserve the method and raw samples.

W3's NVFP4 dequantization helpers moved from `bench.verifier_numerics` into
`sharar.weights.modelopt`, removing a runtime dependency on files absent from
the wheel. Distribution checks now exercise installed W3 imports, its CUDA
source asset and dequantization outside the checkout.

## Target calibration

The candidate retains the current W3 kernel, eight-level E4M3 codebook,
32-weight groups and E4M3 group scales. An activation-weighted clipping grid
and GPTQ column-error updates fit codes from the pinned block-128 FP8 release.
The original NVFP4/FP8 prefill weights and NVFP4 vocabulary head stay in use.
The [calibration guide](../../w3-calibration.md) covers fitting and loading.

The frozen corpus contains 240 training, 60 validation and 40 final test
requests over Sharar source files. Splits use different source files, and
benchmark/parity sources are excluded. This is a synthetic coding proxy,
not recorded user traffic. Calibration uses 64 training files and 53,702
input tokens, excluding both held-out splits and task-accuracy datasets.

All 304 projections were fitted, producing 9,898,500,352 bytes of tensor
files. The median projection fitting-error ratio against nearest rounding
from the FP8 source is 0.200; the largest ratio is 0.470. These diagnostics
reuse calibration rows and do not establish task quality.

The first full attempt failed at the first attention layer after completing
three layers. The offline driver now synchronizes between sequences before
reusing FlashInfer's planner staging buffers for a different sequence length.
The resumed run completed in 1,056.8 seconds, with at least 82.92 GiB of host
memory available. The failed attempt and its logs are retained.

The completed artifact's manifest SHA256 is
`9585ba8c682de346e526aeed0e0c9ba5ec34b341d7e8bfb382da0b81f60991b8`.
The evaluation server verified all file checksums and reported this identity.

## Target evaluation

The baseline uses the actual production profile: W3 target layers and draft,
FP8 DFlash2 k=15, accepted-state reconstruction, and the NVFP4 vocabulary
head. Accuracy requests explicitly select exact acceptance and disable
thinking in both arms.

| Task | Current W3 | Calibrated W3 |
|---|---:|---:|
| HumanEval, 164 tasks | 147 / 164 (89.63%) | 150 / 164 (91.46%) |
| GSM8K, 1,319 tasks | 1,214 / 1,319 (92.04%) | 1,246 / 1,319 (94.47%) |

HumanEval has five newly passing tasks and two newly failing tasks; paired
McNemar p=0.45. The small gain is not statistically conclusive by itself.
GSM8K has 45 newly correct answers and 13 newly incorrect answers, a gain
of 2.43 percentage points with paired McNemar p=0.0000301. The accuracy
window kept at least 28.84 GiB of host memory available.

The full family-v3 gate ran nearest, calibrated, nearest in separate GPU
windows, with fixed pools (60,000 KV pages and 24 state slots), 10,000 input
tokens and 1,000 output tokens per task. Every pass completed all 105 tasks.

| Cohort | Decode ratio, calibrated / nearest | 95% interval |
|---|---:|---:|
| 101 task families | 1.0862 | [1.0513, 1.1230] |
| Four reporting prompts | 0.9519 | [0.8796, 1.0302] |
| All 105 prompts | 1.0807 | [1.0464, 1.1164] |

Across the task families, tokens per round rose by 8.38% [4.88%, 12.06%],
while round cost was essentially unchanged (ratio 0.9979). End-to-end
throughput rose by 6.76% [4.05%, 9.60%]. The four reporting prompts alone do
not establish a gain or regression; their interval includes one.

The repeated nearest baseline reproduced every output exactly and differed
in decode speed by 0.13% [0.086%, 0.179%]. Calibrated outputs differ from the
baseline on every prompt, which is why task accuracy is evaluated separately.
At least 68.34 GiB of host memory remained available during the gate.
The calibrated target passes both the quality and performance gates.

CPU checks: 539 passed, eight skipped. Installed distribution checks pass.
GPU checks: 55 passed, covering calibrated packing and execution, graph
replay, accepted-state reconstruction, draft feedback ownership and
bit-identical BF16/FP8 draft reload. Reload also preserves the W3 planes,
scales and row-scale addresses.

## Drafter adaptation

Fresh traces were collected from the calibrated target at concurrency four,
with exact acceptance and thinking disabled: 240 training requests / 87,157
output tokens, 60 validation requests / 22,390 tokens, and 40 final test
requests / 14,477 tokens. Every request retained its projected features and
target top-k labels. The split was checked against frozen prompt identities;
the trainer's newest 60 requests exactly match the validation files. Trace
collection kept at least 26.74 GiB available.

The 300-step run used seed 20260917, learning rate 1e-4, 128 anchors per
sequence, two sequences per step, and the existing attention training mode.
It updated 403,963,520 parameters while retaining the frozen context
projections. Training steps took 860.6 seconds, excluding validation and
checkpoint writes; minimum available memory was 68.64 GiB.

| Checkpoint | BF16 validation tokens per round |
|---|---:|
| Pinned baseline | 4.863864 |
| Step 100 | 5.088879 |
| Step 200, selected | 5.147533 |
| Step 300 | 5.097010 |

The selected checkpoint improves the validation proxy by 5.83%. Selection
uses only validation; the final 40 requests, from 20 different source files,
are reserved for the actual FP8/W3 serving comparison.

Serving ran pinned, trained, pinned, trained through the supported reload API,
at concurrency one and four in each arm. Pools and captured graph addresses
were unchanged, each cell flushed the prefix cache, and one warmup request was
excluded. All 320 measured requests completed their fixed output budgets with
zero relaxed accepts. The target was calibrated W3 with the NVFP4 vocabulary
head in every arm.

| Concurrency | Pinned mean tokens/s | Trained mean tokens/s | Ratio |
|---|---:|---:|---:|
| One | 66.50 | 68.90 | 1.0360 |
| Four | 157.29 | 164.26 | 1.0443 |

A paired bootstrap of the 20 source-file clusters (20,000 resamples, seed
20260917, averaging repeated runs within each arm) gives tokens-per-round
ratios 1.0369 [1.0137, 1.0600] at concurrency one and 1.0565
[1.0300, 1.0879] at concurrency four. Inverse request-latency ratios are
1.0360 [1.0144, 1.0575] and 1.0493 [1.0257, 1.0783], respectively. The
concurrent request-latency interval is **not** an aggregate-throughput
interval. These are results on a synthetic coding proxy, not user traffic
or evidence of a universal drafter improvement.

At concurrency one, each drafter repeated all 40 outputs exactly; control
throughput differed by 0.39% and trained throughput by 0.13%. Concurrency-four
outputs repeated on 10/40 control and 8/40 trained requests; throughput varied
by +1.02% for the control and -3.18% for the trained drafter. The trained
drafter was faster in both paired runs. The serving window kept at least
31.87 GiB available.

HumanEval with the selected drafter passes 152/164 tasks (92.68%), versus
150/164 for the calibrated target with the pinned drafter: three newly passing
and one newly failing task, paired McNemar p=0.625. This is an inconclusive
small gain and supplies no evidence of a coding-quality regression. GSM8K
was evaluated for target calibration, not rerun after drafter adaptation.

## Qualified artifacts and activation

The runtime, calibrated target and step-200 drafter pass their gates. The
deployment image is `sharar:w3-calibration-20260917`, SHA256
`c8b70c24bd7d6de8860f751af5463f5af33567c9cbff7c52084d2e6b48633bea`.
All 292 packaged code/data files match the tested checkout. The frozen runtime
source archive contains 331 files and has SHA256
`7c4f2fe336050d310013d8e19c5ed360a597f91f8508674a8a83577bece608ce`;
evolving documentation is excluded from that archive.

The completed target is staged at
`${SHARAR_RESULTS}/w3-checkpoints/20260917-gptq-fp8/`. The selected drafter is
`${SHARAR_RESULTS}/reviews/2026-09-17-w3-calibration/draft-run/best/`.
The activation script preserves the original container, configuration and
draft symlink and restores them if startup or generation checks fail.

The local service was activated with explicit user approval on 2026-09-17.
`/get_server_info` confirms the calibrated target identity and the promoted
FP8 k=15 drafter at `/draft`; the running image matches the qualified image.
An exact-acceptance, non-thinking chat request returned the expected Python
addition function, and the memory guard is active. `deployment.json` records
the live configuration, identities and smoke response. The original container
is retained, stopped, as `sharar-qwen38-before-w3cal-20260917`, with its
configuration backed up under the evidence root's `deployment/` directory.
