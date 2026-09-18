# Calibrated W3 QKV row tiling

2026-09-17. Base source `ddd82af1027060758cf776ff1b62f3b99f2d2936`.
Raw evidence: `${SHARAR_RESULTS}/reviews/2026-09-17-w3-qkv-gate/`.
The [compact manifest](w3-qkv-tiling-results.json) identifies the frozen sources, plans, outputs and logs.

**Decision: retain the narrow QKV specialization for its repeatable decode-cost gain. No general HTTP throughput improvement is established.**

## Scope and mechanism

This follows the targeted candidate in the [W3 runtime profile](w3-runtime-profile.md). Its initial
screen was insufficient to distinguish the small whole-round gain from
control drift. The new comparison holds the workload and per-round target
work fixed, interleaves the arms and repeats them in a fresh process.

The CUDA specialization handles exactly `(M,N,K) = (64,14336,5120)`, scalar
or per-row activation scales, and one K split. It gives adjacent blocks the
first and second 32-row halves. Logical verifier rows, proposal width and
each row's K reduction remain unchanged. All other shapes, grouped scales
and split-K launches retain the original dispatch. A short prefill with the
same matrix shape also uses the specialization.

There is no new persistent allocation, scratch requirement, transfer or
synchronization. Both halves use the existing packed weights and current
CUDA stream; graph replay keeps its existing pointers. Register allocation
falls from 198 to 146 registers per thread for this static-scale QKV shape.
Neighboring blocks can reuse weights through L2, while each carries fewer
row accumulators. Register allocation supports this explanation; this study
does not measure achieved occupancy or memory bandwidth. The earlier
188.37 to 145.49 microsecond isolated timing predicts about 0.69 ms saved
over 16 QKV launches, not a comparable percentage gain for an entire round.

## Paired protocol

One GB10, calibrated target manifest
`9585ba8c682de346e526aeed0e0c9ba5ec34b341d7e8bfb382da0b81f60991b8`,
selected step-200 adapted FP8 DFlash2 drafter, k=15, reconstructed accepted
state and exact acceptance. Weight values, sampling, acceptance and the
existing CPU-staged calibrated loader are the same in both arms. The image
is `sha256:c8b70c24bd7d6de8860f751af5463f5af33567c9cbff7c52084d2e6b48633bea`,
with the checkout mounted separately. Recorded software: PyTorch 2.13.0+cu130,
CUDA compiler 13.0.88, driver 580.95.05 and Python 3.12.3.

The frozen family-v3 input checksum is
`b72bdecd50c2fb0a7a00eda1433bb697a305160b8f6e72c1c74cb0b3530124a3`.
Twenty-six four-request groups cover all 101 families and three reporting
prompts. An additional reporting-only group is excluded from the primary
interval. Six families provide six batch-one and three batch-two control
groups. The exact membership was frozen before the run in `plan.json`.

Each independent process uses one model, 60,000 KV pages and 24 state slots,
and captures both verification graph sets. Capture order reverses in the
second process. Within each group, four runs alternate ABBA or BAAB; the
starting arm also reverses between processes. Both arms are warmed up.

Each group primes its original 10,000-token prefixes once and appends the
primer's actual first output token for the measured request. Each run emits
128 tokens, does not replace the retained primer, and asserts full target
prefix hits, draft-cache hits and zero draft fallbacks. Timing excludes the
primer and the tiny resume prefill. This is a cached-prefix decode-cost gate;
the separate HTTP check includes prefill and request handling.

The analysis compares paired throughput and mean round cost, averages log
ratios across process blocks within each group, then bootstraps groups with
20,000 resamples and seed 20260917. Four concurrent requests are one group,
not four independent observations. Native-repeat ratios expose control
variation. Output IDs and per-round batch, physical rows, emitted counts and
accepted counts are checked across all four arms.
The interval resamples workload groups within these two measured process
blocks; it does not estimate variation across machines or days.

An assertion for round-work equality was added after process zero imported
its driver. This only adds a post-timing assertion and report field; it does
not alter execution, inputs or timing. Both driver versions are archived,
and the analyzer independently checks the raw work events for both blocks.

## Decode-cost result

Across both processes, batch-four decode throughput improves **0.66%**,
with a paired 95% interval of **0.61–0.71%**. Mean round cost falls 0.65%.
Every primary batch-four group has a positive paired speed ratio in each
process. The independent process ratios are 1.00743 [1.00657, 1.00852] and
1.00573 [1.00510, 1.00635].

| Batch | Groups | Candidate/native decode TPS | 95% paired interval |
|---|---:|---:|---:|
| 1 | 6 | 1.00094 | [1.00004, 1.00167] |
| 2 | 3 | 1.00138 | [0.99927, 1.00273] |
| 4 | 26 | 1.00658 | [1.00611, 1.00714] |

The batch-four native-repeat ratio is 1.00033 [0.99932, 1.00122]. The
batch-one and batch-two controls show no material slowdown; their small
samples are protection checks, not evidence of a general speedup at those
batch sizes. All 36 groups in each process preserve outputs and round work.
This is a modest cost reduction for the selected calibrated profile, not
an acceptance improvement or a full-server throughput claim.

## HTTP protection with bounded pools

Four fresh processes run native, candidate, candidate, native through the
real HTTP server at concurrency 1, 2 and 4. Each excludes one warmup and
measures cold 256-token requests, three cached turns of 128 tokens with a
controlled 128-token tool suffix and 0.1-second delay, and mixed cold/cached
arrivals. Both arms use 100,000 KV pages and 24 state slots. All processes
complete every cell with zero draft fallbacks and pass four natural-stop
quality canaries. These are synthetic serving fixtures, not agent task-success
measurements. The pairs below are the two fresh-process samples, in tokens/s.

| Workload | Concurrency | Native TPS | Candidate TPS |
|---|---:|---:|---:|
| Cold | 1 | 31.26 / 31.19 | 31.28 / 31.30 |
| Warm | 1 | 137.07 / 136.57 | 136.72 / 137.06 |
| Cold | 2 | 36.40 / 37.07 | 36.43 / 36.48 |
| Warm | 2 | 223.40 / 230.33 | 237.01 / 223.45 |
| Mixed | 2 | 64.52 / 64.52 | 64.32 / 64.37 |
| Cold | 4 | 42.54 / 41.94 | 42.32 / 41.28 |
| Warm | 4 | 116.20 / 172.85 | 337.79 / 117.25 |
| Mixed | 4 | 111.14 / 111.21 | 111.39 / 111.23 |

Cold concurrency-two/four outputs and round counts differ across these
asynchronously admitted requests. The cold-four candidate mean is 1.05%
lower, while the native repeats differ by 1.42% and candidate repeats by
2.45%; this small set does not resolve a cold-serving improvement or
regression. Exact per-row kernel equivalence does not make HTTP arrival
order or the resulting sequence of batch shapes identical.

At warm concurrency four, native runs lose two and one cached prefixes;
candidate runs lose zero and two. Corresponding prefill work is
22,316 / 12,060 native tokens versus 1,548 / 22,316 candidate tokens.
The large apparent candidate advantage is therefore not credited to the
kernel. All warm outputs still match. Cache, prefill, emitted-token, batch
mix, TTFT and stream-gap samples are retained for every arm; small request
counts do not establish tail latency. The live service has 96 state slots,
so a separate warm comparison at that capacity follows this pressure check.

## Warm serving with 96 state slots

After observing the misses, `capacity-plan.json` freezes a separate native,
candidate, candidate, native follow-up with the live service's 96 state
slots. It keeps 100,000 KV pages, exact acceptance and the same synthetic
warm fixture, so it is not a full reproduction of the deployed configuration.
Every request must reuse its entire committed prefix, leaving exactly 129
uncached tokens. Both measured and warmup requests pass that assertion.

The measured native samples are 339.00 / 337.95 tokens/s; the candidate
samples are 332.94 / 340.34 tokens/s. Every arm performs exactly 1,548 prefill
tokens across twelve requests, with identical outputs. The candidate mean
is 0.54% lower; its two repeats differ by 2.22%, while native repeats differ
by 0.31%. Maximum observed stream gaps are 0.364 s native and 0.361 s
candidate. All four processes pass their quality canaries.

This follow-up removes the observed prefix-loss confound and finds no
repeatable serving benefit or material slowdown. Its small sample does
not resolve the sub-percent kernel effect through the HTTP scheduler. Keep
the controlled decode-cost result separate from these serving measurements.

## CPU background load

Four CPU workers each repeatedly add two 64 MiB float32 arrays, sleeping
50 ms between additions. They run throughout model setup and a separate
four-group paired process; all workers exit cleanly after approximately
160 seconds. This is moderate synthetic memory traffic, not recorded tool
execution.

The two batch-four groups improve by 0.68% and 0.48%, respectively.
Batch-one and batch-two control ratios are 0.99917 and 0.99856. All outputs
and per-round work remain exact. This small descriptive check supports the
idle-machine result under the tested interference; it does not establish a
new family interval or behavior under every CPU workload.

## Correctness and limits

The original CUDA source remains an executable control in the proof bundle.
Across six target matrix shapes, eight row counts and three scale layouts,
all 144 calibrated-weight cases match it bit for bit, eagerly and through
three graph replays. Inputs include finite FP8 outliers. Two new regression
tests also use distinct row halves, nonzero storage offsets, scalar/per-row
scales, changed graph inputs and an independent dequantized matmul reference.

The affected GPU suite passes 59 checks. A separate full-model comparison
uses budgets 2, 6, 9 and 17 to shrink batches and compares emitted IDs,
finish reasons, committed lengths and all recurrent/convolution state
tensors after each round. Greedy and mixed-sampling comparisons are exact.
Final CPU checks pass 540 tests with eight skipped; distribution assets and
installed-runtime probes pass.

The row tiling preserves each row's arithmetic. This run does not change
weights or acceptance, and does not repeat or extend the earlier calibrated
target's GSM8K/HumanEval quality claim. Exact kernel and state comparisons
establish execution equivalence on the covered inputs; the family timing
gate separately decides whether the optimization saves work time.

## Build identity and final verification

The initial experiment and the serving image shared a Torch extension
build directory. Restoring older image timestamps can let Ninja reuse an
object from a different source revision. The experiment's loaded W3 cache
was preserved under a separate path, forcing the original service to rebuild;
subsequent benchmark windows use an isolated `TORCH_EXTENSIONS_DIR`.

The W3 loader now includes the CUDA source digest in its compiled module
name. A CPU regression covers changed source bytes with identical sizes and
old timestamps, including rollback to the original contents. This startup
change was added after the performance comparisons. The CUDA source itself
is unchanged: `1dce452331124d23dff06b0565fc2fde46ae95270dd7466b3fc39f7f9888f0f6`.
The final loader reruns all 59 affected GPU tests and all 144 original-kernel
comparisons successfully. A fresh one-group paired smoke has identical
outputs/work and a 1.00503 throughput ratio; it is excluded from the primary
interval. The final distribution and complete CPU checks also pass. A
sandboxed CPU attempt was interrupted before the successful run with local
socket access; both logs are retained.

The retained change is a small, exact execution-cost improvement under the
measured calibrated profile. Cold HTTP results and cache-pressure anomalies
remain in the evidence, without turning their throughput swings into a
kernel gain. Further speed work should measure the large grouped W3 MLP
kernels before choosing another implementation change.

The minimum host `MemAvailable` across the qualification windows is 56.04
GiB, above the 16 GiB floor. The original service is healthy with its memory
guard active. Restoration verifies the image, original CUDA source digest,
original kernel specializations and the inode of the binary actually mapped
by the service. The QKV and build-cache fixes remain local source changes;
they have not been committed, pushed or activated in the live image.
