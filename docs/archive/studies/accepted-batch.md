# Batched accepted-feature feedback

> Dated evidence for the source and protocol recorded below. See the [current documentation](../../README.md) for supported behavior and priorities. Historical commands may require the recorded revision.

The next [direct accepted-state reconstruction experiment](direct-state.md)
removes the final FP32 output scratch and commit copies. Measurements below
remain the preceding feature-batching baseline.

This follow-up shares DSpark/DFlash2 feature and context-KV projections across
the requests in a completed speculative round. It applies the same 350 MiB of
BF16 projection weights in one batch instead of four separate calls at B=4.
This is logical weight-operand reuse; physical DRAM traffic was not measured.
The runtime changes only the feedback contract and proposer implementations.
Target forward math, StateManager ownership, graphs and serving defaults stay
unchanged. MTP/full snapshots remain the production defaults.

## Contract and implementation

`SpecDecoder` finishes every target commit and output-history update before
calling `accepted_batch(requests, emitted, features)`. Each row has its own
stop-truncated count; the last emitted token remains pending. The adapter:

1. Checks counts, private prefix positions, unique draft slots and feature bounds
   before any private KV writes.
2. Concatenates only committed feature rows, excluding verify padding, then
   projects them together. One asynchronous metadata upload describes absolute
   positions and destinations in the existing private feature/KV rings.
3. Runs each draft layer's KV projection once and scatters by request ownership.
   It updates endpoints, pending tokens and any endpoint cache snapshots afterward.

Work is bounded by B*(k+1), at most 32 rows for the current external-draft
profile, independent of cache capacity. There are no new persistent buffers or
weight copies; these small temporaries fit within the existing activation
reservation, which already covers much larger prefill projections.

Single-request feedback retains its original arithmetic. A batch with an
interior checkpoint or a chunk exceeding the ring window falls back to the
sequential helper, preserving snapshots before history can be overwritten.
Ordinary decode/prefill feedback retains its previous path. MTP exposes the
batch callback by looping over its existing teacher-forced calculation;
n-gram feedback remains empty. There is no additional scheduler abstraction.

The proposer contract now requires `accepted_batch`, replacing the old public
per-request `accepted` method. Contributors can begin with a loop inside this
callback, then optimize their own adapter without moving target ownership.
See [interfaces.md](../../interfaces.md) for the feature/commit lifetime contract.

## GB10 comparison

Baseline: `684f62f22a91cffcb5d944bb13f11888d34aacf5`, including the prior GPU
selection kernel. Same pinned target/DFlash2 revisions, k=7, reconstruction,
graphs, 35,000 target pages, 12 state slots and B<=4. The final confirmation
alternates candidate/control/control/candidate on the same fresh model and
graphs. Each cell uses one warmup block, then three measured blocks: six
eight-round samples per method, with fresh requests for every sample.

| Context / batch | Control ms/round | Batched ms/round |
|---|---:|---:|
| 128 / 1 | 110.60 | 110.85 |
| 128 / 4 | 136.48 | 130.58 |
| 8,192 / 1 | 113.62 | 113.88 |
| 8,192 / 4 | 143.57 | 137.27 |

All emitted IDs match across methods and repetitions in these fixed-work
cells. B=4 round time falls 4.3–4.4%; B=1 changes by less than 0.3%.
The earlier separate-process pilot agrees: B=4 improves about 4.2%, with B=1
within 0.4%. In that pilot, B=4 feature projection and context-KV work fall
from approximately 8.1 ms to 2.2 ms, with their call counts falling from four
to one per round.
CUDA-event spans include queue gaps and are diagnostic; uninstrumented wall
timings above are the speed comparison. Neither is a DRAM-counter measurement.
Every two-round trace records two CUDA stream synchronizations in both modes:
one target-acceptance wait per round. Batching adds none. The three explicit
device synchronizations in those traces belong to the profiling harness.

Both separate-process HTTP pilots complete 78 requests without request or
automatic coding/tool/JSON/reasoning failures. All non-mixed token sequences
match, including sampled, cancellation and short-budget cases. Four
mixed-arrival responses differ; one chat continuation grows from 67 to 73
tokens. Preserve those outputs instead of treating these as equal-token work.
Mixed coding/chat/JSON completion medians fall 1.7–2.3%; mixed session rises
1.3%. Other main completion changes range about -0.8% to +1.5%, and the
short-budget group rises 2.3%. This is a small pilot, not a general latency or
10% end-to-end speed claim.

Assistant review covers all 26 rubric-scored outputs per mode. Both retain
22 passes and the same four sampled-chat misses: the analogy omits explicit
recomputation. Changed mixed outputs meet their existing rubrics. These are
bounded local checks, not broad model-quality certification or an SGLang lead.

DSpark also completes all 78 frozen HTTP cases in the standalone candidate
image, without request or automatic-check failures. Assistant review of its
26 rubric-scored outputs records 22 passes and four sampled-chat recomputation
misses. This is a shared-adapter validation run; no DSpark before/after speed
comparison was performed here. Reviews are not independent human assessments.

## Numerical and ownership gates

Batching changes BF16 GEMM shapes, so multi-request feedback is not claimed
bitwise identical to separate projections. Tests compare both implementations
with FP32 matmuls (TF32 disabled), rounding at the same BF16 projection
boundaries and retaining the same normalization/RoPE equations. The gate was
fixed before execution: relative RMS below 1% and cosine above 0.9999 for
projected features and all five layers' K/V. Across both pinned adapters and
B=1/2/3/4, the observed maximum relative RMS is about 0.13%; B=1 is bitwise
equal to the old path. The exact stochastic verifier still uses the actual
conditional draft law, including any numerical changes.

CPU tests use distinct row markers and poisoned padding to check mixed counts,
ring wrap, cache snapshot contents, borrowed-buffer lifetime, and rejection of
bad batches before private writes. GPU tests use real target features, inspect
neighboring KV guards and cached feature copies, and exercise EOS inside a
verified block, mixed sampling, eager/graphs, cache hits and full page/slot
recovery for DSpark and DFlash2. The common MTP/n-gram and recurrent-state
regressions remain required because feedback now runs after all target commits.

The standalone GB10 image passed the full **497-test suite** in 947 seconds,
with no failures or skips. This includes both snapshot/reconstruction strategies
and their bitwise committed-state checks, existing MTP/n-gram and draft lifecycle
coverage, the eight projection-oracle cases, and eight external-draft EOS/cache/
mixed-sampling cases. These are implementation and bounded numerical gates;
the broader quality and competitive promotion gates remain separate.
The final CPU suite passed **185 tests**, with six GPU-only skips.
Available host memory stayed above **75.71 GiB** throughout full GPU validation,
confirmation and the DSpark workload run, against the unchanged 16 GiB floor.
The original production container and memory guard were restored, and
health/model/canary checks passed afterward.

## Next mechanism

The next state experiment is StateManager-authorized reconstruction directly
into validated committed slots. It targets the final 576 MiB FP32 scratch and
roughly 5 ms of B=4 copying measured in the prior profile. Validate every count
before mutation, preserve read-before-write tile ownership, and retain the
bitwise stop/cache oracle. This optimization is not implemented here.

For larger mixed-workload gains, profile shared target projection work across
prefill and verification before changing scheduling. Equal-memory cache
pressure, broader sampled quality and fresh competitive SGLang comparisons
remain open promotion gates.

## Reproduction and evidence

`bench/draft_profile.py --output NEW_DIRECTORY --workloads` records fixed-work
timing, component events, traces, source fingerprints and frozen HTTP outputs.
`--compare-accepted PATH_TO_BASELINE_PARALLEL_PY` alternates the old per-request
method and new batch callback on a fresh model, candidate/control/control/
candidate, with three measured blocks after warmup. It asserts equal fixed-work
outputs. `--name dspark --workloads-only` runs the shared adapter's HTTP gate.

The [evidence manifest](accepted-batch-results.json) links the full archived bundle by SHA-256. Full local evidence is under
`/home/yke/sharar-results/reviews/2026-09-05/accepted-batch/`, including source
snapshots, commands, memory/power logs, numerical reports, test XML, raw HTTP
outputs and review decisions. Run GPU commands in isolation with the 16 GiB
watchdog and restore serving afterward. The standalone candidate image is
`sha256:67cb916cc0355932bb011e754f0dfc485c09a6f57a22ddbcaf9c6a5aeb1facc6`.
