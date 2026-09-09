# GB10 target arithmetic repairs and benchmark iteration

2026-09-09. Final AIPerf validation is complete. The FP8 draft configuration
is faster on all three measured workloads than the faster of two fresh
SGLang means per task. Full cross-engine output equality is still unresolved. Evidence is in
`/home/yke/sharar-results/reviews/2026-09-08-target-fix`.
The fixes are in the working tree. Production retains its original image;
exclusive offline GPU windows automatically restore the service and guard.

## Implemented corrections

The reference is SGLang `5f55db35e` in image
`sha256:60166a99b6610e2b4b3febbeb6411e4cbc9338e996ab6c80d3339f86a7b5b9c3`.
Both engines use target revision `319f741cce68d7914884900c138a1fbb70a42f30`,
DFlash2 revision `dedf8df68adfb1afeaf7b7480c0a0243108177b4`, and 16-row
verification at concurrency one. The retained Sharar draft is BF16.

* Prefill Gemma norm uses the native CUDA FP32 residual sum and reduction.
* FP4 activation scaling retains the CUDA reciprocal order:
  `rcp(sf * rcp(global_scale))`. Algebraically replacing this with
  `global_scale / sf` changed real FP4 midpoint decisions. Fused SiLU/product
  quantization also retains the BF16 product before quantization. Both
  quantizers preserve signed zero and round the zero/0.5 midpoint to even.
* FP8 weight shards pass through the reference's FP16 dequantization boundary
  before requantization, including single shards. Target FP4 GEMMs use CUTLASS.
* Q/K normalization and rotary embedding retain FP32 rotary tables/products.
* Chunked prefill attends to new BF16 K/V and the cached FP8 prefix separately,
  merging their attention states with log-sum-exp. It writes new K/V to the
  FP8 cache afterward. Mixed batches can include rows without a prefix.
* Verification convolution uses BF16 products with FP32 partial sums and the
  reference Triton activation, replacing BF16 partial sums.
* Compiled verification residual norm computes variance from the FP32 sum but
  reloads a BF16 numerator. Its initial norm uses a chunked 2048-element
  reduction. Its GDN norm promotes epsilon and subsequent normalization math
  to FP64. The implementation now preserves these behaviors explicitly.
  FP8 conversion stays separate: attempted fusion changed midpoint rounding.

The compiler behaviors were checked against generated kernels and TTIR. They
are properties of this pinned compiled profile, not a claim about ideal model
math. Benchmark tools now default to current serving arithmetic (`native`);
explicit `bf16`/`fp32` flags retain historical norm ablations. Reproducing an
older complete result requires its recorded source snapshot.

## Exact comparisons and remaining output gap

The frozen 10K code prefill matches all 320 sampled layer-output/residual
pairs: the last 32 rows of all 64 layers in each of five chunks. First-token
and top-logit probes agree. All 48 initial recurrent/convolution states agree.

With SGLang's proposals and verification boundaries, the integrated target
matches **65 successive blocks / 307 accepted rows** exactly:

* Every accepted hidden-state/residual row in all 64 layers.
* All 3,070 accepted top-10 logits and their token IDs.
* Every committed recurrent and convolution state before each block.

This is an untimed teacher-forced diagnostic. Reference tokens never enter
serving. The replay retains Sharar's own computed features and committed
states; it does not replace those with reference tensors.

Independent complete generations still differ. The retained D source's first
mismatches against the fresh uninstrumented SGLang C run are code index 216,
design 44 and reasoning 924 (zero-based). Sharar's complete 1,000-token code
output and 242 rounds are identical with CUDA graphs disabled, enabled and
then enabled again. The accepted-state comparison does not establish
invariance to different draft proposals or verification boundaries.

Reference capture validation matters. Direct source-instrumented and runtime
launch-instrumented SGLang requests produce identical complete code IDs, but
both differ from the uninstrumented AIPerf baseline at index 40. Running the
launch-instrumented server through the exact AIPerf workflow extends agreement
to index 389; that capture supplies the 307-row comparison above. It records
buffers around already-compiled kernels without editing model forward source.
The later **uninstrumented SGLang D load matches all 1,000 output IDs of the
AIPerf launch capture**, providing a positive full-output validation of that
capture. SGLang C and D themselves differ at code index 389, design 108 and
reasoning 69. The cause of this across-process numerical variation is not
isolated; attributing it specifically to the diagnostic hooks would be
unwarranted. Instrumented latencies are excluded. Full cross-engine greedy
parity remains unproven.

## AIPerf results

All requests use exactly 10,000 frozen input IDs and 1,000 emitted output IDs,
concurrency one, temperature zero, seed 42, neutral penalties, no stops,
ignored EOS and no cached prefix. Each task has one excluded full warmup and
two timed requests. Both engines use the same local relay. The auditor checks
payload equality, prompt IDs, output counts, server usage, length finish,
HTTP success and complete streaming termination.

The final D source is measured in two configurations: retained BF16 drafting
(D) and the existing explicit FP8 draft option (E). Both retain the same target
checkpoint, FP8 target KV, FP32 recurrent state, 60,000 KV pages, 24 state slots,
maximum batch four, 2048-token prefill chunks, CUDA graphs and reconstruction
of accepted recurrent state.

| Task | SGLang C, s | SGLang D, s | Sharar BF16 draft, s | Sharar FP8 draft, s | FP8 reduction vs faster SG mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Code | 35.068999 | 32.061113 | 31.704847 | 26.560758 | **17.16%** |
| Design | 34.738705 | 36.794874 | 33.423270 | 33.110768 | **4.69%** |
| Reasoning | 24.617043 | 26.190059 | 21.366782 | 21.659629 | **12.01%** |

The comparison deliberately uses the faster SGLang mean from the two fresh
loads for each task: D for code, C for design and reasoning. The raw loads and
different continuations remain separate. BF16 Sharar is lower by 1.11%, 3.79%
and 13.20%, respectively; its small code lead is not a broad performance claim.
FP8 drafting reduces code latency another 16.22% relative to D and design
0.93%, while reasoning takes 1.37% longer. Both pass all four quality canaries.
BF16 remains the default; FP8 is an explicit tradeoff, not a uniform improvement
against BF16. The conservative code E/SG ratio is approximately 1.21x output throughput.
The preliminary 24.26% / 1.32x code result used SGLang C alone and is superseded
by the faster D reference in the headline comparison.

E's effective decode-round cost is about 107 ms, versus D's 114.6 ms and the
reference's approximately 129.7 ms. These are `(latency - TTFT) / rounds`, not
new isolated kernel profiles. FP8 reduces reported persistent proposer storage
from 4.10 to 2.66 GiB. Sharar's existing accepted-state reconstruction also avoids
the reference's full per-proposal recurrent-state snapshots. The earlier GPU
trace remains dominated by matrix work; no new hardware-counter claim follows
from these timings.

Each profile's three responses per task are internally identical, including
the warmup. Counts, payloads and stream audits pass. Cross-engine text differs,
so these are complete-configuration latency measurements, **not a matched-output
win**. SGLang changes its continuation across the two fresh processes despite the
pinned checkpoint, image and command. Both loads are internally repeatable.
Two timed samples per task per load do not establish tail latency or a broad
workload guarantee; there is no wide matched-output win claim.

Intermediate C predates the final initial-norm repair. Its means were 28.706,
31.496 and 20.847 s, on different continuations. It is retained as an iteration,
not selected over the more complete D arithmetic because its code timing was
lower. SGLang C also differs from its historical code continuation at index
289; historical and fresh latencies must not be pooled silently.

To enable the tested FP8 draft compute path in a newly built runtime:

```bash
sharar serve --proposer dflash2 --spec 15 --verify-state reconstruct --draft-precision fp8
```

The serving launcher sizes its pools from available memory. The exact fixed
benchmark pools, launch commands, raw client payloads and AIPerf artifacts are
recorded in the evidence directory; the command above does not redeploy the
original production image.

## Additional screens

These are full-generation untimed screens, with 10K input and 1K output IDs.
Changing a draft can change the verification boundaries and continuation, so
round counts alone do not establish a latency improvement.

| Draft with repaired D target | Code rounds | Design rounds | Reasoning rounds |
| --- | ---: | ---: | ---: |
| Retained BF16 draft | 242 | 257 | 152 |
| Existing FP8 draft option | 211 | 272 | 165 |
| Reference-style draft math and FP8 KV | 231 | 264 | 153 |

Neither alternative improves all three round counts. The FP8 option advances
to the AIPerf measurements above because cheaper rounds can offset more rounds.
The reference-style draft experiment is not installed in serving. Its code
output matches the direct instrumented SGLang run through index 763, but differs
from the uninstrumented AIPerf baseline at index 40; this still does not establish
a complete baseline parity oracle.

A history-lookup screen uses only the prompt and already-emitted tokens to
select the longest 8–32-token match within a 4096-token window. Future tokens
are used only to score it afterward. Across 984 positions per task it finds
101/14/146 matches (code/design/reasoning), averaging 4.32/2.21/4.74 correct
proposed tokens on a hit. Coverage and accuracy do not justify another serving
path for these workloads. No lookup policy or benchmark-specific heuristic is
added to the engine.

A final B1 experiment uses the exact two-row compiled residual-normalization
layout at the five feature-tap boundaries. Its complete code output and 242
rounds are identical to retained D, so no additional serving variant is added.

## Validation and provenance

* CPU regression: 373 passed, eight skipped, 10.07 s.
* Quantization/numerics GPU checks: 47 passed.
* Verification convolution and mixed-prefix attention: 40 passed.
* Compiled verification normalization and changed-input graph replay: 19 passed.
* Real draft lifecycle, BF16 and FP8: 22 passed in 125.28 s.
* All five Sharar profiles A/B/C/D/E pass all four quality canaries: exact
  instruction, arithmetic, strict JSON and long-context needle.

Source snapshots and hashes are retained for before, C and D, alongside all
failed screens, raw responses, generated reference kernels and resource logs.
The main replay is `verify_launch_replay.py`; its result is
`verify-launch-aiperf-replay.json`. `native-D-screen.json` contains the complete
independent graph/eager outputs. The final benchmark uses `coordinator_d.py`
and `server_d.py` (BF16), or `coordinator_e.py` and `server_e.py` (FP8).
`final-results.json` contains raw sample summaries and every comparison. The
offline window wrapper owns restoration and memory
monitoring. No checkpoint, driver, GPU clock or production image is changed.

The [machine-readable results](gb10-target-fix.json) include every timed sample,
output hash, comparison and validation scope. The evidence directory's
`session-source-changes.diff` isolates this iteration's 23 Python/TOML file
changes from the preexisting working tree. `manifest.json` and `SHA256SUMS`
cover retained scripts, source snapshots, raw responses and tensor captures;
rebuildable caches are excluded, with relevant generated kernels copied to
`reference-generated`.
