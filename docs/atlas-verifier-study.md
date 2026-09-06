# Atlas comparison and target-verifier experiment

The verifier experiment misses its frozen promotion gates and adds no serving
kernel or runtime switch. Sharar's matched cold single-request decode ranges
from 4.2% slower to 9.6% faster than SGLang; shared-prefix B4 throughput is
7.7–17.6% higher. This is a modest, workload-dependent lead. The Atlas diagnostic
exposes draft-context bugs and cannot support a qualified matched-DFlash2 ranking. The Sharar–SGLang
repeat comparison uses the same frozen requests and numerical profile.
Raw samples, failed setups, experimental kernels and the local Atlas patch stay
outside Git; the compact manifest links the full evidence.

## Verifier result

The unchanged baseline is Sharar `5024fc2231c71d5ff73159db1964c74743920c54`,
using the pinned RadixArk Qwen3.8-27B NVFP4 target and incoai DFlash2 checkpoint,
seven drafts plus the anchor, FP8 target KV and FP32 recurrent state. These
runs use the validated source-overlay image `4a7206b66ead`, not a freshly built
dependency image. Production retains its original image and MTP profile.

The new experiment tests masked small-row FP8 tensor-core tiles, split-K
variants and an eight-row extension of the existing FP8 GEMV. It captures all
176 real target FP8 projections from one natural round at each batch size.
This complements the [earlier NVFP4 study](nvfp4-cost.md); it does not retest
those rejected narrow NVFP4 kernels or call shape tuning a novel algorithm.

The fastest numerically eligible selection reduces complete isolated FP8
projection-family time from **33.857 to 31.538 ms at B=1 (6.85%)**, and from
**33.751 to 33.186 ms at B=4 (1.67%)**. Both miss the predeclared 10% component
gate. Of 176 captured outputs, 104 are bitwise identical at B=1 and 137 at B=4. Maximum
relative L2 is 0.0000753/0.0000215 and scaled maximum error is 0.001825/0.000749,
within the exploratory 0.001/0.01 limits. This is not full-model parity.

A second candidate fuses gated RMS normalization and FP8 quantization, removing
one launch and its intermediate BF16 write/read for each of 48 GDN layers.
It explicitly preserves the intermediate BF16 rounding. All 192 actual layer
inputs across the four context/batch cells produce bitwise-identical FP8
outputs. Separate graph sets isolate FP8-only, fusion-only and combined effects.

| Input context | Concurrent requests | Baseline round | Combined round | Reduction |
|---|---:|---:|---:|---:|
| 8,192 | 1 | 113.419 ms | 111.188 ms | 1.97% |
| 8,192 | 4 | 132.447 ms | 131.521 ms | 0.70% |
| 31,744 | 1 | 118.277 ms | 114.847 ms | 2.90% |
| 31,744 | 4 | 141.251 ms | 139.859 ms | 0.99% |

Four ABBA blocks follow an ABBA warmup in each ablation cell: eight measured
rounds per implementation, 192 measured rounds and 48 warmup rounds overall.
Prefix restoration occurs outside the timing. Each timed round includes real
drafting, target verification, acceptance, commit and draft feedback. Output
IDs and committed lengths agree in every paired sample. The experiment does
not establish full continuation/state equivalence or useful-answer TPS.
Fusion alone ranges from a 0.96% regression to a 0.13% reduction; FP8 alone
saves 2.08–2.40% at B=1 and 0.35–0.51% at B=4.

The limited whole-round gain is consistent with a rough component budget:
the isolated FP8 family sums to 33.9 ms against a 113.4 ms B=1 round. Saving
6.85% of that family predicts roughly 2% of the round if its timings transfer
to graph execution; the full-round experiment measures the actual effect. Removing
normalization launches alone contributes little in the captured graph. These
measurements motivate more useful accepted tokens per weight traversal, rather
than another small dispatch hook.

The combined candidate misses its separately frozen 3% whole-round gate.
There is no runtime flag, dispatch change, promotion repeat or new speed claim
for the shipped engine. Preserve the result for a future combined mechanism;
do not lower the threshold after seeing the measurements. An initial Triton
compile failure caused by integer masked-load fill values is retained; the
corrected floating fill compiles for SM121 and produces the numbers above.

## Matched Sharar–SGLang workload

The GB10 is used exclusively, with an eight-core CPU quota, four OMP threads,
no fixed CPU affinity, a 64 GiB
container limit, an additional 64 GiB host-available-memory drop allowance and
a 16 GiB host floor. Production and its memory guard are restored, with
health/model/canary checks, after every window. Device power, temperature and
SM clocks are recorded; memory-clock telemetry reports unavailable.

Both engines use the pinned RadixArk Qwen3.8-27B NVFP4 target and incoai DFlash2
checkpoint, greedy sampling with EOS honored, seven drafts plus an anchor,
FP8 target KV, BF16 draft body/KV, FP32 recurrent state and a 2K draft window.
Their target pools hold 35,000 tokens and their context limit is 32,768.
Sharar uses 12 recurrent slots and accepted-state reconstruction; SGLang uses
24 Mamba cache slots with its extra-buffer strategy. These are system-level
comparisons, not identical scheduling or numerical-reduction implementations.

The frozen workload requests standalone MinHeap code, tests and explanations.
Historical repository material fills exact 8,192/16,384/31,616-token prompts;
short prompts contain 95–99 tokens. Output budgets are 512 or 1,024 for single
requests. Long-context batches have four 512-token requests after priming a
shared prefix; each branch has exactly the stated input length. The primer is
outside timing. These branches share history and fit a 35K pool; they are not
four independent 32K contexts.

Each engine has two independent load windows and two repeats per cell per
window, with reversed Sharar/SGLang order. All paired input-token hashes must
agree. Cold cells have distinct early nonces and must report zero cached tokens.
A retry uses identical prompt IDs; a cache miss remains a valid system result.
Every request must complete the output budget with attested counts to enter an
equal-work ratio. Short outputs and failures remain in the raw report.

Single-request **decode TPS** excludes the first emitted burst and uses the
remaining observed stream span. **Batch aggregate TPS** divides all generated
tokens by client batch completion time, including prefill. No timestamps are
invented inside a speculative burst. Four samples per cell do not establish
tail latency. This is synthetic coding traffic, not executed agent episodes or
a broad output-quality evaluation.

All **120 timed Sharar/SGLang cells complete their output budgets**. Each
table row uses four paired prompt instances across two independent loads.
Ratios above 1 favor Sharar. These are observed medians, not confidence or tail guarantees.

### Single-request cold generation

| Input tokens | Output tokens | Sharar decode TPS | SGLang decode TPS | Sharar / SGLang |
|---|---:|---:|---:|---:|
| 96–99 | 512 | 49.33 | 47.34 | 1.042× |
| 95–98 | 1,024 | 55.58 | 54.16 | 1.026× |
| 8,192 | 512 | 45.13 | 43.15 | 1.046× |
| 8,192 | 1,024 | 49.38 | 49.37 | 1.000× |
| 16,384 | 512 | 44.17 | 46.12 | 0.958× |
| 16,384 | 1,024 | 50.70 | 50.04 | 1.013× |
| 31,616 | 512 | 43.79 | 39.94 | 1.096× |
| 31,616 | 1,024 | 47.24 | 44.60 | 1.059× |

### Shared-prefix batch throughput

Aggregate TPS counts all 2,048 generated tokens and includes batch prefill.

| Input tokens/request | Output tokens/request | Requests | Sharar aggregate TPS | SGLang aggregate TPS | Sharar / SGLang |
|---|---:|---:|---:|---:|---:|
| 8,192 | 512 | 4 | 147.89 | 125.76 | 1.176× |
| 16,384 | 512 | 4 | 139.88 | 129.82 | 1.077× |
| 31,616 | 512 | 4 | 132.24 | 119.67 | 1.105× |

### First-token latency and exact retries

All cells below request 512 output tokens. The retry reuses identical input IDs.

| Input tokens | Sharar cold TTFT | SGLang cold TTFT | Sharar retry TTFT | SGLang retry TTFT |
|---|---:|---:|---:|---:|
| 96–99 | 0.128 s | 0.212 s | 0.126 s | 0.182 s |
| 8,192 | 3.071 s | 3.388 s | 0.148 s | 0.946 s |
| 16,384 | 6.630 s | 7.198 s | 0.156 s | 1.062 s |
| 31,616 | 14.559 s | 15.643 s | 0.169 s | 0.609 s |

The three short JSON sanity probes are repeated once per load:
Sharar passes 2/3 in each load; SGLang passes 3/3 in each load.
These are formatting/arithmetic probes, not broad coding or agent-quality evidence.
The recorded Sharar failure returns correct numeric fields inside a Markdown fence,
violating the requested JSON-only format. No whole-agent quality claim follows.

Maximum observed host-available-memory drops across the two windows are
**31.52 GiB for Sharar** and **43.75 GiB for SGLang**.
These include allocator/cache effects and are not exact CUDA-allocation totals.
Both remain within the common allowance; production restoration passes after each window.

## Atlas: retained diagnostics, no qualified ranking

Atlas is pinned at `136520b4f54b4d681718c0a6fc2770e5d07e1513` and built with
CUDA 13, its GB10/Qwen3.8/NVFP4 kernel target and locked Rust dependencies.
The downloaded public `latest` image predates its current DFlash2 implementation
and is not used as the representative. No Atlas changes were submitted upstream
and no Atlas kernel was copied into Sharar.

The stock DFlash shared vocabulary path interprets this checkpoint's packed
NVFP4 allocation as BF16, requiring four times its 635,699,200 allocated bytes.
Source inspection found this before attempting that unsafe GPU path. A local
compatibility patch uses Atlas's existing W4A16 projection for an aliased packed
head and disables its invalid BF16-to-FP8 mirror. BF16-backed checkpoints retain
their prior behavior. Six focused Rust tests, formatting and the release build
pass; the patched executable passes three short JSON sanity probes. Those tests
do not establish full DFlash2 correctness or long-context quality.

The completed diagnostic window uses prefill 2,048, utilization 0.47,
69,344 target KV tokens, 12 SSM snapshot slots, four requests, FP8 target KV,
FP32 SSM and a BF16 draft body. `ATLAS_NO_MTP_DRAFTER_CONTEXT=1` disables an
unused MTP capture with a 16K allocation/32K advertised-capacity mismatch.
Its default DFlash capture cap remains 16K. Startup reports 55.1 GiB tracked
live; the maximum observed host-available drop is 60.927 GiB. These are different
measurements, not interchangeable allocation totals.

That window contains 30 timed cells. Twenty-six satisfy the harness's budget
and accounting checks; four shared-prefix batches have watchdog-truncated
outputs. All 24 single-request cells complete their requested budgets. The
following medians each contain two samples and are **diagnostics only**:

| Input tokens | Cold decode TPS, 512 outputs | Cold decode TPS, 1,024 outputs | Cold TTFT, 512 | Exact-retry TTFT, 512 |
|---|---:|---:|---:|---:|
| Short (96–98) | 29.07 | 28.99 | 0.43 s | 0.43 s |
| 8,192 | 18.79 | 20.96 | 12.64 s | 12.60 s |
| 16,384 | 19.68 | 22.91 | 25.91 s | 25.78 s |
| 31,616 | 18.22 | 19.74 | 52.91 s | 52.66 s |

These figures cannot support a matched engine ranking. Source audit identifies:

1. **Wrong chunk capture origin.** The per-stream prefill caller passes the
   attention KV write floor to a helper expecting an absolute token position.
   Cold chunks repeatedly overwrite the start of the draft-feature accumulator,
   then finalization advertises uncaptured rows as valid. Increasing capacity
   alone does not repair this. The alternate kernel-batched path has no verified
   correct capture implementation either.
   [Capture caller](https://github.com/Avarok-Cybersecurity/atlas/blob/136520b4f54b4d681718c0a6fc2770e5d07e1513/crates/spark-model/src/model/trait_impl/prefill_b/forward_layers.rs#L214),
   [capture destination](https://github.com/Avarok-Cybersecurity/atlas/blob/136520b4f54b4d681718c0a6fc2770e5d07e1513/crates/spark-model/src/model/impl_b3.rs#L284).
2. **Missing cached draft features.** Prefix hits restore target KV/SSM but
   allocate a fresh zeroed draft accumulator and capture only recomputed suffix
   rows. No restoration of the skipped five-layer draft history was found.
3. **Unapplied draft window.** The native paged Option B path stores the requested
   `--dflash-window-size 2048` but passes `sliding_window=0` to attention. Its
   separate context-window setting bounds scratch chunks, not attention history.
   A legacy debug path lacks DFlash2 convolution handling and is not a faithful
   workaround.
   [Native attention call](https://github.com/Avarok-Cybersecurity/atlas/blob/136520b4f54b4d681718c0a6fc2770e5d07e1513/crates/spark-model/src/layers/dflash_head/forward_block_layer_paged.rs#L668).

The definitive source audit in the evidence records caller/callee references,
B1/B4 routing, an independent CPU index reproduction and conceptual repairs.
It supersedes the earlier incomplete hypothesis that only the 16K cap caused
missing history. These source defects do not prove that any particular malformed
target output was caused by them. A correct speculative verifier can preserve
target outputs despite a poor drafter; that property was not qualified here for
these numerical modes.

Atlas also reports **radix-matched tokens**, not necessarily skipped prefill, in
`cached_tokens`. The immediate 8K retry finds an exact snapshot but declines it
under the default exact-prompt policy, without falling back to the shallower
checkpoint. Its generic “no SSM snapshot” log does not establish eviction.
Shared branches instead restore the 8,160-token intermediate checkpoint.
Measured first-token time is therefore needed to establish actual cache benefit.

The four failed shared-prefix batches contain outputs of 337, 49 or 65 tokens
instead of 512. The 337-token server trace explicitly records a content-loop
watchdog stop; Atlas maps non-timeout guards to `length` and omits the guard name
from its completion stream. This is not EOS, context exhaustion or demonstrated
stream loss. Complete-budget output alone does not certify coding quality.

### Different arithmetic despite shared input checkpoints



| Projection | Sharar | Atlas, 8 target rows | Atlas, 32 target rows |
|---|---|---|---|
| GDN | FP8 weights/activations | Native FP8 weights expanded for BF16 MMA; BF16 activations | Same |
| MLP | NVFP4 weights/activations | Packed NVFP4 weights, BF16 activations, scalar FP32 accumulation | Dequantized weights and activations converted to E4M3 for FP8 MMA |
| Full attention | Native FP8 weights/activations | FP8 weights requantized to NVFP4; BF16 activations | Same requantized weights, E4M3 operands for FP8 MMA |
| Target vocabulary | NVFP4 weights/activations | Packed NVFP4, BF16 activations | Packed NVFP4, E4M3 operands for FP8 MMA |
| Draft vocabulary | NVFP4 weights/activations | Patched conservative NVFP4-to-BF16 projection | Same |


The conservative patched draft vocabulary projection is not claimed to be
Atlas's fastest possible packed-head implementation. Target full-attention
weights are requantized at load. The pinned
[Atlas source](https://github.com/Avarok-Cybersecurity/atlas/tree/136520b4f54b4d681718c0a6fc2770e5d07e1513)
and retained dispatch audit establish these choices: `gemv_tier.rs` selects M8;
`dense_ffn.rs` and attention select M32 tiles whose implementation explicitly
converts operands to E4M3. Function names alone do not establish precision.

### Retained setup failures and amendments

| Attempt | Outcome and disposition |
|---|---|
| Original review-prompt pilot | Sharar often emitted short tool calls with EOS honored. Replaced prospectively by the frozen standalone coding fixture; no competitor timing claim from this pilot. |
| SGLang native API setup | `seed` rejected before any timed cell. Adapter corrected to `sampling_seed`; exact pinned parser validated, inputs and limits unchanged. |
| Atlas 8K prefill arena / .50 utilization | 90,928 target KV tokens, 58.8 GiB startup ledger. Host drop reached 64.52 GiB during 16K B4 warmup; stopped by the common guard before timings. |
| Atlas 2K prefill / .47 | 69,344 target KV tokens, 55.3 GiB startup ledger. Near-32K warmup exposed the unused MTP capture overrun; no timed cells. |
| Atlas with unused MTP context disabled | Completed the diagnostic window above. Later source audit found the DFlash capture/window defects. |
| Atlas cap 32K / .4602 | Startup confirmed 35,712 target KV tokens and 53.9 GiB tracked live. Stopped during warmups once the wrong-offset source bug was established; zero timed cells. Exit 137 records explicit cancellation, not OOM. |

The final cancellation stopped only the isolated container while its live guard
restored production. The earlier review-pilot cancellation had also stopped its
child guard; restoration was explicitly rerun and verified. The corrected runner
supports a checked stop-after-window marker and keeps per-window canary logs.
All failed attempts, commands, hashes and restoration records are retained.

## Decision and proof

Do not promote the verifier kernels or add rejected experimental hooks. The
next speed mechanism is a prospective 16-target-row draft-tree coverage study:
more useful accepted tokens per target traversal, followed by an isolated
verifier only if coverage and measured cost justify it. Require at least 10%
useful-turn completion reduction, output/state checks, protected B4 behavior and
the same memory floor before serving integration. Correct output contracts and
DFlash2 default promotion remain separate gates.

The full CPU suite passes **319 tests, six GPU-only skips**. The verifier's GPU
round checks and limits are recorded above; no new serving code is introduced.
Production retains its original image and MTP profile. The compact result
[manifest](atlas-verifier-results.json) records frozen inputs, image/source identities, the complete artifact
index, raw-report hashes, tests and final restoration checks.
