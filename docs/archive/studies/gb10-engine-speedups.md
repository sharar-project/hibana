# Exact-arithmetic GB10 engine speedups

> Dated evidence for the source and protocol recorded below. See the [current documentation](../../README.md) for supported behavior and priorities. Historical commands may require the recorded revision.

2026-09-08. This follow-up implements and measures draft and target execution
changes against a fresh Sharar baseline. The paired AIPerf evidence is under
`${SHARAR_RESULTS}/reviews/2026-09-08-engine-speedup/`.

The code request's mean latency falls from 35.144 to
33.803 seconds, a **3.82% reduction**.
All measured requests and full-generation warmups preserve the previous
Sharar baseline's exact 1,000 output IDs and decoded text. Verification-round
counts also stay unchanged. These are savings on the same generated work.

This does **not** establish a wide win over SGLang. Its earlier pinned AIPerf
code measurement was 30.904 seconds; the optimized Sharar mean is still
9.38% longer. The earlier cross-engine greedy
text mismatch remains unresolved. SGLang was not rerun in this optimization
comparison; [its full measurement and parity results](gb10-aiperf-results.md)
remain the reference.

## Implemented execution changes

- Draft RoPE and KV scatter use one kernel with explicit BF16 round points.
  Strided input projections and uploaded metadata remain supported. Cache
  writes retain each request's private pages.
- Draft RMS uses fused pointwise kernels around the original PyTorch FP32 mean.
  It preserves both that reduction and the BF16 cast before weight multiplication.
- Accepted-feature projection and draft KV feedback have CUDA graphs for each
  exact accepted length. No row padding changes GEMM geometry. Interior
  checkpoints and eager execution retain the existing path. Graph input
  storage stays alive across replay, and metadata uploads own their pinned storage.
- Target GDN verification fuses four-tap convolution, FP32 SiLU and candidate
  state-window writes. Each original BF16 product/add round point is retained;
  committed state is read-only. SiLU uses libdevice exp and round-to-nearest
  division, checked against Torch for every finite BF16 input.
- GDN gating consumes the strided B/A projection views directly. Its prefill,
  decode and verify consumers already honor token strides, so the two copies
  per layer are unnecessary.

Target weights, GEMM backends, residual-sum arithmetic, draft precision, sampling,
attention and recurrent-state reconstruction are unchanged in this comparison.
The default draft remains BF16. Production restoration uses the original image;
the source changes were exercised in isolated containers with the repository mounted.

## Paired AIPerf results

| Frozen prompt | Baseline | Optimized | Latency reduction | Verify rounds, baseline / optimized |
|---|---:|---:|---:|---:|
| Code | 35.144 s | 33.803 s | 3.82% | 264 / 264 |
| Design | 34.563 s | 33.241 s | 3.82% | 258 / 258 |
| Reasoning | 23.915 s | 23.112 s | 3.36% | 170 / 170 |

Each request has exactly 10,000 frozen input IDs and 1,000 returned output IDs,
temperature zero, seed 42, neutral penalties, no stop strings and ignored EOS.
Both use the same pinned target and DFlash2 revisions, 16 verification rows,
FP8 target KV, BF16 draft body/KV, FP32 recurrent state, 60,000 target KV entries,
24 recurrent slots, a 2,048-token prefill chunk and capacity four. Concurrency is one.

AIPerf 0.12.0 supplies the request timings. Loading, compilation, graph capture,
canaries, client startup, confirmed cache flushing and full-generation warmups
are excluded. Every request's actual payload, returned prompt IDs, output IDs,
server usage, zero cache count, length finish and final DONE are independently audited.
No generated text is retokenized to establish output-token counts.

Load order is baseline / optimized / optimized / baseline. Code has four measured
samples per variant across two loads; design and reasoning have two per variant.
Each load/case also has one excluded full-generation warmup. There are 16 measured
requests and eight warmups. Small sample counts do not establish tail latency or
performance on other workloads. CPU tests and instrumented checks finish before timing.

The code effective decode-round cost falls from
118.920 to
113.880 ms. This is `(latency - TTFT) / rounds`,
including scheduling and streaming; it is not an isolated GPU kernel duration.
The major weight projections still dominate execution. These fusions remove launches
and intermediate traffic without reducing the target's projection weight bytes.

## Validation and retained screens

- 373 CPU tests passed; eight GPU-only checks skipped in that run.
- 83 focused GPU checks passed, including exhaustive BF16 SiLU, convolution
  rounding/state windows, RoPE scatter, RMS and feedback graph input lifetimes.
- 22 real-model lifecycle tests passed across BF16/FP8 drafts, eager/graph modes,
  varying widths, padded batches, sampling, prefix reuse and long batched prefill.
- All 16 final quality canaries passed. The optimized loads also passed nine
  real-weight, whole-arena feedback equivalence cases each.
- Every final baseline/optimized request, including warmups, has identical raw
  output IDs and text. The untimed target-prefill top-ten probes agree across loads.
- The original service and memory guard were restored after each GPU window.

The earlier target FP8 tile screen passed four short canaries but changed generated
text and verification-round counts. Its lower request times are retained as an
unqualified numerical candidate, not credited as an exact-work speedup.
RoPE alone was under 1%; adding feedback graphs gave roughly 1.4% on code in an
initial screen. The intermediate combined screen was about 2.6% versus the prior
baseline. Those sequential screens are ablations, not the final paired comparison.
The first RoPE server startup exposed strided metadata; it was corrected and
covered by GPU tests before timed requests. Its failed setup log is retained.

`protocol.json`, `summary.json`, per-request AIPerf raw exports, commands, source
snapshots, resource telemetry and test logs are retained with the evidence.
The summary JSON records the prospectively specified promotion gates explicitly.
