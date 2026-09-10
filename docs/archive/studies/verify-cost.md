# Controlled DFlash2 costs and compact convolution screen

> Dated evidence for the source and protocol recorded below. See the [current documentation](../../README.md) for supported behavior and priorities. Historical commands may require the recorded revision.

Measured on GB10 on 2026-09-06 (local date), against serving source `63a9b30`.
The controlled cost study and a standalone compact-convolution prototype are
complete. The prototype is bit-exact on the captured cases and substantially
faster in isolation, but fails the frozen integration screen. Serving code and
its speed remain unchanged. This experiment does not update the SGLang results.

## Fixed work and measurement

Both independent model loads use the pinned Qwen3.8-27B-NVFP4 target and
DFlash2 draft, k=7, FP32 accepted-state reconstruction, FP8 target KV, 35,000
target KV tokens and 12 state slots. Each runs alone under the same 64 GiB
host memory-drop/cgroup budget and 16 GiB floor, with eight CPU cores of quota
and four OpenMP threads. The lowest sampled host `MemAvailable` is 80.26 GiB.

The matrix covers 128/8,192 committed prompt tokens, B=1/4 and accepted lengths
1/4/8. Each cell has one warmup and five measured single rounds: **60 measured
rounds per window**. Fresh requests restore the same cached prefix and prefill
one token outside timing. Setup suppresses optional draft snapshot publication
for these noncacheable requests so the primer remains resident within the
unchanged feature-cache budget.

Every round runs the native draft, target forward/logit computation, acceptance,
state commit and draft feedback. The benchmark supplies frozen proposal tokens
and forced acceptance labels: these are synthetic cost controls, not generated
answers or quality evidence. All twelve frozen-work hashes match between the
two windows. Emitted IDs match within each cell and window; page/state lengths
advance by exactly the selected accepted count.

Uninstrumented round timing includes synchronization at its boundaries. CUDA
event spans come from separate diagnostic rounds. Parent spans include their
children: acceptance includes commit and feedback, and commit includes recurrent
reconstruction. Do not add nested spans. The large host duration inside acceptance
includes waiting for queued target work; it is not that much new CPU computation.

For full acceptance, the independent confirmation gives:

| Context / batch | Whole round | Draft event span | Target verify event span | Recurrent reconstruction | Draft feedback | Isolated convolution, 48 layers |
|---|---:|---:|---:|---:|---:|---:|
| 128 / 1 | 111.45 ms | 19.25 ms | 86.68 ms | 1.40 ms | 2.07 ms | 1.44 ms |
| 128 / 4 | 127.54 ms | 21.09 ms | 95.78 ms | 5.39 ms | 2.22 ms | 2.77 ms |
| 8,192 / 1 | 114.45 ms | 20.02 ms | 88.21 ms | 1.41 ms | 2.07 ms | 1.45 ms |
| 8,192 / 4 | 131.89 ms | 22.35 ms | 99.45 ms | 5.39 ms | 2.22 ms | 2.79 ms |

The first complete window gives whole-round medians 112.70, 127.03, 114.13 and
131.95 ms for the same four cells. The artifact retains all counts, samples and
phase spans. Isolated convolution timings are a scale estimate, not a strict
upper bound on its cost in a full model. No physical DRAM counters were collected.

## Prototype and decision

The prototype fuses the four convolution products/additions while preserving
each intermediate BF16 rounding. It retains the existing float SiLU operation.
Verification stores only the raw pre-convolution inputs. A separate kernel
constructs the accepted three-token window from those inputs and the untouched
committed convolution state. Each program owns complete channels, reads their
old entries before writing, and writes only the supplied active slots.

On real captured QKV, weights and history from all 48 GDN layers in each context/
batch cell, **192 convolution/input comparisons and 36 complete state comparisons
are byte-exact**. State checks cover every accepted length 1–8, mixed counts,
three active rows in capacity four, and unchanged inactive/dummy slots. These
checks cover the eight-token block in this prototype; they are not a full
serving, alternate-width, proposer or model-compatibility gate.

Three ABBA blocks give six independent graph measurements per implementation
and cell; each measurement contains five groups of 30 graph replays. Both
implementations keep both buffer sets allocated for timing. Commit timing is
measured separately at full acceptance.

| Context / batch | Reference convolution | Prototype convolution | Estimated net saving / whole round |
|---|---:|---:|---:|
| 128 / 1 | 1.419 ms | 0.298 ms | 1.00% |
| 128 / 4 | 2.771 ms | 0.716 ms | 1.62% |
| 8,192 / 1 | 1.421 ms | 0.292 ms | 0.98% |
| 8,192 / 4 | 2.847 ms | 0.732 ms | 1.61% |

The last column is **an estimate**, computed from the isolated convolution plus
commit difference divided by the separately measured whole-round median. No
integrated whole-round or HTTP speedup was measured for this prototype. Retained
convolution storage is 90 MiB for full windows versus 30 MiB for compact inputs
at B=4, a potential 60 MiB reduction; physical memory recovery and cache-pressure
benefits are unmeasured.

Before writing the prototype, the frozen screen required an estimated saving
of at least **2% in both B=4 context cells** before changing serving interfaces.
The prototype fails that screen. Archive the implementation and proof; do not
add a runtime option or change StateManager for this result. A future measured
memory-pressure benefit or a different dominant shape could justify revisiting it.

## Next speed mechanism

The same instrumented round traces contain 130 NVFP4 GEMM launches totaling
49.35–53.68 ms of kernel durations, including target MLP and target/draft vocabulary
projections. Classification uses the recorded `blockscaled_gemm`/`block_scaled`
kernel names; sums are attribution, not elapsed-time or bandwidth measurements.
Target verification itself spans about 86–99 ms. These costs are much larger
than convolution and identify the next place to investigate.

Next isolate the target's gate/up, down and vocabulary projections at the actual
verification row counts, holding packed inputs/scales/weights fixed. Separate
weight/scale reads, activation packing and matrix multiplication. Compare cold
and warm cache behavior and full-round traces before choosing a kernel mechanism.
The candidate direction is an SM121 kernel and weight layout for small token
blocks that reuse packed weights/scales efficiently and avoid redundant conversion
or staging, wherever measurements establish that overhead. Keep the numerical
contract and freeze a meaningful predicted benefit before implementing. This is
kernel/layout work; earlier backend tuning is already present. AOT packaging
still follows the supported-shape and unified-memory work.

## Reproduction and evidence

Inside the pinned runtime, with exclusive GPU use and the host guard:

```bash
python3 bench/verify_cost.py --output /path/to/new-run --dump-convolution
```

The optional dump retains real per-layer inputs outside git for the standalone
prototype. The [evidence manifest](verify-cost-results.json) records the full external bundle
location and SHA-256; detailed samples, sources and outputs stay outside Git.
Raw evidence is under `$SHARAR_RESULTS/reviews/2026-09-06/verify-conv/`.

The initial incomplete run is retained and excluded: its benchmark setup
published full-window feature snapshots that evicted the primer before the last
cell. Corrected setup retains the same cache capacity and runtime behavior.
Both complete windows and the prototype finish successfully. The original
production image and memory guard are restored, with health/model/canary checks.
Runtime source is unchanged, so the prior runtime/parity proof remains applicable;
this step validates the new measurement and prototype rather than rerunning the
unchanged full model suite.
