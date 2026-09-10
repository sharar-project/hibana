# BS4 engine study

2026-09-10. **No BS4 throughput change is promoted.** Keep k=15, a 2,048-token
prefill chunk, a 4,096-token prefill budget, and one decode step between
prefill blocks. Smaller verification widths and both scheduler candidates
fail the throughput gate. A runtime-extent kernel prototype eliminates
first-use compilation stalls, but its warm HTTP throughput does not improve;
it remains in the frozen experimental source, with production kernels unchanged.

The baseline is main at `4af72042f3ec9a02eb918c40bc56dbbc3a54545c`:
the pinned Qwen3.8-27B NVFP4 target, FP8 DFlash2 with 15 proposals, and the
exact BS1 QKV/MLP integration. Target and draft weights remain fixed.
The restored production container uses the BF16 draft, so these results
are comparisons against the FP8 source baseline, not a deployment claim.

## Verification width

Keep the draft at 15 proposals and truncate only target verification.
Widths 16, 8 and 4 mean k=15, 7 and 3; full BS4 executes 64, 32 and 16
target rows respectively. Every process captures all three depths, with
the same 60,000 KV pages, 24 state slots and batch capacity four.

Four fresh loads run width 16 / 8 / 4 / 16. Each includes three BS1 controls
and six BS4 groups, with 10,000 input and 1,000 output tokens per request.
Prompts are cold, greedy and `ignore_eos=True`. Native BS4 prefills every
request before timing decode; HTTP scheduling is measured separately.
There are 108 timed requests, plus excluded preflight and warmup requests.

Five disjoint groups cover the 18 existing synthetic task families. The
last group has two copies each of its two families. The independent unit
for throughput statistics is the **group**, not each co-batched request.
The separate sixth group contains four copies of the standard code prompt.

| Verifier width | Pooled family decode TPS | Change | Paired group ratio, 95% interval |
|---|---:|---:|---:|
| 16 | 132.70 | baseline | — |
| 8 | 129.25 | −2.60% | 0.9749 [0.9208, 1.0194] |
| 4 | 104.80 | −21.02% | 0.7890 [0.7561, 0.8329] |

Pooled end-to-end TPS is 86.96 / 85.42 / 74.09. Baselines average opening
and closing controls. Intervals bootstrap mean paired log ratios across
five groups (10,000 resamples, seed 0), so they are not intervals around
the pooled estimator or broad population-quality claims.

**Keep width 16.** Smaller rounds cost less, but lose enough emitted tokens
to require more complete target traversals. Width 8 changes individual
group throughput by −12.6% to +3.4%; none reaches the proposed 5% advancement
gate. All 27 output sequences change at either narrower width. The three
BS1 controls at width 8 change by −26.6%, +8.8%, and −20.0%; a blanket
width change is also unsuitable for BS1. No adaptive width policy or true
shorter-draft execution is promoted from this screen.

The standard four-copy code group is particularly misleading on its own:
151.41 → 85.34 / 97.68 TPS. Its forced post-EOS continuation is retained as
evidence, not used to select a changed-shape configuration.

## Initialization observations

Opening and closing full-width loads reproduce **27/27 output sequences**.
The closing family control is 1.08% faster by paired geometric mean, with
individual groups varying from −0.01% to +3.00%. The opening load includes
occasional first-use rounds of 214–403 ms; steady BS4 rounds are around
132 ms. Draft normalization and RoPE specialize their kernels on accepted
row count, which can take many values at BS4.

The candidate makes only those extents runtime kernel arguments with
`do_not_specialize`. It keeps strides, layout, arithmetic and rounding
unchanged. All **135 GPU checks** pass, including exact RMS/RoPE values,
strided KV views, odd counts, graph replay with mutated inputs and ring
positions, all 64 BS4 feedback counts, and real BF16/FP8 draft lifecycles.
The CPU suite passes **425 tests**, with eight skipped.

A separate probe starts each variant with its own empty Triton cache and
warms counts 1–16 before visiting 17–64. It executes the two normalization
geometries and strided RoPE/scatter used by feedback:

| Probe measurement | Old extents | Runtime extents |
|---|---:|---:|
| Compiled variants after counts 1–16 | 80 | 8 |
| Compiled variants after counts 1–64 | 320 | 8 |
| Median first-use call for counts 17–64 | 220.45 ms | 0.086 ms |
| Total first-use time for those 48 calls | 10.67 s | 0.005 s |
| Median already-warm call | 0.086 ms | 0.072 ms |

Variant counts refer to the probe's layouts, not every kernel in the model.
These standalone calls include Python and allocation time; the tiny warm
timing difference is not a whole-model speedup claim. The demonstrated
benefit is eliminating repeated shape compilation. Full-generation results
are checked separately, and this is distinct from a large steady throughput
gain.

The complete candidate pass preserves **27/27 output sequences** from the
closing control, including all three BS1 controls and six BS4 groups.
Pooled family BS4 decode throughput is **133.41 → 133.75 TPS (+0.25%)**.
The three BS1 controls change by +0.23%, +0.21%, and +0.10%. This is one
candidate pass, not a repeated steady-throughput promotion claim; it
supports the absence of a material regression alongside the exactness
checks. All requests complete without draft fallback and recover their pools.

**Do not promote the kernel prototype as a throughput improvement.** The
subsequent warm HTTP confirmation below has lower point estimates. Admission
order and continuations change, so the experiment does not isolate a steady
kernel regression, but it also does not establish a serving benefit. The
runtime kernel edits were reverted; `source-candidate/` in the proof bundle
retains the complete validated prototype for a separate initialization task.
The expanded normalization, RoPE and feedback tests also pass on the final,
unchanged production kernels: **50 GPU checks**.

The original width screen is retained in full, including initialization
cost. The subsequent HTTP screen explicitly warms feedback counts 1–64,
including the strided metadata used by batched feedback, before timing.

## HTTP scheduling and confirmation

The real streaming `/generate` endpoint runs two frozen workloads: four
simultaneous cold 10K/1K requests, and four closed-loop client streams that
each replenish three requests, with prompts of 512–10,000 tokens and output
budgets of 128–1,000 tokens. These are finite, fixed client lanes, **not an
indefinitely saturated shared queue**. Their final drain and workload imbalance
remain in the measured totals. Source, weights, pools and graph depths are
fixed; every server warms feedback counts 1–64 before timing.

The original order is control / runtime extents / budget 8192 / four decode
steps / control. An extra runtime-extents / control pair checks HTTP variation.
All **112 timed requests** and **28 natural-stop canaries** pass across seven
fresh server loads, with no cache hits, draft fallbacks, retractions, or leaks.
Controls below pool all three loads; runtime extents pool two. The scheduler
candidates have one sample per workload, so this is a screen, not a broad
latency or population-throughput claim.

| Configuration | Finite end-to-end TPS | Replenished end-to-end TPS | Finite p95 TTFT, s | Replenished p95 TTFT, s |
|---|---:|---:|---:|---:|
| Control | 81.81 | 76.19 | 17.52–17.62 | 9.09–9.28 |
| Prefill budget 8192 | 75.45 (−7.8%) | 74.08 (−2.8%) | 19.17 | 9.22 |
| Four decode steps | 79.97 (−2.2%) | 71.52 (−6.1%) | 18.24 | 9.17 |
| Runtime extents, experimental | 78.71 (−3.8%) | 75.33 (−1.1%) | 17.51–17.53 | 8.96–8.98 |

Finite-request p95 latency is 48.20–49.04 s for the controls, 52.27 s for the
larger budget, 49.52 s for four decode steps, and 49.48–51.18 s for runtime
extents. Replenished-request p95 latency is 37.84–41.01 / 37.33 / 38.45 /
37.09–37.14 s respectively. These are sample quantiles of four or twelve
requests per load, not production tail guarantees.

The first runtime-extent finite load starts with one prefill request rather
than two, executes 11 rather than 10 prefill forwards, and takes 265 rather
than 239–240 decode rounds. Its second load takes 252 rounds. Full BS4 round
cost remains about 133.6–134.2 ms across the controls and candidates. Even
the unchanged controls have different HTTP outputs when admission and batch
membership change. All raw outputs and memberships are retained; the lower
HTTP point estimates are not discarded or described as a throughput win.

In the controls, about 81% of finite-workload decode time and 44–49% of
replenished-workload decode time uses all four rows. The latter includes
prefill interruptions and the finite client-lane drain. This is evidence
about the measured workload, not proof that the scheduler can fill those
slots with work that has not yet arrived.

## Why smaller rounds did not win

Four instrumented full-BS4 rounds at each width give the following mean GPU
kernel-duration sums. The source and weights are unchanged; each width runs
its own continuation, so these are diagnostics rather than paired latency
savings on identical intermediate states.

| Width | Target verification | Drafting | Acceptance/state work | Draft feedback |
|---|---:|---:|---:|---:|
| 16 | 110.11 ms | 14.33 ms | 5.49 ms | 1.83 ms |
| 8 | 99.67 ms | 14.31 ms | 5.49 ms | 1.77 ms |
| 4 | 94.33 ms | 14.24 ms | 5.47 ms | 1.76 ms |

Reducing target rows fourfold saves only about 16 ms of target kernel time;
drafting and state work persist. Target weight traversal dominates, while
the narrower window loses useful emitted tokens. These kernel sums are not
additive end-to-end latency predictions. Profiler queue gaps include its
own overhead and are not a measured recoverable budget.

No adaptive width controller, extra BS3 graph, or mixed-forward rewrite is
justified by this screen. The previous [mixed-forward prototype](mixed-forward.md)
already failed TTFT and protected quality; it is not a new untested shortcut.
Large cold BS4 gains remain open. Joint target/state/draft cache retention
is a separate opportunity for warm sessions, described in the
[cache plan](request-cache-policy.md), and should not be counted as cold TPS.

## Reproduction and evidence

The external proof bundle is
`$SHARAR_RESULTS/reviews/2026-09-10-bs4-engine/`. It contains frozen inputs,
group membership, protocols and amendments, immutable source snapshots and
hash manifests, complete output IDs, per-forward work and timing records,
resource samples, and service restoration records.

The reusable drivers are `bench/gb10_bs4_engine.py` and
`bench/gb10_bs4_http.py`. The native driver records actual batch size,
physical verifier rows, padding, uncommitted rows, emitted and accepted
tokens, and per-request output IDs. Its per-forward times are host intervals;
complete decode timing includes a final device synchronization. Profiler
diagnostics are separate from throughput measurements.

The width window uses the pinned production image
`sha256:0d849f3f7ebd1309d1ab0828f00d4eaffe410126dd7db8ea31d226095655130a`
with frozen source mounted over it. Minimum observed available memory is
80.53 GiB in the width window and 75.02 GiB across the candidate validation
and HTTP window, above the 16 GiB guard. Every window restores the original
service and memory guard; final restoration details are in `final-health.json`.
