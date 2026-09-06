# Runtime cleanup and clean-image promotion review

This review keeps the shipped scope to the pinned Qwen model on GB10, removes
rejected serving experiments, and repairs MTP feedback before evaluating the
DFlash2 candidate. Default promotion depends on the measured gates below.

## Changes

- Remove mixed forwarding from the scheduler, executor, StateManager and model,
  along with the failed width and latency policies and their public options.
  Their source and benchmark runner remain reproducible at `56605f0`.
- Move checkpoint construction to `proposers/factory.py`. Generic execution
  validates declared features and maximum depth. Normal startup captures only
  the selected verification width; controlled benchmarks explicitly opt into
  additional widths. Preserve draft-aware cache ownership, sampled verification,
  accepted-state reconstruction and the shared GDN prefill plan.
- MTP accepted feedback uploads all pending/accepted input tokens once through
  bounded pinned staging. A copy-completion event protects staging reuse for
  direct callers. Per-request KV correction preserves the original
  row shape, BF16 arithmetic and the required target-conditioned KV
  for accepted inputs, including the fully accepted bonus input. A captured-KV
  prototype adds no consistent gain beyond pinning and is archived in the
  benchmark instead of becoming another runtime path.
- Replace ten large evidence bundles in the current Git tree with compact
  manifests: **7,232,106 bytes to 15,925 bytes**. Every original bundle is preserved
  outside Git and verified by SHA-256. Markdown reports retain the results;
  manifests locate the full evidence. CI enforces a 32 KiB manifest limit.
  This does not rewrite or shrink existing Git history.
- The runtime shrinks from **15,340 to 15,100 Python lines**; `executor.py`
  shrinks from **570 to 493 lines**. These counts include vendored runtime kernels
  and exclude benchmarks/tests.
- Hold additional dense/MoE models and generic target abstractions until the
  current default is proven. Update the roadmap and current interface guide;
  mark obsolete reproduction commands as historical.

## Validation

The CPU suite passes **251 tests**, with six GPU-only skips. The initial clean
image, including the subsequently archived KV graph prototype, passes **573
CPU/GPU tests**, including real state/cache continuations,
stop and output-budget boundaries, retraction, stochastic verification, API
validation, shutdown draining and canaries. Tests for removed experiments were
removed with those experiments; applicable reservation coverage was retained.

The Dockerfile dependency stages were rebuilt with `--no-cache` from the CUDA
13.0.3 development base. The pinned cu130 dependency install, `pip check` and
runtime imports all succeed. Final benchmark/test updates reuse those freshly
built dependency stages. The final pinned-feedback image is:

`sha256:2addcbfc71a422dffc30a51ebc2cc0cb1a91b4f5592ae09edd87843e19bfca3d`

The initial graph-prototype image was
`sha256:225273d394c467f6f76365f1e6a1ef2dcae74e05423d09a9c8018f0933d30823`.
Removing that MTP-only path leaves the validated target and DFlash2 code unchanged.
The final image passes **269 focused checks**: the complete CPU/API suite plus
cached MTP EOS/budget tests at batches 1, 3 and 4 under both state strategies.

The final image contains all 95 current runtime Python files, checked against the build
context. Clean-image validation and Sharar performance runs use the image's
runtime directly, without a source overlay. Independent measurement/control
scripts and output directories are mounted separately.

An earlier source-overlay test launch omitted the Python main guard, causing
multiprocessing spawn to re-enter pytest during the SIGTERM test. It recorded
564 passes and that launcher failure. The corrected clean-image invocation
passes all 573 checks, including SIGTERM; no serving change was required.

## MTP repair results

Twenty cells cover 128/8,192 context tokens, batch one/four, native acceptance
and forced accepted counts 1–4. Each cell runs legacy/pinned/graphs/graphs/pinned/
legacy in one warmup block and three measured blocks: **360 timed rounds plus
120 warmup rounds**. All **480 round comparisons** have identical emitted IDs,
committed lengths, full target recurrent/convolution state, corrected draft KV
and pending target hidden rows. Forced counts measure cost and state alignment;
they are not answer-throughput measurements.

Natural whole-round medians at MTP k=3, including proposal, verification and
acceptance, with the same prepared context:

| Context tokens | Concurrent requests | Blocking uploads (ms) | Pinned batch upload (ms) | Round reduction |
|---:|---:|---:|---:|---:|
| 128 | 1 | 110.760 | 109.383 | 1.24% |
| 128 | 4 | 128.559 | 123.743 | 3.75% |
| 8,192 | 1 | 112.837 | 111.969 | 0.77% |
| 8,192 | 4 | 132.099 | 128.455 | 2.76% |

Pinning supplies the material gain: **3.64–4.82 ms per natural B=4 round**.
The additional KV graph prototype is 0.51% slower than pinning at 128/B=4 and
0.11% faster at 8K/B=4; results across forced counts are also mixed. It stays in
`bench/mtp_feedback.py`, with no runtime field, mode or extra graph capture.
The required target-conditioned KV correction remains asynchronous on the
executor's stream. These costs do not directly predict DFlash2 acceleration.

## DFlash2 cache and lifecycle evidence

Additional real-model checks pass at 8,192 and 32,766 input tokens with two
outputs, both cold and cached. Missing/evicted draft features safely fall back
to ordinary target decoding and match its output IDs for greedy and sampled
requests. Four active prefix readers survive eviction and cache flush, then
retraction preserves delivered output without duplication. Cancellation releases
its request. After flushing, all 34,999 usable KV pages, 11 state slots and four
draft slots are returned.

The 32,766-token repeat reuses only a **20,480-token checkpoint** with the fixed
12-slot pool. It is correct, but must redo over 12K prompt tokens. Endpoint versus
interior checkpoint retention under tight slot budgets is a concrete later
cache-efficiency task; this run does not establish a long-session capacity win.

All **16 historical parity gates pass**. The full clean-image validation window
stays above **75.67 GiB MemAvailable**, within the declared budget. Production
restoration passes its health/model/canary check after the window.

## Frozen comparison protocol

Two independent windows reverse profile order. Each compares repaired Sharar
MTP k=3, Sharar DFlash2 k=7 and native SGLang DFlash2 with eight-token blocks.
Both Sharar controls use accepted-state reconstruction. The target, external
draft, tokenizer inputs and natural workload definitions remain pinned.

Each fixed-work cell uses 128/2,048/8,192 input tokens, **256 output tokens per
request**, concurrency one/four, one warmup and three measured batches per
window. EOS is ignored in these diagnostic cells. Aggregate output throughput
is total emitted tokens divided by client batch makespan, including prefill.
Natural tasks use their original stopping, sampling and quality checks.

Both engines get eight CPU cores, four OpenMP threads, 35,000 target KV tokens,
FP8 target KV, FP32 recurrent state, 32,768 context capacity and four running
requests. Sharar uses 12 state slots; SGLang uses 24 to provide actual running
capacity four. The fresh SGLang comparison uses 2K prefill chunks; the prior
8K results remain historical, so this is not a per-cell best of both settings.
SGLang's default sampler ignores per-request seeds without its deterministic
mode; sampled answers are qualitative comparisons, not matched-seed timings.

Each exclusive run enforces a 64 GiB cgroup cap and host `MemAvailable`-drop
budget, plus the 16 GiB absolute floor. These are common enforced budgets, not
identical allocations or a sum of overlapping CUDA/RSS measures. Restore the
original production image and memory guard after each validation window.

Promotion requires passing correctness and frozen quality checks. Relative to
repaired MTP, no fixed cell may lose over 3% throughput or regress TTFT over 15%;
protected natural completion may regress at most 5%. A selected natural workload
must improve at least 10%. Failed quality cells cannot supply speed wins. Two
windows with six measured batches per cell remain limited evidence for tails;
they do not establish production p95/p99 or a universal speed lead.

## Reproduction

Export `sharar/proposers/mtp.py` from `56605f0` as the frozen control, then run
in the exclusive guarded runtime:

```bash
python3 -m bench.mtp_feedback --reference /path/to/legacy_mtp.py --output /path/to/new-proof
python3 -m bench.competitive --inputs /path/to/inputs.json --url http://127.0.0.1:30000 \
  --profile sharar-dflash2 --output /path/to/new-run
```

Full sources, build logs, commands, runtime hashes and measurements live under
`$SHARAR_RESULTS/reviews/2026-09-06/promotion/`. The archived older evidence is
in its `archive/` directory. Exact host launch/restoration commands are retained
there rather than embedded in committed JSON.
