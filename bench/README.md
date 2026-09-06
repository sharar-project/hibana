# GB10 text-serving baseline

`benchmark_text_server.py` captures a reproducible, text-only serving baseline
for a single DGX Spark/GB10. It records the host, container, model and repository
configuration before running the matrix.

The default controlled matrix is:

- Input lengths: 128, 2,048 and 8,192 exact token IDs.
- Output length: 256 tokens with EOS ignored.
- Concurrency: 1 and 4.
- One warmup and five measured batches per cell.
- Prefix-cache flush before every batch.
- `MemAvailable` sampled four times per second and GPU power sampled once per
  second.
- Thinking disabled and no multimodal inputs.

Run it from the DGX Spark host while the reference SGLang container is listening only
on `127.0.0.1:30000`:

```bash
python3 bench/benchmark_text_server.py
```

Each run creates an immutable timestamped directory under
`${SHARAR_RESULTS:-${XDG_STATE_HOME:-$HOME/.local/state}/sharar/gb10}/controlled`.
`batches.jsonl` contains the raw request and resource samples,
`summary.csv` contains the measured rows,
`manifest.json` captures provenance, and `quality.json` contains deterministic
text canaries. Use a distinct `--label` for every controlled variant; the
launcher and benchmark defaults are pinned to the 27B NVFP4 GB10 engine.

Use shorter settings only for validating the harness itself:

```bash
python3 bench/benchmark_text_server.py \
  --input-lengths 32 \
  --output-length 8 \
  --concurrency 1 \
  --warmups 0 \
  --repetitions 1 \
  --skip-quality
```

Compare every candidate against the frozen production baseline. The comparator
rejects request or quality failures, serial output drift, more than 3%
throughput regression, more than 15% TTFT regression, or less than 16 GiB
available memory. Concurrent hash drift is reported but is diagnostic because
batch scheduling is not bitwise deterministic:

```bash
python3 bench/compare_runs.py BASELINE_RUN_DIR CANDIDATE_RUN_DIR
```

## Local workload pilot (Phase 7C)

`workloads/local_v1.json` freezes seven local-use tasks, greedy/sampled settings,
natural EOS, cold/repeated/session prefixes, four staggered arrivals, a client
disconnect, and output budgets 1–5. Its coding context is a checksum-verified
source fixture at `7aebbe6`; it never changes with the current working tree.
The JSON repeat is an expected miss: its 46-token prompt has no interior
64-token recurrent checkpoint. Coding/session repeats must actually hit.

Against an already prepared Sharar server on the same GB10 host:

```bash
python3 bench/local_workloads.py --url http://127.0.0.1:30001 --output /path/to/new-run
```

The client saves the exact materialized token IDs, tokenizer/template hashes,
server identity and every request/stream event. It measures client TTFT,
completion time and delivery gaps **between SSE bursts**. Several speculative
tokens can arrive together; it does not invent timestamps within that burst.
Cancellation records client-consumed tokens and the server cancellation count.

Automatic quality checks cover bounded clamp-function behavior, exact JSON,
tool name/arguments and a reasoning answer. Free-form chat/session outputs have
explicit review rubrics. These are smoke checks, not a model-quality benchmark.
The process fails on a request or automatic-quality failure; a successful exit
still requires rubric review. Three repetitions establish a pilot only: retain
raw samples and do not report p95/p99 or a statistically established speedup.
The existing synthetic comparator does not certify these workload results.

For the full exclusive-memory source validation in the prepared GB10 image:

```bash
python3 bench/phase7c.py --output /path/to/new-proof
```

This driver first diagnoses cold/split/cache state, starts a temporary loopback
HTTP server for plain and MTP k=3, and then profiles fixed-token decode at
batch 1/4 and prefix 128/8,192. It loads the real model: stop the serving model
before running it, keep the 16 GiB host floor guard active for the test process,
and restore serving afterward. The driver does not deploy an image or stop a
service itself. Host/container identity and memory/power samples belong with
the results (see [the 7C evidence](../docs/phase7c.md)).

`gb10_profile.py` saves independent unprofiled four-step timings and compressed
Chrome traces with proposer, verifier and acceptance scopes. Device activity
uses the union of kernel/copy intervals, not their overlapping duration sum.
Trace gaps include profiler overhead; allocation byte counts are explicitly
separate from physical DRAM traffic, which this profiler does not measure.


## Pinned Sharar / SGLang comparison

`competitive.py` uses `workloads/competitive_v1.json`: identical token inputs,
256-token fixed-work cells at input 128/2,048/8,192 and concurrency 1/4, followed
by natural `local-v1` requests and a primed session. It reports aggregate tokens
per client batch makespan, TTFT, completion and throughput after the first SSE
burst. The fixed-work matrix ignores EOS and is a throughput diagnostic; the
natural requests retain their original stopping, sampling and quality rubric.
SGLang uses `sampling_seed` for the same seed Sharar calls `seed`.

Freeze once and reuse the same file for every engine/profile:

```bash
python3 -m bench.competitive --freeze /path/to/inputs.json
python3 -m bench.competitive --inputs /path/to/inputs.json \
  --url http://127.0.0.1:30000 --profile sglang-dflash2 --output /path/to/new-run
```

Use exclusive GPU windows with the same host memory budget and record the
server/image/source identities and memory samples. Run the client in a separate
process for each engine. The SGLang DFlash2 matrix includes prefill sizes 2,048
and 8,192; preserve both results and label any best-of-two comparison. Check
actual running capacity: SGLang uses several recurrent-state slots per request,
so matching Sharar's internal slot count can silently restrict concurrency.
A cgroup memory limit alone does not account for CUDA allocations on GB10.
Three repetitions per cell are
a pilot snapshot; retain errors and generated outputs, review the manual rubric,
and do not infer tail latency or broad quality equivalence from this suite.
The current launch settings, measurements and excluded attempts are documented
in [the competitive report](../docs/competitive.md).

## Controlled verification costs

`verify_cost.py` holds proposal tokens and accepted lengths fixed while running
the real DFlash2 round. It uses 128/8,192-token contexts, batch 1/4 and accepted
lengths 1/4/8, with prefix restoration outside timing. Separate event spans and
traces attribute draft, target, reconstruction, commit and feedback. These are
synthetic cost controls, not generated-answer or competitive throughput results.

```bash
python3 bench/verify_cost.py --output /path/to/new-run --dump-convolution
```

The optional dump stores real per-layer convolution inputs outside git. Run
exclusively under the host memory guard; the host harness owns service restoration.
The [convolution screen](../docs/verify-cost.md) documents the measurements and
the unpromoted prototype, including the frozen integration threshold.

## NVFP4 projection experiments

`nvfp4_cost.py` captures all 130 packed projection inputs from a real DFlash2
round, with checkpoint revision and input/scale/output hashes. It times packing,
the distinct-weight projection families, flushed/repeated addresses and three
backends at the actual M=7/8/28/32 row counts. Raw tensors stay outside git.
`nvfp4_tiles.py` replays those inputs through isolated narrow-tile adapters;
`nvfp4_round.py` screens the small-M dispatch change with interleaved complete
rounds, natural acceptance and forced counts, and exact work/output accounting.

Use the commands and exclusive-memory procedure in [the NVFP4 report](../docs/nvfp4-cost.md).
The tile adapters use private APIs in the pinned FlashInfer runtime. The direct
variant needs `--retain-full-scale-block`; without it, the original upstream
scale-layout assertion is deliberately reproducible. These experiments failed
their frozen speed gates and expose no serving options. Component timing and
synthetic acceptances are not useful-output throughput claims.
