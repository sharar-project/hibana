# GB10 text-serving baseline

## Quick TPS command

Use an already running server built from the current source, with exclusive
GPU use and the host memory guard. The benchmark client runs on the host;
it flushes the server's prefix cache but does not start or stop services.

```bash
tools/benchmark.sh
# Equivalent installed entry point:
sharar benchmark
```

Defaults: the frozen `code` task, 10,000 input tokens, exactly 1,000 output
tokens, temperature zero, seed 42, concurrency one, one excluded full warmup
and three timed requests. `--workload all` adds `design` and `reasoning`.
The three built-in prompts reproduce the first input row of each task in the
recorded [AIPerf TPS comparison](../docs/gb10-tps.md). The packaged historical
source fixture is independent of edits to current engine code. Prompt
tokenization occurs outside timing; each run saves the exact IDs it uses.

`--input-tokens`, `--output-tokens`, `--repeats` and `--warmups` can shorten a
development run. Changed token budgets define a different benchmark cell.
Run `tools/benchmark.sh --help` for all options.

One-time setup, if AIPerf is not already available:

```bash
python3 -m venv .venv-aiperf
.venv-aiperf/bin/pip install aiperf==0.12.0
```

Discovery checks `--aiperf`, `SHARAR_AIPERF`, `PATH`, the active Python's bin
directory, `.venv-aiperf/bin/aiperf`, and `$SHARAR_RESULTS/aiperf-venv/bin/aiperf`.
Explicit paths take precedence. The client reads `SHARAR_HF_CACHE`,
`SHARAR_RESULTS` and `SHARAR_AIPERF` from `config/gb10.env` when unset; environment
variables override that file. `SHARAR_CONFIG_FILE` selects another config.
AIPerf needs the pinned tokenizer available locally even with server token
counting enabled. Installing Sharar with `.[benchmark]` is another way to
install the pinned AIPerf dependency.

To compare engines, execute these in **separate server/GPU windows**, using
new output directories. Match target weights, context/batch limits and prefill
settings, and record the draft precision selected for each server:

```bash
# While the prepared SGLang server owns the GPU:
tools/benchmark.sh --url http://127.0.0.1:30000 --output /tmp/sglang-code

# After switching to the prepared Sharar server:
tools/benchmark.sh --inputs /tmp/sglang-code/frozen-inputs.json \
  --baseline /tmp/sglang-code --output /tmp/sharar-code
```

`--baseline` rejects differing input IDs, token budgets or benchmark protocols.
It prints percentage changes in both TPS measures and reports whether the
timed output IDs match. Output differences are retained for investigation;
they do not establish a same-output speedup or equal answer quality. The client
benchmarks the server configuration it connects to; use Sharar's explicit
`--draft-precision fp8` serving option for the optimized draft profile.

The table uses only timed requests. End-to-end TPS is total output tokens
divided by total request latency, including prefill. Decode TPS is
`sum(output_tokens - 1) / sum(latency - TTFT)`. This is the same decode-rate
convention as the earlier audit; SSE bursts do not provide individual token
timestamps. TTFT and latency columns are arithmetic means. Model loading,
client startup, tokenization, cache flushing and warmups are outside these
request metrics. Short runs do not establish tail percentiles.

Every result directory contains `summary.json`, `results.json`,
`frozen-inputs.json` and each request's raw AIPerf artifacts, server info and
token audit. Existing directories are never overwritten. The default root is
`$SHARAR_RESULTS/aiperf`, falling back to `$XDG_STATE_HOME/sharar/gb10/aiperf`
(or `~/.local/state/sharar/gb10/aiperf`). The legacy `bench.gb10_aiperf` entry
point remains available for historical commands and uses the shared auditor.

## Detailed profiling and historical studies

Profile the optimized FP8 draft on the same frozen inputs as the TPS client:

```bash
python3 -m bench.gb10_audit --draft-precision fp8 \
  --inputs /results/benchmark-run/frozen-inputs.json \
  --output /results/new-fp8-profile --cases code design reasoning \
  --batches 1 --warmups 1 --repeats 2 --capture-candidates
```

Run inside an exclusive, guarded GPU window. FP8 is the profiler default;
`--draft-precision bf16` selects the control. It records complete native
generations separately from CPU/CUDA traces at output positions 32, 384 and
768, plus full-generation candidate support when requested. Terminal windows
without a complete proposal/bonus horizon are excluded from support ceilings.
Candidate captures and instrumented traces are diagnostics, not HTTP TPS.
The launcher accepts `SHARAR_DRAFT_PRECISION=fp8`; its default remains BF16.
Both require an image built from source containing the FP8 serving option.

The [current 16-row matrix study](../docs/gb10-verifier16.md) uses raw native
reference output IDs in a JSON object keyed by `code`, `design`, and
`reasoning`. Capture operands first, then validate/tune against those captures:

```bash
python3 -m bench.gb10_matrix capture --inputs /proof/frozen-inputs.json \
  --reference /proof/reference-ids.json --output /proof/capture
python3 -m bench.gb10_matrix screen --captures /proof/capture --output /proof/screen
python3 -m bench.gb10_matrix family --captures /proof/capture --output /proof/family
python3 -m bench.gb10_fp4_pipeline --captures /proof/capture --output /proof/fp4-screen
python3 -m bench.gb10_fp4_pipeline --family --captures /proof/capture --output /proof/fp4-family
python3 -m bench.gb10_fp8_tactics --expand --captures /proof/capture --output /proof/fp8-tactics
python3 -m bench.gb10_quant_pack --captures /proof/capture --output /proof/quant-pack
python3 -m bench.gb10_matrix native --fp4-mlp --cases code --batches 1 \
  --inputs /proof/frozen-inputs.json --reference /proof/reference-ids.json \
  --output /proof/combined-native
python3 -m bench.gb10_verify_width --inputs /proof/frozen-inputs.json --output /proof/width-cost
```

Use the exclusive guarded GPU window. Output directories must be new.
`gb10_matrix profile` replays eight labeled matrix calls inside a CUDA profiler
range for Nsight Compute. FP4 pipeline tuning uses private CuTe APIs;
cuBLASLt tuning builds a benchmark-only C++ extension. No tactic is selected
automatically for serving. Matrix/quantizer/flat-path measurements do not
replace complete generation timing, correctness or task-quality checks.

Tree coverage can be reproduced on CPU from the preceding full candidate
captures. Neither policy receives future target tokens during selection:

```bash
python3 -m bench.gb10_tree_screen --captures /proof/native-fp8 \
  --policy probability --budgets 16 32 64 --output /proof/probability-tree.json
python3 -m bench.gb10_tree_screen --captures /proof/native-fp8 \
  --policy protected --budgets 32 64 128 --output /proof/protected-tree.json
```

The optional CPU selector screen trains on code/design and evaluates reasoning
once, using a fixed eight-feature residual ranker:

```bash
python3 -m bench.gb10_selector_fit --captures /results/new-fp8-profile \
  --output /results/new-selector-screen
```

The target compression screen now uses code/design calibration and held-out
reasoning decode blocks at the same early/middle/late positions:

```bash
python3 -m bench.gb10_target_fp4 --draft-precision fp8 \
  --inputs /results/benchmark-run/frozen-inputs.json \
  --output /results/new-target-compression
```

This screens ten projections with both prefill and decode calibration. It
validates every held-out block against a 5% normalized RMS threshold and times
the 16-row shape. It does not change target serving weights or establish
whole-model quality. The new default protocol differs from historical runs.

For the 10,000-input / 1,000-output native-generation audit, run
`python3 -m bench.gb10_audit --output /results/new-gb10-audit` inside the
pinned GPU image with exclusive GPU use and the host memory guard. It saves
exact inputs, full continuations, per-round counts, a logical weight inventory,
and independent phase/kernel traces. `--depth 7 --cases code` is the width
control. This is a component diagnostic; use HTTP measurements for serving
latency. See [the audit and optimization priorities](../docs/gb10-10k1k-audit.md).

The latest [TPS validation](../docs/gb10-tps.md) repeats the exact-ID AIPerf
comparison with five timed requests per task and engine. It reports output TPS
including prefill and decode TPS separately, retains SGLang reload variation,
and documents the recommendation to submit a review PR with FP8 kept opt-in.

The preceding [target arithmetic repairs](../docs/gb10-target-fix.md) cover native
quantization boundaries, prefill KV precision, rotary precision and the pinned
compiled verification norms. The report separates AIPerf latency from exact
same-history target/state comparisons and full-generation token parity.
The explicit FP8 draft configuration measures 26.561 / 33.111 / 21.660 s for
code / design / reasoning, with exact 10K input and 1K output token counts.
These are 17.16% / 4.69% / 12.01% below the faster SGLang mean per task from
two fresh loads; full cross-engine greedy text still differs. Both current
draft precisions pass the quality canaries and lifecycle tests.

The earlier [target-path diagnosis](../docs/gb10-target-path.md) repairs FP4 tie
rounding and audits 27 additional AIPerf requests. It records a 0.58% QKV/Z
fusion screen and a SiLU precision candidate that improves code but regresses
the other prompts; neither performance candidate is promoted. Reproduction
scripts, raw responses and CUDA traces are indexed in the linked manifest.

The earlier [exact-arithmetic fusion results](../docs/gb10-engine-speedups.md)
use AIPerf 0.12.0, fresh baseline/optimized/optimized/baseline loads, cache
flushing and exact returned token-ID audits. Code latency falls 3.82%, design
3.82% and reasoning 3.36%; all 24 requests preserve the old baseline output.
The fusions apply automatically in the current source. Production retains its
original image, and SGLang greedy-output parity is still unresolved.

The implementation follow-up is in
[GB10 optimization results](../docs/gb10-optimization-results.md). Its explicit
FP8 draft option is `sharar serve --draft-precision fp8`; BF16 remains the
default. The fused BF16 draft convolution is automatic. Results include full
10k/1k continuations, held-out task families, failed selector/quantization
screens and numerical limitations.

To reproduce an optimization cell in an exclusive guarded GPU container,
mount the follow-up evidence directory at `/proof` (it contains the frozen
input IDs and original selector source), then use a fresh output directory:

```bash
python3 -m bench.gb10_optimize --profile fp8 --mode http --cases code design reasoning --batches 1 4 --repeats 2 --quality --output /proof/new-fp8-http
```

Use separate processes for `control`, `fused`, `selector`, `fp8` and
`fp8_selector`. `fp8_rowwise` is a separate exploratory weight-scaling screen.
`bf16_f32_math` and `fp8_f32_math` are unpromoted normalization/convolution
rounding ablations; they do not establish SGLang arithmetic parity.
`--depth 7` tests eight physical verify rows; `--depth 15` tests sixteen.
`bench.gb10_acceptance` records fixed-history diagnostics outside timing, and
`bench.gb10_target_fp4` performs layerwise calibration/cost screening without
changing the target serving profile. Full native and HTTP timings remain
separate measurements.

The [acceptance parity follow-up](../docs/gb10-acceptance-parity.md) compares
proposals on a reference engine's identical histories and verification
boundaries. Its validated JSON trace is retained in the dated evidence folder:

```bash
python3 -m bench.gb10_acceptance_replay --reference /proof/reference-code.json --output /proof/new-replay.json --rounds 128
```

This is teacher forcing, not a latency benchmark or ordinary generation.
`--target-residual-sum native` is now the default in both tools and uses current
serving arithmetic. Explicit `bf16` and `fp32` select historical norm ablations;
they do not reproduce an older complete engine after other source changes.
Reproducing a dated result requires its recorded source snapshot. The old
FP32-only ablation failed a JSON canary; the complete current repair is a
separate profile. Use separate processes for each mode. Neither benchmark
changes serving defaults or imports SGLang.

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

## Evidence storage and archived experiments

`agent_workloads.py` freezes agent-shaped repository/log inputs for matched
DFlash2 HTTP runs at 8K/16K/near-32K. It separates fixed 512/1,024-token output
work from strict JSON/tool-call episodes, and records paused retries, shared
history branches, actual growing inputs and cache churn. `agent_profile.py`
attributes native generation on the same frozen inputs. Its instrumented phase
events/traces are separate from HTTP throughput. See the
[protocol and results](../docs/agent-workloads.md) for resource matching and limits.

`two_path_study.py` measures physical batch 1/2 target verification against a
shared immutable prefix and records two-path coverage over full native
512/1,024-token continuations. Freeze its protocol and new inputs before GPU
execution. Each captured continuation must match a separate uninstrumented
control. Coverage and estimated costs cannot establish actual alternative-rollout
speed or winner-only state correctness. This driver exposes no serving option.

Raw JSON, traces, output tokens, embedded sources and model tensors belong under
`$SHARAR_RESULTS`, outside Git. Commit a concise Markdown report and an evidence
manifest with the full bundle's relative path, size and SHA-256. Each committed
results manifest is limited to 32 KiB; CI checks this rule. Existing large bundles
were preserved under `reviews/2026-09-06/promotion/archive/` before replacement.
Removing current-tree copies does not rewrite or shrink Git history.

Mixed target execution, latency scheduling and confidence-width scheduling are
archived at commit `56605f0`. Reproduce those experiments in a checkout of that
commit; they have no current serving hooks. Standalone kernel studies remain in
`bench/`. MTP feedback proof uses `bench/mtp_feedback.py` with the original MTP
adapter exported from that commit and an external output directory.

## Exact-ID AIPerf comparison

See the [measured results and token-parity audit](../docs/gb10-aiperf-results.md).

`gb10_aiperf.py` drives NVIDIA AIPerf 0.12.0 against an exclusively owned,
already started server. It uses `/v1/completions` with frozen token-ID prompts,
temperature zero, fixed seed, and ignored EOS to enforce 10,000 input / 1,000
output tokens. Every trial flushes the cache before a separate one-request
AIPerf invocation. Full-generation warmups are marked and excluded by analysis.

```bash
HF_HUB_CACHE=/path/to/existing/huggingface/cache python -m bench.gb10_aiperf \
  --url http://127.0.0.1:30000 --aiperf /path/to/aiperf \
  --inputs /path/to/frozen-inputs.json --output /path/to/new/results \
  --cases code design reasoning --repeats 2
```

Both servers must support `return_token_ids` and streaming
`stream_options.include_usage`. The client validates the actual AIPerf payload,
returned prompt IDs, output IDs, reported usage, finish reason and zero cached
tokens. It retains all raw responses and metrics, and fails on missing or
inconsistent token evidence. Compare output IDs across engines separately;
greedy sampling alone does not establish exact numerical parity. The Sharar
completions reporting change is in the working source; an older production
image does not acquire it until rebuilt.
