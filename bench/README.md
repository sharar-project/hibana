# Benchmarking Sharar

## Quick TPS command

Use an already running server built from the current source, with exclusive
GPU use and the host memory guard. The benchmark client runs on the host;
it flushes the server's prefix cache but does not start or stop services.

For experiments that mount changed CUDA sources into a container, set a
separate `TORCH_EXTENSIONS_DIR` from the serving container. A shared Ninja
build directory can retain an experimental binary when an older image's
source timestamps precede the compiled object. Verify the loaded extension
when restoring a service; its image identity alone is insufficient.

```bash
tools/benchmark.sh
# Equivalent installed entry point:
sharar benchmark
```

Defaults: the frozen `code` task, 10,000 input tokens, exactly 1,000 output
tokens, temperature zero, seed 42, concurrency one, one excluded full warmup
and three timed requests. `--workload all` adds `design` and `reasoning`.
The three built-in prompts reproduce the first input row of each task in the
recorded [AIPerf TPS comparison](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/gb10-tps.md). The packaged historical
source fixture is independent of edits to current engine code. Prompt
tokenization occurs outside timing; each run saves the exact IDs it uses.

The `code` workload changed on 2026-09-15 (`gb10-coding-v2`): the earlier
prompt made the model answer with a tool call and stop at output token 98, so
902 of its 1,000 tokens were a forced post-EOS continuation. The new prompt
tells the model it has no tools and runs the full budget; `--workload code_v1`
keeps the old prompt for comparisons with reports dated before the change.

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

## The family gate

Any change that alters arithmetic (a kernel, a dispatch threshold, a draft
precision, a verify shape, a proposer) is judged on the frozen 21-task family
set, not on the three benchmark prompts and not on bit-identical outputs.
The gate runs each pass in its own exclusive window and reports paired
per-task log-ratios of decode throughput, round cost and tokens per round
with bootstrap 95% intervals, plus the noise floor between two native passes:

```bash
tools/family-gate.py --label my-change --order N,C1,N --candidates bench.gate_hooks:my_hook
tools/family-gate.py --label stable --order N,C1,N --candidates env:SHARAR_STABLE_NUMERICS=1
```

A candidate is a hook in `bench/gate_hooks.py` (`fn(stage, executor, args)`
at `model`, `spec` and `ready`), or `env:KEY=VALUE`; `|bf16` after a
candidate runs that pass with the BF16 draft. Inputs come from
`$SHARAR_RESULTS/datasets/family-eval-v1/` and are hash-checked;
`python3 -m bench.family_inputs freeze` regenerates them byte-identically from
`bench/workloads/family_v1.json` and the benchmark prompts. Read the
`families` row of the comparison (101 families) as the decision, the `benchmarks`
row as context, and never a single task; the round-cost interval resolves 0.05%,
the tokens-per-round interval is about ±3% wide for any continuation-changing
change on the 101-family set ([study](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/family-set-v3.md); it was ±7%
on the 18-family `v2` set, [study](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/family-gate.md)). Results and logs land in
`$SHARAR_RESULTS/reviews/<date>-family-gate/<label>/`. Row-count changes
should also be compared under `SHARAR_STABLE_NUMERICS=1`, the shape-stable
research baseline, so that continuation drift does not masquerade as a gain.

## The lossy-change gate

A lossy speed option is judged by `tools/lossy-gate.py` (rules and layers in
[CONTRIBUTING](../CONTRIBUTING.md#judge-a-lossy-option) and the tool's docstring):

```bash
tools/lossy-gate.py --target flashnext --name fast --profile fast                  # a profile against exact
tools/lossy-gate.py --target flashnext --name recal --profile fast --env SHARAR_W3_CHECKPOINT=/path/to/candidate  # a new calibration against exact
tools/lossy-gate.py --target qwen38 --name x --profile exact --env SHARAR_X=1 --through inprocess   # layers 1-2 only
tools/lossy-gate.py --target flashnext --name fast --profile fast --report           # rebuild the scorecard
```

Do not wrap it in `flock`: it takes `$SHARAR_RESULTS/.gpu.lock` for each phase (at
most `--hold-minutes`, default 45) and waits while another model container runs.
Layers 1 and 2 run `bench/lossy_gate.py` in a container of the image with only
`bench/` mounted over it (the image's own engine), built as `sharar serve` builds
the executor; layer 3 serves each arm with `run-sharar.sh` as container
`sharar-lossy-gate` on `--port` (30100 by default), checks that
`/get_server_info` reports the in-process arm's configuration, and runs
`sharar-eval`'s `evals.suite` (`--evals`, beside this checkout by default, with its `.venv`):
`quick` as the screen, `full` when the screen is close, and `hard` / `long` (RULER v2) / both with
`--battery hard|long|all`.
The baseline arm is cached in `$SHARAR_RESULTS/reviews/lossy-gate-baselines/`
(keyed by image, baseline settings, prompts, worker and eval code) and reused by
every candidate. A rerun with the same `--name` resumes; `--through` stops after
`inprocess` or `screen`, or forces `full`.

## Current measurement drivers

| Driver | Purpose |
|---|---|
| `sharar benchmark` / `tools/benchmark.sh` | Installed AIPerf cold BS1 iteration command |
| `bench.gb10_family_eval` | Frozen family native comparisons and paired statistics |
| `tools/family-gate.py`, `bench/gate_hooks.py`, `bench/family_inputs.py` | The family gate: windows, candidate hooks, reproducible frozen inputs |
| `tools/lossy-gate.py`, `bench/lossy_gate.py`, `bench/workloads/lossy_v1.json` | The lossy-change gate: distribution distance, speculation health and the paired task battery, one verdict |
| `bench.gb10_bs4_engine` | Controlled verifier-width/native BS4 comparison |
| `bench.gb10_bs4_http` | Finite and replenished BS4 HTTP workloads |
| `bench.gb10_agent_retention` | Cold controls and interleaved native cache resumes |
| `bench.gb10_agent_http` | Growing simulated conversations and mixed HTTP arrivals |
| `bench.gb10_runtime_ab` | Current calibrated target and selected drafter over cold, warm and mixed HTTP workloads at concurrency 1, 2 and 4; optional candidate hook |
| `bench.gb10_round_profile` | Separate unprofiled, Nsight and CUDA-event phase windows on frozen prompts; `--draft` selects the served drafter |
| `bench.gb10_matrix` | Real-operand matrix diagnostics |
| `sharar-eval` (separate repository) | HTTP evaluations in one command (`evals.suite`: quick, full, hard), plus h2h, test-time scaling, long context and output identity |
| `bench/w3_bench.py` | 3-bit decode GEMM: correctness against the dequantized reference and DRAM bandwidth per decode shape at 16, 32 and 64 rows (weights rotated past L2, CUDA-graph timing), split-K sweep |
| `bench/w3_calibrate.py` | Fit checksummed W3 target copies from FP8 source weights and frozen activation inputs; [workflow](../docs/w3-calibration.md) |

Use each driver's `--help` and its linked [performance evidence](../docs/performance.md)
to freeze inputs, output budgets, warmups, precision and geometry before running.
Run candidate and control in separate fresh, guarded GPU processes. A source
reorganization does not make old protocols current or establish new TPS gains.

Shared recording, HTTP, identity, retained-batch and CUDA-timing helpers live in
[support](support/README.md). Completed experiments are in git history
(the `archive/*` tags keep the study branches). Some older
drivers remain top-level because maintained callers still share their workload
or numerical helpers; moving them wholesale would hide an active dependency.

See the [decision archive](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/README.md) for historical studies and
the [old command guide](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/benchmark-guide-2026-09-11.md) for commands
at their original revisions. Raw tensors, outputs and traces stay outside Git.

## Draft adaptation tools

- `bench/trace_corpus.py` generates traced traffic over HTTP against a server
  started with traces on (`./sharar-ctl serve`): `--dataset` replays ShareGPT
  first turns, `--repo DIR` sends coding requests over a repository's files;
  `--exclude LOG` keeps a test set's file/task pairs out of the traffic.
- `bench/draft_ab.py` replays a trace directory's newest held-out prompts (or
  `--all`) at batch one against a running server and reports throughput,
  tokens per round and output hashes, for A/B of drafts across windows.
- `bench/dflash2_reference.py` runs inside the runtime container on an
  exclusive GPU and checks the training forward against the serving proposer
  and the publisher's reference model, and the objective against SpecForge,
  then runs a short training smoke. `bench/support/target_features.py`
  recomputes raw target features for it.
- `bench/draft_validate.py` is the earlier publisher-oracle gate for external drafts.

See [draft adaptation](../docs/draft-adaptation.md).
