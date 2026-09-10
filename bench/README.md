# Benchmarking Sharar

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
recorded [AIPerf TPS comparison](../docs/archive/studies/gb10-tps.md). The packaged historical
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

## Current measurement drivers

| Driver | Purpose |
|---|---|
| `sharar benchmark` / `tools/benchmark.sh` | Installed AIPerf cold BS1 iteration command |
| `bench.gb10_family_eval` | Frozen family native comparisons and paired statistics |
| `bench.gb10_bs4_engine` | Controlled verifier-width/native BS4 comparison |
| `bench.gb10_bs4_http` | Finite and replenished BS4 HTTP workloads |
| `bench.gb10_agent_retention` | Cold controls and interleaved native cache resumes |
| `bench.gb10_agent_http` | Growing simulated conversations and mixed HTTP arrivals |
| `bench.gb10_audit`, `bench.gb10_matrix` | Generation profiling and real-operand matrix diagnostics |

Use each driver's `--help` and its linked [performance evidence](../docs/performance.md)
to freeze inputs, output budgets, warmups, precision and geometry before running.
Run candidate and control in separate fresh, guarded GPU processes. A source
reorganization does not make old protocols current or establish new TPS gains.
The agent HTTP `--schedule` flag explicitly selects the rejected archived
scheduler; it is an experiment and does not change serving defaults.

The old `bench.gb10_prefix_policy` command exits with a migration message:
current retention makes its two capacity labels identical. Reproduce its
duplicate-save study at `ea6a9de`; measure current retention using the drivers above.

Shared recording, HTTP, identity, retained-batch and CUDA-timing helpers live in
[support](support/README.md). Completed experimental kernels and policies live
in [archive](archive/README.md), with their reference tests retained. Some older
drivers remain top-level because maintained callers still share their workload
or numerical helpers; moving them wholesale would hide an active dependency.

See the [decision archive](../docs/archive/README.md) for historical studies and
the [old command guide](../docs/archive/benchmark-guide-2026-09-11.md) for commands
at their original revisions. Raw tensors, outputs and traces stay outside Git.
