# Benchmarking Sharar

## Evaluate a build: `tools/evaluate.py`

One command serves a build and measures it with
[sharar-eval](https://github.com/abdelfattah-lab/sharar-eval): quality suites and/or AIPerf speed,
both versions recorded, the server stopped afterwards.

```bash
tools/evaluate.py --target flashnext --profile fast --suite quick --speed          # the config's image
tools/evaluate.py --target flashnext --profile fast --build --suite hard           # an image built from this commit
tools/evaluate.py --target flashnext --profile exact --suite quick --speed \
    --baseline $SHARAR_RESULTS/reviews/<date>-evaluate-flashnext-fast            # paired with an earlier run
```

- **Quality:** `--suite quick|full|hard|long|agentic|all`, or `--tasks gpqa,mmlupro --limit 50`.
- **Speed:** `--speed` runs `evals.speed` on `--workload code|design|reasoning|all`.
- **Acceptance:** neither pins it, so both measure the profile as served.
- **Output:** `$SHARAR_RESULTS/reviews/<date>-evaluate-<name>/` holds `SUMMARY.md`, the sharar-eval run
  directory, the AIPerf run, `/get_server_info` and the logs. A rerun with the same `--name` resumes.

The lossy gate below is the decision tool for a lossy option. It compares two configurations and adds
in-process layers; `evaluate.py` measures one build, optionally against a baseline run.

## End-to-end speed (sharar-eval)

The HTTP speed and serving benchmarks live in
[sharar-eval](https://github.com/abdelfattah-lab/sharar-eval) (`evals/speed/`, moved 2026-10-02). They
only talk to a running server, so the same commands measure Sharar, SGLang or Atlas:

```bash
cd ../sharar-eval
.venv/bin/python -m evals.speed --url http://127.0.0.1:30004            # AIPerf: frozen 10K/1K, cold prefix, BS1 (formerly `sharar benchmark`)
.venv/bin/python -m evals.speed.sharegpt ...                            # ShareGPT buckets at several concurrencies
.venv/bin/python -m evals.speed.competitive ...                         # engine-neutral native /generate cells
.venv/bin/python -m evals.speed.engine_matrix ...                       # Sharar / SGLang / Atlas token-input matrix
.venv/bin/python -m evals.speed.local ... / evals.speed.agent ...       # local-use and agent-shaped workloads
```

The frozen prompts are byte-identical to the ones here: `sharar/benchmark_workloads.py` stays because the
in-process benchmarks (family inputs, `tools/check_distribution.py`) use it, and both repositories pin
the same digests in their tests.

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
`quick` as the screen, `full` when the screen is close, and `hard` / `long` (RULER v2) / both /
`agentic` (Terminal-Bench 2.1) with `--battery hard|long|all|agentic`. Served arms run with the cold tier
off (`SHARAR_COLD_TIER_GIB=0`) so Terminal-Bench's task containers have host memory.
The baseline arm is cached in `$SHARAR_RESULTS/reviews/lossy-gate-baselines/`
(keyed by image, baseline settings, prompts, worker and eval code) and reused by
every candidate. A rerun with the same `--name` resumes; `--through` stops after
`inprocess` or `screen`, or forces `full`.

## Current measurement drivers

| Driver | Purpose |
|---|---|
| `bench.gb10_family_eval` | Frozen family native comparisons and paired statistics |
| `tools/family-gate.py`, `bench/gate_hooks.py`, `bench/family_inputs.py` | The family gate: windows, candidate hooks, reproducible frozen inputs |
| `tools/lossy-gate.py`, `bench/lossy_gate.py`, `bench/workloads/lossy_v1.json` | The lossy-change gate: distribution distance, speculation health and the paired task battery, one verdict |
| `sharar-eval` (separate repository) | Everything over HTTP: quality suites (`evals.suite`: quick, full, hard, long, agentic), speed and serving benchmarks (`evals.speed`), h2h, test-time scaling, output identity |
| `bench/w3_bench.py` | 3-bit decode GEMM: correctness against the dequantized reference and DRAM bandwidth per decode shape at 16, 32 and 64 rows (weights rotated past L2, CUDA-graph timing), split-K sweep |
| `bench/w3_calibrate.py` | Fit checksummed W3 target copies from FP8 source weights and frozen activation inputs; [workflow](../docs/w3-calibration.md) |
| `bench/flashnext_round.py`, `flashnext_profile.py`, `flashnext_parity.py`, `flashnext_quant.py` | Flash-Next round cost and phase profile, parity against the reference, trunk quantization |

The 27B serving-study drivers (batch-4 and agent HTTP experiments, acceptance replay, matrix and round
profiles, Phase 7 checks) were removed on 2026-10-02; they and their results are on the
[archive branch](https://github.com/abdelfattah-lab/sharar/tree/archive/pre-cleanup-2026-09-26) and in git history.

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

See [draft adaptation](../docs/draft-adaptation.md).
