# Evaluating Sharar

How a build or a change is measured from the outside. For looking inside the engine (round cost, kernel
breakdowns, quantization error) see [profiling](../profiling/README.md).

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

Two runs of the same image differ by under 1% in AIPerf TPS and by about ±1 point on GSM8K (300) and
HumanEval at p ≥ 0.45 (2026-10-03, both targets): served batches change rounding, so quality is compared
item by item with a paired test, never on point estimates.

## Judge a change to arithmetic

A kernel, dispatch threshold, draft precision, verify shape or proposer passes the GPU suite
(`tools/gpu-tests.sh`), then is measured against main: `tools/evaluate.py --build --speed --suite quick`
for the candidate, paired with the same command on main's image (`--baseline`). Bit-identical outputs are not
required (the target's arithmetic changes with the row count), but a quality loss paired against main is a
regression. `profiling/flashnext_round.py` isolates round cost and tokens per round when the speed change
needs explaining. The rules are in [CONTRIBUTING](../CONTRIBUTING.md#run-checks).

## End-to-end speed (sharar-eval)

The HTTP speed and serving benchmarks live in sharar-eval (`evals/speed/`, moved 2026-10-02). They only talk
to a running server, so the same commands measure Sharar, SGLang or Atlas:

```bash
cd ../sharar-eval
.venv/bin/python -m evals.speed --url http://127.0.0.1:30004            # AIPerf: frozen 10K/1K, cold prefix, BS1 (formerly `sharar benchmark`)
.venv/bin/python -m evals.speed.sharegpt ...                            # ShareGPT buckets at several concurrencies
.venv/bin/python -m evals.speed.competitive ...                         # engine-neutral native /generate cells
.venv/bin/python -m evals.speed.engine_matrix ...                       # Sharar / SGLang / Atlas token-input matrix
.venv/bin/python -m evals.speed.local ... / evals.speed.agent ...       # local-use and agent-shaped workloads
```

The frozen prompts are byte-identical to the ones here: `sharar/benchmark_workloads.py` stays because the
in-process tools (`profiling/inputs.py`, `tools/check_distribution.py`) use it, and both repositories pin the
same digests in their tests.

## The lossy-change gate

A lossy speed option is judged by `tools/lossy-gate.py` (rules and layers in
[CONTRIBUTING](../CONTRIBUTING.md#judge-a-lossy-option) and the tool's docstring):

```bash
tools/lossy-gate.py --target flashnext --name fast --profile fast                  # a profile against exact
tools/lossy-gate.py --target flashnext --name recal --profile fast --env SHARAR_W3_CHECKPOINT=/path/to/candidate  # a new calibration against exact
tools/lossy-gate.py --target qwen38 --name x --profile exact --env SHARAR_X=1 --through inprocess   # layers 1-2 only
tools/lossy-gate.py --target flashnext --name fast --profile fast --report           # rebuild the scorecard
```

Do not wrap it in `flock`: it takes `$SHARAR_RESULTS/.gpu.lock` for each phase (at most `--hold-minutes`,
default 45) and waits while another model container runs. Layers 1 and 2 run `tools/gate/lossy_gate.py`
(prompts `tools/gate/workloads/lossy_v1.json`) in a container of the image with only `tools/gate/` mounted
over it (the image's own engine), built as `sharar serve` builds the executor; layer 3 serves each arm with
`run-sharar.sh` as container `sharar-lossy-gate` on `--port` (30100 by default), checks that
`/get_server_info` reports the in-process arm's configuration, and runs sharar-eval's `evals.suite`
(`--evals`, beside this checkout by default, with its `.venv`): `quick` as the screen, `full` when the screen
is close, and `hard` / `long` (RULER v2) / both / `agentic` (Terminal-Bench 2.1) with
`--battery hard|long|all|agentic`. Served arms run with the cold tier off (`SHARAR_COLD_TIER_GIB=0`) so
Terminal-Bench's task containers have host memory. The baseline arm is cached in
`$SHARAR_RESULTS/reviews/lossy-gate-baselines/` (keyed by image, baseline settings, prompts, worker and eval
code) and reused by every candidate. A rerun with the same `--name` resumes; `--through` stops after
`inprocess` or `screen`, or forces `full`.

## Nightly and weekly checks on the GB10

`tools/ci/gb10.sh` runs on the self-hosted `gb10` runner (`.github/workflows/{nightly,weekly}-gb10.yml`,
America/New_York):

- **Nightly, 01:00** (skipped when main is unchanged): builds `sharar:ci`, runs `tests/unit` and `tests/gpu`
  in it, then serves and verifies both targets. When all of that passes on main, it publishes the image to
  `ghcr.io/abdelfattah-lab/sharar` as `main-<commit>` and `latest`, the launchers' default image.
- **Weekly, Sunday 02:00:** `tools/evaluate.py` per target on the fast profile, AIPerf speed plus GSM8K (300)
  and HumanEval, paired with `$SHARAR_RESULTS/ci/baseline-<target>` (re-point the link to adopt a newer run).
  It fails on more than 5% lower TPS or a quality loss of at least 1 point at p < 0.05.

The checks never hold anyone up: they skip when the GPU lock is held or awaited, another GPU container or
process runs, or memory is short, and give way mid-run on the same signs. Settings are in
`~/sharar-ci/gb10.env` on the runner; the script's header has the details.
