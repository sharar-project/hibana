# Evaluating Sharar

How a build or a change is measured from the outside. For looking inside the engine (round cost, kernel
breakdowns, quantization error) see [profiling](../profiling/README.md).

## Evaluate a build: `tools/evaluate.py`

One command serves a build and measures it with
[sharar-eval](https://github.com/abdelfattah-lab/sharar-eval): quality suites and/or AIPerf speed,
both versions recorded, the server stopped afterwards.

```bash
tools/evaluate.py --target flashnext --profile fast --suite quick --speed          # this checkout's engine
tools/evaluate.py --target flashnext --profile fast --suite hard                   # another suite
tools/evaluate.py --target flashnext --profile exact --suite quick --speed \
    --baseline $SHARAR_RESULTS/reviews/<date>-evaluate-flashnext-fast            # paired with an earlier run
```

- **Quality:** `--suite quick|full|hard|long|agentic|all`, or `--tasks gpqa,mmlupro --limit 50`.
  `--max-tokens 32768` replaces every task's token cap (GPQA 8,192, MMLU-Pro and IFEval 4,096 by default),
  so an answer that reasons longer is not scored wrong for being cut off.
- **Speed:** `--speed` runs `evals.speed` on `--workload code|design|reasoning|all`.
- **Acceptance:** neither pins it, so both measure the profile as served.
- **Output:** `$SHARAR_RESULTS/reviews/<date>-evaluate-<name>/` holds `SUMMARY.md`, the sharar-eval run
  directory, the AIPerf run, `/get_server_info` and the logs. A rerun with the same `--name` resumes.

Two runs of the same build differ by under 1% in AIPerf TPS and by about ±1 point on GSM8K (300) and
HumanEval at p ≥ 0.45 (the 27B and Flash-Next): served batches change rounding, so quality is compared
item by item with a paired test, never on point estimates.

## Judge a change to arithmetic

A kernel, dispatch threshold, draft precision, verify shape or proposer passes the GPU suite
(`SHARAR_RUN_GPU=1 uv run pytest tests/gpu`; [CONTRIBUTING](../CONTRIBUTING.md#run-checks)), then is measured against
main: `tools/evaluate.py --speed --suite quick` from the candidate's checkout, paired with the same command from a
checkout of main (`--baseline`; a worktree of main works). The `quick` suite (GSM8K, HumanEval, MGSM, needles) is
compared item by item, and a quality loss paired against main is a regression. Bit-identical outputs are not
required and not a substitute: the target's arithmetic changes with the row count, so outputs drift whenever shapes
move ([numerical limits](operations.md#numerical-limits)). `profiling/flashnext_round.py` isolates round cost and
tokens per round when the speed change needs explaining. A lossy option is judged by the
[lossy-change gate](#the-lossy-change-gate) instead.

## sharar-eval

The quality suites, AIPerf speed and the HTTP serving benchmarks live in
[sharar-eval](https://github.com/abdelfattah-lab/sharar-eval), pinned as the `eval` dependency group in `pyproject.toml`
(`uv sync --group eval`). It only talks to a running server, so the same commands measure other engines; its README
lists them. Run by hand, it needs this configuration's results directory and model cache (`SHARAR_RESULTS`,
`SHARAR_HF_CACHE`, as in `config/gb10.env`). This repository's tools run it with both set and the server managed:
`tools/evaluate.py` (a build), `tools/h2h-pr.py` (a speed change against main;
[CONTRIBUTING](../CONTRIBUTING.md#run-checks)) and `tools/lossy-gate.py` (a lossy option).

## The lossy-change gate

`tools/lossy-gate.py` judges a lossy option ([when](../CONTRIBUTING.md#judge-a-lossy-option)); its docstring is the
canonical description of the three layers (distribution distance, speculation health, a served task battery) and
the verdict rules.

```bash
tools/lossy-gate.py --target flashnext --name fast --profile fast                  # a profile against exact
tools/lossy-gate.py --target flashnext --name recal --profile fast --env SHARAR_W3_CHECKPOINT=/path/to/candidate  # a new calibration against exact
tools/lossy-gate.py --target qwen38 --name x --profile exact --env SHARAR_X=1 --through inprocess   # layers 1-2 only
tools/lossy-gate.py --target flashnext --name fast --profile fast --report           # rebuild the scorecard
```

Do not wrap it in `flock`: it takes `$SHARAR_RESULTS/.gpu.lock` for each phase (at most `--hold-minutes`,
default 45) and waits while another model server or GPU process runs. Layer 3 serves each arm with
`sharar serve <target> --<profile>` on `--port` (30100 by default) without the SSD tier, checks that
`/get_server_info` reports the in-process arm's configuration, and adds sharar-eval's longer suites with
`--battery hard|long|all|agentic`. The baseline arm is cached in `$SHARAR_RESULTS/reviews/lossy-gate-baselines/` and
reused by every candidate. A rerun with the same `--name` resumes; `--through` stops after `inprocess` or `screen`,
or forces `full`. In layer 1, identical distributions read about 99.3% argmax agreement, not 100% (exact logit ties
break by index order; the floor carries the same effect), and relaxed acceptance does not move it: it shows in
layers 2 and 3.

## Nightly and weekly checks on the GB10

`tools/ci/gb10.sh` runs on the self-hosted `gb10` runner (`.github/workflows/{nightly,weekly}-gb10.yml`,
America/New_York):

- **Nightly, 01:00** (skipped when main is unchanged): brings the runner's Python environment up to date
  (`UV_PROJECT_ENVIRONMENT`, default `~/sharar-ci/venv`), runs `tests/unit` and `tests/gpu` from it, then serves and verifies
  the 27B and Flash-Next.
- **Weekly, Sunday 02:00:** `tools/evaluate.py` per target on the fast profile, AIPerf speed plus GSM8K (300)
  and HumanEval, paired with `$SHARAR_RESULTS/ci/baseline-<target>` (re-point the link to adopt a newer run).
  It fails on more than 5% lower TPS or a quality loss of at least 1 point at p < 0.05.

The checks never hold anyone up: they skip when the GPU lock is held or awaited, another Sharar service, GPU
container or GPU process runs, or memory is short, and give way mid-run on the same signs. Settings are in
`~/sharar-ci/gb10.env` on the runner; the script's header has the details.
