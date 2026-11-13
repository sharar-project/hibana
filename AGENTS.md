# Agent instructions for Hibana

Read [CONTRIBUTING.md](CONTRIBUTING.md) first; it binds humans and agents alike. This file sums up what matters when an
agent does the work.

Hibana serves a few pinned models on one NVIDIA GB10 (DGX Spark), or two linked by their ConnectX ports. Prefer a
small implementation with explicit tensor ownership over a registry or framework, and match the code around you.

## AI policy

- A human owns every change: they read every changed line, ran it (on a GB10 when it touches the engine) and can
  explain it without AI help.
- Push, open pull requests or post on GitHub only when the human asks in the session, never on your own.
- Never report a test, benchmark or evaluation as run when it was not. If you could not run something, say so and say
  what you ran instead.
- The pull request says that AI assisted, lists the commands run with their results and carries the measurements the
  change needs (below). Search open pull requests and issues first; do not duplicate work in flight.

## Layout

```
hibana/     the engine, installed as the `hibana` package and command; packages in docs/architecture.md#layout
tests/      unit/ (CPU) and gpu/ (HIBANA_RUN_GPU=1)
tools/      checks, the lossy gate, h2h_pr.py and evaluate.py
profiling/  in-process profilers and their frozen workloads
docs/       operations, API, architecture and one page per model
```

## Environment

Use `uv`, never the system `python3` with `pip`:

```bash
uv sync                                   # the GB10 environment from uv.lock
uv run pytest tests/unit                  # CPU tests
python3 tools/check_links.py              # documentation links
bash tools/check.sh setup-cpu             # off the GB10: a CPU-only environment, then the same unit tests
HIBANA_RUN_GPU=1 uv run pytest tests/unit tests/gpu   # the GPU suite; check that nothing else uses the GPU
```

A version change touches both `constraints/` and `pyproject.toml`'s `constraint-dependencies`, then `uv lock`.

## Rules

- **No history in tracked files.** Code, comments, tests and docs carry no issue or pull-request numbers and no dates
  (pinned revisions and hashes excepted); history goes in the commit message and the pull request.
- **Settings** are `EngineConfig` fields in `hibana/engine/config.py`, read through `current()` where the thing they
  shape is built. Never read a `HIBANA_*` variable from `os.environ` elsewhere; a unit test fails on it.
- **Docs follow behavior.** A changed default or a new setting updates [operations](docs/operations.md) in the same
  change; a change in behavior updates the page that describes it.
- **Arithmetic changes** (kernels, dispatch thresholds, draft precision, verify shapes, proposers) pass the GPU suite
  and are measured against main, paired item by item ([evaluation](docs/evaluation.md#judge-a-change-to-arithmetic)).
- **Lossy options** enter `fast` only through `tools/lossy_gate.py`, never `exact`
  ([judge a lossy option](CONTRIBUTING.md#judge-a-lossy-option)). **Speed changes** post `tools/h2h_pr.py`'s table.
- **Frozen workloads** (`tools/gate/workloads/`, `profiling/workloads/`, `hibana/benchmark_workloads.py`) keep their
  content and checksums.
- **Keep a pull request lean:** one change, only the files it needs, no drive-by edits.

## Tests

[Write a test](CONTRIBUTING.md#write-a-test) has the rules. The parts agents most often get wrong:

- Verify a fix by re-running what failed. Verification alone does not call for a new test.
- Before adding a test, name the future change that would turn it red; if the only answer is editing the test, do
  not write it.
- Extend an existing test file or `parametrize` first. Ask before adding a test file, fixture or helper, and before
  deleting a test.

## Code comments

A comment explains a non-obvious why, with the measurement when one decided it, and never restates the code. Keep it
to a line or two; use ASCII (`-`, `->`).

## Commits

[Conventional Commits](CONTRIBUTING.md#commit-messages), one line, imperative, lowercase, no trailing period:
`perf(bolt): fuse the mamba prefill scan and grouped normalization`. Pull requests are squash-merged, so the title
follows the same form. Commit only when the human asks; for attribution use `Assisted-by: <agent>`, not
`Co-authored-by`.

## Before changing an area

| Change | Read first |
|---|---|
| A setting, a model | [Add a setting](CONTRIBUTING.md#add-a-setting), [add a target](CONTRIBUTING.md#add-a-target) |
| A drafter or speculation | [Add a proposer](CONTRIBUTING.md#add-a-proposer), [change the drafter](CONTRIBUTING.md#change-the-drafter) |
| A kernel, weight format or CUDA graph | [Add a GB10 kernel](CONTRIBUTING.md#add-a-gb10-kernel) |
| Caching, admission or eviction | [Add a cache policy](CONTRIBUTING.md#add-a-cache-policy) |
| Memory or hardware tricks | [Propose a hardware optimization](CONTRIBUTING.md#propose-a-hardware-optimization) |

## Editing these instructions

This file loads on every agent request: keep it near 100 lines. Add a rule only for a mistake agents actually make,
prefer a test or check that enforces it, and put area-specific detail in CONTRIBUTING.md.
