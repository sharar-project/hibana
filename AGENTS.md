# Agent instructions for Hibana

These instructions apply to every AI-assisted change to Hibana. [CONTRIBUTING.md](CONTRIBUTING.md) is the full guide;
this file holds what an agent needs on every task and the rules agents most often break.

Hibana serves a few pinned models on one NVIDIA GB10 (DGX Spark), or two linked by their ConnectX ports. Prefer a
small implementation with explicit tensor ownership over a registry or framework, and match the code around you.

## Contribution policy

- A human submits and owns every change: they read every changed line, ran the checks below and can defend it.
- The pull request says that AI assistance was used, lists the commands run with their results, and carries the
  measurements the change needs (below).
- Search open pull requests and issues first; do not duplicate work in flight.
- No pull requests for a lone typo or style tweak. Keep a pull request to the files the change needs: no drive-by
  edits, few new test files, a handful of logical commits.

## Environment

Use `uv`, never the system `python3` with `pip`:

```bash
uv sync                                   # the GB10 environment from uv.lock
uv run pytest tests/unit                  # CPU tests
python3 tools/check_links.py              # documentation links
bash tools/check.sh setup-cpu             # off the GB10: a CPU-only environment, then the same unit tests
```

The GPU suite needs the GPU to itself; check that nothing else is serving or benchmarking first:

```bash
HIBANA_RUN_GPU=1 uv run pytest tests/unit tests/gpu   # or: tools/check.sh gpu-suite
```

A version change touches both `constraints/` and `pyproject.toml`'s `constraint-dependencies`, then `uv lock`.

## Rules

- **No history in tracked files.** Code, comments, tests and docs carry no issue or pull-request numbers and no
  dates (pinned revisions and hashes excepted). Say what the code does and why; history goes in the commit message
  and the pull request.
- **Settings** are `EngineConfig` fields in `hibana/engine/config.py`, read through `current()` where the thing they
  shape is built. Never read a `HIBANA_*` variable from `os.environ` elsewhere; a unit test fails on it.
- **Docs follow behavior.** A changed default or a new setting updates [operations](docs/operations.md) in the same
  change; a change in behavior updates the page that describes it.
- **Arithmetic changes** (kernels, dispatch thresholds, draft precision, verify shapes, proposers) pass the GPU suite
  and are measured against main, paired item by item ([evaluation](docs/evaluation.md#judge-a-change-to-arithmetic)).
  Bit-identical outputs are neither required nor a substitute.
- **Lossy options** never enter `exact`, and enter `fast` only through `tools/lossy_gate.py`
  ([judge a lossy option](CONTRIBUTING.md#judge-a-lossy-option)).
- **Speed changes** post `tools/h2h_pr.py`'s table in the pull request, run on the targets the change touches.
- **Frozen workloads** (`tools/gate/workloads/`, `profiling/workloads/`, `hibana/benchmark_workloads.py`) keep their
  content and checksums.

## Before changing an area

Read the matching section of CONTRIBUTING.md first:

| Change | Section |
|---|---|
| A new setting | [Add a setting](CONTRIBUTING.md#add-a-setting) |
| A new model | [Add a target](CONTRIBUTING.md#add-a-target) |
| A drafter or speculation | [Add a proposer](CONTRIBUTING.md#add-a-proposer), [change the drafter](CONTRIBUTING.md#change-the-drafter) |
| A kernel, weight format or CUDA graph | [Add a GB10 kernel](CONTRIBUTING.md#add-a-gb10-kernel) |
| Caching, admission or eviction | [Add a cache policy](CONTRIBUTING.md#add-a-cache-policy) |
| Memory, unified-memory or hardware tricks | [Propose a hardware optimization](CONTRIBUTING.md#propose-a-hardware-optimization) |

The design is in [architecture](docs/architecture.md) and [interfaces](docs/interfaces.md).

## Editing these instructions

This file loads on every agent request: keep it under 100 lines. Add a rule only for a mistake agents actually make,
prefer a test or check that enforces it, and put area-specific detail in CONTRIBUTING.md. Keep agent guidance in
this one file rather than in tool-specific copies such as `CLAUDE.md`.
