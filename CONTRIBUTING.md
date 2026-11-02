# Contributing to Sharar

Sharar specializes in local inference on one GB10. Keep the current Qwen
target checkpoint, its vocabulary head and the numerical serving profile fixed
unless a change explicitly validates a new one. Favor a small implementation with
explicit tensor ownership over a registry or framework. Read [the architecture](docs/architecture.md) and
[the interfaces](docs/interfaces.md) first; open work is in the GitHub issues.

Tracked files (code, comments, tests, docs, configs) carry no issue or pull-request numbers and no dates; pinned
revisions and hashes and archive tag names are the exceptions. Say what the code does and why, with the measurement
when it decides a choice. The history (which issue asked for it, which pull request changed it, what was tried and
measured on the way) belongs in the commit message and the pull request, not the docs.

## Run checks

On the GB10, `uv sync` builds the environment from `uv.lock` and `uv run pytest tests/unit` runs the control,
scheduling, cache, sampler, HTTP and transaction tests. Off the GB10, `bash tools/check.sh setup-cpu` creates a
CPU-only environment with pip (the pinned CPU PyTorch wheel and `.[test]`; no model, FlashInfer or GPU) and runs the
same tests; the GitHub CPU workflow uses it. The tested versions are in `constraints/`; `pyproject.toml`'s
`[tool.uv] constraint-dependencies` repeat them for `uv.lock`, so a version change touches both and `uv lock`.

Also run `python3 tools/check_links.py` for documentation links, and `.venv-cpu/bin/python tools/check_distribution.py` to build an sdist and
wheel and probe installed runtime assets outside the checkout. CI runs these
checks; a source-only or editable installation does not validate packaging.

The GPU suite (`tests/gpu`, every target; a module on a real checkpoint skips without it) runs with
`SHARAR_RUN_GPU=1`; without it the GPU modules are not collected:

```bash
SHARAR_RUN_GPU=1 uv run pytest tests/unit tests/gpu      # or: tools/check.sh gpu-suite
SHARAR_RUN_GPU=1 uv run pytest tests/gpu/test_qsa.py -x
```

`tests/conftest.py` points it at the model caches, the calibrated 3-bit checkpoint and `$SHARAR_RESULTS` as
`sharar serve` resolves them (the environment wins). Preserve frozen
workload content and checksums (`tools/gate/workloads/`, `profiling/workloads/`,
`sharar/benchmark_workloads.py`) when moving readers.

The modules on the real 27B share one load (the session fixture `qwen38` in `tests/gpu/conftest.py`), run last,
and build their executors on it with bounded pools: 35,000 KV pages and 12 recurrent slots. Keep the memory guard active for the run (`uv run python -m sharar.launch.guard --pid PID`). The loader
requires a 24 GiB allowance of `MemAvailable` above the host floor (30 GiB at the default 6), based on the measured 21.3 GiB
streaming-load peak. Runtime separately checks the
prospective pool/draft budget before allocation.
Performance measurements need exclusive GPU use. The check command does not
stop or replace a running service.

Anything that changes arithmetic (kernels, dispatch thresholds, draft precision, verify shapes, proposers) passes
the GPU suite, then is measured against main, paired item by item, as [evaluation](docs/evaluation.md#judge-a-change-to-arithmetic)
describes; bit-identical outputs are neither required nor a substitute. When a change alters a default or adds a
setting, update [operations](docs/operations.md) in the same change.

A change meant to make serving faster posts `tools/h2h_pr.py`'s table in its pull request, so changes compare on one
workload and their gains stack:

    tools/h2h_pr.py --target flashnext --pr <branch>      # and --target 27b when the change touches the 27B

It serves main and the branch in turn and runs sharar-eval's h2h on `fast` (1 and 4 requests) and `throughput`
(8 to 128 closed-loop streams), greedy and T 0.7; the tool's docstring has the details. About 1.5 hours a target;
`--repeats 2` when the difference is within a few percent.

## Judge a lossy option

A lossy option changes what the model emits on purpose (low-bit weights, fewer or pruned experts, relaxed acceptance,
a reduced draft vocabulary). It never enters the `exact` profile, and it enters `fast` (and so `throughput`) only
through the lossy-change gate, which pairs the candidate against `exact` on the same engine:

```bash
tools/lossy_gate.py --target flashnext --name my-option --profile exact --env SHARAR_X=1
tools/lossy_gate.py --target qwen38 --name fast --profile fast
```

The layers, the verdict rules and the outputs are in the docstring of [`tools/lossy_gate.py`](tools/lossy_gate.py);
[evaluation](docs/evaluation.md#the-lossy-change-gate) covers running it.

## Add a setting

Settings are fields of `EngineConfig` (`sharar/engine/config.py`): a `SHARAR_*`
name, its default and a one-line comment. Read `current()` where you build the
thing the setting shapes (a model, a linear, a proposer, a graph), not at import
and not per call on the hot path; `tests/unit/test_engine_config.py` fails on any
other `SHARAR_*` environment read under `sharar/`. The launcher keeps forwarding
the same variable names; `/get_server_info` reports the installed value.

## Add a target

A target is a pinned checkpoint with a `Target` value in `sharar/target.py`
(geometry, profile, load budget, snapshot lookup, host validation) and a
model package under `sharar/model/` whose forward returns `ForwardResult`
and exposes `.target`. Register the loader in `target.load_model` under a
new `Target.family`; model-specific facts become optional `Target` fields
(absent for the 27B), per-slot state beyond the GDN's goes through
`StatePool.register_aux` (`register_state`), per-page rows through
`KvPool.register_page_aux` (`register_pages`), side inputs a forward needs
through `forward_extras()` on `ForwardContext.extras` (never new
`ForwardContext` fields or model checks in `engine/`; the one exception is
image input, a request input every target with a vision tower shares: the
model's optional `vision`, typed by `engine/contracts.VisionEncoder`, is all
`engine/` reads of it, and its embeddings reach prefill as
`ForwardContext.media_rows` / `media_embeds`), and a family's model cache is named by
`Target.cache_env`. `sharar/model/flashnext/` is the worked example. The
engine, pools and graphs read `executor.target`; do not add module-level
constants for a new model, and keep `config.py` describing the 27B for the
tools that import it. Validate a new target with layer-by-layer parity
against a reference server, the CPU contracts, the GPU lifecycle suites and a
`tools/evaluate.py` run (quality and speed) before it is served.

## Add a proposer

1. Put the adapter in `sharar/proposers/`. Implement `Proposer` and declare
   width, target-feature requirements and persistent/scratch bytes. Use
   `tests/ngram_proposer.py` (the tests' proposer) for a feature-free example, `mtp.py`
   for a sequential draft, and `parallel.py` for conditional DFlash2 proposals and draft-cache handles.
2. Own draft weights, KV, pending tokens and retained features inside the
   adapter. Declare exactly which post-block layers or final-normalized rows
   you need. Choose the adapter once with `Executor.set_proposer()` before
   graph capture. A CLI shortcut belongs in `Executor.enable_spec()` and the
   CLI choices; scheduler, radix internals and HTTP request handling stay out
   of the change.
3. Declare optional features in `ProposerRequirements.capabilities`
   (`engine/contracts.py`): device acceptance feedback, checkpoint
   feature cache, SSD-tier payload, a configurable draft head.
   The engine reads the flags, never probes for methods, and
   `check_capabilities` rejects a flag without its protocol. End the draft in a
   draft head (`proposers/draft_head.py`: the target's head by default, or a
   shortlist) so cheaper heads apply to every speculator; draft depth comes from
   the spec driver's depth policy (`engine/depth.py`).
4. Add CPU tests for readiness, fallback, boundary/stop behavior and cleanup.
   Exercise the shared verifier using the tiny target in `test_contracts.py`.
   Then run the GPU continuation suite, feature lifetime checks and a fixed
   workload comparison against the unchanged adapter.

The verifier accepts flat proposals: MTP k≤8, DFlash2 k≤15.
External sampled proposals include their actual transformed conditional q,
including selector effects. `engine/rejection.py` is the common reference
algorithm; test tiny distributions independently of model rounding. Sampled
requests are verified as point masses (MTP's argmax drafts) or with their q (DFlash2, and MTP's sampled drafts); a
`greedy_only` proposer's sampled requests use ordinary decode. Tree proposals are not implemented.
Declare prospective memory in `cache/spec_budget.py`, and retain/restore draft
features through identity-keyed metadata without taking ownership of target
pages. Both full snapshots and accepted-state reconstruction use the same
StateManager commit and stop/continuation tests.

## Change the drafter

Sharar serves the publisher's drafters; `--draft DIR` serves another DFlash2
checkpoint of the same architecture, trained elsewhere (a trainer on recorded
traffic is archived at the tag `archive/draft-training`). Trace format changes
bump `FORMAT_VERSION` in `engine/traces.py` and keep the reader accepting older
records.

## Add a GB10 kernel

A new weight format implements the `Linear` interface (`weights/linear.py`:
call, `out_features`/`in_features`, `bytes`, `dequantize()`, optional `w3`
copy) and keeps its own row-count dispatch; models list their projections with
`linears()` so 3-bit copies (`w3.attach`) and calibration work on any model.
Attention calls plan through `kernels/attention_plan.py` (`AttentionPlanner`, one
per KV geometry: host index pointers, `plan_guard` for eager wrappers, no guard
for graph-bound buffers); new CUDA graphs use `engine/graph_set.py` (`pick`,
`capture`) and keep their own static buffers and body.

Keep the semantic operation in `kernels/ops.py` or its existing model module;
put the specialized implementation in `kernels/`. Select it explicitly at
initialization. State the supported shapes, layouts, dtypes, scratch needs,
stream and graph behavior. Preserve an executable reference path for the
comparison. Validate numerical error on representative and adversarial inputs,
then run the affected real-model checks and measure end-to-end emitted tokens
and latency. A faster isolated kernel is insufficient if it adds transfers or
slows prefill/verification. Version the artifact inputs once the profile is
ready for AOT.

Prefill sequence metadata belongs in `model/gdn_plan.py`, once per forward.
Keep CPU lengths available and pass the prepared plan through `ForwardContext`;
layer kernels should not read GPU lengths back or rebuild metadata. The plan's
lifetime is one immutable request layout. Preserve the separate attention-mode
choice and check exact checkpoint/committed state when changing this path.

## Add a cache policy

Policy chooses admission, snapshot placement and eviction candidates inside
`cache/`. `StateManager` transfers request ownership; `RadixCache` owns cached
pages/snapshots and locks. Keep those responsibilities intact. Test shared
prefixes, partial hits, active flush, retraction, speculative truncation and
exact continuation after reuse. Measure saved workload time per total byte,
including recurrent snapshots and draft reconstruction. Returning IDs to a
monolithic pool does not reclaim physical DRAM; a reclaimable allocation scheme
needs its own pressure and graph-address tests.

## Propose a hardware optimization

GB10 shares physical DRAM between CPU and GPU, but not every CUDA allocation is coherently CPU-accessible
([porting notes](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html),
[UMA guidance](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/optimization.html)). Write down the bottleneck
evidence, the mechanism and the predicted gain before building; measure in fresh processes with repeats, intervals
and host `MemAvailable`, and report the results, negative ones included, in the pull request. A microbenchmark alone
does not establish a serving gain.
