# Contributing to Sharar

Sharar specializes in local inference on one GB10. Keep the current Qwen
target checkpoint, its vocabulary head and the numerical serving profile fixed
unless a change explicitly validates a new one; the DFlash2 drafter is the one
component trained on recorded traffic. Favor a small implementation with
explicit tensor ownership over a registry or framework. Read [the interfaces](docs/interfaces.md),
[the roadmap](roadmap.md), and [request ownership](docs/ownership.md).

## Run checks

On Linux with Python 3.12, this creates a CPU-only environment and runs the
control, scheduling, cache, sampler, HTTP and transaction tests:

```bash
bash tools/check.sh setup-cpu
```

It installs the pinned CPU PyTorch wheel and `.[test]`; no model, SGLang,
FlashInfer, Triton or GPU is required. For subsequent runs use
`.venv-cpu/bin/python -m pytest tests/unit -q`, or `tools/check.sh cpu` in an
already prepared environment. Sharar's small GitHub CPU workflow uses the same
setup command. Dependency constraints are in `constraints/`; the GB10 image
uses `gb10-cu130.txt`, CUDA 13.0.3 and the pinned checkpoint revision.

Also run `python3 tools/check_evidence.py` for recursive evidence/link checks,
and `.venv-cpu/bin/python tools/check_distribution.py` to build an sdist and
wheel and probe installed runtime assets outside the checkout. CI runs these
checks; a source-only or editable installation does not validate packaging.

The GPU suite (`tests/gpu`, about 11 minutes, both targets) runs from the host
in the runtime image with this checkout mounted:

```bash
tools/gpu-tests.sh                 # or: tools/check.sh gpu-container
tools/gpu-tests.sh tests/gpu/test_qsa.py -x
```

It mounts the model caches, the calibrated W3 artifact and `$SHARAR_RESULTS`
at their container paths (read from `config/gb10.env` and
`config/flashnext.env`; `SHARAR_TEST_IMAGE` picks another image) and sets
`SHARAR_RUN_GPU=1`; without it the GPU modules are not collected. Inside an
already prepared GB10 container use `bash tools/check.sh gpu`. Shared benchmark
infrastructure lives in `bench/support/`; preserve frozen workload content and
checksums when moving readers.

The real-checkpoint fixture uses bounded pools: 35,000 KV pages and 12 recurrent
slots. Keep the external memory guard active for that container. The loader
requires 40 GiB `MemAvailable`: a 24 GiB allowance based on the measured 21.3 GiB
streaming-load peak plus the 16 GiB host floor. Runtime separately checks the
prospective pool/draft budget before allocation. See [7D evidence](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/phase7d.md).
Performance measurements need exclusive GPU use. The check command does not
stop or replace a running service.

Anything that changes arithmetic (kernels, dispatch thresholds, draft
precision, verify shapes, proposers) is judged by the family gate,
`tools/family-gate.py` ([bench/README.md](bench/README.md#the-family-gate)):
paired 21-task intervals against a native pass, with the noise floor from a
repeated native pass. Bit-identical outputs are not required and not a
substitute: the target's arithmetic already changes with the row count, so
outputs drift whenever shapes move. Compare row-count changes under
`SHARAR_STABLE_NUMERICS=1` as well. Read the gate this way: the round-cost
interval decides cost (it resolves 0.05%); the tokens-per-round interval
decides acceptance only when it excludes 1, and it is about ±3% wide on the
101-family default set (±7% on the old 18-family set) for any change that
moves continuations, so do not select among variants on its point estimate. The three-prompt `tools/benchmark.sh`
remains the reporting cell for end-to-end numbers, never the decision.
A change to the target's arithmetic or to acceptance also runs
the `quick` suite of the `sharar-eval` repository (`evals.suite`; GSM8K, HumanEval and MGSM, paired
against the exact configuration) before it becomes a default; the gate measures speed and
acceptance, not correctness. A lossy option (one that changes what the model
emits on purpose) is judged by the lossy-change gate below instead.
When an optimization lands, changes its default or is retired, update its
row in the [optimization inventory](docs/optimizations.md) in the same change.

## Judge a lossy option

A lossy speed option (low-bit weights, fewer or pruned experts, relaxed
acceptance, a reduced draft vocabulary) never enters the `exact` profile. It
enters `balanced` or `fast` only through the lossy-change gate (#31), one
command for either target:

```bash
tools/lossy-gate.py --target flashnext --name my-option --profile exact --env SHARAR_X=1
tools/lossy-gate.py --target qwen38 --name fast --profile fast
```

The candidate is a serving profile plus `SHARAR_*` overrides on the target's
launcher config, paired against the `exact` profile on the same image. The tool
runs unattended, takes the GPU lock for each phase, resumes after an
interruption, and writes `$SHARAR_RESULTS/reviews/<date>-lossy-gate-<name>/`
(`SUMMARY.md`, `scorecard.json`, one verdict line). Its three layers run
cheapest first:

1. **Distribution distance** (in-process, minutes): argmax agreement and
   KL(exact || candidate) on the exact model's own greedy continuations of 28
   frozen prompts (code, math, prose, multilingual, structured, short chat),
   64 rows per forward (the decode path, where 3-bit copies act). The floor is
   the exact model at 32 rows per forward. Reported per domain.
2. **Speculation health** (same process): tokens per round, round cost and
   decode tok/s per prompt with the profile's acceptance rule; the floor is a
   repeated baseline pass. The verdict's speed delta comes from here.
3. **Task battery** (served over HTTP, paired per item, McNemar; the suites of
   `sharar-eval`'s `evals.suite`):
   - **Screen:** the `quick` suite (GSM8K 300, HumanEval 164, MGSM 24 per language, needles at 16K and 64K).
   - **Full sets, when the screen is close:** the `full` suite (GSM8K 1,319, HumanEval, MBPP 500, MGSM 2,750).
   - **`--battery hard` (opt-in, 4-6 hours per arm):** the `hard` suite (MMLU-Pro, GPQA Diamond, IFEval, WMT24++).
     It is the one that separates close variants on knowledge.
   - **`--battery long` (opt-in, about 3 hours per arm):** the `long` suite (RULER v2 at 32K and 128K, 120 items
     each). Use it for anything that touches attention, the KV cache or prefill. `--battery all` runs both.
   - **`--battery agentic` (opt-in, about 18 hours per arm):** Terminal-Bench 2.1 through Harbor (the 78 tasks
     valid on the GB10, twice). It detects only differences of about 10 points, so it is an agentic check, not a fine gate.

The verdict (intervals are 95%; pooled = all tasks of the stage):

| Profile | Rule |
|---|---|
| `exact` | layers 1 and 2 within their floors in every domain (the change is lossless; layer 3 is skipped) |
| rejected | no measurable speed gain (decode interval lower bound at or below 1.035, the layer's own noise), or a task or the pooled battery significantly worse (McNemar p < 0.05 with more losses than wins; with `--battery` also MMLU-Pro, GPQA, IFEval or a RULER v2 length (paired sign test), while WMT24++ chrF++ is reported only), or a pooled loss above 2.0 points |
| `balanced` | not rejected, full stage run, pooled interval lower bound at or above -1.0 point and every task within -1.0 point |
| `fast` | not rejected otherwise: the loss is not significant and at most 2.0 points |

The screen escalates to the full sets unless it rejects the candidate or its
pooled interval already excludes a loss of more than 1.0 point. Layer 1 grants
nothing beyond `exact`: it says which domain a change hurts and is the cheap
early warning. Layer 2 resolves speed to about 3.5% (a warm baseline repeat reads x1.035, a
second exact process x0.986); a smaller gain needs the served, interleaved h2h
arms. Argmax agreement breaks exact logit ties by index order, so identical
distributions read about 99.3%, not 100%; the floor carries the same effect.
Calibration on Flash-Next (2026-09-28): the nearest-rounded 3-bit trunk separates from
exact on all three layers (layer-1 KL 0.184 against a 0.013 floor, tokens per round x0.65,
screen -4.9 points, p < 0.001); exact against exact stays within every floor (screen
+0.7 points, p 0.44).
Relaxed acceptance does not move layer 1 (the distribution is
unchanged); it shows in layers 2 and 3. Sampled requests are not scored.

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
`ForwardContext` fields or model checks in `engine/`), and a family's model cache is named by
`Target.cache_env`. `sharar/model/flashnext/` is the worked example. The
engine, pools and graphs read `executor.target`; do not add module-level
constants for a new model, and keep `config.py` describing the 27B for the
tools that import it. Validate a new target with layer-by-layer parity
against a reference server, the CPU contracts, the GPU lifecycle suites and
regenerated family-gate inputs before it is served.

## Add a proposer

1. Put the adapter in `sharar/proposers/`. Implement `Proposer` and declare
   width, target-feature requirements and persistent/scratch bytes. Use
   `ngram.py` for a feature-free example, `mtp.py` for a sequential draft, and
   `parallel.py` for conditional DSpark/DFlash2 proposals and draft-cache handles.
2. Own draft weights, KV, pending tokens and retained features inside the
   adapter. Declare exactly which post-block layers or final-normalized rows
   you need. Choose the adapter once with `Executor.set_proposer()` before
   graph capture. A CLI shortcut belongs in `Executor.enable_spec()` and the
   CLI choices; scheduler, radix internals and HTTP request handling stay out
   of the change.
3. Declare optional features in `ProposerRequirements.capabilities`
   (`engine/contracts.py`): device acceptance feedback, reload, checkpoint
   feature cache, cold-tier payload, trace features, a configurable draft head.
   The engine reads the flags, never probes for methods, and
   `check_capabilities` rejects a flag without its protocol. End the draft in a
   draft head (`proposers/draft_head.py`: the target's head by default, or a
   shortlist) so cheaper heads apply to every speculator; draft depth comes from
   the spec driver's depth policy (`engine/depth.py`).
4. Add CPU tests for readiness, fallback, boundary/stop behavior and cleanup.
   Exercise the shared verifier using the tiny target in `test_contracts.py`.
   Then run the GPU continuation suite, feature lifetime checks and a fixed
   workload comparison against the unchanged adapter.

The verifier accepts flat proposals: MTP k≤8, n-gram k≤4, DSpark k≤7, DFlash2 k≤15.
External sampled proposals include their actual transformed conditional q,
including selector/Markov effects. `engine/rejection.py` is the common reference
algorithm; test tiny distributions independently of model rounding. MTP/n-gram
sampled requests are verified as point masses (MTP) or with their q (parallel drafts); n-gram
sampled requests use ordinary decode. Tree proposals are not implemented.
Declare prospective memory in `cache/spec_budget.py`, and retain/restore draft
features through identity-keyed metadata without taking ownership of target
pages. Both full snapshots and accepted-state reconstruction use the same
StateManager commit and stop/continuation tests.

## Change the drafter or its training

Training lives in `sharar/train/`: the differentiable forward must stay a
function of the serving weight dictionary (`draft.py`), the objective must
match the reference formulas the CPU tests carry (`tests/unit/test_train.py`),
and any change to the forward, objective, selector or mask must pass
`bench/dflash2_reference.py` on the GPU (serving proposer, the publisher's
model and SpecForge as references) and `tests/gpu/test_train_forward.py`.
Judge a drafter on held-out tokens per round and served throughput on the
user's own newest traffic (`bench/draft_ab.py`), never on a public corpus.
Keep the feature and K/V projections frozen unless the cache identity, the
cold-tier payloads and the trace format are versioned with it. Trace format
changes bump `FORMAT_VERSION` in `engine/traces.py` and keep the reader
accepting older records.

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

GB10 shares physical DRAM between CPU and GPU, but not every CUDA allocation
is coherently CPU-accessible; a mapped-allocation experiment must establish
its visibility and synchronization contract
([porting notes](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html)).
Use host `MemAvailable` alongside allocator measurements, and benchmark with
background CPU load as well as an idle machine
([UMA guidance](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/optimization.html)).
Use independent repeated runs with intervals appropriate to the sample count;
do not infer tail latency from a handful of requests. Each experiment records
the bottleneck evidence and predicted reduction; pinned source, image, model,
draft, shapes, cache state, sampling and host load; an ablation against the
same target work, separating kernel-cost comparisons from generation runs when
changed continuations alter the amount of work; the affected tests, raw
samples, emitted tokens and memory; and a keep/remove decision against the
declared threshold, negative results included. The full historical plan is
[archived](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/gb10-experiments-2026-09.md). Describe the mechanism,
what traffic or synchronization it removes, the control experiment, and the
condition that would make us abandon it. Keep tuning and algorithm/architecture
experiments distinguishable. Report source/image/compiler identities, all raw
samples, output checks and memory. Avoid claiming novelty or superiority from
a microbenchmark or from the frozen plain-decode reference alone.
