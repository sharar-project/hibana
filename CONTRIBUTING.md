# Contributing to Sharar

Sharar specializes in local inference on one GB10. Keep the current Qwen
checkpoint and numerical profile fixed unless a change explicitly validates a
new one. Favor a small implementation with explicit tensor ownership over a
registry or framework. Read [the interfaces](docs/interfaces.md),
[the roadmap](roadmap.md), and [request ownership](docs/phase7a.md).

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

Use [operations](docs/operations.md) for the current container setup. Shared
benchmark infrastructure lives in `bench/support/`; completed experiments
live in `bench/archive/`. Preserve frozen workload content and checksums when
moving readers. Dated reports and manifests stay together in `docs/archive/`.

Inside a prepared GB10 container with the snapshot, frozen parity artifacts
at `$SHARAR_RESULTS/parity`, writable compilation cache and exclusive model
memory, run:

```bash
bash tools/check.sh gpu
```

The real-checkpoint fixture uses bounded pools: 35,000 KV pages and 12 recurrent
slots. Keep the external memory guard active for that container. The loader
requires 40 GiB `MemAvailable`: a 24 GiB allowance based on the measured 21.3 GiB
streaming-load peak plus the 16 GiB host floor. Runtime separately checks the
prospective pool/draft budget before allocation. See [7D evidence](docs/archive/studies/phase7d.md).
Performance measurements need exclusive GPU use. The check command does not
stop or replace a running service.

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
3. Add CPU tests for readiness, fallback, boundary/stop behavior and cleanup.
   Exercise the shared verifier using the tiny target in `test_contracts.py`.
   Then run the GPU continuation suite, feature lifetime checks and a fixed
   workload comparison against the unchanged adapter.

The verifier accepts flat proposals: MTP/n-gram k≤4, DSpark k≤7, DFlash2 k≤15.
External sampled proposals include their actual transformed conditional q,
including selector/Markov effects. `engine/rejection.py` is the common reference
algorithm; test tiny distributions independently of model rounding. MTP/n-gram
sampled requests still use ordinary decode. Tree proposals are not implemented.
Declare prospective memory in `memory/spec_budget.py`, and retain/restore draft
features through identity-keyed metadata without taking ownership of target
pages. Both full snapshots and accepted-state reconstruction use the same
StateManager commit and stop/continuation tests.

## Add a GB10 kernel

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
`memory/`. `StateManager` transfers request ownership; `RadixCache` owns cached
pages/snapshots and locks. Keep those responsibilities intact. Test shared
prefixes, partial hits, active flush, retraction, speculative truncation and
exact continuation after reuse. Measure saved workload time per total byte,
including recurrent snapshots and draft reconstruction. Returning IDs to a
monolithic pool does not reclaim physical DRAM; a reclaimable allocation scheme
needs its own pressure and graph-address tests.

## Propose a hardware optimization

Use [the GB10 experiment plan](docs/gb10-experiments.md). Describe the mechanism,
what traffic or synchronization it removes, the control experiment, and the
condition that would make us abandon it. Keep tuning and algorithm/architecture
experiments distinguishable. Report source/image/compiler identities, all raw
samples, output checks and memory. Avoid claiming novelty or superiority from
a microbenchmark or from the frozen plain-decode reference alone.
