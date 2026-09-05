# Contributing to Sharar

Sharar specializes in local inference on one GB10. Keep the current Qwen
checkpoint and numerical profile fixed unless a change explicitly validates a
new one. Favor a small implementation with explicit tensor ownership over a
registry or framework. Read [the interfaces](docs/interfaces.md),
[the roadmap](roadmap.md), and [the 7A invariants](docs/phase7a.md).

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

Inside a prepared GB10 container with the snapshot, frozen parity artifacts
at `$SHARAR_RESULTS/parity`, writable compilation cache and exclusive model
memory, run:

```bash
bash tools/check.sh gpu
```

The real-checkpoint fixture uses bounded pools: 35,000 KV pages and 12 recurrent
slots. Keep the external memory guard active for that container. The loader
requires 64 GiB `MemAvailable`: a conservative 48 GiB load-peak allowance plus
the 16 GiB host floor. This is a startup estimate, not an allocator guarantee.
Do not load a second copy beside an already large serving process. The check
command does not stop or replace a running service.

## Add a proposer

1. Put the adapter in `sharar/proposers/`. Implement `Proposer` and declare
   width, target-feature requirements and persistent/scratch bytes. Use
   `ngram.py` for a feature-free example and `mtp.py` for a learned draft.
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

The current verifier accepts flat greedy proposals, width 1–4. Sampled requests
fall back to ordinary decoding. DSpark/DFlash2, tree proposals and stochastic
rejection sampling need their own explicit algorithm and correctness work;
the contract is an extension seam, not an assertion that those are implemented.

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
