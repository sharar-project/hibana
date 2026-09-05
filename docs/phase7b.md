# Phase 7B: Extension contracts

Date: 2026-09-05. This phase continues the repaired 7A working tree at git
baseline `21019b01780df17658ade146c85da0757ac6df0d`; the before-performance tree
is a saved copy **including the uncommitted 7A repairs**, not that older commit.

## Changes

`StateManager` now owns admission, cache loans, page reservations, processed
prefix commits, rollback/replay and release. It preserves the 7A invariants and
also returns a slot/cache loan if admission allocation fails. Abandoning a
decode reservation disables caching because an in-place forward may have
partially written the recurrent slot. Adapter cleanup runs once on release.
The scheduler, executor and proposers do not manipulate pool free lists.

`SpecDecoder` now owns only target verification/acceptance and its buffers.
MTP owns draft KV, pending hidden/token state and private draft graphs in
`proposers/mtp.py` and `mtp_graphs.py`. A bounded n-gram adapter uses the same
verification and transaction path without loading MTP or retaining target
features. It is a reference/contract adapter, not a measured speedup claim.

Typed target outputs expose final normalization and opt-in post-block features.
Only selected residual streams allocate feature storage; the verifier clones
only requested graph outputs before callbacks may retain them. The new GPU
tests check the normalization point, shape, dtype and replay lifetime.

Model facts, GB10 constraints and validated serving limits are separate files.
The loader now checks a conservative transient load budget before loading
weights: 48 GiB plus the existing 16 GiB host floor. Graph and context bounds
retain the existing 32,768-token, batch-4, maximum-width-4 profile.

`CONTRIBUTING.md`, `docs/interfaces.md`, pinned dependency constraints,
`tools/check.sh` and a small CPU GitHub workflow make these boundaries usable
without inheriting a framework. The declared Triton dependency now matches the
validated image's 3.7.1. A fresh CPU-only installation does not install Triton
or FlashInfer. The Dockerfile consumes the GB10 constraints; no runtime image
was promoted as part of this source change.

## Correctness and setup evidence

The host-local evidence directory is
`${SHARAR_RESULTS}/reviews/2026-09-05/phase7b/`.

| Check | Result | Evidence |
|---|---|---|
| Fresh CPU-only environment, Python 3.12 / torch 2.13.0+cpu | 129 passed, 9 GPU/snapshot skips | `cpu-setup.log`, `cpu-tests.xml`; `tools/check.sh setup-cpu` |
| GB10 unit/kernel and real-checkpoint suite | 220 passed, including 89 real-checkpoint cases | `gpu.log`, `gpu-tests.xml`, `gpu-command.json` |
| Host memory during the passing GPU suite | Minimum `MemAvailable` 73.98 GiB | `gpu-memory.jsonl` |
| Dockerfile/build-context validation | Passed, no warnings | `docker build --check -f docker/Dockerfile .` |

The GPU run includes all 84 maintained 7A continuation cases and five new
real-target adapter/feature cases. Seven later CPU checks cover abandoned
transaction/one-time adapter cleanup and rejection of incomplete or misleading
benchmark evidence; they are included in the 129 CPU result, not the earlier
GPU collection. The GitHub workflow is configured; the equivalent setup/check
command was validated locally, not on a GitHub runner in this session.

An initial n-gram test used an untemplated `hello ` repeated 128 times and
required identical cold/cached continuations on that prompt. The same mismatch
reproduces with ordinary decoding on the saved pre-7B tree: cold first token
109266, cached first token 248046 (prefix lengths 128 and 64). This establishes
that it predates the refactor; it does not establish whether its cause is
kernel rounding or a pre-existing cache/prefill defect. Its failure is
preserved in `initial-repetitive-prompt-test/`. The maintained
contract check uses the frozen structured canary and an explicitly primed
partial prefix; it still requires identical target output, real n-gram rounds
and correct ownership. The adversarial prompt remains in the before/after
benchmark diagnostics rather than being silently discarded.
Phase 7C must isolate the first diverging layer/state at fixed token prefixes
before treating this case as acceptable numerical variation.

## Refactor performance gate

`bench/contract_bench.py` measures both source trees in the same pinned image,
sequentially with exclusive model memory. The check has 12 cells: plain and
MTP k=3, input lengths 128/2,048/8,192, batches 1/4. Each uses one warmup and
three measured repetitions, 128 generated tokens, greedy sampling, ignored EOS,
cache disabled, identical bounded pools and graph buckets. Prefill in this
driver is sequential per request; this is an engine regression check, not a
scheduler or HTTP latency benchmark.

The predeclared gate requires identical output hashes across all samples,
no more than 5% median regression in decode or total emitted throughput,
all 12 nonempty cells, valid repeated samples and the 16 GiB host floor.
`bench/compare_contracts.py` rejects missing/duplicate cells, insufficient
samples, invalid numbers, output drift and a failed timing gate.

**The gate passed in all 12 cells.** Every output hash matched across both
trees and all three repetitions (72 measured runs total). The largest median
slowdown was 0.47% in total throughput; decode changes ranged from -0.28% to
+0.84%. These small differences are consistent with an unchanged hot path;
three repetitions in one process per tree do not establish a new speedup or
production tail latency. Run order was baseline then candidate, on the same
machine without clock locking.

| Mode | Input | Batch | Before decode tok/s | After decode tok/s | Decode change | Total throughput change |
|---|---:|---:|---:|---:|---:|---:|
| plain | 128 | 1 | 12.17 | 12.15 | -0.17% | -0.44% |
| plain | 128 | 4 | 45.03 | 45.02 | -0.02% | -0.37% |
| plain | 2,048 | 1 | 12.11 | 12.07 | -0.26% | -0.18% |
| plain | 2,048 | 4 | 44.38 | 44.25 | -0.28% | -0.33% |
| plain | 8,192 | 1 | 11.83 | 11.93 | +0.84% | +1.00% |
| plain | 8,192 | 4 | 42.75 | 42.71 | -0.09% | -0.15% |
| mtp | 128 | 1 | 35.31 | 35.24 | -0.18% | -0.07% |
| mtp | 128 | 4 | 119.38 | 119.34 | -0.04% | +0.99% |
| mtp | 2,048 | 1 | 31.34 | 31.30 | -0.14% | -0.09% |
| mtp | 2,048 | 4 | 84.58 | 84.50 | -0.09% | -0.47% |
| mtp | 8,192 | 1 | 35.78 | 35.71 | -0.19% | -0.11% |
| mtp | 8,192 | 4 | 117.17 | 116.93 | -0.21% | approximately 0% |

Minimum sampled `MemAvailable` was 74.41 GiB for baseline and 74.44 GiB for
candidate, with a 16 GiB floor. The adversarial cold/cache diagnostic also
produced identical before/after results, confirming the retained issue predates
the refactor.

Reproduce the comparison with:

```bash
python3 bench/compare_contracts.py \
  "$SHARAR_RESULTS/reviews/2026-09-05/phase7b/baseline-bench.json" \
  "$SHARAR_RESULTS/reviews/2026-09-05/phase7b/candidate-bench.json"
```

Raw samples are in those JSON files; `comparison.json` holds the gate result,
`comparison-gate.json` the predeclared limits, and `*-command.json`,
`*-source-sha256.json`, `*-memory.jsonl` and `validation_runner.py` the execution
identities and memory samples. `baseline/` and `candidate/` preserve source
snapshots. The pinned image used for both trees is
`sha256:b893cdc2e3171bc2b977949305535e1cc62ebb41cee7d607638d628588d50b1b`.

After validation, the same `sharar-qwen38` container and
`sharar-memory-guard.service` were restored. Health, pinned model/profile and
the `SPARKLANG_OK` chat canary passed (`restored-service.log`). The new source
was not deployed; runtime promotion remains separate.

## What follows

The first remote CPU workflow exposed a packaging omission: the artifact
ignore rule had also hidden the locally present `sharar/parity` source. The
[7C follow-up](phase7c.md) anchors that rule, tracks the package, and verifies
both a clean exported source tree and the remote CPU workflow.

Phase 7C freezes realistic local workloads and competitive baselines while
starting hardware-specific experiments. First measure exposed round-trip
synchronization and recurrent-snapshot traffic, then prototype device-resident
progress or selective checkpoints where the trace shows headroom. Session
cache value, reclaimable unified-memory slabs, prepacked weight layouts and
adaptive local scheduling follow the hypotheses and discard criteria in
[docs/gb10-experiments.md](gb10-experiments.md).

DSpark/DFlash2 adapters, draft-prefix restoration, ordinary-decode feature
feedback, adaptive width and stochastic proposal verification remain explicit
7D work. This phase provides tested seams for that work and makes no new
competitive speed or algorithmic novelty claim.
