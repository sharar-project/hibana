# Profiling Sharar

In-process tools that look inside the engine: they load the model and kernels themselves, so they run in the
runtime image (`python3 -m profiling.<tool>`, the checkout at `/workspace/sharar`), on an exclusive, guarded GPU.
To measure a served build (quality suites, AIPerf speed, a lossy option) use the tools in
[evaluation](../docs/evaluation.md) instead; HTTP benchmarks live in
[sharar-eval](https://github.com/abdelfattah-lab/sharar-eval).

| Tool | Purpose |
|---|---|
| `flashnext_round.py` | Where a Flash-Next speculative round spends its time: draft, verify, acceptance and GPU idle gaps (`--gap-profile`, `--kernel-gaps`) |
| `flashnext_profile.py` | Kernel-level breakdown of eager Flash-Next decode steps (torch profiler) plus graph-replay step time |
| `flashnext_quant.py` | Flash-Next trunk quantization error against the BF16 trunk on the serving path: argmax agreement, truncated KL and NLL, dense and sparse regions, with a rounding-noise control |
| `w3_bench.py` | 3-bit decode GEMM: correctness against the dequantized reference and DRAM bandwidth per decode shape at 16, 32 and 64 rows, split-K sweep |
| `inputs.py`, `workloads/family_v1.json` | The frozen long prompts the profilers read: 18 synthetic task families and the three benchmark prompts, 10,000 tokens each, built from source with any tokenizer |

Each tool's `--help` and docstring give its inputs and output. Freeze inputs, output budgets, warmups, precision
and geometry before running; run candidate and control in separate fresh processes. Flash-Next loads stall
when earlier loads left the model files in the page cache (CUDA on the GB10 sees MemFree): drop them first
(`posix_fadvise(DONTNEED)` over the model cache, as `tools/ci/gb10.sh` does).

The 3-bit calibration builders that produce the serving artifacts moved into the package:
`python3 -m sharar.weights.w3_calibrate` (27B), `sharar.weights.flashnext_w3_calibrate` and
`sharar.weights.w3_calibration_inputs` ([workflow](../docs/w3-calibration.md)).

## Draft adaptation tools (`draft/`)

- `draft/trace_corpus.py` generates traced traffic over HTTP against a server started with traces on
  (`./sharar-ctl serve`): `--dataset` replays ShareGPT first turns, `--repo DIR` sends coding requests over a
  repository's files; `--exclude LOG` keeps a test set's file/task pairs out of the traffic.
- `draft/draft_ab.py` replays a trace directory's newest held-out prompts (or `--all`) at batch one against a
  running server and reports throughput, tokens per round and output hashes, for A/B of drafts across windows.
- `draft/dflash2_reference.py` runs inside the runtime container on an exclusive GPU and checks the training
  forward against the serving proposer and the publisher's reference model, and the objective against
  SpecForge, then runs a short training smoke. `draft/target_features.py` recomputes raw target features for it.

See [draft adaptation](../docs/draft-adaptation.md).

## Removed

On 2026-10-04 `bench/` became this directory. Finished one-off studies (KV page size, batch throughput cells,
draft-head variants), the SGLang parity check and the family gate (`tools/family-gate.py`, the family
evaluator, its hooks, the 101-family set and the evidence helper) were removed; they are at the tag
[`archive/bench-pre-split-2026-10-04`](https://github.com/abdelfattah-lab/sharar/tree/archive/bench-pre-split-2026-10-04/bench),
and their conclusions are in [optimizations](../docs/optimizations.md). Older studies are on the
[archive branch](https://github.com/abdelfattah-lab/sharar/tree/archive/pre-cleanup-2026-09-26). Raw tensors,
outputs and traces stay outside Git.
