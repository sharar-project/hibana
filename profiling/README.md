# Profiling Sharar

In-process tools that look inside the engine: they load the model and kernels themselves, so they run from the
Python environment (`uv run python -m profiling.<tool>` in the checkout), on an exclusive, guarded GPU.
To measure a served build (quality suites, AIPerf speed, a lossy option) use the tools in
[evaluation](../docs/evaluation.md) instead; HTTP benchmarks live in
[sharar-eval](https://github.com/abdelfattah-lab/sharar-eval).

| Tool | Purpose |
|---|---|
| `flashnext_round.py` | Where a Flash-Next speculative round spends its time: draft, verify, acceptance and GPU idle gaps (`--gap-profile`, `--kernel-gaps`), kernels by phase (`--phase-kernels`); `SHARAR_BENCH_VERIFY_STATE=reconstruct` and `--accept 3:2.0` reproduce the served `fast` round |
| `flashnext_draft_step.py` | One MTP draft step on the head alone (loads in seconds): graph-replay time, kernels per step, and a bitwise check of the step's outputs across code versions (`--save` / `--check`) |
| `ab_identity.py` | Greedy outputs of the speculative engine on fixed prompts (batch one and four, exact and relaxed acceptance, each twice): run on two builds and compare for identity |
| `flashnext_profile.py` | Kernel-level breakdown of eager Flash-Next decode steps (torch profiler) plus graph-replay step time |
| `flashnext_quant.py` | Flash-Next trunk quantization error against the BF16 trunk on the serving path: argmax agreement, truncated KL and NLL, dense and sparse regions, with a rounding-noise control |
| `prefill_gemm_tactics.py` | Every CUTLASS FP4 tactic for the 27B's MLP GEMMs at prefill row counts: bitwise equality with the default dispatch and time; the input to `sharar/kernels/prefill_gemm.py`'s table |
| `tp_comm.py` | Two Sparks: the RDMA one-shot all-reduce against NCCL's ring per message size, bitwise check, inside a CUDA graph after GPU work and back to back (`sharar dev tp-comm` runs both ranks) |
| `tp_draft_head.py` | Two Sparks: the vocabulary-parallel draft head against the whole-vocabulary one (`sharar dev tp-draft-head`) |
| `tp_serve_bench.py` | Served speed over HTTP on one Spark or two: single-stream decode, 4 and 16 concurrent requests, long-prompt time to first token |
| `decode_curve.py` | Decode speed against concurrency at a populated 10K-token context, prefill excluded (the benchmark prompts of `sharar.benchmark_workloads`, request index varied): steady-state ms per round, tokens per round and output tokens/s per batch size, served `throughput`-profile settings by default. The common curve for comparing a change (run control and candidate alternately) |
| `w3_bench.py` | 3-bit decode GEMM: correctness against the dequantized reference and DRAM bandwidth per decode shape at 16, 32 and 64 rows, split-K sweep |
| `inputs.py`, `workloads/family_v1.json` | The frozen long prompts the profilers read: 18 synthetic task families and the three benchmark prompts, 10,000 tokens each, built from source with any tokenizer |

Each tool's `--help` and docstring give its inputs and output. Freeze inputs, output budgets, warmups, precision
and geometry before running; run candidate and control in separate fresh processes. Flash-Next loads stall
when earlier loads left the model files in the page cache (CUDA on the GB10 sees MemFree): drop them first
(`posix_fadvise(DONTNEED)` over the model cache, as `tools/ci/gb10.sh` does).

The 3-bit calibration builders that produce the serving artifacts moved into the package:
`python3 -m sharar.weights.w3_calibrate` (27B), `sharar.weights.flashnext_w3_calibrate` and
`sharar.weights.w3_calibration_inputs` ([workflow](../docs/w3-calibration.md)).

Raw tensors, outputs and traces stay outside Git. Finished one-off studies are at the tags
`archive/bench-pre-split-2026-10-04` and `archive/pre-cleanup-2026-09-26`.
