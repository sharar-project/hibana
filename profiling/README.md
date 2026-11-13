# Profiling Hibana

In-process tools that look inside the engine: they load the model and kernels themselves, so they run from the
Python environment (`uv run python -m profiling.<tool>` in the checkout), on an exclusive, guarded GPU.
To measure a served build (quality suites, AIPerf speed, a lossy option) use the tools in
[evaluation](../docs/evaluation.md) instead; HTTP benchmarks live in
[sharar-eval](https://github.com/abdelfattah-lab/sharar-eval).

| Tool | Purpose |
|---|---|
| `flashnext_round.py` | Where a Flash-Next speculative round spends its time: draft, verify, acceptance and GPU idle gaps (`--gap-profile`, `--kernel-gaps`), kernels by phase (`--phase-kernels`); `HIBANA_BENCH_VERIFY_STATE=reconstruct` and `--accept 3:2.0` reproduce the served `fast` round |
| `flashnext_draft_step.py` | One MTP draft step on the head alone (loads in seconds): graph-replay time, kernels per step, and a bitwise check of the step's outputs across code versions (`--save` / `--check`) |
| `ab_identity.py` | Greedy outputs of the speculative engine on fixed prompts (batch one and four, exact and relaxed acceptance, each twice): run on two builds and compare for identity |
| `flashnext_profile.py` | Kernel-level breakdown of eager Flash-Next decode steps (torch profiler) plus graph-replay step time |
| `flashnext_quant.py` | Flash-Next trunk quantization error against the BF16 trunk on the serving path: argmax agreement, truncated KL and NLL, dense and sparse regions, with a rounding-noise control |
| `prefill_gemm_tactics.py` | Every CUTLASS FP4 tactic for the 27B's MLP GEMMs at prefill row counts: bitwise equality with the default dispatch and time; the input to `hibana/kernels/prefill_gemm.py`'s table |
| `tp_comm.py` | Two Sparks: the RDMA one-shot all-reduce against NCCL's ring per message size, bitwise check, inside a CUDA graph after GPU work and back to back (`hibana dev tp-comm` runs both ranks) |
| `tp_draft_head.py` | Two Sparks: the vocabulary-parallel draft head against the whole-vocabulary one (`hibana dev tp-draft-head`) |
| `ssd_turns.py` | Concurrent multi-turn conversations over HTTP with a small KV pool, so returning turns restore from the SSD tier: time to first token of later turns, gaps in everyone else's stream (a restore that pauses the batch), restore counters. Compares the SSD tier against `HIBANA_SSD=0` |
| `tp_serve_bench.py` | Served speed over HTTP on one Spark or two: single-stream decode, 4 and 16 concurrent requests, long-prompt time to first token |
| `decode_curve.py` | Decode speed against concurrency at a populated 10K-token context, prefill excluded (the benchmark prompts of `hibana.benchmark_workloads`, request index varied): steady-state ms per round, tokens per round and output tokens/s per batch size, served `throughput`-profile settings by default. The common curve for comparing a change (run control and candidate alternately) |
| `flashnext_moe_w3.py` | Flash-Next's 3-bit routed experts against FlashInfer's CUTLASS fused MoE on real expert weights: time, distinct experts and effective bandwidth per layer at each verify row count, inside CUDA graphs |
| `w3_bench.py` | 3-bit decode GEMM: correctness against the dequantized reference and DRAM bandwidth per decode shape at 16, 32 and 64 rows, split-K sweep |
| `mamba_prefill.py` | Separate versus fused Mamba prefill core through the public entry point: state/checkpoint parity, paired CUDA-graph GPU time and direct-call wall time, FP32/FP16 and int8/int4 window state; excludes projection GEMMs and plan preparation |
| `bolt_b12x_moe.py` | Compare b12x 1.5.0's routed IQ2_XXS/ReLU² experts (Bolt's default backend) with the BF16 kernel on real weights; `--sibling` adds the grouped FP8 IQ2 kernel from a frozen source tree. Checks lossless packing size, independent FP8 reference, changed expert ids, padding and graph replay; rotates timing order across route sets larger than L2 |
| `inputs.py`, `workloads/family_v1.json` | The frozen long prompts the profilers read: 18 synthetic task families and the three benchmark prompts, 10,000 tokens each, built from source with any tokenizer |

Each tool's `--help` and docstring give its inputs and output. Freeze inputs, output budgets, warmups, precision
and geometry before running; run candidate and control in separate fresh processes. Flash-Next loads stall
when earlier loads left the model files in the page cache (CUDA on the GB10 sees MemFree): drop them first
(`posix_fadvise(DONTNEED)` over the model cache, as `tools/ci/gb10.sh` does).

The 3-bit calibration builders are in the package ([calibration](../docs/w3-calibration.md)). Raw tensors, outputs
and traces stay outside Git; finished one-off studies are at the `archive/` tags.
