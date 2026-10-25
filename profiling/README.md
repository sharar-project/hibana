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
| `ssd_turns.py` | Concurrent multi-turn conversations over HTTP with a small KV pool, so returning turns restore from the SSD tier: time to first token of later turns, gaps in everyone else's stream (a restore that pauses the batch), restore counters. Compares the SSD tier against `SHARAR_SSD=0` |
| `tp_serve_bench.py` | Served speed over HTTP on one Spark or two: single-stream decode, 4 and 16 concurrent requests, long-prompt time to first token |
| `decode_curve.py` | Decode speed against concurrency at a populated 10K-token context, prefill excluded (the benchmark prompts of `sharar.benchmark_workloads`, request index varied): steady-state ms per round, tokens per round and output tokens/s per batch size, served `throughput`-profile settings by default. The common curve for comparing a change (run control and candidate alternately) |
| `flashnext_moe_w3.py` | Flash-Next's 3-bit routed experts against FlashInfer's CUTLASS fused MoE on real expert weights: time, distinct experts and effective bandwidth per layer at each verify row count, inside CUDA graphs |
| `nemotron_gap_profile.py` | Nemotron's served round (CUDA graphs): device time per graph replay and eager phase, GPU idle between them and host time per phase (`run`, CUDA events); with `--nsys` under `nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node`, kernels per round by phase, gaps inside and between replays, kernels under 5 us and the top kernels (`nsys` on the exported SQLite). `--fused off,on` and `--device-accept off,on` A/B in one load |
| `nemotron_ncu_profile.py` | Nsight Compute over every kernel of one Nemotron round (`--graphs`: graph nodes): DRAM bytes from the L2's system-memory fills and writes (GB10 has no `dram__` counters), GB/s against peak, occupancy; `table` aggregates the CSV. The counters are admin-only on this driver (`RmProfilingAdminOnly=1`) |
| `nemotron_quickstack.py` | Nemotron stacking screen, ~5 min a configuration: in-process decode tok/s at 1 and 4 requests on h2h prompts, teacher-forced KL / top-1 against the exact profile's frozen outputs, GSM8K-50; one load per weight set, takes the GPU lock itself (see below) |
| `w3_bench.py` | 3-bit decode GEMM: correctness against the dequantized reference and DRAM bandwidth per decode shape at 16, 32 and 64 rows, split-K sweep |
| `bolt_b12x_moe.py` | Compare optional b12x 1.5.0 routed IQ2_XXS/ReLU² experts with Bolt's BF16 kernel on real weights; `--sibling` adds the grouped FP8 IQ2 kernel from a frozen source tree. Checks lossless packing size, independent FP8 reference, changed expert ids, padding and graph replay; rotates timing order across route sets larger than L2. Serving dispatch stays unchanged |
| `inputs.py`, `workloads/family_v1.json` | The frozen long prompts the profilers read: 18 synthetic task families and the three benchmark prompts, 10,000 tokens each, built from source with any tokenizer |

Each tool's `--help` and docstring give its inputs and output. Freeze inputs, output budgets, warmups, precision
and geometry before running; run candidate and control in separate fresh processes. Flash-Next loads stall
when earlier loads left the model files in the page cache (CUDA on the GB10 sees MemFree): drop them first
(`posix_fadvise(DONTNEED)` over the model cache, as `tools/ci/gb10.sh` does).

## Quick-stack proxy (Nemotron)

`nemotron_quickstack.py` screens serving configurations before a full `tools/h2h-pr.py` and GSM8K-300: about three
minutes a configuration on a loaded weight set (executor ~15 s, speed ~55 s, KL ~60 s, GSM8K-50 ~55 s; ~1 min more
for the first configuration of a set, ~50-80 s per weight-set load).

    python profiling/nemotron_quickstack.py --name stack1 --config fast+w3 \
        --config fp16=fast+w3:SSM_STATE=fp16 --config fp16-share=fp16:MOE_KEEP=12,MOE_BUDGET=64:192
    python profiling/nemotron_quickstack.py --name stack1 --table

A configuration is `NAME=BASE[:KEY=VALUE,...]`: a profile, a built-in (`exact`, `fast`, `fast+w3` = fast with the
trunk's and the experts' calibrated 3-bit copies) or an earlier name, plus SHARAR_* settings, resolved as
`sharar serve nemotron` resolves them. Settings read at load (3-bit copies, expert kernels) form weight sets, each
loaded once in a worker; the rest (executor, state pool, depth, draft head, device acceptance, MoE sharing on the
loaded blocks, acceptance rule) is applied per configuration. It takes the GPU lock itself (do not wrap it in flock)
in windows of 25 min with 90 s yields, and resumes by configuration. Rows go to
`/home/brian/Desktop/research/nemotron-logs/quickstack/<name>/results.jsonl`, the table to `table.md` there.

Columns: batch-1 median decode tok/s over six h2h_v1 prompts and batch-4 aggregate over h2h's concurrency prompts
(384 tokens), `h2h~` = proxy / CALIBRATION; tokens per round; KL / top-1 / NLL teacher-forced against the exact
profile's frozen continuations of 24 prompts (256 tokens), computed as a verify round computes them (MoE routed as
one verify call, Mamba stepped through the decode kernel) and compared with exact at the same call shape; GSM8K-50
with paired wins / losses against exact. Relaxed acceptance and depth do not move the KL (they change which tokens
are emitted); GSM8K-50 and tokens per round see them.

Validation (Sharar, spark-9931, the proxy against served h2h_v1, greedy):

| config | proxy b1 | h2h b1 | ratio | proxy b4 | h2h b4 | ratio | KL | GSM8K-50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exact | 48.3 | 45.9 | 1.052 | 79.8 | 79.8 | 1.000 | 0 | 49 |
| fast | 56.7 | 54.7 | 1.037 | 94.2 | 92.6 | 1.017 | 0 | 48 |
| fast + both 3-bit copies | 83.3 | 81.0 | 1.028 | 130.4 | 132.7 | 0.982 | 0.099 | 49 |

The proxy orders the three as h2h does, within 2.5% of one calibration (b1 1.039, b4 1.000); a rerun of exact reads
48.6 / 79.4 tok/s, so differences under ~2% are noise. Two findings behind the KL's design: a wide NVFP4 MoE call
(more than 16 rows, b12x) is not reproducible run to run (one prompt prefilled whole and scored three times:
log-probabilities up to 1.5 apart, KL ~0.015 between runs), so prompts prefill in 16-row calls and every
configuration scores bitwise the same twice; and the call shape alone moves the logits as much as a lossy option
(exact at 6-row calls against 4-row: KL 0.017), so the reference is kept at 4-7 rows. GSM8K-50 resolves only large
losses: exact against itself served twice answers 24% of items token for token alike and scores 49 / 48.

The 3-bit calibration builders are in the package ([calibration](../docs/w3-calibration.md)). Raw tensors, outputs
and traces stay outside Git; finished one-off studies are at the `archive/` tags.
