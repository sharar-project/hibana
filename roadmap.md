# Sharar roadmap

Last updated: 2026-09-11 (checkpoint-owned draft features and warm-session benchmarks; production retains its original image)

This is the durable handoff for humans and coding agents working on Sharar. It
records the goal, the facts that constrain the design, the architecture, every
phase with its concrete tasks and definition of done, the promotion gates, and
the decisions already made. Update it whenever a milestone changes the proven
state.

## 1. Mission and end state

Build a thin, contributor-friendly local text-inference engine specialized
for GB10, with performance measured against the best validated alternatives
on the same hardware. Aim for a substantial end-to-end performance lead on
the declared local workloads while keeping the code easy to extend. The first
supported profile is:

- One NVIDIA GB10, compute capability 12.1, `aarch64`, unified memory.
- `RadixArk/Qwen3.8-27B-NVFP4`, revision
  `319f741cce68d7914884900c138a1fbb70a42f30`.
- Local, offline, OpenAI-compatible streaming chat and completions.

The finished engine has these properties:

1. `sharar serve` is the server entry point. Hardware capabilities, immutable
   model facts and serving limits are separate, explicit profile records.
   Hold implementation to the pinned Qwen model on GB10 until the DFlash2
   default passes promotion. Preserve narrow contribution seams; defer second
   models, MoE kernels and generic target abstractions. A compatible trained
   draft remains a separate requirement from target-model support.
2. The process contains no SGLang. All model math, memory management,
   scheduling and serving are Sharar modules with narrow interfaces and tests.
3. MTP, n-gram, DSpark, DFlash2 and future proposers share narrow lifecycle
   and verification contracts. Choose the production proposer and draft width
   from measured workload results. Greedy and sampled decoding both have
   correctness gates; current MTP acceleration is greedy-only.
4. Memory is planned as one unified pool with a hard host floor, and the
   prefix/session cache uses the capacity a discrete GPU cannot offer.
5. Every kernel needed by the supported profiles and shape buckets has a
   versioned AOT artifact. The final runtime carries no compiler and never
   compiles on a request. Final packaging follows the shape/interface work.
6. State ownership, output quality, determinism under the stated numerical
   profile and memory safety pass promotion gates. Performance claims specify
   workload, latency, quality and memory budgets, with reproducible comparisons.

Multimodal, multi-GPU, multi-node, arbitrary model discovery, LoRA, generic
hardware fallbacks and inherited CI are outside the target. Additional model
profiles are deferred until the current single-model default is proven.

Runtime pruning and the MTP feedback repair are complete with
[measured proof](docs/promotion.md). The clean-image matrix completes 414 measured
requests and passes all 40 performance checks against repaired MTP. DFlash2
promotion remains withheld by a frozen quality-rubric failure. Against SGLang,
five fixed cells win in both windows and qualifying natural answers finish
4.7–13.3% sooner; the 128-input / 256-output / concurrency-four cell reverses
between windows. Controlled prefill partitions reproduce different first tokens
and long acceptance tails. The [fixed-token trace](docs/agent-context.md) now
locates early recurrent-layer rounding sensitivity. Long-session cache repairs
pass 584 clean-image checks; a near-32K scheduler retry falls from 14.48 s to
0.145 s TTFT in the bounded study. The [agent-shaped study](docs/agent-workloads.md)
now records a modest, workload-dependent generation lead and failed structured
turns in both engines. Three general candidate-path screens and the full-length
two-path completion screen fail. A fixed 16-target-row tree is the next coverage
hypothesis and requires fresh evidence before implementation. Repair
the quality evaluation prospectively before default promotion. Mixed forwarding,
latency policy and confidence-based width policy are archived in Git history.
Capture only the selected verify width during normal startup. Preserve accepted
state reconstruction, selective features, draft-aware prefix caching and the
shared GDN prefill plan. The compact convolution and narrow NVFP4 tile studies
are complete and failed their frozen integration screens. Their evidence points
toward more useful tokens per target weight traversal as the next speed direction.

## 2. Relationship to the `sparklang` repository

The packaged `sharar benchmark` command (or `tools/benchmark.sh` from a
checkout) provides the AIPerf iteration path: built-in frozen coding/design/
reasoning workloads, exact token audits, end-to-end and decode TPS, and
optional baseline comparisons. It targets an already running server and saves
raw evidence outside Git. See [the quick benchmark guide](bench/README.md#quick-tps-command).

`sparklang` (github.com/yahya010/sparklang, branch `gb10-engine`) contains a
full SGLang checkout, the flattened reference image
`sparklang-gb10-engine:qwen38-27b`, and the launcher/verify/guard scripts. It
is frozen. Its roles for Sharar are:

- Correctness oracle: per-layer hidden states and logits, serial response
  hashes, and the four quality canaries.
- Historical performance reference: the six-cell controlled matrix and the
  32,766-token boundary result.
- Source of ported code and vendored kernels, with attribution.

Sharar never imports `sglang`. Parity is measured over the network against the
running reference container or against tensor dumps produced inside it. The
reference was retired from daily use in Phase 6. Keep its correctness dumps
frozen. Maintain an evolving performance comparison separately, including
validated SGLang MTP, DSpark and DFlash2 configurations and other relevant
GB10 engines. Beating the frozen plain-decode image is an intermediate result.

## 3. Facts that drive the design

### Hardware

| Property | Value |
|---|---:|
| GPU | NVIDIA GB10, SM 12.1, driver 580.95.05 |
| Memory | 128 GB LPDDR5X unified, about 119.7 GiB usable, about 273 GB/s |
| Runtime stack in the reference image | CUDA 13.0, torch 2.13.0+cu130, FlashInfer 0.6.17, Triton 3.8 |
| Production host floor | 16 GiB `MemAvailable` |

### Model

The checkpoint's `config.json` declares `model_type: qwen3_5` and
`Qwen3_5ForConditionalGeneration`; Qwen3.8 reuses the Qwen3.5 architecture code
path. The model is Qwen3.8-27B.

| Property | Value |
|---|---:|
| Layers | 64: 48 Gated DeltaNet (GDN) linear attention, 16 full attention (every 4th) |
| Hidden / MLP intermediate | 5,120 / 17,408, SiLU gated MLP, dense (no MoE) |
| Full attention | 24 query heads, 4 KV heads, head dim 256, output gate, partial rotary 0.25 |
| GDN | 16 key heads x 128, 48 value heads x 128, conv kernel 4, swish output gate |
| Vocabulary | 248,320, untied embeddings |
| Norms | Gemma-style RMSNorm (1 + weight), eps 1e-6, q/k norm in attention |
| MTP | 1 layer, full attention with q/k norm, bf16 (unquantized), shares embeddings |
| Max positions | 262,144 |

Quantization (ModelOpt mixed recipe):

| Tensors | Format |
|---|---|
| MLP gate/up/down (192 tensors) and lm_head | NVFP4, W4A4, block size 16, per-tensor global scale |
| Attention q/k/v/o and GDN in_proj_qkv, in_proj_z, out_proj (208 tensors) | FP8 per-tensor |
| Embeddings, norms, GDN conv/A_log/dt bias/ba projections, MTP | bf16 |
| KV cache | FP8 E4M3 |

Loaded weights are about 19.2 GB. Safetensors total 20.4 GiB because they
include the vision encoder, which is never loaded.

### Recorded baseline: Sharar with MTP speculative decoding

Runtime image `sharar:qwen38-27b` launched by `SHARAR_SPEC=3 run-sharar.sh`
with the memory guard. Result directories under `${SHARAR_RESULTS}/controlled/`:
`controlled-27b-sharar-spec3g-20260904T153017Z` (matrix, speculative path in
CUDA graphs) and `controlled-27b-sharar-spec3-32766-20260904T135132Z`
(boundary, pre-graph build). The plain (non-speculative) run
`controlled-27b-sharar-baseline-20260904T125049Z` is kept as the fallback
baseline for `SHARAR_SPEC=0`. The same-machine comparison against SGLang
plain and SGLang MTP, with GSM8K accuracy, is in `docs/comparison.md`.

These runs predate the latest synchronization audit and do not cover the
correctness defects recorded in Phase 7A. Keep them as historical measurements;
do not interpret the phase-completion labels as evidence that those defects
are fixed.

| Input | Concurrency | Aggregate tok/s | Median TTFT (s) | p95 TTFT (s) |
|---:|---:|---:|---:|---:|
| 128 | 1 | 32.7 | 0.16 | 0.16 |
| 128 | 4 | 105.1 | 0.40 | 0.40 |
| 2,048 | 1 | 22.8 | 0.86 | 0.86 |
| 2,048 | 4 | 61.8 | 2.54 | 3.39 |
| 8,192 | 1 | 19.5 | 3.65 | 3.65 |
| 8,192 | 4 | 38.5 | 10.7 | 14.4 |

Boundary: 31,742 in + 1,024 out at 20.2 tok/s, TTFT 17.1 s. Quality canaries
4/4. Draft acceptance 79.5% over the matrix. Plain decoding (`SHARAR_SPEC=0`)
gives 11.95 / 43.7 / 11.49 / 38.4 / 10.15 / 26.9 tok/s in the same cells.
Profile: 32,768-token context, concurrency 4, queue 16, prefill chunk 2,048
per request and 4,096 per step, KV pages and state slots planned by the
ledger, FlashInfer attention, Triton GDN, CUDA FP8 GEMV, MTP chain drafts of
3, warm restart 96 s, offline cold start 131 s.

### Historical reference baseline (SGLang)

Frozen result directories under `${SPARKLANG_RESULTS}` on the host:
`controlled-27b-sparklang-cli-candidate-20260903T005850Z` (matrix) and
`controlled-27b-sparklang-cli-candidate-32766-usable-20260903T011622Z`
(boundary). Kept for comparison and for the parity oracles only.

| Input | Concurrency | Aggregate tok/s | Median TTFT (s) | p95 TTFT (s) |
|---:|---:|---:|---:|---:|
| 128 | 1 | 12.74 | 0.13 | 0.13 |
| 128 | 4 | 46.43 | 0.31 | 0.31 |
| 2,048 | 1 | 12.22 | 0.81 | 0.81 |
| 2,048 | 4 | 40.67 | 2.34 | 3.07 |
| 8,192 | 1 | 10.80 | 3.33 | 3.33 |
| 8,192 | 4 | 18.30 | 8.46 | 31.7 |

Reference profile: 32,768-token budget, concurrency 4, queue 16, chunked
prefill 2,048, 20 FP32 Mamba slots, static fraction 0.70, cold start about 2
minutes of which weight load is 99 s; image 21.1 GB.

### Earlier SGLang DSpark result: include in the competitive target

Saved run: `/home/yke/sparklang-dspark-results/` followed by
`controlled-27b-nvfp4-dspark-20260902T182512Z`. Its manifest records the same
target revision, FP8 KV, FP32 SSM, 32K context, concurrency bound four, seven
draft proposals and draft revision `b9a5dbdf03bc999c6c73c426b19c2d9041cea393`.
The table reports median aggregate output tokens per second.

| Input | Concurrency | Sharar MTP recorded tok/s | Earlier SGLang DSpark tok/s |
|---:|---:|---:|---:|
| 128 | 1 | 32.7 | 62.7 |
| 128 | 4 | 105.1 | 188.2 |
| 2,048 | 1 | 22.8 | 52.3 |
| 2,048 | 4 | 61.8 | 56.7 |
| 8,192 | 1 | 19.5 | 22.9 |
| 8,192 | 4 | 38.5 | 36.4 |

The earlier DSpark run passed 4/4 canaries, had no matrix request failures
and recorded about 72 GiB minimum MemAvailable. These are historical runs
with differing configurations and repetitive prompts, not a fresh controlled
comparison or a general speedup claim. Phase 7C must reproduce and broaden
the comparison before choosing a production proposer.

### What the numbers establish and what still needs measurement

- Decode profiles support a bandwidth bottleneck, but the claimed 90% roofline
  is not established. `engine/run.py` hardcodes `19.2 * 2**30` as bytes per
  step; allocated bytes differ from traffic, and embeddings are indexed.
  Measure hot weights, scales, KV/recurrent traffic and sustained bandwidth.
- Plain batch decode scales about 3.6x at concurrency four. At the recorded
  109 ms MTP round time, k=3 permits at most four emitted tokens, about
  36.7 tok/s before prefill. Larger gains need better/wider proposals or reuse
  as well as kernel work. That development measurement is not a new full matrix.
- Prefill runs at about 2,500 tok/s. Profile the mixed NVFP4/FP8/GDN workload
  rather than inferring utilization from one advertised peak.
- Owning the scheduler improved long-input TTFT. It can still execute two
  4,096-token prefill budgets before decode, allowing seconds of streaming
  interruption. Inter-token latency and arrival patterns need their own gates.
- The 87–99 s weight-load measurements include more than storage reads.
  Attribute faults, transfers, concatenation, swizzling and initialization
  before selecting prepacked loading or direct mapping.
- Target FP8 KV costs 32 KiB/token (1 GiB for 32,768 tokens); a GDN state slot
  costs about 147 MiB. Useful cache capacity also depends on snapshot placement,
  draft KV/features, verify scratch, graphs, allocator slack and the host floor.
  One physical DRAM pool does not make allocation APIs interchangeable.

## 4. Architecture

```text
sharar serve
  -> config (pinned model, hardware, profile)
  -> weights (safetensors map -> NVFP4/FP8/bf16 tensors in GPU-resident layout)
  -> model (Qwen3.8 hybrid forward, MTP head)
  -> kernels (vendored Triton GDN/conv, FlashInfer attention/GEMM/sampling)
  -> memory (unified ledger, KV pages, GDN state slots, radix prefix cache)
  -> engine (requests, scheduler, executor with CUDA graphs, sampler, spec)
  -> api (HTTP, OpenAI schemas, chat template, reasoning/tool parsers)
```

### Module plan and line budget

| Module | Owns | Budget (lines) | Origin |
|---|---|---:|---|
| `sharar/config.py`, `model/qwen38_config.py`, `hardware/gb10.py`, `engine/profile.py` | Separate model facts, device constraints and validated serving limits; compatibility imports and validation | 250 | new, from sparklang `gb10/` |
| `sharar/weights/` | Safetensors reader, ModelOpt layouts, NVFP4 block-scale swizzle, FP8 scales, tensor name mapping | 500 | port of `modelopt_quant.py` pieces |
| `sharar/model/` | `qwen38.py` (layers, forward), `gdn.py`, `attention.py`, `mlp.py`, `norm.py`, `rope.py`, `mtp.py` | 1,000 | rewrite, reference `qwen3_5.py`, `qwen3_5_mtp.py` |
| `sharar/kernels/fla/` | GDN Triton: `chunk`, `chunk_fwd`, `chunk_delta_h`, `chunk_o`, `chunk_intra`, `cumsum`, `wy_fast`, `l2norm`, `layernorm_gated`, `fused_gdn_gating`, `fused_sigmoid_gating_recurrent`, `fused_recurrent`, `index`, `op`, `utils` | 4,000 | vendored (fla-org via SGLang) |
| `sharar/kernels/conv1d.py` | Causal Conv1D prefill and state update | 1,200 | vendored (Tri Dao / vLLM via SGLang) |
| `sharar/kernels/ops.py` | FlashInfer wrappers: ragged/paged attention with FP8 KV, `mm_fp4`, FP8 GEMM, `fp4_quantize`, fused SiLU-mul-quant, top-k/top-p sampling | 400 | new |
| `sharar/memory/` | `ledger.py` (one pool, host floor), `kv_pool.py` (paged FP8), `state_pool.py` (GDN conv+SSM slots), `request_state.py` (ownership/transactions), `radix.py` (prefix tree with snapshots) | 900 | rewrite, reference `mamba_radix_cache.py` |
| `sharar/engine/` | `request.py`, `scheduler.py`, `executor.py`, `contracts.py`, `sampler.py`, `spec.py` (shared verification/acceptance), target graphs | 1,300 | new |
| `sharar/proposers/` | MTP adapter and private graphs, feature-free n-gram adapter; future draft-specific implementations | 250 | extracted MTP plus new adapters |
| `sharar/api/` | `http.py`, `openai.py`, `chat_template.py`, `parsers/reasoning.py`, `parsers/tools.py` | 1,000 | new; parsers ported |
| `sharar/cli.py` | `serve`, `describe`, `check`, `parity` | 200 | new |

Total about 6,500 engine lines plus about 5,200 vendored kernel lines. Budgets
are review signals. Keep ownership, names and tests clear; do not compress
code to meet a line count. The table is a module plan, not a claim that all
interfaces already exist. Phase 7B separates the MTP adapter from `spec.py`;
draft-aware prefix restoration and richer proposal algorithms follow in 7D.

### Contribution boundaries to implement in Phase 7B

| Boundary | Owns | Extension it permits |
|---|---|---|
| Target model and profile | Immutable model facts, forward results, selected feature taps; device capabilities and serving limits in separate records | A curated model profile without editing scheduling policy |
| Proposer | Requirements, prefill/resume hooks, candidates, valid lengths, confidence and proposal probabilities when needed | MTP, n-gram, DSpark, DFlash2 and research proposers |
| Verifier and acceptance | Target forward and greedy/stochastic acceptance under the request's sampling transforms | New proposals without changing target semantics |
| State transaction and prefix entry | KV, recurrent and draft resource ownership; reserve, commit, rollback, share, release | New cache layouts and wider verification |
| Scheduling policy | Admission, prefill budgets and selected verification widths from a measured cost model | Interactive latency and confidence scheduling |
| Kernel and graph execution | Tensor layouts, persistent buffers, kernel choices and graph buckets | GB10 tuning independent of request lifecycle |

Use small typed records and narrow adapters; select implementations at
initialization and keep dispatch out of per-layer hot loops. Prove the
contracts with MTP and a simple n-gram proposer before adding larger drafters.
Document buffer lifetime, aliasing, valid token counts and mutation ownership.

### Design rules and invariants

- One process. HTTP/tokenization/detokenization use the asyncio side; one
  engine thread owns scheduling and GPU work. Commands cross an explicit
  queue. No ZMQ, worker processes or distributed groups.
- Committed and scratch state. Every request owns one committed GDN state slot
  (conv + SSM) and the executor owns scratch slots. Prefill and accepted decode
  write committed state; speculative verification writes scratch and commits on
  accept. Record committed, emitted and streamed token counts separately;
  cached KV and recurrent snapshots must describe the same exact prefix.
  Stop/budget truncation happens before choosing the committed snapshot.
- One terminal path releases each request exactly once. Active prefix
  references survive cache flush; retraction preserves already emitted history
  and reconstructs it without publishing those tokens again.
- Every backend and dtype has an explicit measured choice. Adaptive draft
  widths and prefill budgets are permitted through a bounded, documented
  cost policy; every selectable shape must have a validated execution path.
- Hot buffers in device allocations, unified pool for capacity. Weights, KV
  pages and state slots use `cudaMalloc`-backed tensors. Managed or mapped
  memory is used for the weight load path and for cache tiers only after a
  microbenchmark shows no bandwidth loss.
- No network at runtime. The tokenizer, template and weights come from the local
  snapshot; a missing file is a startup error.

## 5. Engineering rules

1. One coherent change per commit, with the gate output that justified it in
   the commit message.
2. Parity before performance. A module is not done until it matches the
   reference within the tolerance in section 8.
3. Never optimize an unmeasured path. Every performance change names the
   microbenchmark or matrix cell it moved.
4. Keep raw benchmark, trace and parity dumps outside git. Commit sanitized
   summaries only.
5. Vendored code keeps its header and gets a row in `THIRD_PARTY_NOTICES.md`
   in the same commit.
6. Do not add multimodal, distributed, LoRA, arbitrary-model discovery or
   generic hardware fallback paths. Curated profiles need independent gates.
7. The host memory guard runs whenever any engine is serving on the machine.
8. Do not import `sglang` anywhere in `sharar/`, `tests/` or `bench/`. Tests
   talk to the reference over HTTP or read its tensor dumps.
9. Keep inherited workflows disabled. Phase 7B defines Sharar's own small CPU
   CI and opt-in GB10 checks; GPU benchmarks must use an isolated measured host.
10. Commits carry no attribution trailers.

## 6. Phases

Phases 0 through 7 record implemented milestones and their historical gates.
The 2026-09-04 review found defects those gates did not exercise; their repair
is required before further promotion. No bugs below are marked fixed by this
roadmap update.

Work order: **7A correctness -> 7B contracts -> 7C benchmarks -> 7D speculative
and cache extensions -> 8 unified memory -> 9 remaining kernels/AOT -> 10
release hardening**. Profiling and benchmark preparation may proceed during
repairs; new runtime features and compiler removal follow the relevant gates.
Draft/scratch memory reservations are prerequisites within 7D, not deferred
until after allocating a larger draft. Each checklist item is a focused commit
or small series.

### Phase 0: Bootstrap `[complete]`

- [x] Create the private repository `abdelfattah-lab/sharar`.
- [x] Copy the benchmark harness, comparator, reference verifier and memory
      guard; rename environment variables to `SHARAR_*`.
- [x] Write this roadmap, README and third-party notices.
- [x] Choose and document the development environment: a dev container from
      the reference image with this repository bind-mounted, so torch,
      FlashInfer and Triton match the reference exactly. Versions are in
      `docs/env.md`.
- [x] Add `tools/dev-shell.sh` that starts that container with the model
      cache mounted read-only, the GPU devices and driver libraries mounted
      the same way `sparklang/run-gb10-engine.sh` does, and no network.
- [x] Add `tests/conftest.py` with fixtures for the model snapshot path, the
      reference URL, and a results directory from `config/gb10.env`.
- [x] Add `sharar/config.py` with the pinned model and hardware facts and a
      `check` command that validates GPU name, compute capability, driver,
      memory and snapshot completeness.
- [ ] Rename the `sparklang` repository on GitHub to mark it as the reference
      (for example `sparklang-reference`) and note the new URL here. (Owner's
      call; everything else in Phase 0 is done.)

Definition of done: `sharar check` passes inside the dev container, and
`tools/verify-reference.sh` passes against the running reference.

### Phase 1: Offline forward parity `[complete]`

Goal: load the checkpoint and match the reference's logits on fixed prompts
without any serving code.

Oracle plumbing:

- [x] Reference dumps: `tools/reference/dump_reference.py` runs the offline
      SGLang engine in the dev container with a patched tensor dumper
      (`tools/reference/sitecustomize.py`) and writes per-layer
      (hidden_states, residual), embedding, final norm and last logits for
      every prompt up to 2,048 tokens. `tools/reference/http_oracle.py` records
      top-8 logprobs at every position and the greedy continuation from the
      serving container for all prompts, and `engine_oracle.py` does the same
      from the offline engine for self-consistency measurements.
- [x] Dumps keep the reference's dtypes (bf16 streams, bf16 logits).

Weights:

- [x] `weights/safetensors.py`: shard map, lazy `safe_open`, vision tensors
      never read.
- [x] `weights/modelopt.py`: NVFP4 packed weights with FP8 block scales and a
      global scale swizzled into the CUTLASS SM120 layout; FP8 per-tensor
      linears with fused shards requantized to the maximum scale exactly as the
      reference does (q/k/v and qkv/z carry different scales in this
      checkpoint; gate/up share one).
- [x] Name mapping lives in the layer constructors (`model/attention.py`,
      `model/gdn.py`, `model/mlp.py`); `tools/reference/inspect_checkpoint.py`
      prints every tensor family's shape, dtype and scales.
- [x] `tests/unit/test_modelopt_layout.py` checks the swizzle is a padded
      permutation; block-level parity replaces a per-layer shape test.

Model:

- [x] `model/norm.py` (Gemma RMSNorm in fp32), `model/rope.py` (partial rotary
      over 64 of 256 dims, NeoX pairing; text positions are 0..T-1).
- [x] `model/mlp.py`: NVFP4 gate/up via `mm_fp4` (CUTLASS backend, the
      reference's `auto` resolution on SM121), fused SiLU-mul-FP4 quantization,
      NVFP4 down.
- [x] `model/attention.py`: FP8 q/k/v/o, per-head q/k Gemma norm, rope, sigmoid
      output gate, FlashInfer ragged causal prefill in bf16.
- [x] Vendored `sharar/kernels/fla/` (16 files) and `sharar/kernels/conv1d.py`
      with original headers; SGLang-only imports replaced by
      `sharar/kernels/_compat.py`. KDA, replay-SSM and spec-decode kernels left
      out.
- [x] `model/gdn.py`: fused FP8 qkv/z projection, bf16 b/a projection, causal
      Conv1D with state write-back, fused gating, chunked gated delta rule with
      state write-back, gated RMSNorm, FP8 out projection. Conv and SSM slots
      are cleared before a fresh prefill because the chunk kernel reads the
      initial state from the pool.
- [x] `model/qwen38.py`: embeddings, 64 layers, final norm, NVFP4 lm_head,
      per-layer capture in the reference's (hidden_states, residual) layout,
      chunked per-position argmax/top-k.
- [x] `sharar parity` (block cosine with reference inputs, confident top-1
      agreement, sure next tokens), plus `parity.blocks`, `parity.truth`,
      `parity.layerprobe`, `parity.chaos` for diagnosis.

Definition of done, as measured (see `docs/parity.md`): every probed block
matches the reference at cosine at least 0.9994 given the reference's input;
confident-position top-1 agreement is 84% to 100% per prompt, equal to the
reference's own agreement with itself under a different prefill chunk size;
the next token matches on every prompt where the reference's margin exceeds
5 nats. The original all-position argmax-equality gate did not hold across
the measured implementations/chunk configurations and was replaced by these
empirical criteria. This is not proof that all future differences are rounding;
new discrepancies require the section 8 diagnosis.

### Phase 2: Decode loop and CUDA graphs `[complete]`

Goal: generate tokens for one request at reference speed.

- [x] `memory/kv_pool.py`: paged FP8 KV for the 16 attention layers, page
      size 1 like the reference (the 1-versus-16 measurement moves to Phase 3
      with the radix cache, which decides page granularity).
- [x] `memory/state_pool.py`: conv and FP32 SSM slots with committed slots
      and reserved scratch slots.
- [x] `engine/executor.py`: admit, prefill, batched decode step (FlashInfer
      tensor-core paged decode with FP8 KV, packed GDN recurrent kernel,
      Conv1D state update), generate. `engine/graphs.py`: CUDA graphs for
      batch 1, 2, 4 with padded dummy rows.
- [x] `engine/sampler.py`: greedy fast path, temperature, top-k, top-p, seeds,
      stop tokens from `generation_config.json`. Repetition and presence
      penalties are deferred to Phase 5 with the API surface.
- [x] `sharar bench-decode`: tokens per second, ms per step, effective weight
      bandwidth, and the determinism checks.
- [x] Determinism: rerun identical, eager identical to graph at equal batch
      size, rows identical within a batch. Batch-size-dependent rounding is
      expected (GEMM tiling) and documented.
- [x] Fused kernels (norm + residual + FP8 quant, gated FP8 quant, q/k norm +
      RoPE, NVFP4 quant and SiLU-fused NVFP4 quant, CUDA FP8 GEMV for M <= 4)
      and per-shape `mm_fp4` backends. See `docs/phase2.md`.

Definition of done, as measured (`docs/phase2.md`): greedy canaries 4/4 with
exact, arithmetic and needle continuations token-identical to the reference
and JSON valid; decode 12.06 tok/s at batch 1 (reference 12.74) and 44.6
aggregate at batch 4 (reference 46.4); prefill of 2,048 tokens 813 ms on the
GPU against the reference's 810 ms TTFT. The 5% decode shortfall is the FP8
GEMV streaming at 194 GB/s in the model versus 225 in isolation, with every
available GEMM kernel capped near 225 GB/s on GB10; that is Phase 9 work and
is recorded there. Byte-identical continuations were replaced by the section 8
criteria for the reasons documented in Phase 1.

### Phase 3: Memory ledger and prefix cache `[complete]`

Goal: own every allocation and make the cache use the unified pool.

- [x] `memory/ledger.py`: weights, KV pool, state pool, workspaces and graphs
      recorded as allocated; pools planned from `MemAvailable` with the 16 GiB
      floor and an 8 GiB prefill reserve; `check_floor()` and `describe()`.
      The health endpoint arrives with the API in Phase 5.
- [x] `memory/radix.py`: radix tree over KV pages with GDN snapshots at node
      boundaries, LRU eviction of unlocked leaves, locked prefix paths for
      running requests. Written from scratch; the reference's `extra_buffer`
      strategy informed the snapshot placement.
- [x] Cache sized from the ledger: on this host with the reference running,
      about 14 GiB of KV pages and 50 to 96 snapshot slots.
- [x] `sharar cache-check`: checkpoint correctness against a fresh prefill,
      identical-prompt re-send, multi-turn reuse, pool accounting and flush.
      Cached-prefix continuations equal cold ones where the reference's margin
      is large and differ within the noise floor elsewhere (FP8 pages versus
      bf16 ragged attention), as documented in `docs/phase3.md`.
- [x] `sharar soak`: 30 rounds of 8K prompts at concurrency 1 to 4 with
      partial shared prefixes; no leak, 302 evictions, `MemAvailable` never
      below 37.9 GiB. Categories track CUDA allocation deltas; full reconciliation
      with reserved and non-PyTorch memory remains Phase 8 work. The memory
      guard was active throughout.

Historical result (`docs/phase3.md`): the ledger sizes the pools and the cache
tracks prefix resources; a re-sent 9,642-token prompt drops from 4.4 s to 158 ms TTFT
with the same first token, the next conversation turn reuses the previous one,
and the soak passes with the guard active. Phase 7A subsequently repaired and
tested active-flush ownership and speculative-stop snapshots.

### Phase 4: Scheduler `[complete]`

Goal: continuous batching for concurrency 1 to 4 with better long-input
behavior than the reference.

- [x] `engine/request.py`: lifecycle states, cancellation, stop token IDs,
      max tokens, streaming cursor. Stop strings need detokenization and
      arrive with the API in Phase 5.
- [x] `engine/scheduler.py`: FCFS admission with queue and running bounds,
      pool check before admission, chunked prefill with a per-request chunk
      (2,048) and a per-step budget (4,096) shared oldest first, retraction of
      the newest request under memory pressure.
- [x] Interleaved steps: up to two prefill steps then decode in the current
      scheduler. Interactive latency under arrivals remains Phase 7D work.
      The merged "mixed chunk"
      forward (decode rows inside the prefill forward) is documented as an
      option in `docs/phase4.md`, not implemented; it would give up the
      CUDA-graph decode path on those steps.
- [x] Varlen prefill: several requests' chunks in one forward with
      per-sequence state slots, `cu_seqlens` for the GDN kernels, one paged
      attention plan and per-sequence checkpoints.
- [x] `tests/unit/test_scheduler.py`: simulated-time policy tests for
      concurrency 1 to 4, queue overflow, cancellation while queued and while
      running, stop tokens, the 31,742 + 1,024 token request under pressure.
- [x] Metrics on the scheduler; cache statistics included.

Definition of done, as measured (`docs/phase4.md`): p95 TTFT at 8K x 4 is
13.9 s (reference 31.7 s) with 47% more throughput in that cell; the other
cells are within 1 to 9% of the reference on TTFT and 5 to 7% below it on
throughput, which is the Phase 2 decode GEMV gap, not the scheduler
(overhead under 0.5 ms per step). All simulation cases pass.

### Phase 5: API and CLI `[complete]`

Goal: a drop-in replacement for the reference's HTTP surface as used by the
harness and by local clients.

- [x] `api/server.py`: FastAPI over one engine thread that owns the
      scheduler; handlers await per-request queues; disconnects cancel.
      Binds 127.0.0.1:30001 by default.
- [x] `/v1/chat/completions`, `/v1/completions` (streaming and not),
      `/v1/models`, `/health`, `/v1/tokenize`, `/get_server_info` (alias
      `/describe`) with profile, ledger and scheduler stats, `/metrics`,
      `/flush_cache`, and the SGLang-style `/generate` the harness speaks.
- [x] Chat template with `chat_template_kwargs.enable_thinking`, `tools`,
      `stop` strings, `stream_options.include_usage`.
- [x] `api/parsers/reasoning.py` (streaming `<think>` split) and
      `api/parsers/tools.py` (the model's XML tool-call blocks to OpenAI
      `tool_calls`, schema-typed arguments), with unit tests.
- [x] `sharar serve`; `describe`, `check`, `parity` existed already.
- [x] `tools/verify-sharar.sh`; `tools/dev-shell.sh --serve PORT`; the
      harness runs against Sharar unchanged with `--base-url` and
      `--container`.

Definition of done, as measured (`docs/phase5.md`): the frozen harness runs
unchanged against Sharar and reports 4/4 quality canaries; the controlled
matrix (`controlled-27b-sharar-phase5c-20260904T034427Z`) shows throughput
5.4 to 6.3% below the reference in five cells (the Phase 2 GEMV gap) and 46%
above at 8,192 x 4 with p95 TTFT 13.9 s against 31.7 s; the 32,766-token
boundary completes at 9.62 tok/s. Serial hash equality with the reference is
not a goal (section 8); Sharar's own serial hashes become the baseline in
Phase 6.

### Phase 6: Promotion and reference retirement `[complete]`

- [x] Gates run against Sharar through the frozen harness (Phase 5) and again
      against the runtime image (basenames in section 3).
- [x] 32,766-token boundary and cold restart measured from the image.
- [x] `docker/Dockerfile` from `nvidia/cuda:13.0.3-devel-ubuntu24.04` with only
      the pinned packages (14.3 GB, no SGLang, transformers or vLLM);
      `sharar check`, the unit tests, an offline cold start with an empty JIT
      cache and a chat all pass inside it. `run-sharar.sh` is the production
      launcher (preflight, devices, driver libraries, loopback port, memory
      guard); `docker/build.sh` builds.
- [x] Section 3 now carries Sharar's baseline; the SGLang reference is
      historical and used only through `tools/reference/`.

Definition of done, as measured (`docs/phase6.md`): Sharar is the production
engine on the machine (`run-sharar.sh`, container `sharar-qwen38`, guard unit
`sharar-memory-guard.service`); the reference container is started only for
parity work.

### Phase 7: Speculative decoding with MTP `[implemented; follow-ups 7A–7D]`

Goal: more emitted tokens per weight read. Expected 1.5x to 2x at concurrency 1.

- [x] `model/mtp.py`: the bf16 MTP module sharing the target's embeddings
      and NVFP4 lm_head, with its own bf16 paged KV addressed by the target's
      page ids.
- [x] `engine/spec.py`: chain drafting with MTP (k = 3 in the production
      profile; 2 and 4 measured) and target verification. Extracting a reusable
      proposer/verifier contract remains Phase 7B work.
- [x] Verify: one target forward over k + 1 tokens per request; the GDN
      layers run the recurrent kernel from the committed state without
      updating it and record the state after every token in per-request
      buffers (the vendored kernel's `intermediate_states_buffer` path), the
      conv window is recomputed in torch; accept copies the chosen state
      back. No replay-SSM kernels were needed.
- [x] CUDA graphs for the verify forward and the chain draft steps
      (`engine/spec_graphs.py`), including the single pending row in step 1.
- [x] Acceptance rate, tokens per round and draft/verify time in
      `SpecDecoder.stats()` and `/get_server_info`; `/generate` now reports
      per-request statistics too. Earlier saved matrices have empty columns;
      Phase 7A repairs stop-aware emitted/committed accounting.
- [x] Greedy verification accepts exactly the target's argmax; sampled
      requests take the plain decode step (rejection sampling deferred).
- [ ] Make the verifier proposer-independent and prove it with a second
      proposer (Phase 7B). The target accepts token chains, but `SpecDecoder`
      and `DraftGraphs` still access MTP, request and executor internals.

Definition of done, as measured (`docs/phase7.md`): concurrency-1 decode
improves 1.9x to 2.4x and concurrency-4 aggregate 1.9x to 2.2x at k = 3 on
text, needle and JSON prompts; canaries 4/4 with every reference token
matched; memory within the ledger (2.1 GiB draft head and KV, 2.4 GiB verify
buffers). Identity with non-speculative greedy output holds where the
target's margin is large and diverges within the noise floor elsewhere, as
documented in the phase notes. This historical result does not cover stopping
inside an accepted block or draft-aware prefix reuse; see Phases 7A and 7D.

### Phase 7A: Request and state correctness `[implemented and validated]`

Review baseline: `21019b01780df17658ade146c85da0757ac6df0d`. Existing CPU
tests passed (21 passed, 9 GPU/snapshot tests skipped), but additional isolated
CPU reproductions exposed the first six defects below. The retraction check
used the scheduler test double plus inspection of `Executor.retract`; the
speculative check ran the actual acceptance/cache-insertion method bodies
against tiny CPU state buffers. These establish lifecycle/ownership defects,
not a GPU numerical evaluation. Host-local evidence is under
`${SHARAR_RESULTS}/reviews/2026-09-04/` (`review.md`, `audit.py`).

All eight repairs below are implemented. Maintained tests and the invariants
for subsequent refactors are documented in [docs/phase7a.md](docs/phase7a.md).
The external audit remains historical evidence.

| ID | Priority | Repaired defect and implementation location | Passing regression check |
|---|---|---|---|
| A1 | P1 | `_prefill_step()` drops requests finishing on the first token before `_reap()` can release them. Queued `cancel()` sets the enum but leaves `finished=False`. `engine/scheduler.py`, `engine/request.py`, `api/server.py` | One-token limit and first-token EOS release once; queued/running cancellation terminates its ticket; no lost request when the scheduler becomes idle |
| A2 | P1 | `_accept()` commits the full accepted block before EOS/limit truncates returned tokens; `_insert_into_cache()` labels that state with the shorter prefix. `engine/spec.py`, `engine/executor.py` | Force every acceptance length and every stop position; assert cached KV, GDN state and token IDs refer to the same prefix, then resume it |
| A3 | P1 | `RadixCache.flush()` frees locked pages still used by requests. `memory/radix.py`, `api/server.py` | Flush during active shared-prefix requests; active IDs cannot be reallocated, and deferred resources return after the final release |
| A4 | P1 | `Executor.retract()` clears already emitted output and the stream cursor. `engine/executor.py`, `engine/scheduler.py` | Retract after streaming tokens; reconstruct the exact emitted history, preserve output budget/RNG progress, and never emit history twice |
| A5 | P2 | `Sampler._generators` shares one evolving RNG per seed across requests. `engine/sampler.py`, `engine/request.py` | Independent same-seed requests get independent reproducible draws; use fixed logits to isolate RNG behavior from batch-dependent model rounding |
| A6 | P2 | Spec stats count proposed output after stopping; scheduler counts one token per request despite multi-token rounds. `engine/spec.py`, `engine/scheduler.py`, `api/server.py` | Distinguish proposed, accepted, committed, emitted and streamed counts; reconcile per-request totals and serving metrics under truncation |
| A7 | P1 | Source inspection: admission does not enforce the profile limit while graph index buffers hardcode `32768 + 64`. `engine/scheduler.py`, `engine/graphs.py`, `engine/spec_graphs.py`, `api/server.py` | Reject invalid token IDs, empty/oversized prompts and invalid budgets before allocation; validate prompt+output limits and internal speculative headroom; test exact boundary and one over |
| A8 | P1 | Source inspection: an unhandled engine exception can strand tickets while `/health` still reports OK. `api/server.py` | Inject executor failures; terminate pending tickets, expose unhealthy/unready state, reject new work and release resources where safe |

Validation: **103 CPU tests passed, 9 GPU/snapshot tests skipped**. The GB10
run passed **194 tests**, including **84 real-checkpoint lifecycle cases**:
eager/graphs, batches 1/2/3/4, cache disabled/miss/hit, mixed EOS/length stops,
exact committed-state comparisons, cached continuation, active flush and
retraction. Both 32K boundary cases and the four frozen canaries in both
execution modes passed. Two additional CPU pressure tests were added after
that GPU run collected its tests; they are included in the 103-test CPU result.
The same seven selected behavioral reproductions failed on `21019b0` and
passed with the repairs. Final GPU minimum `MemAvailable`: **73.99 GiB**.

An initial concurrent load reached the memory floor and the test/serving guards
stopped their respective containers. The passing run used exclusive GPU
memory; service restoration is recorded in the phase notes. This is correctness
evidence, not a refreshed throughput matrix, full layer-oracle rerun or a new
competitive performance claim. Runtime promotion remains a separate action.

### Phase 7B: Extension contracts and contributor workflow `[done — 2026-09-05]`

Goal: a new proposer or GB10 kernel can be contributed through the section 4
boundaries without changing unrelated scheduling, cache or HTTP internals.

- [x] Separate target model facts, GB10 capabilities and serving profile
      limits. Keep the existing Qwen revision as the only validated profile.
      Graph sizes, context bounds and memory requirements derive from it.
      Include transient weight-load peaks before loading another model;
      steady-state allocated bytes alone do not establish startup headroom.
- [x] Define typed forward results and optional selected-feature outputs;
      document normalization point, shape, dtype, aliasing and lifetime.
- [x] Extract proposer lifecycle and requirements from `SpecDecoder` and
      `DraftGraphs`. Draft-specific KV and pending state belong to the proposer.
- [x] Extract target verification/acceptance and a state transaction for
      reservation, commit, rollback and release. Preserve the A1–A8 invariants;
      do not expose pool free lists as cross-module mutation interfaces.
- [x] Adapt MTP and add a simple n-gram proposer as a contract test. A proposer
      requiring no target features must not pay feature-capture overhead.
- [x] Separate scheduling policy from execution buffers/graph replay. Use
      explicit initialization-time choices and bounded graph buckets.
- [x] Add `CONTRIBUTING.md` and ownership/interface documentation with recipes
      for a proposer, a kernel and a cache policy, plus their required checks.
- [x] Declare serving dependencies, add reproducible dependency constraints,
      and provide one-command CPU checks and opt-in GB10 checks. Introduce
      Sharar's own small CPU CI; keep inherited workflows disabled.

Definition of done: MTP and the second proposer share verification/state
logic; changing either requires no scheduler/cache-internal changes; focused
tests and unchanged-profile benchmarks pass. Interfaces remain small and
hot execution paths retain their measured specialization.

Implementation boundaries and contributor recipes are in
[docs/interfaces.md](docs/interfaces.md) and [CONTRIBUTING.md](CONTRIBUTING.md).
The hardware experiment plan is [docs/gb10-experiments.md](docs/gb10-experiments.md).
New mechanisms must have a measured hypothesis and a condition for discarding
them; tile sizes and launch tuning are only one part of the optimization work.

Validation: **129 CPU tests passed, 9 skipped** in a fresh CPU-only install;
**220 tests passed on GB10**, including **89 real-checkpoint cases**. The
12-cell plain/MTP unchanged-profile comparison passed with identical outputs
across all 72 measured runs. Largest median slowdown: **0.47%** against a
predeclared 5% limit. Details, the retained pre-existing cold/cache diagnostic
and raw-evidence locations are in [docs/phase7b.md](docs/phase7b.md).

### Phase 7C: Workload and competitive performance gates `[in progress — 2026-09-05]`

Measurement and hardware research begin together here. First attribute the
per-round synchronization and recurrent-snapshot traffic, then prototype the
mechanism with demonstrated headroom. Do not defer all optimization ideas to
Phase 9 or require finishing every competitor integration before a bounded
experiment. Freeze each experiment's workload and decision threshold first.

- [x] Keep the synthetic six-cell matrix for continuity. Document its limits:
      repetitive prompts, cold prefix cache, ignored EOS, fixed output lengths,
      greedy decoding and thinking disabled.
- [x] Freeze representative coding, chat, tool/structured output and thinking
      prompts with greedy and sampled settings. Add identical/partial-prefix
      reuse, multi-turn sessions, mixed arrivals, cancellation
      and natural stopping. Include budgets 1 through verify-width+1.
- [ ] Add cache-pressure/eviction session traces under equal physical budgets.
- [ ] Migrate restored historical parity probes before using them as new
      oracles; `parity.blocks` still references the former fused GDN members.
      The current cache investigation uses the tested `parity.cache_trace`.
- [x] Triage the `hello ` ×128 cold/cached divergence recorded in
      `docs/phase7b.md`. It reproduces on the pre-7B tree with ordinary decode;
      compare fixed-prefix recurrent state and per-layer outputs before
      attributing it to kernel rounding or accepting broader cache quality.
      Manual/radix restoration is bit-identical at all 64 layers; a live
      split alone reproduces the mismatch. First drift is the FP8 projection
      shape path. Retain checkpoint-precision and broader quality limitations
      in `docs/phase7c.md`; the diagnosis is not universal token-equivalence.
- [x] Record actual streaming TTFT, completion, delivery bursts, emitted IDs,
      cache-hit length and per-request speculation in the local pilot. Bound
      server TTFT storage to 2,048 recent samples with explicit window counts.
- [ ] Measure TTFT, client-visible inter-token gaps, prefill/decode time,
      completion time, actual emitted tokens, accepted tokens/round, cache-hit
      length, draft reconstruction cost and peak/available memory. Bound metric
      storage for long-running servers.
- [ ] Refresh same-machine comparisons for SGLang plain, MTP, DSpark and
      DFlash2 when validated, plus other relevant GB10 engines. Reproduce the
      earlier DSpark run before using it as a current baseline.
      The fresh plain/DFlash2 matrix and natural-request pilot are in
      [the competitive report](docs/competitive.md). MTP/DSpark, eviction
      pressure and independent comparison windows remain open.
- [ ] Pin target/draft/tokenizer revisions, sampling, cache state, concurrency,
      memory budget, graph settings and kernel/compiler/image identities.
      Repeat independent runs; report sample counts and uncertainty. Five
      tiny batches do not establish production p95/p99 latency.
- [ ] Extend the comparator to enforce workload-specific quality, latency,
      memory and throughput limits fixed before tuning. Require the expected
      nonempty case set; separate kernel-cost studies from end-to-end generation
      when differing continuations change acceptance or work performed.
- [ ] Recompute decode byte traffic and the roofline from hot tensors and
      measured counters. Remove the allocated-bytes-as-traffic estimate.
- [x] Trace ordinary/MTP decode at B=1/4, T=128/8,192, keeping independent
      unprofiled samples separate. Freeze the first GB10 experiment brief:
      compact verified recurrence inputs plus accepted-prefix state replay.
      MTP's exposed trace gaps are small; its 2.25 GiB full-snapshot allocation
      provides a concrete capacity target. See `docs/gb10-experiments.md`.
- [x] Implement that experiment and its ablation. Preserve full FP32 snapshots
      as the control, test every accepted/stop position and report negative
      results. Require at least 25% lower total verification snapshot storage
      at fixed B/k and no more than 5% protected latency regression; do not
      substitute allocated bytes for physical traffic or a measured speedup.
      [7D](docs/phase7d.md) measures 73.04% smaller verify buffers and retains
      the control; [direct accepted state](docs/direct-state.md) removes the
      final scratch copy. Broad speed/quality promotion remains separate.

Definition of done: reproducible workload manifests, comparator coverage and
an updated `docs/comparison.md` distinguish historical observations from fresh
comparisons, synthetic performance from local-use performance, and algorithm
gains from kernel/engine gains. The correctness oracle stays frozen.

### Phase 7D: Draft-aware caching and speculative decoding `[implemented; promotion withheld after repeat evaluation]`

Prerequisites: 7A ownership fixes, 7B interfaces and 7C workload gates. Preserve
the current MTP path while evaluating each extension independently.

Implementation and measured decisions: [docs/phase7d.md](docs/phase7d.md).
Native adapters, shared stochastic verification, draft-aware prefix metadata,
prospective budgets and fixed-width CUDA graphs are present. MTP and full
snapshots remain the source defaults until the promotion gates pass. External
drafts and reconstruction are supported candidates. Rejected adaptive policies
and mixed target execution have been removed from the shipped runtime. A passed implementation gate is distinct
from a broad quality, competitive-speed or production-promotion gate.

- [x] Implement selective target feature capture for prefill and verification.
      The reviewed DSpark and DFlash2 checkpoints request layers 5/19/33/47/61;
      pin the draft revision and read its requirements rather than hardcoding
      these into the executor. Match feature semantics against the draft oracle.
- [x] Add draft-aware prefix entries keyed by target/draft identity, feature
      schema and relevant numerical layout. Track valid draft KV/features;
      cache hits reconstruct only missing requirements. Bound retained features
      by a drafter's supported window and measure the total hit-path cost.
- [x] Reserve draft weights/KV/features, verify scratch, graph pools and
      activation peaks before optional cache sizing. Current FP32 SSM verify
      intermediates alone cost about 2.25/4.5/9 GiB at batch four and verify
      width 4/8/16; increasing width must pass memory gates before allocation.
- [x] Add exact stochastic acceptance/rejection sampling using the actual
      conditional proposal distribution, including selector/Markov effects.
      Match temperature/top-k/top-p transforms and use request-local RNG.
      Test the algorithm on tiny known distributions separately from GPU
      numerical parity. Positive-temperature MTP/n-gram requests use plain
      decode; DSpark/DFlash2 use the shared sampled reference path.
- [x] Add batch/verify-width buckets, including tails and padded rows.
      Select the fixed startup width from measured useful emitted tokens and
      cost. Additional width captures are restricted to controlled benchmarks.
- [x] Implement DFlash2 through the proposer interface: parallel draft,
      candidate selector, local convolution and its attention/window semantics.
- [x] Implement DSpark through the same interface: parallel backbone,
      sequential correction. Confidence-based width scheduling was investigated
      and is now archived; no adaptive serving option remains.
- [x] Profile DFlash2's graph, vocabulary/selector and accepted feature/KV work;
      remove measured greedy predecessor scalar transfers with a small exact
      device traversal. Interleaved confirmation reduces round time 2.8–5.4%
      with identical emitted IDs; all 78 HTTP outputs match in the separate
      HTTP pilot. [Evidence and limits](docs/draft-selection.md).
- [x] Batch accepted feature/KV projection through an explicit proposer callback,
      preserving private rings and endpoint snapshots after all target commits.
      Measured feedback cost falls from about 8.1 to 2.2 ms at B=4. Single-row
      arithmetic stays unchanged; interleaved B=4 round time falls 4.3–4.4%
      with identical fixed-work outputs. BF16 projection changes pass a pinned
      FP32 oracle; all 497 GB10 checks pass. [Evidence and limits](docs/accepted-batch.md).
- [x] Implement StateManager-authorized direct accepted-state reconstruction;
      validate all counts against the current verified width before mutation.
      Remove 576 MiB of FP32 scratch at B=4 and the following SSM copy. Preserve
      the bitwise oracle; all 548 GB10 checks pass, including stop/cache recovery.
      Interleaved B=4 round time falls 3.6–3.9%; B=1 results are mixed.
      [Evidence and measurements](docs/direct-state.md).
- [x] Profile shared target projections under frozen mixed arrivals. Replay all
      64 layers with real activations, include packing and B=3 graph padding,
      and test the observed session tail. Short-prefill MLP/FP8 savings pass the
      feasibility gate; whole-request speed remains unproven. Keep distinct
      prefill/verify state and attention cores. [Experiment](docs/mixed-projection.md).
- [x] Build a bounded, opt-in mixed target forward through the existing model,
      executor and StateManager boundaries. Share MLP work first as an ablation,
      then FP8 projections; prepare existing requests' proposals before the
      combined target pass. Preserve explicit row/feature ownership, graph
      padding, separate commits and the separate-forward control. Prove full
      state/EOS/cache/sampling/failure gates before measuring HTTP improvement.
      The implemented eager path saves 24–27% on a fixed prefill/verify pair,
      but its HTTP pilot regresses arrival TTFT 13–20%; the broader workload
      adds two mixed-chat rubric misses. All 654 CPU/GPU checks and 16 historical
      parity comparisons pass, which does not override these serving gates.
      This is the initial pilot; the metadata follow-up below archives public serving.
      [Implementation, measurements and limits](docs/mixed-forward.md).
- [x] Prepare GDN prefill metadata/chunk indices once per forward. CPU lengths
      and a context-owned plan replace reconstruction across 48 layers. The
      129-token control drops from 290 to 7 stream synchronizations; ordinary
      HTTP TTFT improves 18–29% with identical output IDs and exact recurrent
      state/checkpoints. [Implementation and proof](docs/prefill-plan.md).
- [x] Apply the bounded mixed-serving cutoff. After metadata planning, arrival
      TTFT still regresses 16.9–24.4% against a 5% limit. Remove the public
      serving option, followed by the internal executor/scheduler/model hooks.
      Reproduce the archived experiment at commit 56605f0; preserve its evidence
      outside Git and retain the useful ordinary prefill improvement.
- [x] Attribute DFlash2 rounds with fixed tokens and accepted lengths at B=1/4
      and 128/8K contexts. Two complete model-load windows agree on frozen work.
      The standalone fused/compact convolution passes 192 exact convolution
      and 36 complete-state comparisons and saves 74–79% of that component.
      Its estimated whole-round benefit is only 1.0–1.6%, below the frozen 2%
      screen. Archive it without changing serving; NVFP4 GEMMs dominate the
      next kernel investigation. [Cost study and decision](docs/verify-cost.md).
- [x] Compare adapters on the pinned quantized target, cold and reused prefixes,
      greedy/sampled requests and mixed concurrency. Promote a proposer/profile
      only when quality and measured end-to-end performance justify it.
- [x] Add a bounded prefill policy using an interactive decode-latency target.
      Measure requests arriving during generation. Keep graph execution where
      useful; mixed prefill/decode is an experiment, not a requirement.
- [x] Archive rejected mixed-forward, latency and confidence-width policies.
      Consolidate proposer construction at startup; expose maximum depth through
      the proposer contract instead of adapter names in generic execution.
- [x] Batch MTP accepted-token uploads through owned pinned staging. Preserve
      required target-conditioned KV writes. All 480 controlled round/state
      comparisons match; native B=4 round time falls 2.76–3.75%. Archive the
      extra KV graph prototype, whose additional benefit is inconsistent.
- [x] Build clean pinned dependencies and validate the candidate: 573 full-suite
      checks, 269 final focused checks after pruning the KV graph prototype,
      16 historical parity gates and DFlash2 eviction/replay/context checks.
      [Details and limits](docs/promotion.md).
- [x] Complete clean-image DFlash2 evaluation against repaired MTP: repeated
      fixed/natural requests, cache restoration and eviction, pressure, stopping,
      cancellation, sampled correctness and equal memory budgets. Retain every
      failed quality/performance cell. Two windows complete 414 measured requests;
      all 40 MTP performance checks pass. The sampled-chat rubric still fails,
      so the candidate is not a deployed default. [Results](docs/promotion.md).
- [x] Reconcile the generic cache-miss prompt with its rubric (the rubric
      demanded "computation must run" for a prompt whose correct answer is a
      fetch from a slower source; the MTP control failed it too). With the
      rubric fixed, the frozen workloads pass with 0 automatic failures and
      DFlash2 k=7 with accepted-state reconstruction is the production
      default since 2026-09-07 (661 GPU-suite checks on the built image,
      six-cell matrix 2x the MTP production at batch 1).
      [docs/default-dflash2.md](docs/default-dflash2.md). The failed frozen
      run stays in `docs/promotion.md`.
- [x] Attribute 128 input / 256 output at concurrency four with nine measured
      HTTP batches and repeated direct prefill partitions. Slow batches add
      51–84 B=1 rounds. First tokens can differ before speculative decoding;
      waiting for all four admissions does not remove the tail.
- [x] Trace fixed-token target prefill across the observed partitions: logits,
      margins, first divergent layer/state and draft features. Repeated traces
      locate early drift in recurrent layers; fixed operands expose shape-sensitive
      BF16 b/a projection rounding. Corrected paged-attention controls preserve
      the first-token differences. [Evidence and limits](docs/agent-context.md).
- [x] Screen greater candidate coverage per weight traversal in standalone
      benchmarks. The two-path and [16-row tree](docs/tree-coverage-study.md)
      full-continuation screens fail their completion gates; neither adds a
      serving mechanism. The tree gains 18.87% coverage but estimates only
      7.71% / 7.68% completion improvement with the frozen overhead allowance.
- [x] Isolate fixed-b/a prefill controls and the repeated eight/sixteen-row
      verifier difference. [Numerical controls](docs/verifier-numerics.md) locate
      verifier drift at layer 3 attention; fixed reduction geometry removes
      batch dependence in isolated attention. The b/a prefill clamp restores
      first tokens but leaves some late state drift. No arithmetic change is
      promoted; fixed-split B4 planning fails the current graph capacity.
- [ ] Validate graph-safe stable attention reduction geometry across growing
      KV lengths and B1/B2/B4, then measure complete-round/workload cost. Keep
      a shape-stable b/a candidate and residual prefill trace separate, with
      independent numerical references and prospective quality checks.

Definition of done: one declared production proposer/profile passes all quality,
state, memory and protected latency gates on a clean image in repeated windows.
An implementation pass or a fixed-work speed win alone does not promote it.

### Phase 8: Long-session inference and unified memory `[started; bounded 7D follow-ups alongside]`

Goal: reduce useful agent-turn completion time as history grows, with bounded
physical memory on GB10. Long-context prefill and sustained generation are
first-class work here, not deferred until final kernel packaging.

Research allocation paths and prepacked layouts alongside 7C; promote capacity
and serving changes after the draft/state planning prerequisites in 7D. Use the
session-cache, reclaimable-slab and immutable-weight experiments in
[the experiment plan](docs/gb10-experiments.md), including their stop criteria.

Work order: checkpoint retention and agent-turn evidence first, sustained decode
second, physical-memory pressure and allocation experiments third. Keep the
remaining 7D trace bounded; a noisy short synthetic cell must not hold up this work.

- [x] Retain useful near-end checkpoints within the existing state-slot budget.
      Keep target KV, recurrent state and draft features at the same boundary;
      cover identical retries, appended tool turns, branching, active loans and
      eviction. Cache-aware admission and independent snapshot reclamation keep
      useful history under slot pressure. The fourth appended turn now completes;
      near-32K retries recompute at most 64 tokens. [Repair and proof](docs/agent-context.md).
- [x] Freeze and run an agent-shaped pilot at 8K/16K/near-32K against matched
      SGLang DFlash2: 512/1,024-token fixed work, pauses, shared-history concurrent
      branches, simulated tool turns and cache churn. Record 244 measured requests
      plus 84 warmup/priming/churn requests. Retain SGLang's four clipped boundary
      outputs and a separate equal-output headroom correction. The corrected
      task checks pass 8/12 in Sharar and 6/12 in SGLang; neither has an all-pass
      episode. This is not a quality-qualified whole-agent speedup.
      [Protocol, results and limits](docs/agent-workloads.md).
- [x] Profile sustained generation at all three contexts and batch 1/4. Target
      verification spans 88–109 ms, draft work 19–22 ms and reconstruction
      1.4–5.4 ms. Retain the failed one-shot priming setup and corrected six-cell
      run. Three frozen general candidate-path screens fail; no serving hook
      is added. Kernel durations are attribution, not physical DRAM counters.
- [x] Measure the narrower single-agent two-path cost/coverage hypothesis over
      full 512/1,024-token held-out continuations at 8K/16K/near-32K, with two
      independently loaded windows. The extra path costs about 2–3 ms and eight
      writable target KV entries (256 KiB); both paths read unchanged shared
      history without a second committed recurrent slot. The optimistic
      completion estimate fails the 10% gate. [Protocol and proof](docs/two-path-study.md).
      The earlier +23.15% short-screen coverage is not a sustained speed result.
- [ ] Screen a fixed 16-target-row tree on fresh held-out full continuations:
      one anchor, seven greedy nodes and eight alternatives. Allocate the latter across
      likely corrections, sharing common prefixes. The measured low incremental
      target cost motivates better coverage; it does not validate tree attention
      or recurrent execution. Freeze selection and gates before new outputs.
      Require enough coverage to cover actual tree/state/selection overhead and
      at least 10% useful-turn completion reduction before serving integration.
      Keep winner-only KV/state publication in StateManager and candidate
      selection in the proposer; preserve the existing concurrency-four path.
      Selection must stay on the GPU with bounded reusable buffers and no extra
      per-round host synchronization or allocation.
- [x] Make chat output-control validation explicit. `tool_choice` accepts only
      omitted/null/`auto`; `response_format` accepts only omitted/null/`{"type":"text"}`.
      Other values return HTTP 400 before rendering, tokenization or submission.
      Forty-two HTTP checks cover streaming/non-streaming and accepted defaults.
      [Repair and proof](docs/two-path-study.md). Production retains its original
      image during these studies; this code change does not promote DFlash2.
- [ ] Implement and test output/tool-choice enforcement with speculative
      acceptance and constraint-state rollback. Explicit rejection establishes
      the supported API subset; it does not repair unconstrained task failures.
      Extend the pilot to actual tool execution, independent growing histories
      and CPU/tool contention. Preserve the unconstrained task failures.
- [ ] Reconcile `MemAvailable`, PyTorch allocated/reserved memory and
      non-PyTorch allocations. Budget every draft/verify/graph/activation category
      and allocator slack before optional capacity; extend the 7D reservations.
- [ ] Wire internal pressure checks and admission backpressure into serving
      before the external guard has to act. `Ledger.check_floor()` currently
      has no serving-loop caller. Keep the 16 GiB floor and external guard.
- [ ] Design reclaimable optional cache slabs with graph-safe addresses if
      runtime shrinking is needed. Returning IDs to current monolithic pools
      does not return physical DRAM. Prove reclamation under external pressure.
- [ ] Benchmark device, managed and mapped allocation paths for GEMV/GEMM,
      attention and state copies, including first touch, steady bandwidth,
      CPU contention and pressure. Record `docs/unified-memory.md`; a mapped
      tier shares the same physical pool and does not create extra capacity.
- [ ] Attribute load time to file reads/faults, tensor materialization,
      transfers, concatenation, quantization, swizzling and initialization.
      Compare a versioned prepacked artifact with direct mapping and the
      current safetensors path. Validate tensor identity, peak memory and hot
      inference bandwidth. Under 30 s cold weight load is an experimental goal.
- [ ] Measure cache value as saved workload time per byte, accounting for
      recurrent snapshots, draft features/KV and reconstruction. Compare snapshot
      placement/eviction policies and page sizes 1 versus larger blocks; useful
      hybrid-prefix reuse matters more than raw allocated KV token count.
- [ ] Raise the context profile: validate 65,536 and 131,072 as memory-safe
      profiles, and record prefill time and decode tok/s at each so the limit
      is documented as a latency choice.
- [ ] Evaluate pinned/mapped output and completion buffers with explicit CUDA
      completion, visibility and CPU ordering. CPU consumption still depends
      on GPU completion; require a measured latency benefit over current copies.

Platform constraints: [NVIDIA CUDA porting notes](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html)
and [UMA memory reporting guidance](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/optimization.html).

Definition of done: documented load-time and useful cache-reuse improvements,
validated context limits, allocation reconciliation and pressure/reclamation
tests. The host floor remains intact and hot inference does not regress.

### Phase 9: SM121 kernels and AOT `[partial; kernel studies from 7D, final AOT after 8]`

Goal: prefill toward the compute roofline and a runtime with no compiler.

Keep the measured improvements already made. Prioritize further kernels from
the [current FP8 profile](docs/gb10-fp8-profile.md); finalize AOT only once supported model/draft profiles,
prefill tails and graph/verify buckets are explicit.

- [x] Speculative round in CUDA graphs: verify forward at batch 1, 2, 4 and
      every single-row draft step (`engine/spec_graphs.py`). 2026-09-04.
- [x] Reduced speculative-round synchronization: one pinned metadata upload (pending
      token, position, slot), positions derived on device, page frees from
      host ids, FlashInfer plans fed host index pointers (its `plan()` reads
      them on the CPU, so device tensors cost a sync each), phase timing
      opt-in (`SpecDecoder.timing`). GDN state pool and verify buffers are
      stacked over layers so committing a request's state is one strided
      copy per kind. Round wall time at batch 1 went from 119 to 109 ms with
      107 ms of GPU work; the draft step is at 6.5 ms against a 6.3 ms
      measured bandwidth estimate. Acceptance still performs a host `.tolist()`
      wait; the round is not synchronization-free. These are development
      measurements. `docs/phase7.md`. 2026-09-04.
- [x] Profile the optimized FP8-draft 10K/1K workload with independent native
      timing, early/middle/late CUPTI traces and full candidate captures.
      Target verification is about 90 ms, drafting 12.3 ms and reconstruction
      1.4–1.5 ms per approximately 107 ms complete round. GPU activity is
      95.4–97.5% in instrumented spans; this is not DRAM utilization.
      Code prefill's kernel time is about 54.7% matrix operations, unlike
      decode's approximately 91%. The subsequent 16-row study collects L2
      system-memory fills, occupancy and stalls; physical LPDDR counters
      remain unavailable.
- [x] Complete the [current 16-row matrix study](docs/gb10-verifier16.md):
      nine real-operand captures, eight Nsight calls, FP4 pipeline tuning,
      940 cuBLASLt tactics and exact FP4 encoder screens. Attention-QKV and
      K512 MLP candidates preserve 144/1,152 captured matrices; their combined
      code pilot is only 1.014x faster in decode. No serving promotion.
- [ ] Profile prefill at 2,048 and 8,192 tokens with CUPTI on a separate
      profiling container; attribute time to NVFP4 GEMM, FP8 GEMM, GDN chunk
      kernels, attention, norms and quantization. (`sharar profile` gives the
      torch-profiler view already; Phase 2 numbers are in `docs/phase2.md`.)
- [ ] Close the decode FP8 GEMV gap: 194 GB/s in the model versus 225 GB/s in
      isolation for the same kernel (`sharar/kernels/csrc/fp8_gemv.cu`).
      Candidates: TLB/page locality of the FP8 weights, clock behavior under
      sustained load, and fusing the norm-quant producer into the GEMV.
- [ ] Remeasure NVFP4 packing at large prefill shapes before further work in
      `sharar/kernels/nvfp4_quant.py`. The Phase 2 54 ms/2K-token figure is
      historical. Current verification packing, including fused SiLU, costs
      only 0.4–0.9 ms for 130 calls in the isolated replay; it does not justify
      another small-row packing optimization by itself.
- [ ] Microbenchmark FlashInfer `mm_fp4` on SM121 for decode, prefill tails
      and B*verify-width shapes from the supported profiles, with the target
      and draft N/K dimensions; evaluate tile and
      cluster configs in the SM120 CUTLASS path; upstream or vendor a tuned
      config.
      The [2026-09-06 real-input study](docs/nvfp4-cost.md) completes the
      M=7/8/28/32 screen: gate/up, down and vocabulary costs, flushed/repeated
      addresses, three backends, narrow live MMA/scale-staging variants and
      a whole-round dispatch ablation. Both tile variants are exact on 520
      captured outputs each but fail the speed gate. cuDNN-through-32 saves
      only 1.3–1.6% in natural B=4 rounds, below the frozen 2% gate. Archive
      the experiments; no serving change. Other widths and large-prefill
      tails remain unmeasured here. Nsight system-memory fills into L2 are
      1.000–1.012× one packed weight/scale footprint in 16 sampled launches;
      physical LPDDR controller traffic/peak utilization remain unmeasured.
- [ ] Tune the GDN chunk kernels (`chunk_delta_h`, `chunk_intra`, `chunk_o`)
      block sizes and stages for SM121 SMEM and register budgets.
- [ ] Fuse small kernels on the decode path where a trace shows launch-bound
      gaps: gating, norms, rope, quantization.
- [ ] Build required FlashInfer, Triton and custom CUDA artifacts for every
      supported batch/verify bucket and prefill specialization. Version the
      manifest by profile, tensor layout, source/compiler/library versions and
      validated architecture flags for each backend. Load at startup and fail
      if any supported path would require compilation; test tail shapes too.
- [ ] Remove `nvcc`, `cicc`, `ptxas`, CUDA headers and JIT caches from the
      runtime image; verify with a cold, offline start.

Definition of done: single-stream prefill exceeds 3,500 tok/s at 8,192 tokens,
the realistic workload gates pass, and a cold offline image never compiles
on any supported path. The prefill target must not sacrifice decode latency,
cache reuse, quality or the memory floor.

Research candidates, admitted only through measured hypotheses and the same
gates: adaptive verification widths; selective GDN checkpoints plus
accepted-prefix replay; compressed cold snapshots with quality validation;
draft-aware cache eviction and bounded feature windows; prepacked/quantized
draft weights; efficient vocabulary selection. Record wins and losses without
claiming novelty or speedup before evidence.

### Phase 10: Product hardening `[planned]`

- [ ] Harden the health/readiness and failure behavior introduced in 7A;
      integrate live profile and pressure reporting from Phase 8.
- [ ] Soak tests: cancellation storms, repeated long contexts, cache churn,
      mixed concurrency for hours with the guard active.
- [ ] Clean shutdown and restart tests; corrupt or incomplete snapshot tests.
- [ ] Runtime image below 8 GB with an SBOM and verified notices.
- [ ] Release checklist and signed tags.
- [ ] Optional separate local chat client; never an engine dependency.

## 7. Promotion gates

Run gates in proportion to the change. Anything touching model math, memory
or scheduling runs the full sequence.

Documentation-only changes require a diff/link/consistency review, without
starting the GPU runtime. Implementation promotion requires focused CPU tests
for the changed invariants as well as the GPU/integration gates below.
Record skipped tests and unrun gates explicitly.

Fast checks:

```bash
python3 -m pytest tests/unit -q
python3 -m py_compile $(git ls-files '*.py')
bash -n tools/*.sh
git diff --check
```

Parity (Phase 1 onward):

```bash
sharar parity --prompts all --reference "${SHARAR_RESULTS}/parity/reference"
```

Live verification:

```bash
./run-sharar.sh --check
SHARAR_URL=http://127.0.0.1:30001 tools/verify-sharar.sh
```

Controlled matrix:

```bash
python3 bench/benchmark_text_server.py --label CANDIDATE_LABEL
python3 bench/compare_runs.py BASELINE_RUN_DIR CANDIDATE_RUN_DIR
```

Existing synthetic comparison thresholds (retained for continuity):

- No request failures.
- 4/4 quality canaries.
- Exact serial response hashes against the current baseline, which is
  Sharar's own frozen run (section 3); hash equality with the SGLang reference
  is not a goal (section 8).
- No cell loses more than 3% aggregate throughput.
- No cell regresses median TTFT by more than 15%.
- At least 16 GiB `MemAvailable` throughout.
- Concurrent hash drift is reported but diagnostic.

Additional required gates to implement in 7A–7D:

- Resource and lifecycle invariants: exactly-once release and terminal
  publication, no active-page reuse, correct committed prefix after every
  acceptance/stop position, preserved streamed history under retraction, bounded
  requests and failure propagation. Test cache on/off and padded batches.
- Exact speculative algorithm checks on known distributions, plus separate
  numerical parity and representative coding/chat/tool/thinking quality checks.
  Test request-local RNG independently of model rounding.
- Realistic workload TTFT, inter-token and completion latency, useful cache-hit
  behavior, emitted throughput and memory. Phase 7C records per-suite targets
  and statistical confidence before tuning; the frozen local pilot is in
  `bench/workloads/local_v1.json`, with explicit limits in `docs/phase7c.md`.
- Historical baselines stay immutable. When a verified correctness repair or
  a deliberate numerical-path change alters serial hashes, document the cause,
  before/after failing-case evidence, parity/quality results and workload costs
  before adopting a new baseline. Do not reset hashes just to pass a comparator.

Do not claim equivalent accuracy from four canaries or a 100-item subset.
Use those as smoke checks and report the limits of larger quality evaluations.

Maximum usable context:

```bash
python3 bench/benchmark_text_server.py \
  --input-lengths 31742 --output-length 1024 --concurrency 1 \
  --warmups 0 --repetitions 1 --timeout 300 --skip-quality
```

## 8. Parity methodology

- Prompts are fixed token-ID sequences, never text, so tokenizer differences
  cannot mask model differences (`sharar/parity/prompts.py`).
- Block parity: each probed layer is fed the reference's own dumped input and
  must match at cosine at least 0.995 (measured 0.9994 to 1.0000).
- Output parity: top-1 agreement at positions where the reference's top-1
  leads its top-2 by at least 0.5 nats must be at least 80% on prompts with at
  least 20 such positions, and the next token must be equal whenever the
  reference's margin is at least 5 nats.
- These thresholds come from the reference's own self-agreement: SGLang with a
  512-token prefill chunk versus the serving container agrees at 86.0% of
  confident positions on the text prompt and 98.3% on random prompts, and
  flips 4.6% of its own predictions with margins above 5 nats. These are
  empirical diagnostics for those configurations, not a universal noise floor
  or an exemption for state bugs. Details are in `docs/parity.md`.
- Tolerance failures are triaged per block with `sharar.parity.layerprobe`
  (which layers) and `sharar.parity.truth` (distance from an fp32 dequantized
  truth for both sides), before touching the full model.
- Once Sharar is production, its own frozen run becomes the baseline and the
  reference dumps stay as the historical oracle.
- Separate algorithmic exactness from kernel rounding: speculative acceptance
  must preserve the target distribution under the defined sampling transforms.
  Greedy verify/decode differences need controlled numerical comparisons;
  cached state/prefix mismatches are correctness failures regardless of margins.

## 9. Risks and open questions

- The ModelOpt NVFP4 scale layout expected by FlashInfer's SM120 CUTLASS path
  is reproduced exactly (bit-identical GEMM output against SGLang's wrapper,
  `tools/reference/ab_nvfp4.py`). Keep that A/B when the loader changes.
- Phase 7A's repaired lifecycle/cache invariants now have maintained CPU and
  GPU coverage. Preserve them through the proposer/state refactor; existing
  draft-prefix validity and broader numerical/performance questions remain.
- FP32 SSM state was required for validated quality in the reference. Any
  state dtype change is a correctness experiment.
- Cold ragged attention and cached paged attention use different numerical
  paths. Compare fixed prefixes and state boundaries when diagnosing drift.
- Wider speculative verification increases recurrent intermediate storage.
  Scratch limits, draft feature windows and graph buckets must be planned
  together; neither a larger k nor a higher acceptance rate guarantees speed.
- Managed memory on GB10 may page-fault on first touch and lose bandwidth;
  never place hot buffers there without the Phase 8 microbenchmark.
- The reference's cold start JIT-compiles FlashInfer SM120 modules; Sharar's
  dev environment inherits that until Phase 9 builds artifacts ahead of time.
- Dev environment drift: if torch, FlashInfer or Triton versions differ from
  the reference image, parity failures become ambiguous. Preserve reproducible
  constraints and artifact identities when experimenting with newer kernels.

## 10. Decision log

- 2026-09-03: Build a new engine beside SGLang instead of carving SGLang down.
  Subtractive pruning of a 408K-line framework could not reach a few thousand
  lines in reasonable time, and the coupled registries made every deletion a
  full-gate event.
- 2026-09-03: New repository `abdelfattah-lab/sharar`; the `sparklang`
  repository is frozen as the reference oracle.
- 2026-09-03: Speculative decoding is in scope from the start because
  single-stream decode is already at about 90% of the memory-bandwidth ceiling.
  The checkpoint's bf16 MTP head is the first proposer.
- 2026-09-03: Unified memory is used for capacity and copy removal, not for
  hot buffers, pending the Phase 8 microbenchmark.
- 2026-09-03: The bench canary prompt text `SPARKLANG_OK` is kept so serial
  hashes remain comparable with the frozen reference runs.
- 2026-09-03: Parity is measured against tensor dumps and HTTP responses;
  `sglang` is never imported by Sharar code or tests.
- 2026-09-03: Parity criteria were replaced by measured ones after showing
  that the reference disagrees with itself at the same level Sharar disagrees
  with it (docs/parity.md). Exact hash parity with SGLang is not a goal.
- 2026-09-03: Development happens in a container from the reference image so
  the CUDA stack is identical; the runtime image is a Phase 6 deliverable.
- 2026-09-04: Review at `21019b0` keeps the engine and inserts 7A–7D before
  remaining kernel/AOT work: correctness, contribution contracts, realistic
  competitive gates, then speculative/cache extensions.
- 2026-09-04: Phase 7A repairs A1–A8 and records before/after, CPU, HTTP failure
  and GB10 continuation evidence in `docs/phase7a.md`. Keep the historical
  performance baseline and existing runtime image until separate promotion.
- 2026-09-04: GB10 stays the hardware target; Qwen is the first curated model
  profile. Separate model/device/profile facts without arbitrary model discovery.
- 2026-09-04: Frozen correctness reference and evolving performance target are
  distinct. Include the earlier DSpark results; MTP parity is insufficient.
- 2026-09-04: Reassess the earlier 90% bandwidth estimate using actual traffic.
  Unified-memory allocation and zero-copy loading benefits require measurements;
  optional cache eviction alone does not reclaim monolithic allocations.
- 2026-09-06: The agent-shaped DFlash2 comparison has a modest generation lead,
  larger exact-retry benefits and structured-turn failures in both engines.
  Preserve failed tool-template setup and clipped output counts; use separate
  prospective corrections. No whole-agent quality or universal speed claim.
- 2026-09-06: All three general draft-path coverage screens fail. The two-path
  single-agent observation motivates fresh cost/coverage measurement under a
  narrower protocol, not a retroactive pass or a new runtime policy.
- 2026-09-06: The full-length two-path follow-up completes 48 generations and
  10,304 native speculative request-rounds. Extra target cost is 2–3 ms with
  unchanged shared history, but estimated completion reductions of 6.39% and
  6.47% fail the 10% gate. Add no two-path serving hook; screen a fixed 16-row
  candidate tree prospectively. [Proof](docs/two-path-study.md).
- 2026-09-06: Chat now rejects unsupported output controls before prompt work
  or submission. Forty-two HTTP cases pass; actual constraint enforcement and
  deployment through a validated image remain separate work.
- 2026-09-06: The 16-row tree study completes 48 generations and 8,852 native
  speculative request-rounds, with exact control/capture output and committed
  state comparisons. Coverage improves 18.87%, including 14.39% after token 512,
  but projected completion gains of 7.71% / 7.68% fail the frozen 10% gate.
  Add no tree serving hook. An 8K first-chain emission change under physical
  eight/sixteen-row verification repeats in both loads; extend the pending
  numerical controls before changing arithmetic. [Proof](docs/tree-coverage-study.md).
- 2026-09-06: Repeated verifier controls locate the first shape difference at
  layer 3 FA2 attention with identical Q/K/V. Stable reduction geometry removes
  isolated attention batch dependence, but fixed splitting fails B4 planning
  on the current graph buffers. FP64 references support reduction rounding;
  no KV/state corruption is found. Fixed b/a restores prefill first tokens
  with residual late state differences. Keep serving unchanged and validate
  graph-safe geometry before measuring a new speed path.
  [Proof](docs/verifier-numerics.md).

## 11. Immediate next task

The [BS1 implementation study](docs/gb10-bs1-speedups.md) supersedes the
unmeasured kernel stack in the original 1.5x plan. Exact M=16 attention QKV
and K512 MLP dispatch is integrated: family decode improves 1.43%
[1.37%, 1.49%], with all 21 outputs identical. The integrated source passes
53 GPU tests and 1,296 captured-matrix comparisons. The report records the
paired HTTP result and quality canaries. The numerical FP8, two compressed
draft and top-32 candidates do not establish reliable family throughput gains
and remain experimental. A full real-weight FP8 sequence saves 2.04 ms,
not the isolated estimate's 6 ms; shape tuning adds only 0.033 ms.
The [BS4 study](docs/gb10-bs4-engine.md) completes width and scheduling measurements:
16 / 8 / 4 verifier rows give 132.70 / 129.25 / 104.80 pooled family decode TPS.
The larger prefill budget and four-decode-step policies lose their HTTP screens.
Runtime-count normalization/RoPE removes first-use compilation stalls and passes
135 GPU checks with all 27 fixed native outputs preserved, but its warm HTTP
point estimates are lower; it remains in the frozen experimental source.
No BS4 serving change is promoted. Continue with [engine-only work](docs/gb10-engine-next.md),
with training deferred. BS1 host overlap has about 1% round-time headroom;
no large cold BS1/BS4 engine gain or 1.5x result is established.

The [FP8 profile and follow-up experiments](docs/gb10-fp8-profile.md) now
complete the immediate profiling task. Native 10K/1K B1 decode reproduces
44.32 / 34.16 / 56.31 TPS for code / design / reasoning. Full-generation
candidate-support ceilings are 7.85 / 7.29 / 9.66 emitted tokens per round
on captured native histories; reaching 100 decode TPS at current cost needs
about 10.7. These ceilings are diagnostics, not achievable policies.

Fresh three-repeat AIPerf measures 37.61 / 30.11 / 46.06 end-to-end TPS
(44.15 / 34.16 / 56.31 decode TPS). Gains versus the fastest task means
across four pinned SGLang loads are 20.58% / 2.61% / 13.39%, with different
cross-engine outputs and draft precisions. All 24 requests pass raw-stream
audits, 397 CPU tests pass with 8 skipped, and all four Sharar canaries pass.
The original service and memory guard are restored.

The fixed eight-feature selector failed held-out reasoning coverage (-1.62%)
and is not integrated. Expanded FP4 calibration still passes only five of
ten sampled projections; later layers fail the 5% held-out error screen.
Do not repeat score/rank-only fitting or uniform target conversion without
a changed mechanism. The [16-row engine follow-up](docs/gb10-verifier16.md)
now completes the counter and matrix studies. The requested 1.5x–2x goal
remains open: the combined matrix pilot improves code decode by only 1.4%.
A 32-row probability-tree screen adds 19–32% fixed-history coverage, while
flat verifier cost rises from about 90 to 94 ms. Actual branching cost,
winner commit and cross-shape numerical behavior remain unresolved.
Continue engine verification/fusion work alongside repository-separated
target/draft supervision and richer prefix-length training. Validate
surviving mixed-precision candidates with whole-model
quality and long continuations. Retain separate cold-request and warm-session
latency goals. Launcher support now makes FP8 an explicit environment option.

The [request-cache study and API proposal](docs/request-cache-policy.md) finds
a separate agent-session performance cliff: target KV/GDN remains cached while
draft features are evicted. Reusing duplicate checkpoint handles preserves the
tested 10K/1K retry after one intervening request at the original 80 MiB feature
budget: 11.40 -> 42.38 decode TPS, 87.76 -> 23.68 s completion. Fixed outputs
match the immediate speculative control, while the old ordinary-decode
fallback produces a different continuation. The subsequent
[checkpoint-retention study](docs/gb10-agent-retention.md) binds draft features
to the exact recurrent slot, including free/recycle and cold-tier restore.
Two independent loads per variant recover resumed BS1 completion throughput
from 11.34 to 36.73 TPS (3.24×), with all 12 candidate probes matching their
healthy immediate-hit oracles. At 96 slots, four concurrent simulated
conversations improve from 28.78 to 78.48 TPS (2.73×), measured in two timed
blocks per implementation in one fresh process each. Cold throughput is
essentially unchanged. These recoveries are not additive to the earlier repair.

The feature cap grows to one window per usable recurrent slot: 460 MiB at
24 slots and 1,900 MiB at 96 slots, reserved before automatic KV planning.
An enlarged independent LRU also recovers the tested warm throughput;
checkpoint ownership makes lifetime and reclamation predictable. Target-state
pressure can still evict a whole prefix. The scheduler experiment reduces
mixed throughput 7.27% and worsens streaming gaps, so it stays in `bench/`.
Validation passes 447 CPU tests (eight skipped), 45 affected GPU tests and
32 HTTP canaries. Production retains its original image. Per-request
reuse/store modes, namespaces, expiring priority and latest/branch checkpoint
policies remain to implement. This does not close the cold TPS goal.

The preceding [TPS validation](docs/gb10-tps.md) measures the repaired target with
the explicit FP8 draft option at 37.65 / 30.14 / 46.09 output tokens/s including
prefill for code / design / reasoning. AIPerf uses exact 10K input and 1K output
IDs, greedy decoding, concurrency one and five timed requests per task/engine.
Gains versus the fastest task mean across three pinned SGLang loads are
20.71% / 3.30% / 13.46%. All 36 requests pass the raw-stream count audit;
373 CPU tests pass, and the preceding 128 targeted GPU tests cover identical
source. FP8 remains opt-in and production retains its original image.

The [complete target repairs](docs/gb10-target-fix.md) supersede the partial
norm/weight ablations below. They establish exact target and committed-state
agreement for 65 common-history blocks / 307 accepted rows, while independent
greedy outputs still differ. SGLang itself varies across fresh processes.
The next correctness work is to isolate the remaining full-generation gap
and expand held-out answer-quality coverage. Four passing canaries do not
establish equal overall quality; [the distinction is documented here](docs/greedy-quality.md).
The following investigation entries retain their historical results.

The [SGLang acceptance investigation](docs/gb10-acceptance-parity.md) corrects
the earlier acceptance diagnosis: the 264/266-versus-205 round comparison
follows different continuations. On 128 identical SGLang histories, original
Sharar matches 481 draft prefix tokens versus 489, or 609 versus 617 including
correction tokens. This does not support a large selector porting defect or
make retraining the next priority. Target norm and weight-loading differences
were isolated and executed; matching more reference arithmetic does not yield
a quality-qualified speedup. The FP32 residual-sum variant fails strict JSON
and regresses reasoning. Its benchmark-only switch defaults off; existing
serving arithmetic and the production image are preserved. Use the new
teacher-forced replay to evaluate future parity changes before attributing
free-generation round counts to draft accuracy.

The [GB10 implementation follow-up](docs/gb10-optimization-results.md) adds
an exact fused BF16 draft convolution and an explicit `--draft-precision fp8`
profile with two-kernel activation packing and released BF16 matrix storage
(1.44 GiB less live allocation). The paired 10k/1k HTTP run reduces B1 code
time 4.50%, B4 code batch time 5.83%, and design time 6.52%; reasoning regresses
2.51%. BF16 remains the default: the primary code result misses the frozen
5% gate. An eight-row FP8 profile cuts B4 code time 8.67% and design time
16.92% but regresses reasoning 3.01%. Selector FP32 arithmetic fails the
held-out regression limit; it stays out of serving. The target-FP4 layerwise
screen measures 1.4–1.7× local projection speedups, but five of ten sampled
projections exceed the 5% numerical-error threshold. Target conversion stays
disabled. The fixed-history study finds most first-rejection targets inside
the existing top-16 support, with almost no improvement from FP32 score
addition. Better draft selection remains a research option, but the later
reference-history comparison does not establish it as the cause of the
SGLang gap. Properly calibrated target quantization remains a potential way
to reduce per-round traffic. The production image has not been rebuilt.

The [original exact 10k/1k audit](docs/gb10-10k1k-audit.md) supplies the kernel
and bandwidth evidence. B1 code emits about 3.8 tokens per target traversal,
with no full sixteen-token rounds. Trace gaps are only a few milliseconds,
so the older short-input 16 ms acceptance-gap estimate below is not a 10k
speedup prediction. Its matched B1 control was 35.41 s versus SGLang-16's
30.89 s, with 3.79 versus 4.88 emitted tokens per round. The apparent B4
reference gap includes a delayed fourth admission and is not a certified
saturated-B4 advantage. The implementation follow-up replaces the initial
single-sample FP8 feasibility result with paired HTTP and held-out evidence.

[docs/batch4-and-sharegpt.md](docs/batch4-and-sharegpt.md) closes this
round: the batch-4 verify round is GPU-bound (136 ms, kernels at their
floors, about 16 ms of launch gaps in the accept path), cuDNN serves NVFP4
GEMMs up to 64 rows, and the ShareGPT comparison against SGLang DFlash2 and
Atlas DFlash2 is recorded (Sharar leads 7 of 9 cells, lowest TTFT
everywhere, loses short prompts at concurrency 4 and 8 by about 30%).
Next, in order: capture the accept path (draft feature projection, ring
writes, accepted-state reconstruction) as a fixed-shape CUDA graph with
masked rows (about 10% of every batched round, the short x 4 gap); the
GEMM-epilogue NVFP4 quantization as a CUTLASS project (two-GEMM design with
an auxiliary-input epilogue, about 6% of prefill); the conv-kernel q/k/v
layout (2% of prefill); then draft quality on hard text, which is training
work and the only remaining 2x.

Hold the shipped engine to Qwen on GB10. Do not extract a generic target layout
or add dense/MoE models. Proposer construction lives in
`proposers/factory.py`; execution consumes features and capacity requirements.
StateManager owns target state/cache loans; proposers own their draft state.

Runtime pruning, the MTP repair, clean-image correctness and the repeat matrix
are complete ([proof](docs/promotion.md)). Full raw evidence stays under `$SHARAR_RESULTS`,
with compact manifests and report summaries in Git.

The [agent-shaped comparison and generation profile](docs/agent-workloads.md)
and [two-path follow-up](docs/two-path-study.md) are complete. The chat handler now
rejects unsupported output controls before admission. Target/executor/proposer/
kernel sources are unchanged. The two-path cost/coverage screen fails its 10%
completion gate, so it adds no serving mechanism. Sharar's existing generation
lead remains workload-dependent, with no quality-qualified whole-agent claim.

The [FP8 verifier follow-up](docs/atlas-verifier-study.md) also fails its
frozen promotion gates: combined full-round savings are 1.97–2.90% at B=1
and 0.70–0.99% at B=4. It adds no serving kernel or experimental switch.
The next speed mechanism must increase useful accepted tokens per target
traversal or demonstrate a larger measured reduction in state movement.

The same study adds a frozen HTTP comparison and audits Atlas's DFlash2 path
on our checkpoint. Atlas's recorded rates remain diagnostics: chunked capture
uses the wrong feature offset, cached draft history is incomplete, and its
requested 2K draft window is not applied. Qualifying that implementation would
require separate repairs and GPU validation. Keep that work outside Sharar;
use the matched Sharar–SGLang repeat results to assess our current speed gap.
The four-sample coding matrix puts cold single-request decode between 4.2%
slower and 9.6% faster, and shared-prefix B4 throughput 7.7–17.6% higher.
That is a modest lead; a larger useful-turn gain still needs a new mechanism.

The [16-target-row tree study](docs/tree-coverage-study.md) is now complete.
It preserves the full greedy chain and spends eight extra nodes on draft-scored
prefixes. Coverage rises 18.87%, but estimated completion falls only 7.71% /
7.68% with the frozen overhead allowance. Even the optimistic projection leaves
only 0.60–0.66 ms per round for additional tree work to retain a 10% gain.
The screen fails; it adds no tree verifier, selector policy or serving flag.
These are native-trajectory estimates, not measured alternative-rollout speed.

The [numerical control study](docs/verifier-numerics.md) is complete: 168 verifier
forwards, 160 request prefills and 54 complete attention reference evaluations.
Verifier drift starts at layer 3 attention with identical Q/K/V; a common
single-chain attention wrapper removes all recorded downstream differences.
Fixed-split and unsplit attention are batch-invariant on the isolated cases,
with FP64 error consistent with reduction rounding. Fixed b/a restores prefill
first-token choices but leaves some late tensor/state drift. This does not
promote a new arithmetic profile or establish a speed gain.

1. Study **graph-safe stable attention reduction geometry**. Fixed splitting
   cannot simply be enabled: the current B4 graph plan exceeds its padded
   batch capacity, and changing KV lengths can change the launched work.
   Provision and validate the geometry across context growth and B1/B2/B4,
   shared prefixes, private suffixes and graph replay. Compare native, fixed
   and unsplit controls against eager/reference outputs before measuring
   complete verifier rounds and completed workload time. Keep shape-stable
   b/a arithmetic and the residual prefill trace separate. Any numerical-path
   change needs prospective quality validation; do not select a favorable
   partition by its benchmark token. Greater useful coverage or less state/KV
   movement still needs a fresh mechanism and cost/quality gates. Future trees
   must demonstrate actual full-rollout improvement of at least 10%, correct
   winner-only KV/recurrent commit and draft feedback, task quality, the memory
   floor and protected B4 behavior before integration.
2. Release the API validation repair through a validated image, then implement
   output/tool-choice enforcement with request-owned constraint state and
   explicit speculative acceptance/rollback tests. Validation currently
   accepts only unconstrained defaults; the prior structured-task failures remain
   failures. These experiment windows restore production's original image and
   do not deploy that change.
3. Complete the remaining 7D prospective quality evaluation after the numerical
   controls above.
   The target-prefill trace is complete, including corrected attention controls;
   no favorable synthetic partition is forced. Preserve the failed generic
   cache-miss rubric and new unconstrained task failures, align new prompts/rubrics
   and output contracts, then obtain independent review before default promotion.
   Initial unwanted tool calls begin with the target prefill token; they do not
   establish a speculative rejection bug. The earlier 40 MTP performance gates
   do not establish a universal SGLang lead.
4. Follow with physical-memory pressure/admission checks, reclaimable optional
   cache slabs and measured allocation/prepacked-loading experiments. Current
   snapshot eviction reuses slots; it does not release physical DRAM. Measure
   first touch and physical recovery under CPU/tool contention while preserving
   graph references, active loans and the host floor. Targeted SM121 kernels
   accompany these measurements; final AOT follows stable profiles.

The shared GDN prefill plan improves ordinary TTFT 18–29% in the original study.
Compact convolution and narrow NVFP4 tiles failed their integration screens.
Measured GEMM fills are approximately one packed weight/scale footprint into L2;
this is not proof of physical LPDDR saturation. Prioritize useful accepted tokens
per target traversal and less state movement, with exact work and state evidence.

Production changes require a completed promotion result. Validation windows
restore the original image and memory guard after testing.

### Exact-arithmetic GB10 fusions — 2026-09-08

The [paired AIPerf follow-up](docs/gb10-engine-speedups.md) implements fused
GDN verification convolution/SiLU/state-window writes, strided B/A projection
views, draft RMS/RoPE/KV fusions and exact-count accepted-feedback graphs.
Code 10k/1k latency falls from 35.144 to 33.803 s (3.82%); design improves
3.82% and reasoning 3.36%. All 24 final requests, including warmups, preserve
Sharar's previous 1,000 output IDs and text. Code remains at 264 verify rounds;
its effective round cost falls from 118.920 to 113.880 ms. The combined change
passes its predeclared 3% primary gate without relaxing earlier standalone
kernel gates. Target FP8 tiles changed text and remain an unqualified screen.

Validation passes 373 CPU tests, 83 focused GPU checks, 22 real-model lifecycle
tests and all 16 final quality canaries. SiLU matches Torch for every finite
BF16 input. Graph tests cover varying accepted counts, private rings, retained
input storage and checkpoint/eager fallbacks. Extra warmed CUDA allocation is
under 1 MiB. The original production image and memory guard are restored;
these results exercise the updated source in isolated benchmark containers.

The earlier SGLang code result is 30.904 s: updated Sharar remains 9.38% slower
on code, and cross-engine greedy-output parity remains unresolved. A wide win
is still open. Preserve exact-output checks when pursuing target arithmetic
alignment, better learned proposals or calibrated target compression.

### GB10 target arithmetic and remaining cost — 2026-09-08

The [target-path diagnosis](docs/gb10-target-path.md) repairs E2M1 quarter-tie
rounding and signed zero in both NVFP4 quantizers. Exact boundary tests now
cover the standalone and fused-normalization paths. The existing SiLU test
also freezes expected scale bytes before the reusable buffer is overwritten.
Validation passes 373 CPU tests and 22 retained-kernel GPU tests; another
16 GPU tests validate a diagnostic SiLU candidate. Production is restored
with its original image and memory guard after every window.

Fresh profiling is 96.82% GPU-active, with GEMMs accounting for approximately
92.65% of GPU activity. AIPerf audits 27 fixed-token requests across three
profiles. QKV/Z fusion preserves complete corrected-baseline IDs/text but
saves only 0.58% on code, below its 3% gate. It remains outside the engine.

An actual-operand comparison isolates the missing BF16 product boundary in
the fused SiLU/FP4 path: 0.81% of packed bytes and 1.76% of scale bytes differ
from CUDA. Restoring that boundary matches the sampled native bytes exactly
and passes all canaries, but changes continuations: code improves 9.27%,
design regresses 9.91%, and reasoning regresses 21.80%. That partial arithmetic
alignment is not promoted. Complete target intermediate parity before
treating a changed continuation's acceptance gain as an engine speedup.
