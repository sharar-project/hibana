# Sharar roadmap

Last updated: 2026-09-03 (Phase 3 complete)

This is the durable handoff for humans and coding agents working on Sharar. It
records the goal, the facts that constrain the design, the architecture, every
phase with its concrete tasks and definition of done, the promotion gates, and
the decisions already made. Update it whenever a milestone changes the proven
state.

## 1. Mission and end state

Build the smallest practical, fastest local text-inference engine for one
system and one model:

- One NVIDIA GB10, compute capability 12.1, `aarch64`, unified memory.
- `RadixArk/Qwen3.8-27B-NVFP4`, revision
  `319f741cce68d7914884900c138a1fbb70a42f30`.
- Local, offline, OpenAI-compatible streaming chat and completions.

The finished engine has these properties:

1. `sharar serve` is the only server entry point. Model, revision, quantization
   and hardware are facts of the package, not flags.
2. The process contains no SGLang. All model math, memory management,
   scheduling and serving are Sharar modules with narrow interfaces and tests.
3. Speculative decoding with the checkpoint's MTP head is the default decode
   path, and the proposer is an interface so other proposers can be added.
4. Memory is planned as one unified pool with a hard host floor, and the
   prefix/session cache uses the capacity a discrete GPU cannot offer.
5. Every kernel is built ahead of time for SM121. The runtime image carries no
   CUDA compiler, no JIT cache, and cold start is a weight load.
6. Output quality, serial determinism and memory safety meet the frozen
   reference on every promoted change; throughput and latency exceed it.

Multimodal, multi-GPU, multi-node, arbitrary models, LoRA, generic hardware
fallbacks and inherited CI are outside the target and must not be added.

## 2. Relationship to the `sparklang` repository

`sparklang` (github.com/yahya010/sparklang, branch `gb10-engine`) contains a
full SGLang checkout, the flattened reference image
`sparklang-gb10-engine:qwen38-27b`, and the launcher/verify/guard scripts. It
is frozen. Its roles for Sharar are:

- Correctness oracle: per-layer hidden states and logits, serial response
  hashes, and the four quality canaries.
- Performance oracle: the six-cell controlled matrix and the 32,766-token
  boundary result.
- Source of ported code and vendored kernels, with attribution.

Sharar never imports `sglang`. Parity is measured over the network against the
running reference container or against tensor dumps produced inside it. The
reference is retired from daily use only after Phase 6.

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

### Measured reference baseline

Frozen result directories live under `${SPARKLANG_RESULTS}` on the host:
`controlled-27b-sparklang-cli-candidate-20260903T005850Z` (matrix) and
`controlled-27b-sparklang-cli-candidate-32766-usable-20260903T011622Z`
(boundary).

| Input | Concurrency | Aggregate tok/s | Median TTFT (s) | p95 TTFT (s) |
|---:|---:|---:|---:|---:|
| 128 | 1 | 12.74 | 0.13 | 0.13 |
| 128 | 4 | 46.43 | 0.31 | 0.31 |
| 2,048 | 1 | 12.22 | 0.81 | 0.81 |
| 2,048 | 4 | 40.67 | 2.34 | 3.07 |
| 8,192 | 1 | 10.80 | 3.33 | 3.33 |
| 8,192 | 4 | 18.30 | 8.46 | 31.7 |

Reference profile: 32,768-token budget, concurrency 4, queue 16, chunked
prefill 2,048, 20 FP32 Mamba slots, static fraction 0.70, FlashInfer
attention, Triton GDN, cold start about 2 minutes of which weight load is 99 s.

### What the numbers mean

- Decode at concurrency 1 reads about 19 GB of weights per token, which is
  about 245 GB/s effective, about 90% of the bandwidth ceiling. Kernel work
  cannot make single-stream decode much faster; only fewer weight reads per
  emitted token can, which means speculative decoding.
- Batch decode scales 3.6x at concurrency 4 and is also near the ceiling.
- Single-stream prefill runs at about 2,500 tok/s, roughly half of dense FP8
  peak. Prefill kernels (NVFP4 GEMM on SM120-class tiles, GDN chunk kernels)
  are the real kernel optimization surface.
- The p95 TTFT of 31.7 s at 8K x 4 is prefill serialization: decode pauses for
  every 2,048-token chunk and later requests wait behind earlier prefills.
  This is scheduler policy, fixed by owning the loop.
- Weight load at 99 s is a 200 MB/s read path. On unified memory it should be
  a zero-copy map, several times faster.
- KV is about 32 KB per token at FP8 (16 layers x 4 heads x 256 x 2 x 1 byte),
  so 32K tokens is 1 GB. Each GDN state slot is about 147 MB. After weights,
  about 85 GiB remain: the prefix/session cache can hold over a million tokens
  and hundreds of state slots. There are no host/device tiers on this machine.

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
| `sharar/config.py` | Pinned model/hardware facts, profile validation | 250 | new, from sparklang `gb10/` |
| `sharar/weights/` | Safetensors reader, ModelOpt layouts, NVFP4 block-scale swizzle, FP8 scales, tensor name mapping | 500 | port of `modelopt_quant.py` pieces |
| `sharar/model/` | `qwen38.py` (layers, forward), `gdn.py`, `attention.py`, `mlp.py`, `norm.py`, `rope.py`, `mtp.py` | 1,000 | rewrite, reference `qwen3_5.py`, `qwen3_5_mtp.py` |
| `sharar/kernels/fla/` | GDN Triton: `chunk`, `chunk_fwd`, `chunk_delta_h`, `chunk_o`, `chunk_intra`, `cumsum`, `wy_fast`, `l2norm`, `layernorm_gated`, `fused_gdn_gating`, `fused_sigmoid_gating_recurrent`, `fused_recurrent`, `index`, `op`, `utils` | 4,000 | vendored (fla-org via SGLang) |
| `sharar/kernels/conv1d.py` | Causal Conv1D prefill and state update | 1,200 | vendored (Tri Dao / vLLM via SGLang) |
| `sharar/kernels/ops.py` | FlashInfer wrappers: ragged/paged attention with FP8 KV, `mm_fp4`, FP8 GEMM, `fp4_quantize`, fused SiLU-mul-quant, top-k/top-p sampling | 400 | new |
| `sharar/memory/` | `ledger.py` (one pool, host floor), `kv_pool.py` (paged FP8), `state_pool.py` (GDN conv+SSM slots, committed/scratch), `radix.py` (prefix tree with state snapshots) | 900 | rewrite, reference `mamba_radix_cache.py` |
| `sharar/engine/` | `request.py`, `scheduler.py`, `executor.py` (batch assembly, CUDA graphs), `sampler.py`, `spec/` (`proposer.py`, `mtp.py`, `verify.py`) | 1,300 | new |
| `sharar/api/` | `http.py`, `openai.py`, `chat_template.py`, `parsers/reasoning.py`, `parsers/tools.py` | 1,000 | new; parsers ported |
| `sharar/cli.py` | `serve`, `describe`, `check`, `parity` | 200 | new |

Total about 6,500 engine lines plus about 5,200 vendored kernel lines. Budgets
are ceilings for review, not targets to fill.

### Design rules that must hold from the first commit

- One process. Tokenization, scheduling, execution and detokenization run in one
  asyncio process. No ZMQ, no worker processes, no distributed groups.
- Committed and scratch state. Every request owns one committed GDN state slot
  (conv + SSM) and the executor owns scratch slots. Prefill and accepted decode
  write committed state; speculative verification writes scratch and commits on
  accept. This is what makes MTP and rollback cheap later.
- No `auto`. Every backend, dtype and shape is chosen explicitly in code with a
  comment naming the measurement that chose it.
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
6. Do not add multimodal, distributed, LoRA, arbitrary-model or generic
   fallback code paths, even behind flags.
7. The host memory guard runs whenever any engine is serving on the machine.
8. Do not import `sglang` anywhere in `sharar/`, `tests/` or `bench/`. Tests
   talk to the reference over HTTP or read its tensor dumps.
9. No CI/CD until Sharar defines its own; GitHub Actions stay disabled.
10. Commits carry no attribution trailers.

## 6. Phases

Phases 1 through 6 replace the reference. Phases 7 through 10 exceed it. The
order inside a phase is the recommended order of work; each task is meant to be
one commit or one small series.

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
5 nats. The original wording (per-layer tolerance, all-position argmax
equality) is unattainable for this W4A4 checkpoint by any second
implementation, SGLang included, and was replaced by the measured criteria.

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
      below 37.9 GiB. The ledger-versus-allocator comparison is covered by the
      recorded categories; the memory guard was active throughout.

Definition of done, as measured (`docs/phase3.md`): the ledger and cache own
all allocations, a re-sent 9,642-token prompt drops from 4.4 s to 158 ms TTFT
with the same first token, the next conversation turn reuses the previous one,
and the soak passes with the guard active.

### Phase 4: Scheduler `[planned]`

Goal: continuous batching for concurrency 1 to 4 with better long-input
behavior than the reference.

- [ ] Implement `engine/request.py`: lifecycle states, cancellation, stop
      strings and stop token IDs, max tokens, streaming cursor.
- [ ] Implement `engine/scheduler.py`: FCFS admission with queue bound, running
      set bound, token-budget check against the ledger, chunked prefill, and
      retraction on out-of-memory.
- [ ] Implement mixed steps: a prefill chunk and the running decode batch run
      in the same forward so decode never stalls for a whole chunk. Make the
      per-step prefill token budget a profile value.
- [ ] Batch the GDN and attention paths for mixed extend/decode batches
      (varlen prefill with per-request state slots).
- [ ] Add a deterministic scheduler simulation (no GPU) for concurrency 1 to 4,
      queue overflow, cancellation mid-prefill and mid-decode, and a 31,742 +
      1,024 token request.
- [ ] Expose metrics: queue depth, running, prefill and decode tokens per step,
      cache hit rate, retractions, step latency.

Definition of done: the six-cell matrix meets the acceptance thresholds in
section 7, p95 TTFT at 8K x 4 is below 15 s, and all simulation cases pass.

### Phase 5: API and CLI `[planned]`

Goal: a drop-in replacement for the reference's HTTP surface as used by the
harness and by local clients.

- [ ] Implement `api/http.py` on a minimal async server (uvicorn plus a small
      router, or plain asyncio HTTP if it stays under budget), bound to
      127.0.0.1 by default.
- [ ] Implement `/v1/chat/completions`, `/v1/completions` (streaming and not),
      `/v1/models`, `/health`, `/tokenize`, `/describe` (profile and ledger),
      and `/flush_cache` because the harness flushes between batches.
- [ ] Implement `api/chat_template.py` with the snapshot's `chat_template.jinja`
      and the `enable_thinking` switch the harness uses.
- [ ] Port the Qwen3 reasoning parser and tool-call parser into
      `api/parsers/`, text-only, with unit tests from captured reference
      outputs.
- [ ] Implement `sharar serve`, `describe`, `check`, `parity`.
- [ ] Point `bench/` and `tools/verify-reference.sh` at Sharar via
      configuration and add `tools/verify-sharar.sh`.

Definition of done: the harness runs unchanged against Sharar, the four quality
canaries pass, and serial response hashes equal the reference's.

### Phase 6: Promotion and reference retirement `[planned]`

- [ ] Run the complete gate sequence (section 7) against Sharar with the
      reference matrix as baseline; record result basenames here.
- [ ] Run the 32,766-token boundary and a cold restart.
- [ ] Build a first Sharar container image from a CUDA 13 aarch64 base with
      only the pinned packages, and run the gates offline (no network
      namespace).
- [ ] Update this roadmap's baseline table with Sharar's numbers and mark the
      reference as historical.

Definition of done: Sharar is the production engine on the machine; the
reference container is started only for parity work.

### Phase 7: Speculative decoding with MTP `[planned]`

Goal: more emitted tokens per weight read. Expected 1.5x to 2x at concurrency 1.

- [ ] Load the bf16 MTP module (`mtp.fc`, `pre_fc_norm_embedding`,
      `pre_fc_norm_hidden`, one full-attention layer with q/k norm, `mtp.norm`)
      sharing embeddings and lm_head.
- [ ] Implement `engine/spec/proposer.py` as the interface: given committed
      hidden state and last token, propose up to k tokens. Implement
      `spec/mtp.py` as a chain proposer (k = 1 to 3, measured).
- [ ] Implement `spec/verify.py`: target forward over the k + 1 tokens in one
      step using scratch state slots and a verify attention mask; accept the
      longest matching prefix; commit scratch on accept; rollback is a no-op
      because committed state was never touched.
- [ ] Vendor the GDN verify kernel path only if a scratch-slot approach cannot
      express it; the reference's replay-SSM kernels are the fallback design.
- [ ] Extend CUDA graphs to the verify batch shapes.
- [ ] Add acceptance-rate and accept-length to metrics and to the harness rows
      (columns already exist).
- [ ] Keep exact sampling semantics: greedy verification must reproduce the
      non-speculative greedy output; sampled verification uses standard
      rejection sampling.
- [ ] Leave room for a second proposer (n-gram or an SMC-style proposer from the
      `smcsd` work) by keeping the verifier proposer-agnostic.

Definition of done: greedy outputs are identical with and without MTP on the
canaries and the 2,048-token prompt, concurrency-1 decode improves by at least
40% on the 128 and 2,048 cells, no cell regresses, and memory stays within the
ledger.

### Phase 8: Unified memory `[planned]`

Goal: use the shared pool where it helps and prove where it does not.

- [ ] Microbenchmark `cudaMalloc` versus `cudaMallocManaged` versus
      `cudaHostRegister`-mapped memory for a 19 GB streaming read on GB10.
      Record bandwidth and first-touch cost in `docs/unified-memory.md`.
- [ ] Zero-copy weight load: map the safetensors, convert directly into
      GPU-resident tensors without a host staging copy, and drop page cache
      after materialization. Target under 30 s cold weight load.
- [ ] Decide from the microbenchmark whether cold cache tiers (evicted radix
      nodes) may live in mapped memory to enlarge the cache beyond the device
      allocation budget.
- [ ] Extend the ledger with a runtime planner: derive KV pages and state
      slots from current `MemAvailable`, keep the 16 GiB floor, and apply
      admission backpressure before the external guard has to act.
- [ ] Raise the context profile: validate 65,536 and 131,072 as memory-safe
      profiles, and record prefill time and decode tok/s at each so the limit
      is documented as a latency choice.
- [ ] Remove CPU to GPU synchronizations in the step loop: sampled tokens and
      finish flags read through mapped memory rather than `.cpu()` copies.

Definition of done: weight load time, cache capacity and longest safe context
are all improved and documented, and the host floor is never crossed in the
soak.

### Phase 9: SM121 kernels and AOT `[planned]`

Goal: prefill toward the compute roofline and a runtime with no compiler.

- [ ] Profile prefill at 2,048 and 8,192 tokens with CUPTI on a separate
      profiling container; attribute time to NVFP4 GEMM, FP8 GEMM, GDN chunk
      kernels, attention, norms and quantization. (`sharar profile` gives the
      torch-profiler view already; Phase 2 numbers are in `docs/phase2.md`.)
- [ ] Close the decode FP8 GEMV gap: 194 GB/s in the model versus 225 GB/s in
      isolation for the same kernel (`sharar/kernels/csrc/fp8_gemv.cu`).
      Candidates: TLB/page locality of the FP8 weights, clock behavior under
      sustained load, and fusing the norm-quant producer into the GEMV.
- [ ] Make the NVFP4 activation quantizer memory bound
      (`sharar/kernels/nvfp4_quant.py`): 54 ms per 2K-token prefill today for
      about 7 GB of traffic; bit-trick e2m1 rounding and vectorized packing
      should bring it under 15 ms.
- [ ] Microbenchmark FlashInfer `mm_fp4` on SM121 for the exact shapes
      (M in {1, 2, 4, 256, 2048}, N/K from the MLP) and evaluate tile and
      cluster configs in the SM120 CUTLASS path; upstream or vendor a tuned
      config.
- [ ] Tune the GDN chunk kernels (`chunk_delta_h`, `chunk_intra`, `chunk_o`)
      block sizes and stages for SM121 SMEM and register budgets.
- [ ] Fuse small kernels on the decode path where a trace shows launch-bound
      gaps: gating, norms, rope, quantization.
- [ ] Build all FlashInfer JIT modules and Triton kernels ahead of time for
      `sm_121a` into a versioned artifact directory; load from it at startup;
      fail startup if a compile would be needed.
- [ ] Remove `nvcc`, `cicc`, `ptxas`, CUDA headers and JIT caches from the
      runtime image; verify with a cold, offline start.

Definition of done: single-stream prefill exceeds 3,500 tok/s at 8,192 tokens,
decode does not regress, and the image never compiles.

### Phase 10: Product hardening `[planned]`

- [ ] Health, readiness, profile and memory-pressure endpoints.
- [ ] Soak tests: cancellation storms, repeated long contexts, cache churn,
      mixed concurrency for hours with the guard active.
- [ ] Clean shutdown and restart tests; corrupt or incomplete snapshot tests.
- [ ] Runtime image below 8 GB with an SBOM and verified notices.
- [ ] Release checklist and signed tags.
- [ ] Optional separate local chat client; never an engine dependency.

## 7. Promotion gates

Run gates in proportion to the change. Anything touching model math, memory
or scheduling runs the full sequence.

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

Live verification (Phase 5 onward):

```bash
tools/verify-sharar.sh
```

Controlled matrix:

```bash
python3 bench/benchmark_text_server.py --label CANDIDATE_LABEL
python3 bench/compare_runs.py BASELINE_RUN_DIR CANDIDATE_RUN_DIR
```

Acceptance thresholds, unchanged from the reference project:

- No request failures.
- 4/4 quality canaries.
- Exact serial response hashes against the current baseline. From Phase 6 the
  baseline is Sharar's own frozen run; hash equality with the SGLang reference
  is not a goal (section 8).
- No cell loses more than 3% aggregate throughput.
- No cell regresses median TTFT by more than 15%.
- At least 16 GiB `MemAvailable` throughout.
- Concurrent hash drift is reported but diagnostic.

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
  flips 4.6% of its own predictions with margins above 5 nats. A second
  implementation of a W4A4 model cannot do better than that floor, and Sharar
  is at it. Details and every measurement are in `docs/parity.md`.
- Tolerance failures are triaged per block with `sharar.parity.layerprobe`
  (which layers) and `sharar.parity.truth` (distance from an fp32 dequantized
  truth for both sides), before touching the full model.
- Once Sharar is production, its own frozen run becomes the baseline and the
  reference dumps stay as the historical oracle.

## 9. Risks and open questions

- The ModelOpt NVFP4 scale layout expected by FlashInfer's SM120 CUTLASS path
  is reproduced exactly (bit-identical GEMM output against SGLang's wrapper,
  `tools/reference/ab_nvfp4.py`). Keep that A/B when the loader changes.
- The canary_json prompt's greedy next token differs from the reference's
  (code fence versus `{`) inside the measured noise floor. Phase 2 must run
  the decoded canary through the harness's JSON check; if it fails there, the
  fix is a quality decision (for example prompt or parser handling), not a
  parity bug.
- FP32 SSM state was required for validated quality in the reference. Any
  state dtype change is a correctness experiment.
- FP8 KV with FlashInfer decode on SM121 is what the reference runs; confirm
  the ragged prefill path also supports FP8 KV, otherwise prefill writes bf16
  and quantizes on append.
- Speculative verification over GDN layers needs the scratch-slot design to
  hold; if the recurrent kernels cannot run k + 1 steps against a scratch copy
  cheaply, the reference's replay-SSM kernels become necessary and add about
  1.5K vendored lines.
- Managed memory on GB10 may page-fault on first touch and lose bandwidth;
  never place hot buffers there without the Phase 8 microbenchmark.
- The reference's cold start JIT-compiles FlashInfer SM120 modules; Sharar's
  dev environment inherits that until Phase 9 builds artifacts ahead of time.
- Dev environment drift: if torch, FlashInfer or Triton versions differ from
  the reference image, parity failures become ambiguous. Pin them in Phase 0.

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

## 11. Immediate next task

Phase 4, the scheduler. The executor already has admit / prefill /
decode_step / release with the cache behind them; the scheduler owns the loop
around them: a request queue with the running and queued bounds, chunked
prefill with a per-step prefill token budget, mixed steps that run a prefill
chunk alongside the running decode batch so decode never stalls, cancellation,
retraction on memory pressure, and metrics. Start with `engine/request.py`
(lifecycle states) and `engine/scheduler.py`, then the deterministic CPU
simulation for concurrency 1 to 4, queue overflow and cancellation, then the
mixed extend/decode batching in the model path (varlen prefill with per-row
state slots, which the GDN kernels already support through `cu_seqlens`).
