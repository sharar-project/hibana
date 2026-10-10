# Engine architecture

The engine as it is, what each target can do with it, and the shape it should
move to so that every capability works on both targets from shared parts
(issue #24). Scope: one GB10 (DGX Spark), the two pinned targets
(Qwen3.8-27B and Qwen3.8-Flash-Next), one target per process. Other hardware
is out of scope until it exists here. Written 2026-09-25 against main
`e3d7746`; the ownership rules this page relies on are in
[interfaces](interfaces.md) and [ownership invariants](ownership.md).

## Current shape

```
api/        HTTP (OpenAI chat, Anthropic messages, generate), templates, detokenizer, validation
engine/     loop + scheduler (admission, chunked prefill, decode batches)
            executor (pools, prefill/decode steps, attention plans, graphs, warmup, cold tier setup)
            spec (draft -> verify -> accept round), rejection/accept (acceptance rules)
            graphs, spec_graphs, accept_graphs (CUDA graphs), traces, contracts
memory/     KV pool, state pool, radix prefix cache, request state, cold tier, NVMe sessions, ledger, spec budget
model/      Qwen38 (27B) + its layers; flashnext/ (second model package); mtp.py (27B MTP head);
            parallel_draft.py (DFlash2/DSpark body); draft_linear.py
proposers/  factory, DFlash2/DSpark (parallel*), MTP (mtp*), n-gram, draft_vocab (draft-only heads)
kernels/    FlashInfer/cuDNN/CUTLASS dispatch (ops.py), W3, FP8 GEMV, fused Triton kernels, vendored fla/qsa
weights/    safetensors loading, ModelOpt NVFP4/FP8 linears, W3 calibration and checkpoints
train/      DFlash2 drafter trainer (data, loss, differentiable draft, promotion)
target.py   Target value per checkpoint (geometry, optional family fields), registry, loader switch
```

About 15,000 lines of Python in `sharar/` (engine 3.8K, kernels 3.5K, memory
1.9K, Flash-Next package 1.6K, api 1.2K, model 1.2K, train 1.1K, parity 1.1K,
proposers 1.0K).

What holds up and should not change:

- **Ownership.** `StateManager` alone moves pages, slots and the committed
  prefix; the scheduler never touches tensors; proposers own their draft state.
  Every recent feature (sessions on NVMe, device acceptance, sampled
  speculation) landed without breaking these rules.
- **Memory.** Pools sized from the `Target` and the ledger; the radix cache,
  cold tier and session store are model-agnostic and already carry Flash-Next's
  extra state through `StatePool.register_aux` and `KvPool.register_page_aux`.
- **The speculative round.** One driver (`SpecDecoder`) for every proposer:
  reserve, propose, verify (graph or eager), accept (exact, relaxed, sampled
  rejection), commit, feed back. Acceptance rules are already shared.
- **The API layer and the scheduler.** Target-agnostic.

## Capability matrix

| Capability | 27B | Flash-Next |
|---|---|---|
| DFlash2 parallel drafter | served default, k=15, trained drafter | missing (see pain point 3) |
| MTP drafter | works; BF16, unfused, untuned, untrained (`model/mtp.py`) | served default, k=4 (`model/flashnext/mtp.py`, a second head class) |
| Draft-only head (3-bit + rescore, reduced vocabulary) | no | yes, MTP only (`proposers/draft_vocab.py`) |
| Drafter training on traffic, promotion, hot reload | DFlash2 only | none |
| CUDA graphs: decode, verify, draft | yes | yes, dense and selecting variants |
| Device-resident acceptance | DFlash2, opt-in | MTP, on in `fast` (#36, 2026-09-28) |
| Exact / relaxed / sampled acceptance | shared | shared |
| Adaptive draft depth (#43) | no | no |
| 3-bit decode weights | GPTQ-calibrated layers and draft, default on | GPTQ-calibrated trunk copies, on in `fast` (#6, 2026-09-27) |
| Weight formats | NVFP4 MLPs, FP8 projections, W3 copies (`weights/modelopt.py`, `kernels/ops.py`) | FP8-at-load trunk, NVFP4 experts (`model/flashnext/linear.py`, `moe.py`) |
| Prefix cache, cold tier, NVMe sessions, 262K context | yes | yes |
| Shape-stable numerics | yes | yes |

## Pain points

Measured by what the second target and the last features needed.

**1. Model specifics live in the engine core.** The executor, the three graph
modules and `SpecDecoder` know about the PLE table and the sparse-attention
indexer: 18 `_ple` hooks in `engine/` and `proposers/`, `qsa_rows` /
`within_budget` / `fill_qsa` / `sparse_ctx` in the executor and graph code,
and 12 Flash-Next fields on `ForwardContext` (`ple_history`, `ple_rows`, eight
`qsa_*`). Static graph buffers for the PLE rows are allocated by
`DecodeGraphs._make_static` and `VerifyGraphs._static` with a hardcoded row
width (160). A third model with its own side inputs would add a third set.

**2. Attention planning is spread over 17 call sites.** FlashInfer `plan()`
with the target's head shapes is called from the executor (four places),
`graphs.py`, `spec.py`, `spec_graphs.py`, `ops.py`, the 27B attention module
(two), both MTP heads (two each), the MTP graphs, the parallel proposer and a
parity tool. Each repeats the host-pointer rule, the KV dtype and the scale; the re-planning race fixed on 2026-09-23 (`plan_guard`)
had to be threaded through all of them.

**3. Speculators are built per model, not composed.**
- DFlash2 hardcodes the 27B: `hidden_size 5120` is asserted in
  `ParallelDraftModel` and written as a literal in `proposers/parallel.py`,
  `parallel_graphs.py`, `parallel_feedback.py`, `spec_budget.py` and
  `train/draft.py`; Flash-Next's forward raises on any layer-feature request,
  and its stream is the hyper-connected `[T, 4 x 2560]`, not `[T, H]`.
- Two MTP head classes with the same surface and no shared parts; only the
  Flash-Next one has fused kernels, FP8 experts and a draft-only head.
- Capabilities are discovered with `hasattr`/`getattr`: `bind_checkpoint_cache`,
  `auxiliary_codec`, `projected_rows`, `feature_cache`, `accepted_prepare` /
  `feedback_device` / `accepted_publish` (device acceptance), `draft_argmax`,
  `set_draft_head`, `model.load_weights` (reload). Whether a proposer supports
  device acceptance or reload is not visible in its requirements.
- The draft head (which lm_head rows, which precision) is a Flash-Next MTP
  detail; DFlash2 calls the target's full head (`target.logits`) for its
  candidates.
- Depth is one number per server; per-request or per-round depth has no home.

**4. Two weight stacks.** The 27B uses `Nvfp4Linear` / `Fp8Linear` with
`ops.nvfp4_linear` / `ops.fp8_linear` dispatch and W3 attached by
`w3.attach_model` over `w3_checkpoint.model_linears` (Qwen38 layer names).
Flash-Next uses `TrunkLinear` (FP8 at load, its own W3 branch) and
`ReducedHead`. GPTQ calibration (`weights/w3_calibration.py`) only knows the
27B's linears, which is why calibrated W3 for Flash-Next (#6) would today mean
a second calibration pipeline.

**5. Training is DFlash2-shaped.** `sharar/train/` hardcodes the DFlash2 pin,
the block objective, the 27B target (`trainer.py` sets `QWEN38`) and the
feature rings. Traces record `projected_rows` only when the proposer has that
method. Nothing can train an MTP head (#40) or a Flash-Next DFlash2 (#7).

**6. Configuration is import-time globals.** At least 33 `SHARAR_*` variables are
read by 20 modules, many at import (`w3.ENABLED`, `w3.DRAFT`, `accept_graphs.ENABLED`,
`KEEP_BF16`, the MTP draft head in the factory), plus the launcher's own. A
test or benchmark cannot hold two configurations in one process, the server
cannot report its effective profile, and the `--fast` / `--exact` profiles
(#27) have nothing to attach to.

**7. Graph capture is written four times.** Decode, verify, accept, the
parallel draft and MTP draft each implement static buffers, padding to
captured sizes, plan-then-replay and the dense/selecting split separately
(seven `CUDAGraph()` sites).

## Proposed shape

Five seams. Each is a small interface with both targets as its first two
implementations; none is a plugin registry.

### Model contract

What the engine may assume about a target model, replacing the `hasattr`
checks and the Flash-Next fields:

```python
class TargetModel(Protocol):
    target: Target
    def forward(self, ids, ctx: ForwardContext) -> ForwardResult: ...
    def logits(self, stream) -> Tensor: ...
    def stream_width(self) -> int: ...                    # 5120, or 4 x 2560 for the hyper stream
    def features(self, ctx, layers) -> dict[int, Tensor]: ...  # layer taps, defined for both models
    def state_spec(self) -> StateSpec: ...                # per-slot aux and per-page aux rows (PLE window, index keys)
    def step_extras(self) -> StepExtras | None: ...       # side inputs a forward needs (PLE rows, QSA rows)
    def linears(self) -> Iterable[tuple[str, Linear]]: ...    # for quantization and calibration
```

`StepExtras` owns its static buffers (shape per row count), fills them for a
batch (`prepare(requests, tokens, positions, pages)`) and hands the forward an
opaque object on `ForwardContext.extras`. The executor, graphs and
`SpecDecoder` call `prepare` and copy into the static buffers without knowing
what is inside; `ple_*` and `qsa_*` move out of `ForwardContext` and out of
`engine/`. Dense versus selecting graphs becomes a variant the extras object
declares (`variants() -> ("dense", "selecting")`, `variant_for(seqs)`).

### Attention planner

One object per executor that owns the FlashInfer wrappers, their workspaces
and `plan_guard`, with three calls (`plan_prefill`, `plan_decode`,
`plan_verify`) taking pages and lengths; graph code asks it for a
graph-bound wrapper. The target's head shapes and the KV dtype are read once.
Draft heads with their own KV (both MTP heads) use the same planner type with
their own geometry.

### One linear layer

```python
class Linear:          # formats: bf16, fp8 (static or dynamic activation scale), nvfp4, w3 (nearest or calibrated)
    def __call__(self, x): ...     # dispatch by row count: w3 copy <= 64 rows, fp8 GEMV <= 4, GEMM otherwise
    def attach(self, fmt, weights): ...   # add a decode copy beside the served weights
```

`Nvfp4Linear`, `Fp8Linear`, `TrunkLinear`, `DraftFp8Linear` become formats of
it; the dispatch rules in `ops.py` and `TrunkLinear.__call__` become one
function. `TargetModel.linears()` gives quantization and GPTQ calibration a
model-independent list, so calibrated W3 for Flash-Next reuses
`weights/w3_calibration.py`. The draft-only head is a `Linear` over selected
rows with a format and a rescore count, usable by any speculator.

### Speculator parts

A speculator is assembled from parts, with its capabilities declared:

| Part | Implementations |
|---|---|
| Draft body | DFlash2/DSpark block (parallel), MTP layer (sequential), n-gram (host) |
| Draft head | target head, reduced rows, 3-bit + rescore (`Linear` over rows) |
| Feature source | target layer taps or final stream, width from `TargetModel.stream_width()` |
| Depth policy | fixed k (today), confidence-based (#43), per batch size |
| Acceptance policy | exact, relaxed greedy, sampled rejection, typical (already shared; becomes a value per request) |
| Graphs | draft graphs per body; verify and accept graphs shared |

```python
@dataclass(frozen=True)
class Capabilities:
    sampled: bool; max_depth: int; graphs: bool
    device_accept: bool; reloadable: bool; trainable: bool; cold_payload: bool
```

`ProposerRequirements` gains `capabilities`; the executor, `AcceptGraphs.supported`,
the cold tier and `reload_draft` read it instead of probing attributes. The
Proposer protocol keeps its lifecycle methods; the optional ones become a
second protocol (`DeviceFeedback`, `Reloadable`, `ColdPayload`) that a
capability promises. One `MtpLayer` body serves both targets (the 27B's head
gets the fused kernels, FP8 experts and draft head for free); DFlash2 reads
its width and taps from the target, so a Flash-Next DFlash2 is a checkpoint
and a feature source, not a fork.

### Trainable speculator

```python
class Trainable(Protocol):
    def trace_spec(self) -> TraceSpec: ...          # which features and target logits to record per position
    def train_forward(self, batch) -> Tensor: ...   # differentiable body over recorded features
    def parameters(self, subset: str) -> list: ...
    def save(self, dir) / load(self, dir) / reload(self, dir): ...
```

`sharar/train/` keeps the loop, data split, objective terms, evaluation,
promotion and the on/off launcher; the DFlash2 specifics move behind this
interface and MTP implements it second. Traces record what `trace_spec`
names instead of `projected_rows` when present.

### Configuration and profiles

One `EngineConfig` built at startup from the CLI, the profile (#27) and the
environment, passed to the model, the executor and the proposer factory;
modules stop reading the environment at import. `/get_server_info` reports
it. Profiles (`--exact`, default, `--fast`) are named bundles of fields in
one table with each field's measured speed and quality delta per target.

### Graph sets

One helper for static buffers per captured size, padding, plan-then-replay and
variants, used by decode, verify, accept and both draft bodies. Each graph
keeps its own body; the helper removes the duplicated bookkeeping.

## Repository boundaries

`sharar/cache/` is the engine's memory manager, not an application feature:
the paged FP8 KV pool, the GDN state pool, the radix prefix cache, the
`StateManager` ownership boundary, the unified-memory ledger, the cold tier,
the NVMe session store and the speculation budget. It stays in the engine
and is renamed `sharar/cache/` in the cleanup (#23).

| Code | Where | Why |
|---|---|---|
| Engine, models, kernels, weights, memory, **proposers** | this repo, `sharar/` | Proposers sit on the hot path: they share the verify graphs, state ownership, kernels and the draft head with the target. Every speed change touches both sides; a cross-repo interface would have to be versioned per change. |
| **Drafter training** (`sharar/train/`) | this repo, optional extra (`pip install sharar[train]`) | Training imports the serving draft body (`model/parallel_draft.py`, `proposers/parallel.py`) so the differentiable forward matches serving exactly, and promotion hot-reloads into the running server. One-way dependency: `train` imports the engine, the engine never imports `train`. |
| **Target-model training** (OPD #34, LoRA, RL) | a separate repo when it starts | Large, slow-moving, its own dependencies; uses Sharar as the rollout server and loads the target through the model contract. |
| **HTTP-level evals and benchmarks** (`evals.suite` with the accuracy, hard, RULER v2 and Terminal-Bench tasks; `evals.speed` with the AIPerf, ShareGPT, cross-engine, local-use and agent-shaped speed benchmarks; `h2h`, `tts_eval`, `identity`; pinned dataset hashes) | `abdelfattah-lab/sharar-eval` (split 2026-09-26; speed benchmarks moved 2026-10-02) | They import nothing from `sharar` and already run against SGLang, vLLM and Atlas; frozen inputs and results are pinned by their own versions. Every result records both SHAs. |
| **In-process profilers and gate layers** (`profiling/flashnext_round.py`, `flashnext_quant.py`, `w3_bench.py`; `tools/gate/lossy_gate.py`) | this repo, `profiling/` and `tools/gate/` | They reach into engine internals and must version with them. |

## Performance contract

The refactor must not cost speed and should make the next gains cheaper.

**Rules.**
- Abstractions resolve at construction and capture time, never per call: a
  `Linear` picks its kernel for each captured row count once; capabilities
  are read when the speculator is installed; the model's `StepExtras` fills
  static buffers the graphs already own. Inside a CUDA graph the Python layer
  costs nothing; what matters is host work between replays.
- Host time per round is a first-class metric: every refactor PR reports the
  GPU idle gaps (`profiling/flashnext_round.py --gap-profile` on Flash-Next, the
  engine-loop round on the 27B) and may not increase them.
- Gate per PR, both targets: greedy outputs bit-identical, round cost within
  the family gate's noise (it resolves 0.05%), h2h single-stream and
  four-stream tok/s within run noise, load time and memory unchanged.

**What each step buys** (measured opportunities it unblocks):

| Step | Opportunity it makes cheap | Measured size |
|---|---|---|
| 1 `EngineConfig` | per-request profiles; `--fast` bundles | +10-22% from relaxed acceptance and deeper drafts already measured, today behind env vars |
| 2 speculator parts | adaptive depth (#43); draft-only head on the 27B | k=5 gave code +16% / prose -8% on Flash-Next: adaptive keeps the gain; 27B head ~1% |
| 3 planner + graph sets | plans prepared before the draft sync; one graph per round (draft + verify + accept), the step toward #25 | Flash-Next GPU idle 6.5 ms of 65 ms per round (#36) |
| 4 model contract | PLE row reads and QSA rows prepared off the critical path | ~2 ms of the 3.5 ms draft-to-verify gap |
| 5 one `Linear` | calibrated low-bit trunk for Flash-Next (#6) through the existing GPTQ pipeline; the small-M MoE kernel (#37) as a format | ~9 ms per forward (#25); ~4-6 ms per step |
| 6-8 features, MTP body, `Trainable` | Flash-Next DFlash2 (#7); MTP trained on traffic (#40); device acceptance for MTP (#18) | 27B drafter training gave +12.9%; device acceptance ~2-3 ms per round |

## Migration order

Small pull requests, each keeping both targets serving. Gate for every
refactor: greedy outputs bit-identical on both targets (exact profile, h2h
prompts and the family set), round cost within noise, CPU and GPU suites
green. Numerics-changing work is not mixed into these.

| Step | Change | Unblocks |
|---|---|---|
| 1 | `EngineConfig`: one object, no import-time environment reads; `/get_server_info` shows it (#44, done) | every later step; #27 |
| 2 | Speculator capabilities and optional protocols; shared draft head (`Linear` over rows) usable by DFlash2 and both MTP heads (#45, done: `Capabilities`, `proposers/draft_head.py`, `engine/depth.py`) | #43, draft head on the 27B |
| 3 | Attention planner and graph-set helper (#46, done: `kernels/attention_plan.py`, `engine/graph_set.py`) | steps 4 and 6; fewer plan races |
| 4 | Model contract: `StepExtras` and `state_spec`; PLE and QSA leave `engine/` (#47, done: `forward_extras()`, `ForwardContext.extras`, `model/flashnext/extras.py`) | a third model; Flash-Next DFlash2 |
| 5 | One `Linear` with formats; `TargetModel.linears()`; generic calibration (#48, done as one interface: `weights/linear.py`, `linears()` on both models, `w3.attach` over any linears; each format keeps its dispatch, the calibration driver stays per checkpoint) | calibrated W3 for Flash-Next (#6) |
| 6 | Feature taps on both models (`stream_width`, layer taps through the hyper stream); DFlash2 width from the target | Flash-Next DFlash2 (#7) |
| 7 | One `MtpLayer` body for both targets | 27B MTP at Flash-Next quality |
| 8 | `Trainable` interface; MTP trainer second | #40, #7 training |
| 9 | Profiles (`--exact`, `--balanced`, `--fast` = default) (#52: launcher-resolved `SHARAR_PROFILE`, reported in `EngineConfig.profile`) | #27 |

Then the capability gaps, in order of expected gain: adaptive depth (#43) on
both targets, MTP training on Flash-Next (#40), Flash-Next DFlash2 (screen the
public PixelML drafter, then train our own), calibrated W3 for Flash-Next
(#6), device acceptance for MTP (#18). The #23 cleanup (bench scripts, archive,
parked branches) runs beside the refactors.

## Non-goals

Serving two targets in one process, other hardware, a general model zoo or a
plugin registry, and any change to request ownership or the memory contracts.
