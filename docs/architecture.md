# Engine architecture

One GB10 (or two, tensor-parallel), two pinned targets (Qwen3.8-27B and Qwen3.8-Flash-Next), one target per
process. The contracts between the parts are in [interfaces](interfaces.md).

## Layout

```
api/        HTTP: OpenAI chat, Anthropic Messages, /generate; chat templates, detokenizer, validation,
            tool-call and reasoning parsers, the refusal guard
engine/     loop (engine thread, request tickets) and scheduler (admission, chunked prefill, decode batches)
            executor (pools, prefill/decode steps, attention plans, graphs, warmup, cold tier setup)
            spec (draft -> verify -> accept), accept/rejection (acceptance rules), depth (draft depth per batch)
            graphs, spec_graphs, accept_graphs, graph_set (CUDA graphs); grammar (constrained output)
            config (EngineConfig), contracts, traces
cache/      KV pool, state pool, radix prefix cache, request state, cold tier, NVMe session store, ledger,
            speculation memory budget
model/      Qwen38 (the 27B) and its layers; flashnext/ (the second target); mtp.py (27B MTP head);
            parallel_draft.py (DFlash2/DSpark body); draft_linear.py
proposers/  factory; DFlash2/DSpark (parallel*), MTP (mtp*), n-gram; draft heads (draft_head, draft_vocab)
kernels/    dispatch (ops.py), attention planner, 3-bit GEMM and MoE, FP8 GEMV, fused Triton kernels, the RDMA
            one-shot all-reduce, vendored fla and qsa kernels
weights/    safetensors loading, ModelOpt NVFP4/FP8 linears, the Linear interface, tensor-parallel sharding,
            3-bit calibration and checkpoints
train/      DFlash2 drafter trainer (data, loss, differentiable draft, promotion)
distributed.py  tensor-parallel rank group and collectives
target.py   Target value per checkpoint (geometry, optional family fields), registry, loader
```

## What holds it together

- **Ownership.** `StateManager` alone moves pages, slots and the committed prefix; the scheduler never touches
  tensors; proposers own their draft state ([interfaces](interfaces.md)).
- **Memory.** Pools are sized from the `Target` and the ledger. The radix cache, cold tier and session store
  are model-agnostic and carry Flash-Next's extra state through `StatePool.register_aux` and
  `KvPool.register_page_aux`.
- **One speculative round.** `SpecDecoder` drives every proposer: reserve, propose, verify (graph or eager),
  accept (exact, relaxed or sampled rejection), commit, feed back.
- **Configuration.** Every `SHARAR_*` setting is a field of one `EngineConfig` (`engine/config.py`), built at
  startup and reported by `/get_server_info`. The launcher's serving profiles (`tools/serving-profile.sh`) set
  the defaults ([operations](operations.md#serving-profiles)).
- **Model contract.** A model exposes `forward`, `logits`, `linears()`, `register_pages`, `register_state` and
  `forward_extras()`; side inputs (Flash-Next's PLE rows and sparse-attention bookkeeping) travel on
  `ForwardContext.extras`, so `engine/` holds no model-specific code.
- **Speculator parts.** A proposer declares its `Capabilities` (sampled, depth, graphs, device acceptance,
  reload, cold payload, ...); the engine reads the flags instead of probing methods. Draft heads
  (`proposers/draft_head.py`) and the depth policy (`engine/depth.py`) are shared.
- **Kernels.** One `Linear` interface over every weight format (`weights/linear.py`), one `AttentionPlanner`
  per KV geometry (`kernels/attention_plan.py`), one graph-set helper (`engine/graph_set.py`).
- **Tensor parallel.** Every rank runs the whole engine; rank 0 serves HTTP and broadcasts each scheduler
  command, so ranks step in lockstep ([tensor parallel](tensor-parallel.md)).

## Capability matrix

| Capability | 27B | Flash-Next |
|---|---|---|
| Served drafter | DFlash2, k=15 | MTP, k=4, 3-bit draft head |
| Other drafters | MTP (BF16, unfused), DSpark, n-gram | none (DFlash2 needs feature taps, #49) |
| Drafter training on traffic, hot reload | DFlash2 | none (#40, #51) |
| CUDA graphs: decode, verify, draft | yes | yes, dense and sparse-selecting variants |
| Device-resident acceptance | opt-in | on in `fast` |
| Exact / relaxed / sampled acceptance | yes | yes |
| Constrained output (JSON schema, forced tool calls) | yes | yes |
| Calibrated 3-bit weights | layers and draft (decode copies) | trunk (decode copies) and routed experts (the only copy) |
| Weight formats | NVFP4 MLPs, FP8 projections | FP8 trunk; 3-bit routed experts (NVFP4 in `exact` and on two Sparks) |
| Prefix cache, cold tier, NVMe sessions, 262K context | yes | yes (sparse attention past 2,048 tokens) |
| Tensor parallel on two Sparks | yes | yes |

## Repository boundaries

| Code | Where | Why |
|---|---|---|
| Engine, models, kernels, weights, cache, proposers | `sharar/` | Proposers share the verify graphs, state ownership, kernels and draft heads with the target. |
| Drafter training | `sharar/train/` | It imports the serving draft body so its forward matches serving exactly, and promotion hot-reloads into the running server. The engine never imports `train`. |
| HTTP evals and speed benchmarks | [sharar-eval](https://github.com/abdelfattah-lab/sharar-eval) | They import nothing from `sharar` and run against any server. |
| In-process profilers and the lossy gate's in-process layers | `profiling/`, `tools/gate/` | They reach into engine internals and version with them. |
| Target-model training (distillation, LoRA) | a separate repository when it starts | Its own dependencies; uses Sharar as the rollout server. |

## Performance rules

- Abstractions resolve at construction and capture time, never per call: a `Linear` picks its kernel per
  captured row count once, capabilities are read when the speculator is installed, model side inputs fill
  static buffers the graphs own. Inside a CUDA graph the Python layer costs nothing; host work between replays
  is what matters.
- Host time per round is a metric: report GPU idle gaps (`profiling/flashnext_round.py --gap-profile` on
  Flash-Next, the engine-loop round on the 27B) for changes to the round, and do not increase them.

## Non-goals

Serving two targets in one process, a general model zoo or plugin registry, and hardware other than the GB10.
