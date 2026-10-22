# Engine architecture

One GB10 (or two, tensor-parallel), two pinned targets (Qwen3.8-27B and Qwen3.8-Flash-Next), one target per
process. The contracts between the parts are in [interfaces](interfaces.md).

## Layout

```
api/        HTTP: OpenAI chat, Anthropic Messages, /generate; chat templates, detokenizer, validation,
            tool-call and reasoning parsers, the refusal guard
engine/     loop (engine thread, request tickets) and scheduler (admission, chunked prefill, decode batches)
            executor (pools, prefill/decode steps, attention plans, graphs, warmup, SSD tier setup)
            spec (draft -> verify -> accept), accept/rejection (acceptance rules), depth (draft depth per batch)
            graphs, spec_graphs, accept_graphs, graph_set (CUDA graphs); grammar (constrained output)
            config (EngineConfig), contracts, traces
cache/      KV pool, state pool, radix prefix cache (GPU and SSD), request state, SSD tier and store, ledger,
            speculation memory budget
model/      Qwen38 (the 27B) and its layers; flashnext/ (the second target); mtp.py (27B MTP head);
            parallel_draft.py (DFlash2 body); draft_linear.py
proposers/  factory; DFlash2 (parallel*), MTP (mtp*); draft heads (draft_head, draft_vocab)
kernels/    dispatch (ops.py), attention planner, 3-bit GEMM and MoE, FP8 GEMV, fused Triton kernels, the RDMA
            one-shot all-reduce, vendored fla and qsa kernels
weights/    safetensors loading, ModelOpt NVFP4/FP8 linears, the Linear interface, tensor-parallel sharding,
            3-bit calibration and checkpoints
distributed.py  tensor-parallel rank group and collectives
cli.py      the `sharar` command (serve, chat, launch, status, ...; `engine` is the server with explicit settings)
launch/     what `sharar serve` does before the engine: settings and profiles, preflight and downloads, the memory
            guard, --detach services, two Sparks (tp.py), the client commands
target.py   Target value per checkpoint (geometry, optional family fields), registry, loader
config.py   the 27B's constants and paths for kernels, tools and tests; check(), the preflight
hardware/   GB10 facts (compute capability, driver floor, memory budget)
tokenizer.py  the pinned snapshot's tokenizer (`tokenizers`, no transformers)
benchmark_workloads.py  the fixed benchmark prompts (code, design, reasoning) the profilers read
```

## What holds it together

- **Ownership.** `StateManager` alone moves pages, slots and the committed prefix; the scheduler never touches
  tensors; proposers own their draft state ([interfaces](interfaces.md)).
- **Memory.** The ledger sizes the pools from the `Target`. The radix cache and the SSD tier
  are model-agnostic and carry Flash-Next's extra state through `StatePool.register_aux` and
  `KvPool.register_page_aux`.
- **One speculative round.** `SpecDecoder` drives every proposer: reserve, propose, verify (graph or eager),
  accept (exact, relaxed or sampled rejection), commit, feed back.
- **Configuration.** Every `SHARAR_*` setting is a field of one `EngineConfig` (`engine/config.py`), built at
  startup and reported by `/get_server_info`. The serving profiles (`sharar/launch/settings.py`) set
  the defaults ([operations](operations.md#serving-profiles)).
- **Model contract.** A model exposes `forward`, `logits`, `linears()`, `register_pages`, `register_state` and
  `forward_extras()`; side inputs (Flash-Next's PLE rows and sparse-attention bookkeeping) travel on
  `ForwardContext.extras`, so `engine/` holds no model-specific code.
- **Speculator parts.** A proposer declares its `Capabilities` (sampled, depth, graphs, device acceptance,
  checkpoint cache, stored payload, ...); the engine reads the flags instead of probing methods. Draft heads
  (`proposers/draft_head.py`) and the depth policy (`engine/depth.py`) are shared.
- **Kernels.** One `Linear` interface over every weight format (`weights/linear.py`), one `AttentionPlanner`
  per KV geometry (`kernels/attention_plan.py`), one graph-set helper (`engine/graph_set.py`).
- **Tensor parallel.** Every rank runs the whole engine; rank 0 serves HTTP and broadcasts each scheduler
  command, so ranks step in lockstep ([tensor parallel](tensor-parallel.md)).

## Capability matrix

| Capability | 27B | Flash-Next |
|---|---|---|
| Served drafter | DFlash2, k=15 | MTP, k=4 (in `fast` up to 8, chosen per request), 2-bit draft head |
| Other drafters | MTP (BF16, unfused) | DFlash2, k=9 (`exact` 7, `throughput` 4), FP8, on one Spark (`SHARAR_PROPOSER=dflash2`) |
| CUDA graphs: decode, verify, draft | yes | yes, dense and sparse-selecting variants |
| Device-resident acceptance | opt-in | on in `fast` and `throughput` |
| Exact / relaxed / sampled acceptance | yes | yes |
| Constrained output (JSON schema, forced tool calls) | yes | yes |
| Calibrated 3-bit weights | layers and draft (decode copies) | trunk (decode copies) and routed experts (the only copy) |
| Weight formats | NVFP4 MLPs, FP8 projections | FP8 trunk; 3-bit routed experts (NVFP4 in `exact`; split by expert on two Sparks) |
| Prefix cache, SSD tier, 262K context | yes | yes (sparse attention past 2,048 tokens) |
| Tensor parallel on two Sparks | yes | yes |

## Repository boundaries

| Code | Where | Why |
|---|---|---|
| Engine, models, kernels, weights, cache, proposers | `sharar/` | Proposers share the verify graphs, state ownership, kernels and draft heads with the target. |
| HTTP evals and speed benchmarks | [sharar-eval](https://github.com/abdelfattah-lab/sharar-eval) | They import nothing from `sharar` and run against any server. |
| In-process profilers and the lossy gate's in-process layers | `profiling/`, `tools/gate/` | They reach into engine internals and version with them. |
| Target-model and drafter training (distillation, LoRA, drafter adaptation) | a separate repository when it starts | Its own dependencies; uses Sharar as the rollout server and `--draft DIR` to serve a drafter. Drafter training on recorded traffic left this repository at the tag `archive/draft-training`. |

## Performance rules

- Abstractions resolve at construction and capture time, never per call: a `Linear` picks its kernel per
  captured row count once, capabilities are read when the speculator is installed, model side inputs fill
  static buffers the graphs own. Inside a CUDA graph the Python layer costs nothing; host work between replays
  is what matters.
- Host time per round is a metric: report GPU idle gaps (`profiling/flashnext_round.py --gap-profile` on
  Flash-Next, the engine-loop round on the 27B) for changes to the round, and do not increase them.

## Non-goals

Serving two targets in one process, a general model zoo or plugin registry, and hardware other than the GB10.
