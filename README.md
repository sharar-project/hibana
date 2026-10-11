# Sharar

Sharar is a local inference engine for the NVIDIA GB10 (DGX Spark). It serves **Qwen3.8-27B** and
**Qwen3.8-Flash-Next** (125B MoE) through OpenAI- and Anthropic-compatible APIs, with speculative decoding, prefix
caching, a 262K context and an NVMe session store, on one Spark or two.

## Quick start

Install [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`), then:

```bash
git clone https://github.com/abdelfattah-lab/sharar && cd sharar
uv sync                       # the Python environment: pinned CUDA 13 PyTorch, FlashInfer (a few GB)
uv run sharar serve 27b       # Qwen3.8-27B on http://127.0.0.1:30001 (Ctrl-C stops it)
```

In another terminal:

```bash
uv run sharar chat            # chat in the terminal
uv run sharar launch claude   # Claude Code on the model (also opencode, aider; `show` for other tools)
```

(`source .venv/bin/activate` lets you drop the `uv run`.) The first serve downloads the model (26 GB for the 27B,
135 GB for Flash-Next) and its calibrated 3-bit checkpoint. Any OpenAI-compatible client works against
`http://127.0.0.1:30001/v1`, and Anthropic clients against `http://127.0.0.1:30001` ([API](docs/api.md)).

## Commands

| | |
|---|---|
| `sharar serve 27b` / `sharar serve flashnext` | Serve a model in the foreground (Flash-Next on port 30004) |
| `sharar serve 27b --detach` | The same as a background service; `sharar stop`, `sharar logs -f` |
| `sharar serve 27b --exact` | The model's own outputs, for benchmarks (the default `fast` profile is faster and slightly lossy; also `--balanced`, `--throughput`) |
| `sharar serve 27b --tp 2` | Tensor-parallel on two Sparks linked by their ConnectX ports ([tensor parallel](docs/tensor-parallel.md)) |
| `sharar status` | What is serving, with which settings |
| `sharar chat`, `sharar launch claude\|opencode\|aider\|show` | Use the serving model |
| `sharar train` | Train the 27B's drafter on your recorded traffic (Ctrl-C finishes and promotes; [draft adaptation](docs/draft-adaptation.md)) |

`sharar <command> --help` lists the options. Settings (cache and results locations, profiles, ports) are flags, then
`SHARAR_*` environment variables, then an optional `config/gb10.env` ([example](config/gb10.env.example),
[operations](docs/operations.md)).

## Which model

| | Qwen3.8-27B | Qwen3.8-Flash-Next |
|---|---|---|
| Download / weights in memory | 26 GB / 28 GiB | 135 GB / 53 GiB |
| MMLU-Pro / GPQA Diamond / IFEval | 75.4 / 59.6 / 78.7 | 81.6 / 73.7 / 84.8 |
| Decode at 128K / 256K context | 54 / 46 tok/s | 66 / 57 tok/s |
| A five-turn Claude Code session (30/30 tests both) | 31 min | 19 min |

Flash-Next is the stronger model and the better agent, and it slows less with context. The 27B is smaller and loads
faster (measured 2026-10-05/06, #11; Flash-Next has since moved to 3-bit routed experts, about 20% faster at the same
quality, #73).

## Speed

One request at a time, median decode rate over 22 prompts (chat, code, math, multilingual, prose, structured),
thinking off ([h2h benchmark](https://github.com/abdelfattah-lab/sharar/issues/26)):

| tok/s | Sharar | SGLang (fastest public setup) |
|---|---:|---:|
| Qwen3.8-27B, greedy | **92.3** | 44.8 |
| Qwen3.8-27B, temperature 0.7 | **66.5** | 43.7 |
| Qwen3.8-Flash-Next, greedy | **85.4** | 41.8 |
| Qwen3.8-Flash-Next, temperature 0.7 | **74.4** | 40.5 |

Sharar 2026-10-05/06, `fast` profile; SGLang 2026-09-24, same GB10. Two Sparks: 27B 77 → 119 tok/s, Flash-Next
74 → 99. Quality and speed benchmarks live in [sharar-eval](https://github.com/abdelfattah-lab/sharar-eval)
([evaluation](docs/evaluation.md)).

## Develop

```bash
uv run pytest tests/unit                      # CPU tests
SHARAR_RUN_GPU=1 uv run pytest tests/gpu      # GPU suite (the GPU to itself)
```

The engine is `sharar/` ([architecture](docs/architecture.md)); in-process profilers are in `profiling/`, checks and
gates in `tools/`, and every doc is listed in [docs/](docs/README.md). Read [CONTRIBUTING.md](CONTRIBUTING.md) before
changing the engine. Work in progress is tracked in [issues](https://github.com/abdelfattah-lab/sharar/issues).
Apache 2.0 ([license](LICENSE), [third-party notices](THIRD_PARTY_NOTICES.md)).
