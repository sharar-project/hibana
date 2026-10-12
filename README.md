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
uv run sharar serve 27b --tp 2  # or across two Sparks linked by their ConnectX ports (one Spark unless asked)
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
| `sharar serve 27b --exact` | The model's own outputs, for benchmarks (the default `fast` profile is faster and slightly lossy; also `--throughput`) |
| `sharar serve 27b --tp 2` | Tensor-parallel on two Sparks linked by their ConnectX ports ([tensor parallel](docs/tensor-parallel.md)) |
| `sharar status` | What is serving, with which settings |
| `sharar chat`, `sharar launch claude\|opencode\|aider\|show` | Use the serving model |
| `sharar train` | Train the 27B's drafter on your recorded traffic (Ctrl-C finishes and promotes; [draft adaptation](docs/draft-adaptation.md)) |

`sharar <command> --help` lists the options. Settings (cache and results locations, profiles, ports) are flags, then
`SHARAR_*` environment variables, then an optional `config/gb10.env` ([example](config/gb10.env.example),
[operations](docs/operations.md)).

## Develop

```bash
uv run pytest tests/unit                      # CPU tests
SHARAR_RUN_GPU=1 uv run pytest tests/gpu      # GPU suite (the GPU to itself)
```

The engine is `sharar/` ([architecture](docs/architecture.md)); in-process profilers are in `profiling/`, checks and
gates in `tools/`, and every doc is listed in [docs/](docs/README.md). Quality and speed benchmarks live in
[sharar-eval](https://github.com/abdelfattah-lab/sharar-eval) ([evaluation](docs/evaluation.md)). Read [CONTRIBUTING.md](CONTRIBUTING.md) before
changing the engine. Work in progress is tracked in [issues](https://github.com/abdelfattah-lab/sharar/issues).
Apache 2.0 ([license](LICENSE), [third-party notices](THIRD_PARTY_NOTICES.md)).
