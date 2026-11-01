# Sharar

Sharar is a local inference engine for the NVIDIA GB10 (DGX Spark). It serves **Qwen3.8-27B**,
**Qwen3.8-Flash-Next** (125B MoE), **Nemotron 3.5 Super** (120B-A12B) and **Nemotron 3.5 Bolt** (74B-A9B) through
OpenAI- and Anthropic-compatible APIs, with speculative decoding, prefix caching (with an optional NVMe tier for
evicted prefixes) and a 262K context: the Qwen models on one Spark or two, the Nemotron models on one.

## Requirements

- One NVIDIA GB10 (SM 12.1, a DGX Spark; two linked by their ConnectX ports for
  [tensor parallel](docs/tensor-parallel.md)), `aarch64`, about 128 GB unified memory.
- NVIDIA driver 580 or newer and the CUDA 13 toolkit in `/usr/local/cuda` (DGX OS ships it; the GB10 kernels compile
  on first use).
- [uv](https://docs.astral.sh/uv/), which installs the tested Python 3.12.
- Disk for the model cache: about 26 GB for the 27B and 135 GB for Flash-Next, plus 10 GiB that the download check
  keeps spare ([operations](docs/operations.md#prerequisites)).

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
uv run sharar launch claude   # Claude Code on the model (also opencode, aider, codex; `show` for other tools)
```

(`source .venv/bin/activate` lets you drop the `uv run`.) The first serve downloads the model (26 GB for the 27B,
135 GB for Flash-Next) and its calibrated 3-bit checkpoint. Any OpenAI-compatible client works against
`http://127.0.0.1:30001/v1`, and Anthropic clients against `http://127.0.0.1:30001` ([API](docs/api.md)).

## Commands

| | |
|---|---|
| `sharar serve 27b` / `sharar serve flashnext` | Serve a model in the foreground (Flash-Next on port 30004) |
| `sharar serve nemotron` | Serve Nemotron 3.5 Super on port 30005, one Spark ([Nemotron](docs/nemotron.md)) |
| `sharar serve bolt` | Serve Nemotron 3.5 Bolt's packed text model on port 30006, one Spark ([Bolt](docs/bolt.md)) |
| `sharar serve 27b --detach` | The same as a background service; `sharar stop`, `sharar logs -f` |
| `sharar serve 27b --exact` | The model's own outputs, for benchmarks (the default `fast` profile is faster and slightly lossy; also `--throughput`) |
| `sharar serve 27b --tp 2` | Tensor-parallel on two Sparks linked by their ConnectX ports ([tensor parallel](docs/tensor-parallel.md)) |
| `sharar status` | What is serving, with which settings |
| `sharar cache` | What is cached on the GPU and on the SSD tier (`$SHARAR_RESULTS/ssd/<target>`, up to `SHARAR_SSD_GIB`) |
| `sharar chat`, `sharar launch claude\|opencode\|aider\|codex\|show` | Use the serving model |

`sharar <command> --help` lists the options. Settings (cache and results locations, profiles, ports) are flags, then
`SHARAR_*` environment variables, then an optional `config/gb10.env` ([example](config/gb10.env.example),
[operations](docs/operations.md)).

## Develop

```bash
uv run pytest tests/unit                      # CPU tests
SHARAR_RUN_GPU=1 uv run pytest tests/gpu      # GPU suite (the GPU to itself)
```

The engine is `sharar/`, in-process profilers are in `profiling/`, and checks and gates in `tools/`. Quality and
speed benchmarks live in [sharar-eval](https://github.com/abdelfattah-lab/sharar-eval). Read
[CONTRIBUTING.md](CONTRIBUTING.md) before changing the engine. Work in progress is tracked in
[issues](https://github.com/abdelfattah-lab/sharar/issues). Apache 2.0 ([license](LICENSE),
[third-party notices](THIRD_PARTY_NOTICES.md)).

## Documentation

Each page describes the code as it is; measurements and plans live in issues and pull requests. When a change alters
behavior, update its page in the same pull request (`python3 tools/check_links.py` after moving or deleting one).

| Page | Contents |
|---|---|
| [Operations](docs/operations.md) | Prerequisites, launcher settings, serving profiles, traces, the SSD tier |
| [API](docs/api.md) | Endpoints, request controls, tool calls, constrained output, Anthropic Messages |
| [Architecture](docs/architecture.md) | Package layout, the seams the targets share, capability matrix |
| [Cache design](docs/cache_design.md) | GPU pools, the radix prefix cache and the SSD tier |
| [Memory plan](docs/memory_plan.md) | How startup sizes the KV and state pools around the host floor |
| [Interfaces](docs/interfaces.md) | Request ownership, proposer lifecycle, device-resident acceptance, failure handling |
| [Flash-Next](docs/flashnext.md) | Qwen3.8-Flash-Next: model, package, conventions, sparse attention, its MTP and DFlash2 drafts |
| [Nemotron](docs/nemotron.md) | Nemotron 3.5 Super: the Mamba-2 and latent-MoE model as served, formats, its MTP draft |
| [Nemotron Bolt](docs/bolt.md) | Packed IQ2/Q8/NVFP4 text inference on the Nemotron Mamba runtime |
| [Tensor parallel](docs/tensor-parallel.md) | Two Sparks: setup, sharding, the RDMA all-reduce, limits |
| [3-bit calibration](docs/w3-calibration.md) | Fitting and serving the calibrated 3-bit copies |
| [Evaluation](docs/evaluation.md) | Measuring a build or a change, the lossy-change gate, nightly and weekly GB10 checks |
| [Profiling](profiling/README.md) | In-process profilers |
