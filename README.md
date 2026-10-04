# Sharar

Sharar is a local inference engine for one NVIDIA GB10 (a DGX Spark). Currently, it serves
**Qwen3.8-27B** and **Qwen3.8-Flash-Next** (125B MoE) through OpenAI- and
Anthropic-compatible APIs, with speculative decoding, prefix caching, 262K
context and an NVMe session store, and it trains its speculative drafter on your
own traffic so it gets faster at the work you do.

## Speed

Batch Size 1
([h2h benchmark](https://github.com/abdelfattah-lab/sharar/issues/26)):

| tok/s | Sharar | SGLang (fastest public setup) |
|---|---:|---:|
| Qwen3.8-27B, greedy | **92.5** | 44.8 |
| Qwen3.8-27B, temperature 0.7 | **74.5** | 43.7 |
| Qwen3.8-Flash-Next, greedy | **72.9** | 41.8 |
| Qwen3.8-Flash-Next, temperature 0.7 | **67.9** | 40.5 |

## Setup

Copy [config/gb10.env.example](config/gb10.env.example) to `config/gb10.env`,
set the model cache and results paths, then:

```bash
./sharar-ctl serve                                        # Qwen3.8-27B on 127.0.0.1:30001
SHARAR_CONFIG_FILE=config/flashnext.env ./sharar-ctl serve   # Flash-Next on 127.0.0.1:30004
./sharar-ctl chat
```

The first serve pulls the runtime image `ghcr.io/abdelfattah-lab/sharar:latest`, which the nightly GB10 check
publishes from main after its GPU suite and serve checks pass; `./sharar-ctl pull` updates it. While the
repository is private the image is too: log in once with a GitHub token that can read packages
(`gh auth refresh -s read:packages`, then `gh auth token | docker login ghcr.io -u <github user> --password-stdin`).
To serve your own build, run `docker/build.sh` and set `SHARAR_IMAGE=sharar:qwen38-27b`.

The default `fast` profile downloads its calibrated 3-bit checkpoint on first serve
([calibration](docs/w3-calibration.md)); the preflight warns when it serves without one.

`sharar-ctl` needs only Python 3 on the host. [Operations](docs/operations.md)
covers prerequisites, the launcher settings and the memory guard.

| Command | What it does |
|---|---|
| `./sharar-ctl pull` | Fetch the newest published image |
| `./sharar-ctl serve [--exact]` | Start serving (fastest settings by default; `--exact` for the model's own outputs, e.g. benchmarks) |
| `./sharar-ctl chat` | Terminal chat against the running server |
| `./sharar-ctl status` / `stop` / `logs` | Server state, stop, container logs |

Any OpenAI-compatible client works against `http://127.0.0.1:30001/v1`, and
Anthropic Messages clients against `http://127.0.0.1:30001`
([API guide](docs/api.md)).

## Coding agents

```bash
tools/claude-sharar.sh                      # Claude Code on Sharar (SHARAR_URL=http://127.0.0.1:30004 for Flash-Next)
tools/claude-sharar.sh -p "fix the failing test and run the suite"
```

The script points Claude Code at the server with its own config directory
(`~/.claude-sharar`), so your Anthropic login and settings are untouched.
OpenAI-compatible harnesses (OpenCode, Cline, Aider, Continue, Zed) use base URL
`http://127.0.0.1:30001/v1` with any API key and model name. After the first
request of a session, later turns start in under a second.
For **Pi**, see the [Pi configuration](docs/api.md#pi).

## Train the drafter on your traffic

Requests are recorded locally while you use the server (nothing leaves the
machine; `"trace": false` opts a request out).

```bash
./sharar-ctl train          # stop serving, train on the recorded traffic
./sharar-ctl train stop     # evaluate on your newest traffic, promote if better, serve again
./sharar-ctl draft list     # runs, their held-out gains, the draft in use
```

Serving follows the promoted draft and swaps it in without a restart. On a
repository-coding workload, four serve-then-train rounds raised serving
throughput by 12.4% ([draft adaptation](docs/draft-adaptation.md)).

## Develop

```bash
bash tools/check.sh setup-cpu     # CPU tests, no GPU or model needed
tools/gpu-tests.sh                # GPU suite in the runtime image (exclusive GPU)
python3 tools/check_evidence.py   # documentation links
```

| Path | Purpose |
|---|---|
| `sharar/` | Engine (`engine/`), caches (`cache/`), models (`model/`, `model/flashnext/`), kernels, weights, proposers, trainer, API |
| `sharar-ctl`, `run-sharar.sh` | Host command and the container launcher |
| `profiling/`, `tests/`, `tools/`, `docker/` | In-process profilers, tests, checks and gates, image build |
| `docs/` | [Architecture](docs/architecture.md), [optimizations](docs/optimizations.md), [performance](docs/performance.md), [Flash-Next](docs/flashnext.md), [roadmap](roadmap.md) |

Quality and speed benchmarks live in [sharar-eval](https://github.com/abdelfattah-lab/sharar-eval) and
measure any running server over HTTP. `tools/evaluate.py --target flashnext --profile fast --suite quick --speed`
serves this build and runs them ([evaluation](docs/evaluation.md); [profiling](profiling/README.md) for in-process tools). Read
[CONTRIBUTING.md](CONTRIBUTING.md) before changing the engine. Apache 2.0
([license](LICENSE), [third-party notices](THIRD_PARTY_NOTICES.md)).
