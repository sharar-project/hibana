# Sharar

Sharar is a local inference engine for one NVIDIA GB10 (a DGX Spark). It
serves Qwen3.8-27B in NVFP4 through an OpenAI-compatible API with prefix
caching and DFlash2 speculative decoding, and it can train the speculative
drafter on your own traffic so it gets faster at the work you actually do.
Outputs are the target model's own: the drafter only changes speed. By default a
greedy request also accepts a drafted token the target ranks second within one
logit of its first choice (about +10% throughput, one token in 25); set
`accept_top_k: 1` on a request, or `SHARAR_ACCEPT_DEFAULT=off`, for the exact
greedy continuation.

## Setup

Copy [config/gb10.env.example](config/gb10.env.example) to `config/gb10.env`,
set the model cache and results paths, then build the runtime image:

```bash
docker/build.sh
./sharar-ctl serve
./sharar-ctl chat
```

`sharar-ctl` is a host-side command with no dependencies beyond Python 3. See
[operations](docs/operations.md) for prerequisites, the launcher environment
and the memory guard.

## Everyday use

| Command | What it does |
|---|---|
| `./sharar-ctl serve` | Serve on `127.0.0.1:30001`; records request traces unless `--no-traces` |
| `./sharar-ctl serve --exact` / `--balanced` / `--fast` | Serving profile; `--fast` (every measured speed option, lossy ones included) is the default. Costs: [serving profiles](docs/optimizations.md#serving-profiles) |
| `./sharar-ctl chat` | Terminal chat against the running server (`--think`, `/max N`, `/reset`) |
| `./sharar-ctl status` | Serving, training, the draft in use, traces on disk |
| `./sharar-ctl stop` | Stop serving |
| `./sharar-ctl logs [--train]` | Container logs |

Any OpenAI-compatible client works against `http://127.0.0.1:30001/v1`
(chat completions, completions, streaming, thinking, tool-call parsing), and
Anthropic Messages clients against `http://127.0.0.1:30001`; see the
[API guide](docs/api.md). Context is 262K tokens, at most four concurrent
requests.

## Use it from coding agents

Agent harnesses run on Sharar unchanged: tool calls, streaming, thinking and
long sessions. **Claude Code**, from the directory you want it to work in:

```bash
tools/claude-sharar.sh                      # interactive
tools/claude-sharar.sh -p "fix the failing test and run the suite"
```

The script points Claude Code at the server (`SHARAR_URL`, default
`http://127.0.0.1:30001`) under a model name Claude Code accepts, declares the
server's context length, turns off telemetry, and keeps its own config
directory (`~/.claude-sharar`) so your Anthropic login and settings are
untouched. Claude Code warns that the model is not in its catalog and
compacts sessions at 200K tokens; that is expected. **OpenAI-compatible
harnesses** (OpenCode, Cline, Aider, Continue, Zed): base URL
`http://127.0.0.1:30001/v1`, any API key and model name. From another machine,
forward the port first: `ssh -N -L 30001:127.0.0.1:30001 <user>@<spark>`.

The first request of a session prefills the harness's system prompt and tools
(about 16K tokens for Claude Code, 8 s); later turns reuse the cache and start
in under a second. Forced tool choice and JSON-schema output return 400 for
now. Agent traffic is recorded like any other (`"trace": false` opts out), so
`./sharar-ctl train` makes the drafter faster at your own agent work.

## Train the drafter on your traffic

While you use the server, each request is recorded locally: token IDs, every
verification round, and the context features the drafter saw. Nothing leaves
the machine, and a request can opt out with `"trace": false`.

```bash
./sharar-ctl train          # stops serving, trains on the recorded traffic until told to stop
./sharar-ctl train stop     # final held-out evaluation, promote if better, serving comes back
./sharar-ctl draft list     # every run, its held-out gain, and the draft in use
```

Serving follows the promoted checkpoint automatically; `./sharar-ctl serve
--pinned` returns to the publisher's draft, `draft use RUN` picks another;
both swap the draft inside a running server without a restart.
The held-out set is always your newest traffic. On a repository-coding
workload, four serve-then-train rounds over 1,000 requests raised acceptance
from 27.5% to 31.5% of proposals and serving throughput by 12.4%; a single
300-step session gave +12.9% on held-out prompts. Details, limits and the
design are in [draft adaptation](docs/draft-adaptation.md).

## Benchmark

With a server running and exclusive GPU use:

```bash
tools/benchmark.sh                  # code, 10K input / 1K output, batch one
tools/benchmark.sh --workload all   # code, design and reasoning
```

The launcher's defaults are the measured serving profile: FP8 DFlash2 at k=15,
relaxed acceptance, 3-bit decode copies of the layers and the draft. The
[3-bit quantization guide](docs/3bit-quantization-explained.md) explains the
format, calibration and prior work from the basics. The
[optimization inventory](docs/optimizations.md) lists what is on,
opt-in and parked. [Performance](docs/performance.md) links every measured claim to its
evidence; [bench/README.md](bench/README.md) covers protocols and tools.

## Develop

```bash
bash tools/check.sh setup-cpu               # CPU contracts, no GPU or model needed
.venv-cpu/bin/python tools/check_distribution.py
python3 tools/check_evidence.py
bash tools/check.sh gpu                     # inside the runtime container, exclusive GPU
```

| Path | Purpose |
|---|---|
| `sharar/` | Engine, API, kernels, model, trainer (`sharar/train/`), installed CLI; `sharar/target.py` names the served checkpoint and its geometry |
| `sharar-ctl`, `run-sharar.sh`, `run-sharar-train.sh` | Host command and the container launchers it drives |
| `bench/` | Measurement drivers, corpus and A/B tools, shared support |
| `tests/` | CPU contracts and opt-in GPU validation |
| `tools/`, `docker/` | Checks, image build, host memory guard |
| `docs/` | Current guides, [roadmap](roadmap.md), indexed archive of dated studies |

Read [CONTRIBUTING.md](CONTRIBUTING.md) and the [interfaces](docs/interfaces.md)
before changing the engine. Sharar is licensed under [Apache 2.0](LICENSE);
vendored code keeps its headers, see [third-party notices](THIRD_PARTY_NOTICES.md).
