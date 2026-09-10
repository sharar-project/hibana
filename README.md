# Sharar

Sharar is a local text-inference engine for one NVIDIA GB10 (SM 12.1), on an
`aarch64` DGX Spark with unified memory. The supported target is
`RadixArk/Qwen3.8-27B-NVFP4`, revision
`319f741cce68d7914884900c138a1fbb70a42f30`.

It serves offline, OpenAI-compatible streaming chat and completions with
prefix caching and speculative decoding. Sharar runs independently of SGLang;
the separate frozen reference is used for numerical and performance comparisons.

## Run

Set model-cache and results paths in `config/gb10.env` using
[the example](config/gb10.env.example), then build and launch on GB10:

```bash
docker/build.sh
./run-sharar.sh --check
./run-sharar.sh
tools/verify-sharar.sh
```

The launcher serves on `127.0.0.1:30001` and starts the host memory guard.
The default is **DFlash2, 15 proposals, accepted-state reconstruction, BF16 draft**.
For the optimized measured draft profile, select FP8 when launching:

```bash
SHARAR_DRAFT_PRECISION=fp8 ./run-sharar.sh
```

Use `./run-sharar.sh --stop` before replacing an existing container. Building
or updating source does not update an already running image. See
[operations](docs/operations.md) for prerequisites, configurations and development.

```bash
curl -s http://127.0.0.1:30001/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":64,"chat_template_kwargs":{"enable_thinking":false}}'
```

See the [API guide](docs/api.md) for streaming, thinking, tools, stopping and
the supported output controls. Forced tool calls and constrained JSON output
are not implemented. The supported context limit is 32,768 tokens and maximum
concurrent batch is four.

## Measure

With a server already running and exclusive GPU use:

```bash
tools/benchmark.sh                  # code, 10K input / 1K output, BS1
tools/benchmark.sh --workload all   # code, design and reasoning
```

The installed command is `sharar benchmark`. It uses AIPerf and reports
end-to-end TPS, decode TPS, TTFT and latency with exact token audits.
See [benchmark setup and protocols](bench/README.md).

**FP8 DFlash2 at k=15 is the current measured performance baseline.** The
BS4 width study favors 16 verification rows over 8 or 4. Checkpoint retention
recovers 3.24× resumed BS1 throughput and 2.73× on a simulated four-session
conversation trace, with a larger bounded feature cache; cold throughput is
essentially unchanged by retention. These are workload-specific results.
[Performance and limitations](docs/performance.md) links the measured evidence.

## Develop

```bash
bash tools/check.sh setup-cpu
.venv-cpu/bin/python tools/check_distribution.py
python3 tools/check_evidence.py
```

Start with [CONTRIBUTING.md](CONTRIBUTING.md), the [ownership contracts](docs/interfaces.md)
and the [current roadmap](roadmap.md). GPU tests require an isolated, guarded
GB10 environment. CPU checks and distribution validation need no model or GPU.

| Path | Purpose |
|---|---|
| `sharar/` | Engine, API, kernels, model and installed benchmark client |
| `bench/` | Current measurement drivers and shared support |
| `bench/archive/` | Completed experiment implementations and reference tests |
| `tests/` | CPU contracts and opt-in GPU/numerical validation |
| `tools/`, `docker/` | Checks, development/runtime containers and host guard |
| `docs/` | Current guides, proposals and indexed historical evidence |

The [documentation index](docs/README.md) separates current guidance from
[archived studies](docs/archive/README.md). Raw benchmark artifacts stay outside Git.

Sharar is licensed under [Apache 2.0](LICENSE). Vendored code retains its
original headers; see [third-party notices](THIRD_PARTY_NOTICES.md).
