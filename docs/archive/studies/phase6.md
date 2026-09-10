# Phase 6 notes: promotion and runtime image

> Dated evidence for the source and protocol recorded below. See the [current documentation](../../README.md) for supported behavior and priorities. Historical commands may require the recorded revision.

## What exists

- `docker/Dockerfile`: the Sharar runtime image from `nvidia/cuda:13.0.3-devel-ubuntu24.04`
  (arm64) with a venv holding only the pinned packages: torch 2.13.0+cu130
  and its CUDA runtime wheels from the PyTorch cu130 index, FlashInfer 0.6.17
  with its cu130 JIT cache, Triton 3.7.1, safetensors, tokenizers, jinja2,
  numpy, einops, packaging, fastapi, uvicorn, pytest. No SGLang, no
  transformers, no vLLM. `nvcc` stays until Phase 9 builds the SM121 modules
  and the CUDA GEMV extension ahead of time.
- `docker/build.sh`, `run-sharar.sh` (preflight, container with the GB10
  devices and driver libraries bind-mounted, loopback-only port 30001,
  persistent JIT cache under `${SHARAR_RESULTS}/runtime-state`, the
  unified-memory guard as a user systemd unit; `--check`, `--stop`).
- `tools/verify-sharar.sh` verifies a running server.

## Results

| Check | Result |
|---|---|
| Image | `sharar:qwen38-27b`, `sha256:745c17e43d59783ca235b72b4e0bb1b3f2229d1d7f74b176314fd6e40ce6ed94`, 14.3 GB (reference image 21.1 GB) |
| Inside the image | `sharar check` OK, 30 unit tests pass, no sglang / transformers / vllm installed (80 packages) |
| Offline cold start (no network, empty JIT cache) | healthy after 131 s; chat canary answers `SPARKLANG_OK` |
| Warm restart (JIT cache kept) | healthy after 96 s, of which weight load is 87 s |
| Production restart via `run-sharar.sh --stop` / `run-sharar.sh` | healthy after 96 s, guard active |
| Controlled matrix from the image vs the dev-container run | PASS: all six cells within 1% on throughput, TTFT within 4%, exact serial hashes, canaries 4/4 |
| 32,766-token boundary from the image | 9.61 tok/s, TTFT 16.4 s, minimum MemAvailable 38.5 GiB |

Sharar baseline (image run `controlled-27b-sharar-baseline-20260904T125049Z`,
boundary `controlled-27b-sharar-baseline-32766-20260904T130641Z`):

| Input | Conc | tok/s | TTFT p50 / p95 (s) | min MemAvailable |
|---:|---:|---:|---|---:|
| 128 | 1 | 11.95 | 0.19 / 0.19 | 38.9 GiB |
| 128 | 4 | 43.7 | 0.35 / 0.38 | 39.1 GiB |
| 2,048 | 1 | 11.49 | 0.85 / 0.85 | 39.0 GiB |
| 2,048 | 4 | 38.4 | 2.45 / 3.26 | 38.6 GiB |
| 8,192 | 1 | 10.15 | 3.55 / 3.55 | 38.5 GiB |
| 8,192 | 4 | 26.9 | 10.3 / 13.8 | 38.6 GiB |

Serial response hashes were identical between the dev container and the
runtime image, so Sharar's serial output is deterministic across environments
and the hash gate is meaningful from now on. The 8 GiB weight-load reserve and
the cold-start time are Phase 8 and Phase 9 work (zero-copy load, AOT kernels).

## Decisions

- The image is built from a clean CUDA base rather than derived from the
  reference image, so its contents are exactly the pinned list and nothing
  inherited.
- The reference container is retired from daily use: Sharar's own controlled
  run is the baseline from now on and `bench/compare_runs.py` measures Sharar
  against itself. The reference stays available for parity work
  (`tools/reference/`).
