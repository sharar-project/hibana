# Sharar

Sharar is a thin inference engine for exactly one target:

- One NVIDIA GB10 (SM 12.1) on `aarch64`, one DGX Spark-class machine with
  unified LPDDR5X memory.
- `RadixArk/Qwen3.8-27B-NVFP4` at revision
  `319f741cce68d7914884900c138a1fbb70a42f30`.
- Local, offline, OpenAI-compatible text generation with speculative decoding.

It is written from scratch beside SGLang rather than carved out of it. The
`sparklang` repository holds the frozen SGLang-based reference container; Sharar
uses that container only as a correctness and performance oracle and never
imports the `sglang` package.

Target size is about 6K lines of engine code plus about 5K lines of vendored,
attributed Triton kernels, on top of `torch`, `flashinfer`, `triton`,
`safetensors` and `tokenizers`.

## Layout

| Path | Purpose |
|---|---|
| `sharar/` | The engine package (see the roadmap for the module plan) |
| `bench/` | Controlled six-cell benchmark and comparator, shared with the reference |
| `tools/` | Reference-container verification and the host memory guard |
| `tests/` | Unit tests and parity tests against the reference container |
| `config/` | Host-local paths (`gb10.env` is ignored by git) |
| `docs/` | Design notes that outlive a single phase |

## Status

Phase 0. The repository, oracle plumbing and roadmap exist; no engine code yet.
Read [roadmap.md](roadmap.md) before doing anything else.

## Attribution and license

Sharar is licensed under the Apache License 2.0. Vendored kernels and ported
code keep their original headers and are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
