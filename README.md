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

## Try it

From a DGX Spark with the model snapshot and the reference image present
(`config/gb10.env` points at them):

```bash
tools/dev-shell.sh --tty -- python3 -u -m sharar chat
```

This loads the model (about 90 s), captures the decode CUDA graphs, and gives
an interactive `you>` prompt. Replies stream token by token and every turn ends
with a stats line: prompt tokens, time to first token, tokens generated and
decode tokens per second. `/reset` clears the history, `/exit` quits. Useful
flags: `--temperature 0.7` to sample, `--thinking` to enable the model's
reasoning mode, `--system "..."`, and `--prompt "..."` for a one-shot answer.

Other commands: `python3 -m sharar canaries --reference /results/parity/reference`
(quality canaries against the reference), `python3 -m sharar bench-decode
--prompts /results/parity/prompts.json` (decode throughput and determinism),
`python3 -m sharar parity --reference /results/parity/reference` (prefill
parity), `python3 -m sharar profile --prompts /results/parity/prompts.json`
(kernel time breakdown). All run inside `tools/dev-shell.sh`.

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

Phases 0 to 2 are complete. The engine loads the checkpoint, prefills, and
decodes with CUDA graphs at 12.1 tok/s single-stream and 44.6 aggregate at
batch 4 (reference 12.7 and 46.4), passes the four quality canaries, and
matches the reference to within the model's own quantization noise floor
([docs/parity.md](docs/parity.md), [docs/phase2.md](docs/phase2.md)). Phase 3
(memory ledger and radix prefix cache) is next. Read [roadmap.md](roadmap.md)
before doing anything else.

## Attribution and license

Sharar is licensed under the Apache License 2.0. Vendored kernels and ported
code keep their original headers and are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
