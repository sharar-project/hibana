# Development environment

Sharar is developed inside a container started from the frozen reference image
so that the CUDA stack matches the reference bit for bit. `tools/dev-shell.sh`
starts it with this repository at `/workspace/sharar`, the model cache
read-only at `/cache/huggingface`, `${SHARAR_RESULTS}` at `/results` (also the
container `HOME`, so FlashInfer/Triton JIT caches persist across runs), the GB10
devices and driver libraries bind-mounted the way the reference launcher does,
and no network.

| Component | Version (reference image `sparklang-gb10-engine:qwen38-27b`) |
|---|---|
| Base image | `lmsysorg/sglang@sha256:616a3e97f45191af975896cfa644279096cb31bd408a071c2e99ca7209c3cafe` |
| CUDA toolkit | 13.0.3 (`nvcc` 13.0.r13.0), driver 580.95.05 on the host |
| Python | 3.12.3 |
| torch | 2.13.0+cu130 |
| flashinfer-python | 0.6.17 (JIT cache built for `sm_121a` at first use) |
| triton | 3.7.1 (`tokenspeed-triton` 3.8.10 is also installed by the base image) |
| safetensors | 0.8.0 |
| tokenizers | 0.22.2 |
| jinja2 | 3.1.6 |
| numpy | 2.3.5 |
| einops | 0.8.2 (used by the vendored fla kernels) |
| sglang | 0.0.0.dev1+g5f55db35e, editable at `/sgl-workspace/sglang/python` |

`sglang` is importable in the container because the base image installs it as
an editable package. Sharar code and tests never import it; the parity tools
under `tools/reference/` are the only scripts that do, and they run with
`tools/dev-shell.sh --with-reference`.

Useful commands:

```bash
tools/dev-shell.sh                      # interactive shell
tools/dev-shell.sh -- python3 -m sharar check
tools/dev-shell.sh -- python3 -m sharar parity-prompts --out /results/parity/prompts.json
python3 tools/reference/http_oracle.py --prompts "$SHARAR_RESULTS/parity/prompts.json" --out "$SHARAR_RESULTS/parity/reference"
SHARAR_DEV_CONTAINER=refdump tools/dev-shell.sh --with-reference -- \
  env SHARAR_REFERENCE_DUMP=1 python3 tools/reference/dump_reference.py \
  --prompts /results/parity/prompts.json --out /results/parity/reference
tools/dev-shell.sh -- python3 -m sharar parity --reference /results/parity/reference
tools/dev-shell.sh -- python3 -m pytest tests/unit -q
```

The reference dump engine needs about 25 GB of unified memory next to the
running reference server; the parity forward needs about 22 GB. Both fit with
the production 16 GiB floor intact on the 119.7 GiB host.
