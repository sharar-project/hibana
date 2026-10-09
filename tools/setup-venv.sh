#!/usr/bin/env bash
# The runtime image's Python environment (docker/Dockerfile) as a host venv built with uv: .venv with the pinned
# cu130 torch, FlashInfer and the serving packages, and sharar installed editable. Needs the host CUDA 13 toolkit
# (/usr/local/cuda) for FlashInfer's SM121 JIT modules. Usage: tools/setup-venv.sh
set -euo pipefail
cd "$(dirname "$0")/.."
C=constraints/gb10-cu130.txt
uv venv --python 3.12 .venv
uv pip install -p .venv -c $C torch --index-url https://download.pytorch.org/whl/cu130
uv pip install -p .venv -c $C flashinfer-python flashinfer-jit-cache --extra-index-url https://flashinfer.ai/whl/cu130 \
  --index-strategy unsafe-best-match
uv pip install -p .venv -c $C safetensors tokenizers jinja2 numpy einops packaging fastapi uvicorn pytest httpx xgrammar \
  huggingface_hub ninja
.venv/bin/python docker/repair_wheel_tag.py
uv pip install -p .venv --no-deps -e .
# ibverbs headers for the two-Spark RDMA all-reduce (sharar/kernels/rdma_oneshot.py), unpacked into the venv: no root
(cd .venv && apt-get download libibverbs-dev && dpkg -x libibverbs-dev_*.deb ibverbs && rm libibverbs-dev_*.deb)
.venv/bin/python -c "import sharar, flashinfer, torch; print(torch.__version__, flashinfer.__version__, torch.cuda.is_available())"
