#!/usr/bin/env bash
# Sharar's Python environment: .venv in this checkout (or SHARAR_VENV), built with uv from the pinned constraints:
# cu130 torch, FlashInfer and the serving packages, with sharar installed editable. The launchers run it first when
# the venv is missing or older than this script, the constraints or pyproject.toml (the stamp below).
# Needs uv (https://docs.astral.sh/uv/) and the host CUDA 13 toolkit in /usr/local/cuda (FlashInfer's SM121 JIT modules
# and Sharar's CUDA kernels compile on first use). Usage: tools/setup-venv.sh [--check]
#   --check   exit 0 when the venv is current, 1 when it needs (re)building; changes nothing
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=${SHARAR_VENV:-$PWD/.venv}
C=constraints/gb10-cu130.txt
STAMP="$VENV/.sharar-setup"
want=$(cat tools/setup-venv.sh "$C" pyproject.toml | sha256sum | cut -c1-16)
if [[ "${1:-}" == --check ]]; then
  [[ -x "$VENV/bin/python" && -f "$STAMP" && "$(cat "$STAMP")" == "$want" ]]
  exit
fi
UV=$(command -v uv || echo "$HOME/.local/bin/uv")
[[ -x "$UV" ]] || { echo "uv is not installed: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }
[[ -x /usr/local/cuda/bin/nvcc ]] || { echo "the CUDA 13 toolkit is missing from /usr/local/cuda (DGX OS ships it)" >&2; exit 1; }
echo "building the Python environment in $VENV (torch, FlashInfer and the serving packages; a few GB the first time)"
# A uv-managed interpreter carries its C headers, which the CUDA kernels' first-use builds need.
"$UV" venv --allow-existing --managed-python --python 3.12 "$VENV"
"$UV" pip install -p "$VENV" -c $C torch --index-url https://download.pytorch.org/whl/cu130
"$UV" pip install -p "$VENV" -c $C flashinfer-python flashinfer-jit-cache --extra-index-url https://flashinfer.ai/whl/cu130 \
  --index-strategy unsafe-best-match
"$UV" pip install -p "$VENV" -c $C safetensors tokenizers jinja2 numpy einops packaging fastapi uvicorn pytest httpx xgrammar \
  huggingface_hub ninja
# NVIDIA's cusparselt 0.8.1 wheel carries an sbsa platform tag that pip check rejects (tools/repair_wheel_tag.py)
grep -qs manylinux2014_sbsa "$VENV"/lib/python3.12/site-packages/nvidia_cusparselt_cu13-*.dist-info/WHEEL \
  && "$VENV/bin/python" tools/repair_wheel_tag.py
"$UV" pip install -p "$VENV" --no-deps -e .
# ibverbs headers for the two-Spark RDMA all-reduce (sharar/kernels/rdma_oneshot.py), unpacked into the venv: no root
if [[ ! -d "$VENV/ibverbs" ]]; then
  (cd "$VENV" && apt-get download libibverbs-dev >/dev/null 2>&1 && dpkg -x libibverbs-dev_*.deb ibverbs && rm libibverbs-dev_*.deb) \
    || echo "note: libibverbs-dev could not be fetched; only the two-Spark RDMA all-reduce needs it" >&2
fi
"$VENV/bin/python" -c "import sharar, flashinfer, torch; print('torch', torch.__version__, 'flashinfer', flashinfer.__version__, 'cuda', torch.cuda.is_available())"
echo "$want" > "$STAMP"
