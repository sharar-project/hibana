#!/usr/bin/env bash
# CPU setup is standalone. GPU mode runs inside the prepared GB10 environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${1:-cpu}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
  setup-cpu)
    python3 -m venv .venv-cpu
    .venv-cpu/bin/python -m pip install -c constraints/cpu.txt torch --index-url https://download.pytorch.org/whl/cpu
    .venv-cpu/bin/python -m pip install -c constraints/cpu.txt -e '.[test]'
    exec env CUDA_VISIBLE_DEVICES= SHARAR_RUN_GPU=0 OMP_NUM_THREADS=1 .venv-cpu/bin/python -m pytest tests/unit -q -p no:cacheprovider "$@"
    ;;
  cpu)
    exec env CUDA_VISIBLE_DEVICES= SHARAR_RUN_GPU=0 OMP_NUM_THREADS=1 python3 -m pytest tests/unit -q -p no:cacheprovider "$@"
    ;;
  gpu)  # inside a prepared GB10 environment (paths already set)
    exec env SHARAR_RUN_GPU=1 python3 -m pytest tests/unit tests/gpu -q -p no:cacheprovider "$@"
    ;;
  gpu-container)  # from the host: the runtime image, this checkout mounted (tools/gpu-tests.sh)
    exec tools/gpu-tests.sh "$@"
    ;;
  *) echo 'usage: tools/check.sh {setup-cpu|cpu|gpu|gpu-container} [pytest arguments]' >&2; exit 2 ;;
esac
