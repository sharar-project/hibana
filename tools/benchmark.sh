#!/usr/bin/env bash
# Benchmark an already running server using the repository's host paths.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec python3 -m sharar benchmark "$@"
