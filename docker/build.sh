#!/usr/bin/env bash
# Build the Sharar runtime image. Usage: docker/build.sh [tag]
set -Eeuo pipefail
cd "$(dirname "$0")/.."
TAG="${1:-sharar:qwen38-27b}"
[[ "$(uname -m)" == "aarch64" ]] || { echo "build on the aarch64 host" >&2; exit 1; }
docker build --file docker/Dockerfile --tag "$TAG" .
docker image inspect "$TAG" --format 'Built {{.RepoTags}} {{.Id}} size={{.Size}} layers={{len .RootFS.Layers}}'
