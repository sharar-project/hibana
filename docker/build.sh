#!/usr/bin/env bash
# Build the Sharar runtime image. Usage: docker/build.sh [tag]
set -Eeuo pipefail
cd "$(dirname "$0")/.."
TAG="${1:-sharar:qwen38-27b}"
[[ "$(uname -m)" == "aarch64" ]] || { echo "build on the aarch64 host" >&2; exit 1; }
REVISION="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
[[ "$REVISION" == unknown || -z "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]] || REVISION+="+dirty"
docker build --file docker/Dockerfile --build-arg REVISION="$REVISION" --tag "$TAG" .
docker image inspect "$TAG" --format 'Built {{.RepoTags}} {{.Id}} size={{.Size}} layers={{len .RootFS.Layers}} revision={{index .Config.Labels "org.opencontainers.image.revision"}}'
