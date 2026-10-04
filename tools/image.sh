# Sharar runtime image: the default and how a launcher gets it. Sourced by run-sharar.sh and run-sharar-train.sh;
# sharar-ctl reads SHARAR_DEFAULT_IMAGE from this file.
#
# The default is the image the nightly GB10 check publishes from main after its GPU suite and serve checks pass
# (tools/ci/gb10.sh, stage publish). The package is private while the repository is, so pulling it needs
# `docker login ghcr.io` with a GitHub token that can read packages. SHARAR_IMAGE=sharar:<tag> serves a local
# build (docker/build.sh) instead.
SHARAR_DEFAULT_IMAGE=ghcr.io/abdelfattah-lab/sharar:latest

# ensure_image IMAGE [--no-pull]: succeed when IMAGE is present, pulling a registry image that is missing.
ensure_image() {
  docker image inspect "$1" >/dev/null 2>&1 && return 0
  if [[ "$1" != */* ]]; then
    echo "image is missing: $1 (build it with docker/build.sh $1, or unset SHARAR_IMAGE to pull $SHARAR_DEFAULT_IMAGE)" >&2
    return 1
  fi
  if [[ "${2:-}" == --no-pull ]]; then
    echo "image is missing: $1 (the launcher pulls it; sharar-ctl pull fetches it now)" >&2
    return 1
  fi
  echo "pulling $1 (about 15 GB the first time)" >&2
  docker pull "$1" >&2 && return 0
  echo "could not pull $1: while the package is private, run \`gh auth token | docker login ghcr.io -u <github user> --password-stdin\` with a token that has read:packages, or build locally with docker/build.sh and set SHARAR_IMAGE" >&2
  return 1
}
