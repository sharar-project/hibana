#!/usr/bin/env bash
# Run a command so PyTorch can start inside an LXD/Incus container whose CPU set has holes.
#
# lxcfs numbers /proc/cpuinfo's processors 0..N-1, while /sys/devices/system/cpu keeps the host's numbering (e.g.
# online 0-11,14-19 of possible 0-19). PyTorch's cpuinfo indexes one by the other and corrupts the heap on
# `import torch` ("malloc(): corrupted top size"). This presents CPU lists that agree with /proc/cpuinfo, in a private
# user and mount namespace: nothing outside the command changes. Usage: tools/lxcfs-run.sh COMMAND [ARGS...]
set -euo pipefail
grep -q " /sys/devices/system/cpu fuse.lxcfs " /proc/mounts || exec "$@"
list=$(mktemp)
echo "0-$(($(grep -c ^processor /proc/cpuinfo) - 1))" > "$list"
exec unshare --user --map-root-user --mount bash -c '
  for f in possible present online; do mount --bind "$0" /sys/devices/system/cpu/$f; done
  rm "$0"
  exec "$@"' "$list" "$@"
