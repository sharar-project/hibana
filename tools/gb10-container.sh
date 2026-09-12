#!/usr/bin/env bash
# Shared by run-sharar.sh and run-sharar-train.sh: append the GB10 devices and
# driver libraries a runtime container needs to the caller's `args` array.
# Requires DRIVER_LIB_DIR and a `fail` function in the caller.
gb10_container_args() {
  local dev src gpucomp
  for dev in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset /dev/dri/renderD128; do
    [[ -e "$dev" ]] || fail "missing device $dev"
    args+=(--device "$dev")
  done
  add_lib() { src="$(readlink -f "$DRIVER_LIB_DIR/$1")"; [[ -f "$src" ]] || fail "missing driver library $1"; args+=(--mount "type=bind,src=$src,dst=$DRIVER_LIB_DIR/$2,readonly"); }
  add_lib libcuda.so.1 libcuda.so.1; add_lib libcuda.so.1 libcuda.so
  add_lib libnvidia-ptxjitcompiler.so.1 libnvidia-ptxjitcompiler.so.1; add_lib libnvidia-ptxjitcompiler.so.1 libnvidia-ptxjitcompiler.so
  add_lib libnvidia-ml.so.1 libnvidia-ml.so.1; add_lib libnvidia-ml.so.1 libnvidia-ml.so
  add_lib libnvidia-nvvm.so.4 libnvidia-nvvm.so.4; add_lib libnvidia-nvvm.so.4 libnvidia-nvvm.so
  add_lib libcudadebugger.so.1 libcudadebugger.so.1
  for gpucomp in "$DRIVER_LIB_DIR"/libnvidia-gpucomp.so.*; do [[ -f "$gpucomp" ]] && args+=(--mount "type=bind,src=$gpucomp,dst=$gpucomp,readonly"); done
  args+=(--mount type=bind,src=/usr/bin/nvidia-smi,dst=/usr/bin/nvidia-smi,readonly)
}
