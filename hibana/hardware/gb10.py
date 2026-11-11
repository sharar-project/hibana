"""The only hardware target. Allocation strategies must share this host budget."""

GPU_NAME = "NVIDIA GB10"
COMPUTE_CAPABILITY = (12, 1)
MIN_DRIVER_MAJOR = 580
MIN_TOTAL_MEMORY_KIB = 120_000_000
HOST_FLOOR_KIB = 6 * 1024 * 1024  # 6 GiB MemAvailable the pools leave for the OS and other programs (SHARAR_HOST_FLOOR_GIB);
# admission holds and the memory guard act at half of it
UNIFIED_PHYSICAL_MEMORY = True
