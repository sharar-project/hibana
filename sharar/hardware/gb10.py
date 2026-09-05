"""The only hardware target. Allocation strategies must share this host budget."""

GPU_NAME = "NVIDIA GB10"
COMPUTE_CAPABILITY = (12, 1)
MIN_DRIVER_MAJOR = 580
MIN_TOTAL_MEMORY_KIB = 120_000_000
HOST_FLOOR_KIB = 16 * 1024 * 1024  # 16 GiB MemAvailable, never crossed
UNIFIED_PHYSICAL_MEMORY = True
