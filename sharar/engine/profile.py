"""Validated serving limits, separate from model facts and device capabilities."""

from dataclasses import dataclass


# Contexts up to the target's native 262,144 positions are served; batches of up to
# four requests (sixteen in the throughput profile, #59) share the KV pool, and
# admission budgets each request's actual tokens.
MAX_CONTEXT = 262144
MAX_BATCH = 128
BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128)  # CUDA-graph batch sizes; a batch pads to the smallest that holds it


@dataclass(frozen=True)
class ServingProfile:
    max_context: int = 32768
    max_batch: int = 4
    max_queued: int = 16
    chunk_size: int = 2048
    prefill_budget: int = 4096

    def __post_init__(self):
        for name, upper in (("max_context", MAX_CONTEXT), ("max_batch", MAX_BATCH)):
            value = getattr(self, name)
            lower = 2 if name == "max_context" else 1
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} must be in [{lower}, {upper}] for the validated profile")
        for name in ("max_queued", "chunk_size", "prefill_budget"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def graph_sizes(self) -> tuple[int, ...]:
        return tuple(sorted({n for n in BUCKETS if n <= self.max_batch} | {self.max_batch}))

DEFAULT_PROFILE = ServingProfile()
