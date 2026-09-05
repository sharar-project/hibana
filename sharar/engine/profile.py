"""Validated serving limits, separate from model facts and device capabilities."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ServingProfile:
    max_context: int = 32768
    max_batch: int = 4
    max_spec_depth: int = 7  # target verification; each adapter also limits its width
    max_queued: int = 16
    chunk_size: int = 2048
    prefill_budget: int = 4096

    def __post_init__(self):
        for name, upper in (("max_context", 32768), ("max_batch", 4), ("max_spec_depth", 7)):
            value = getattr(self, name)
            lower = 2 if name == "max_context" else 1
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} must be in [{lower}, {upper}] for the validated profile")
        for name in ("max_queued", "chunk_size", "prefill_budget"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def graph_sizes(self) -> tuple[int, ...]:
        return tuple(sorted({n for n in (1, 2, 4) if n <= self.max_batch} | {self.max_batch}))

    def verify_capacity(self, depth: int) -> int:
        if type(depth) is not int or not 1 <= depth <= self.max_spec_depth:
            raise ValueError("unsupported speculative depth")
        return self.max_context + depth


DEFAULT_PROFILE = ServingProfile()
