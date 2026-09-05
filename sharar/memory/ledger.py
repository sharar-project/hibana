"""One unified-memory budget for the whole process.

GB10 has a single physical pool shared by the OS, the host process and the
GPU allocator, so the only true constraint is host MemAvailable minus a hard
floor. The ledger records every large allocation category, plans the KV page
and state slot counts from what is available at startup, and can re-check the
floor at any time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from sharar import config
from sharar.memory.state_pool import CONV_DIM, CONV_STATE_LEN

GIB = 2**30
KV_BYTES_PER_TOKEN = 2 * config.NUM_KV_HEADS * config.HEAD_DIM * 16  # K+V, fp8, 16 attention layers
STATE_BYTES_PER_SLOT = 48 * (CONV_DIM * CONV_STATE_LEN * 2 + config.LINEAR_NUM_VALUE_HEADS * config.LINEAR_VALUE_HEAD_DIM * config.LINEAR_KEY_HEAD_DIM * 4)


def mem_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


def check_load_headroom(peak_bytes: int) -> None:
    """Reject a second model load before transient allocations reach the floor."""
    available = mem_available_bytes()
    floor = config.HOST_FLOOR_KIB * 1024
    if available < floor + peak_bytes:
        raise MemoryError(f"model load needs {(floor + peak_bytes) / GIB:.1f} GiB MemAvailable including the host floor; have {available / GIB:.1f} GiB")


@dataclass
class Plan:
    kv_tokens: int
    state_slots: int
    reserved_bytes: int  # activation workspace kept free for the largest prefill


@dataclass
class Ledger:
    floor_bytes: int = 16 * GIB
    entries: dict[str, int] = field(default_factory=dict)
    start_available: int = field(default_factory=mem_available_bytes)

    def record(self, name: str, nbytes: int) -> None:
        self.entries[name] = self.entries.get(name, 0) + int(nbytes)

    def record_cuda_allocated(self, name: str) -> None:
        """Attribute the growth of torch's allocation since the last call."""
        cur = torch.cuda.memory_allocated()
        prev = getattr(self, "_last_cuda", 0)
        self.record(name, cur - prev)
        self._last_cuda = cur

    @property
    def total(self) -> int:
        return sum(self.entries.values())

    def plan(
        self,
        max_running: int = 4,
        context_len: int = config.CONTEXT_LENGTH,
        prefill_reserve_bytes: int = 8 * GIB,
        state_slot_target: int = 96,
        kv_token_target: int = 1 << 20,
        cache_fraction: float = 0.5,
        future_bytes: int = 0,
        extra_page_bytes: int = 0,
    ) -> Plan:
        """Size the pools from what is available now.

        Guarantees room for `max_running` requests at `context_len` (KV pages
        and committed state slots) plus the prefill activation reserve and the
        floor; the remainder, times `cache_fraction`, goes to prefix-cache
        capacity. A cached request costs about three state snapshots (441 MiB)
        against 256 MiB of KV for an 8K prompt, and the Phase 3 soak showed
        slots, not pages, drive eviction, so the split is 40% pages / 60% slots.
        """
        available = mem_available_bytes()
        budget = available - self.floor_bytes - prefill_reserve_bytes - future_bytes
        page_bytes = KV_BYTES_PER_TOKEN + extra_page_bytes
        must_kv = max_running * context_len * page_bytes
        must_state = (max_running + 1) * STATE_BYTES_PER_SLOT  # +1 dummy slot 0
        if budget < must_kv + must_state:
            raise MemoryError(
                f"only {available / GIB:.1f} GiB available; need {(must_kv + must_state + prefill_reserve_bytes + future_bytes + self.floor_bytes) / GIB:.1f} GiB"
            )
        spare = int((budget - must_kv - must_state) * cache_fraction)
        kv_tokens = min(kv_token_target, max_running * context_len + int(spare * 0.4) // page_bytes)
        state_slots = min(state_slot_target, max_running + 1 + int(spare * 0.6) // STATE_BYTES_PER_SLOT)
        return Plan(kv_tokens=kv_tokens, state_slots=state_slots, reserved_bytes=prefill_reserve_bytes + future_bytes)

    def check_floor(self) -> int:
        available = mem_available_bytes()
        if available < self.floor_bytes:
            raise MemoryError(f"MemAvailable {available / GIB:.1f} GiB is below the {self.floor_bytes / GIB:.0f} GiB floor")
        return available

    def describe(self) -> str:
        lines = [f"  {name:<28}{nbytes / GIB:8.2f} GiB" for name, nbytes in self.entries.items()]
        lines.append(f"  {'total recorded':<28}{self.total / GIB:8.2f} GiB")
        lines.append(f"  {'MemAvailable now':<28}{mem_available_bytes() / GIB:8.2f} GiB  (floor {self.floor_bytes / GIB:.0f} GiB)")
        return "\n".join(lines)
