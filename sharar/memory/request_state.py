"""The ownership boundary for request KV, recurrent state and cache loans.

Kernels may write through supplied tensor views. Only StateManager transfers
ownership, advances the processed prefix, or returns request resources.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sharar.engine.request import Request
    from sharar.memory.kv_pool import KvPool
    from sharar.memory.state_pool import StatePool
    from sharar.memory.radix import Match, RadixCache


@dataclass
class Reservation:
    pages: torch.Tensor
    ids: list[int]
    full_pages: torch.Tensor | None = None


@dataclass(eq=False, kw_only=True)
class RequestState:
    slot: int = -1
    pages: torch.Tensor | None = None
    prefix_len: int = 0  # borrowed, locked prefix; the rest is request-owned
    committed_len: int = 0  # prefix represented by both KV and recurrent state
    match: Match | None = None
    checkpoints: dict[int, int] = field(default_factory=dict)
    reservation: Reservation | None = None
    cacheable: bool = True
    cache_enabled: bool = False
    released: bool = False
    cache_auxiliary: dict[int, dict] = field(default_factory=dict)


class StateManager:
    def __init__(self, kv_pool: KvPool, state_pool: StatePool, cache: RadixCache | None = None):
        self.kv = kv_pool
        self.recurrent = state_pool
        self.cache = cache

    def authorize_mixed(self, chunks, verifying, max_batch):
        """Validate both destinations before any mixed-pass reservation/write."""
        requests = [r for r, _ in chunks] + list(verifying)
        if not chunks or not verifying or len(requests) > max_batch:
            raise ValueError("invalid mixed request count")
        if len({id(r) for r in requests}) != len(requests) or len({r.slot for r in requests}) != len(requests):
            raise ValueError("mixed requests must own distinct slots")
        for req in requests:
            if (req.released or req.finished or req.cancel_requested or req.reservation is not None
                    or req.pages is None or type(req.slot) is not int or not 0 < req.slot < self.recurrent.num_slots):
                raise ValueError("mixed request has no available private state")
        for req, count in chunks:
            if (type(count) is not int or count < 1 or req.prefill_pos + count > req.prefill_len
                    or req.committed_len != req.prefill_pos or len(req.pages) < req.prefill_pos + count):
                raise ValueError("invalid mixed prefill extent")
        for req in verifying:
            if not req.prefill_done or not req.output_ids or req.committed_len != req.context_len - 1:
                raise ValueError("mixed verify pending input does not follow committed state")

    def can_admit(self, prompt_len: int, reserve: int = 0) -> bool:
        need = prompt_len + reserve
        if self.cache is not None:
            self.cache.evict(need_pages=need, need_slots=1)
        return self.kv.free_count >= need and self.recurrent.free_count >= 1

    def ensure_pages(self, count: int) -> None:
        if self.kv.free_count < count and self.cache is not None:
            self.cache.evict(need_pages=count)
        if self.kv.free_count < count:
            raise MemoryError(f"KV pool exhausted: need {count}, free {self.kv.free_count}")

    def admit(self, req: Request, ids: list[int]) -> None:
        if req.slot >= 0 or req.pages is not None or req.reservation is not None or req.released:
            raise ValueError("request already owns resources or has been released")
        match, slot, own = None, -1, None
        try:
            if self.cache is not None:
                match = self.cache.match(ids[:-1])  # last token must produce logits
                if match.length:
                    self.cache.lock(match.node, 1)
                else:
                    match = None
            prefix = match.length if match is not None else 0
            if self.cache is not None:
                self.cache.evict(need_pages=len(ids) - prefix, need_slots=1)
            slot = self.recurrent.alloc()
            if match is not None:
                self.recurrent.copy_slot(match.snapshot_slot, slot)
            own = self.kv.alloc(len(ids) - prefix)
            pages = torch.cat([match.pages, own]) if match is not None else own
        except Exception:
            if own is not None:
                self.kv.free(own)
            if slot >= 0:
                self.recurrent.free(slot)
            if match is not None:
                self.cache.lock(match.node, -1)
            raise
        req.match, req.slot, req.pages = match, slot, pages
        req.cache_enabled = self.cache is not None
        req.prefix_len = req.committed_len = req.prefill_pos = prefix

    def checkpoint(self, req: Request, position: int, *, copy: bool = False) -> int | None:
        if self.cache is None:
            return None
        if not self.recurrent.free_count:
            self.cache.evict(need_slots=1)
        if not self.recurrent.free_count:
            return None
        slot = self.recurrent.alloc()
        req.checkpoints[position] = slot
        if copy:
            self.recurrent.copy_slot(req.slot, slot)
        return slot

    def commit_prefill(self, req: Request, end: int) -> None:
        if not req.committed_len <= end <= len(req.pages):
            raise ValueError("prefill commit outside owned pages")
        req.prefill_pos = req.committed_len = end

    def reserve(self, requests: list[Request], width: int) -> torch.Tensor:
        if width < 1 or any(r.reservation is not None or r.finished for r in requests):
            raise ValueError("invalid or overlapping reservation")
        self.ensure_pages(len(requests) * width)
        pages, ids = self.kv.alloc_with_ids(len(requests) * width)
        for i, req in enumerate(requests):
            lo, hi = i * width, (i + 1) * width
            req.reservation = Reservation(pages[lo:hi], ids[lo:hi])
        return pages

    def step_pages(self, req: Request) -> torch.Tensor:
        """Borrow the committed prefix plus the pending reservation for a forward."""
        if req.reservation is None:
            raise ValueError("forward requires a reservation")
        if req.reservation.full_pages is None:
            req.reservation.full_pages = torch.cat([req.pages, req.reservation.pages])
        return req.reservation.full_pages

    def commit(self, req: Request, count: int, *, ssm=None, conv=None) -> None:
        """Keep count verify inputs, or publish a successful in-place decode.

        ssm/conv are borrowed snapshots after input count-1; together they
        describe exactly the prefix that owns the retained pages.
        """
        pending = req.reservation
        if pending is None or not 1 <= count <= len(pending.ids):
            raise ValueError("commit outside reservation")
        if (ssm is None) != (conv is None):
            raise ValueError("recurrent commit needs both SSM and conv state")
        pages = pending.full_pages if count == len(pending.ids) and pending.full_pages is not None else torch.cat([req.pages, pending.pages[:count]])
        if ssm is not None:
            self.recurrent.ssm_all[:, req.slot].copy_(ssm)
            self.recurrent.conv_all[:, req.slot].copy_(conv)
        self._publish(req, count, pages)

    def commit_reconstructed(self, requests, counts, replay, conv_windows, *, verified_steps: int) -> None:
        """Validate the whole batch before allowing reconstruction into private slots.

        Prepare retained page tensors before mutation. A failed in-place commit
        poisons every affected request; rollback can free pages but cannot undo
        recurrent writes. Publication and proposer feedback follow all writes.
        """
        batch = len(requests)
        if (not 1 <= batch <= replay.batch or len(counts) != batch
                or type(verified_steps) is not int or not 1 <= verified_steps <= replay.steps):
            raise ValueError("invalid reconstruction batch")
        slots = [r.slot for r in requests]
        if len(set(slots)) != batch or any(not 0 < slot < self.recurrent.num_slots for slot in slots):
            raise ValueError("reconstruction requires unique private slots")
        conv = self.recurrent.conv_all
        if (conv_windows.ndim != conv.ndim + 1 or conv_windows.shape[0] != conv.shape[0]
                or conv_windows.shape[1] < batch or conv_windows.shape[3:] != conv.shape[2:]
                or conv_windows.dtype != conv.dtype or conv_windows.device != conv.device):
            raise ValueError("invalid reconstruction convolution windows")
        for req, count in zip(requests, counts):
            pending = req.reservation
            if (req.finished or req.released or req.pages is None or len(req.pages) != req.committed_len
                    or pending is None or type(count) is not int
                    or not 1 <= count <= min(len(pending.ids), verified_steps, conv_windows.shape[2])):
                raise ValueError("reconstruction commit outside reservation")
        pages = [r.reservation.full_pages if count == len(r.reservation.ids) and r.reservation.full_pages is not None
                 else torch.cat([r.pages, r.reservation.pages[:count]]) for r, count in zip(requests, counts)]
        metadata = torch.tensor([slots, counts], dtype=torch.int32, pin_memory=conv.device.type == "cuda")
        metadata = metadata.to(conv.device, non_blocking=True)
        try:
            replay.run(self.recurrent, metadata[0], metadata[1])
            for row, (req, count) in enumerate(zip(requests, counts)):
                conv[:, req.slot].copy_(conv_windows[:, row, count - 1])
            for req, count, retained in zip(requests, counts, pages):
                self._publish(req, count, retained)
        except BaseException as exc:
            for req in requests:
                req.cacheable = False
            if isinstance(exc, MemoryError):
                # The scheduler retries pool-pressure MemoryError. A partially
                # written recurrent slot requires failure cleanup instead.
                raise RuntimeError("reconstruction failed after state mutation") from exc
            raise

    def _publish(self, req, count, pages) -> None:
        self.kv.free_ids(req.reservation.ids[count:])
        req.pages = pages
        req.committed_len += count
        req.reservation = None

    def rollback(self, req: Request) -> None:
        """Discard pages, not in-place kernel writes; retract before retrying.

        Verification preserves the committed slot until acceptance. A failed
        decode or direct reconstruction may have changed it, so an abandoned
        transaction must never be cached.
        """
        if req.reservation is not None:
            req.cacheable = False
            self.kv.free_ids(req.reservation.ids)
            req.reservation = None

    def retract(self, req: Request) -> None:
        self.rollback(req)
        self._drop(req)
        req.prefix_len = req.committed_len = req.prefill_pos = 0

    def release(self, req: Request) -> None:
        if req.released:
            return
        if req.reservation is not None:
            # In-place decode could have partially advanced a slot on failure.
            req.cacheable = False
            self.rollback(req)
        cached = False
        if (self.cache is not None and not self.cache.flushing and req.cacheable
                and req.slot >= 0 and req.pages is not None and req.prefill_done):
            cached = self._insert(req)
        self._drop(req, cached=cached)
        req.released = True

    def _drop(self, req: Request, *, cached: bool = False) -> None:
        if req.match is not None:
            self.cache.lock(req.match.node, -1)
            req.match = None
        if not cached:
            if req.pages is not None:
                self.kv.free(req.pages[req.prefix_len:])
            if req.slot >= 0:
                self.recurrent.free(req.slot)
            for slot in req.checkpoints.values():
                self.recurrent.free(slot)
        req.pages, req.slot, req.checkpoints = None, -1, {}
        req.cache_auxiliary = {}

    def _insert(self, req: Request) -> bool:
        ids = req.prompt_ids + req.output_ids[:-1]
        n = len(ids)
        if req.committed_len != n or not n or len(req.pages) < n:
            return False
        snapshots = {n: req.slot, **{p: s for p, s in req.checkpoints.items() if p < n}}
        for p, slot in req.checkpoints.items():
            if p >= n:
                self.recurrent.free(slot)
        new_start = self.cache.insert(ids, req.pages[:n], snapshots, req.cache_auxiliary)
        if new_start > req.prefix_len:
            self.kv.free(req.pages[req.prefix_len:new_start])
        if len(req.pages) > n:
            self.kv.free(req.pages[n:])
        return True

    def flush(self) -> None:
        if self.cache is not None:
            self.cache.flush()
