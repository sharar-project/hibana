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
    from sharar.cache.kv_pool import KvPool
    from sharar.cache.state_pool import StatePool
    from sharar.cache.radix import Match, Node, RadixCache


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
    page_ids: list[int] = field(default_factory=list)  # host ids of the pages after prefix_len (-1: the cache owns it)
    prompt_node: Node | None = None  # where `cache_prompt` ended the prompt in the cache; locked until release
    cacheable: bool = True
    cache_enabled: bool = False
    released: bool = False
    cache_auxiliary: dict[int, dict] = field(default_factory=dict)


class _Lap:
    """Accumulate synchronized wall time per named sub-phase into a dict (profiling only)."""

    def __init__(self, sink: dict, conv_device):
        import time
        self.sink, self.time, self.sync = sink, time, (torch.cuda.synchronize if conv_device.type == "cuda" else (lambda: None))
        self.sync()
        self.t = time.perf_counter()

    def __call__(self, name: str) -> None:
        self.sync()
        now = self.time.perf_counter()
        self.sink[name] = self.sink.get(name, 0.0) + (now - self.t)
        self.t = now


class StateManager:
    commit_profile: dict | None = None  # set to a dict to time commit_reconstructed sub-phases (benchmarks)
    # Wide batches (the throughput profile): prefill snapshots never take the recurrent slots that
    # admitting up to `admission_slots` live requests needs (0: snapshots take any free slot, as at batch four).
    admission_slots = 0
    live = 0  # admitted requests that still own a slot

    def __init__(self, kv_pool: KvPool, state_pool: StatePool, cache: RadixCache | None = None):
        self.kv = kv_pool
        self.recurrent = state_pool
        self.cache = cache


    def can_admit(self, ids: list[int], reserve: int = 0) -> bool:
        if self.cache is None:
            return self.kv.free_count >= len(ids) + reserve and self.recurrent.free_count >= 1
        # Obtain a private slot before pinning a path that might otherwise own
        # every state slot. Budget only the uncached suffix and decode reserve.
        if not self.cache.restore_async(ids[:-1], reserve + 1):  # a stored prefix comes back before it is matched
            return False  # its reads are still landing; the engine serves everyone else meanwhile
        self.cache.evict(need_slots=1)
        match = self.cache.match(ids[:-1], touch=False)
        self.cache.lock(match.node, 1)
        try:
            need = len(ids) - match.length + reserve
            self.cache.evict(need_pages=need)
            return self.kv.free_count >= need and self.recurrent.free_count >= 1
        finally:
            self.cache.lock(match.node, -1)

    def forget(self, ids: list[int]) -> None:
        """A queued request is gone: a restore holding its path lets go."""
        if self.cache is not None:
            self.cache.release(ids[:-1])

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
                self.cache.restore(ids[:-1])
                self.cache.evict(need_slots=1)
                match = self.cache.match(ids[:-1])  # last token must produce logits
                if match.length:
                    self.cache.lock(match.node, 1)
                else:
                    match = None
                self.cache.release(ids[:-1])  # a restore that held this path for the request: the match holds it now
            prefix = match.length if match is not None else 0
            if self.cache is not None:
                self.cache.evict(need_pages=len(ids) - prefix, need_slots=1)
            slot = self.recurrent.alloc()
            if match is not None:
                self.recurrent.copy_slot(match.snapshot_slot, slot)
            own, own_ids = self.kv.alloc_with_ids(len(ids) - prefix)
            pages = torch.cat([match.pages, own]) if match is not None else own
        except Exception:
            if own is not None:
                self.kv.free_ids(own_ids)
            if slot >= 0:
                self.recurrent.free(slot)
            if match is not None:
                self.cache.lock(match.node, -1)
            raise
        req.match, req.slot, req.pages = match, slot, pages
        req.page_ids = list(own_ids)
        self.live += 1
        req.cache_enabled = self.cache is not None
        req.prefix_len = req.committed_len = req.prefill_pos = prefix

    def checkpoint(self, req: Request, position: int, *, copy: bool = False) -> int | None:
        if self.cache is None:
            return None
        if position in req.checkpoints:
            return req.checkpoints[position]
        keep = max(0, self.admission_slots - self.live)  # slots the next admissions need
        if self.recurrent.free_count <= keep:
            self.cache.evict(need_slots=keep + 1)
        if self.recurrent.free_count <= keep:
            # Only recycle this request's completed snapshots. Positions beyond
            # committed_len are destinations of the upcoming forward; reusing
            # them would alias two kernel writes. Cache loans are never ours.
            candidates = [p for p in req.checkpoints if p <= req.committed_len and p < position]
            if copy and candidates:
                # An exact prompt-end copy must not displace the last interior
                # boundary needed by an identical retry (which needs logits).
                candidates = [p for p in candidates if p != max(req.checkpoints)]
            if not candidates:
                return None
            oldest = min(candidates)
            self.recurrent.free(req.checkpoints.pop(oldest))
            req.cache_auxiliary.pop(oldest, None)
        slot = self.recurrent.alloc()
        req.checkpoints[position] = slot
        if copy:
            self.recurrent.copy_slot(req.slot, slot)
        return slot

    def prefill_checkpoints(self, boundaries: list[tuple[Request, list[int]]]) -> list[tuple[int, int, int]]:
        """Reserve (row, position, slot), giving each row its latest boundary first.

        Then fill earlier intervals round-robin. The fixed pool bounds retention;
        allocation cannot let one long prompt consume every row's endpoint.
        """
        priorities = [positions[-1:] + positions[:-1] for _, positions in boundaries]
        planned = []
        for rank in range(max(map(len, priorities), default=0)):
            for row, ((req, _), positions) in enumerate(zip(boundaries, priorities)):
                if rank < len(positions):
                    pos = positions[rank]
                    slot = self.checkpoint(req, pos)
                    if slot is not None:
                        planned.append((row, pos, slot))
        return planned

    def commit_prefill(self, req: Request, end: int) -> None:
        if not req.committed_len <= end <= len(req.pages):
            raise ValueError("prefill commit outside owned pages")
        req.prefill_pos = req.committed_len = end

    def cache_prompt(self, req: Request) -> None:
        """With an SSD tier, hand the cache the prompt's whole blocks as soon as the prompt is prefilled, locked for the
        request, and queue them for backup: the tier writes them while the request decodes instead of after it
        finishes, so eviction finds them stored. A prompt's KV is final once it is prefilled, a drafter's stored pages
        included (`stored_pages`): decoding writes only the positions after it."""
        end = self.cache.backup_span(req.prefill_len) if self.cache is not None and req.cacheable else 0
        if end <= req.prefix_len or req.prompt_node is not None:
            return
        ids = req.prefill_ids[:end]
        owned = [-1] * req.prefix_len + req.page_ids
        new_start = min(self.cache.insert(ids, req.pages[:end], {}, page_ids=owned[:end]), end)
        req.page_ids[new_start - req.prefix_len:end - req.prefix_len] = [-1] * (end - new_start)  # the cache's from now on
        req.prompt_node = self.cache.node_at(ids)
        self.cache.lock(req.prompt_node, 1)
        self.cache.backup(ids)

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

    def commit(self, req: Request, count: int, *, ssm=None, conv=None, aux: dict | None = None) -> None:
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
            for name, value in (aux or {}).items():
                self.recurrent.aux[name][req.slot].copy_(value)
        self._publish(req, count, pages)

    def _validate_reconstruction(self, requests, replay, conv_windows, verified_steps: int, counts=None) -> list[int]:
        """The batch-wide checks shared by the host and device commit paths; returns the slots."""
        batch = len(requests)
        if (not 1 <= batch <= replay.batch or (counts is not None and len(counts) != batch)
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
        for row, req in enumerate(requests):
            pending = req.reservation
            if (req.finished or req.released or req.pages is None or len(req.pages) != req.committed_len
                    or pending is None or len(pending.ids) < verified_steps or conv_windows.shape[2] < verified_steps):
                raise ValueError("reconstruction commit outside reservation")
            if counts is not None:
                count = counts[row]
                if type(count) is not int or not 1 <= count <= min(len(pending.ids), verified_steps, conv_windows.shape[2]):
                    raise ValueError("reconstruction commit outside reservation")
        return slots

    def commit_reconstructed(self, requests, counts, replay, conv_windows, *, verified_steps: int, aux_windows: dict | None = None) -> None:
        """Validate the whole batch before allowing reconstruction into private slots.

        Prepare retained page tensors before mutation. A failed in-place commit
        poisons every affected request; rollback can free pages but cannot undo
        recurrent writes. Publication and proposer feedback follow all writes.
        """
        prof = self.commit_profile  # optional sub-phase timing (synchronizes; benchmarks only)
        lap = _Lap(prof, conv_device=self.recurrent.conv_all.device) if prof is not None else None
        slots = self._validate_reconstruction(requests, replay, conv_windows, verified_steps, counts)
        conv = self.recurrent.conv_all
        pages = self._retained_pages(requests, counts)
        lap and lap("pages")
        # Two aligned rows: Triton specializes kernels on pointer alignment, and a [2, B] row view
        # of counts sits at 4*B bytes, which recompiled the reconstruction per batch size.
        metadata = torch.zeros((2, (len(slots) + 3) // 4 * 4), dtype=torch.int32, pin_memory=conv.device.type == "cuda")
        metadata[0, :len(slots)] = torch.tensor(slots, dtype=torch.int32)
        metadata[1, :len(counts)] = torch.tensor(counts, dtype=torch.int32)
        metadata = metadata.to(conv.device, non_blocking=True)
        lap and lap("metadata")
        try:
            replay.run(self.recurrent, metadata[0, :len(slots)], metadata[1, :len(counts)])
            lap and lap("replay")
            for row, (req, count) in enumerate(zip(requests, counts)):
                conv[:, req.slot].copy_(conv_windows[:, row, count - 1])
                for name, win in (aux_windows or {}).items():
                    self.recurrent.aux[name][req.slot].copy_(win[row, count - 1])
            lap and lap("windows")
            for req, count, retained in zip(requests, counts, pages):
                self._publish(req, count, retained)
            lap and lap("publish")
        except BaseException as exc:
            self.poison(requests, exc)

    def authorize_reconstruction(self, requests, replay, conv_windows, *, verified_steps: int) -> list[int]:
        """Validate a batch whose counts the device will decide (`engine/accept_graphs.py`).

        Every count the device can produce lies in [0, verified_steps]; the
        reservation covers verified_steps pages and the windows verified_steps
        rows, so the only host checks left are ownership and uniqueness, made
        before any in-place write. Returns the private slots in batch order.
        """
        return self._validate_reconstruction(requests, replay, conv_windows, verified_steps)

    def publish_reconstructed(self, requests, counts) -> None:
        """Publish counts the device decided after `authorize_reconstruction` and the writes landed."""
        for req, count in zip(requests, counts):
            pending = req.reservation
            if pending is None or type(count) is not int or not 1 <= count <= len(pending.ids):
                raise ValueError("reconstruction commit outside reservation")
        for req, count, retained in zip(requests, counts, self._retained_pages(requests, counts)):
            self._publish(req, count, retained)

    def poison(self, requests, exc: BaseException):
        """A failure after in-place recurrent writes: nothing in the batch may be cached or retried."""
        for req in requests:
            req.cacheable = False
        if isinstance(exc, MemoryError):
            # The scheduler retries pool-pressure MemoryError. A partially
            # written recurrent slot requires failure cleanup instead.
            raise RuntimeError("reconstruction failed after state mutation") from exc
        raise exc

    def _retained_pages(self, requests, counts) -> list[torch.Tensor]:
        return [r.reservation.full_pages if count == len(r.reservation.ids) and r.reservation.full_pages is not None
                else torch.cat([r.pages, r.reservation.pages[:count]]) for r, count in zip(requests, counts)]

    def _publish(self, req, count, pages) -> None:
        self.kv.free_ids(req.reservation.ids[count:])
        req.page_ids += req.reservation.ids[:count]
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
        if req.slot >= 0:
            self.live -= 1
        if req.match is not None:
            self.cache.lock(req.match.node, -1)
            req.match = None
        if req.prompt_node is not None:
            self.cache.lock(req.prompt_node, -1)
            req.prompt_node = None
        if not cached:
            if req.pages is not None:
                self.kv.free_ids([p for p in req.page_ids if p >= 0])  # host ids: no device read-back
            if req.slot >= 0:
                self.recurrent.free(req.slot)
            for slot in req.checkpoints.values():
                self.recurrent.free(slot)
        req.pages, req.slot, req.checkpoints = None, -1, {}
        req.page_ids = []
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
        ids_full = [-1] * req.prefix_len + req.page_ids  # -1: cache-owned pages (never freed here)
        new_start = self.cache.insert(ids, req.pages[:n], snapshots, req.cache_auxiliary, page_ids=ids_full[:n])
        if new_start > req.prefix_len:
            self.kv.free_ids([p for p in ids_full[req.prefix_len:new_start] if p >= 0])
        if len(req.pages) > n:
            self.kv.free_ids(ids_full[n:])
        req.page_ids = []  # ownership moved to the cache
        # With an SSD tier, back up its full KV blocks and its resume points: its end and, if snapshotted, its prompt's end.
        prompt_end = req.prefill_len if req.prefill_len in snapshots else n
        self.cache.backup(ids, (n, prompt_end))
        return True

    def flush(self) -> None:
        if self.cache is not None:
            self.cache.flush()
