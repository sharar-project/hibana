"""Experimental hot-resume scheduling, evaluated separately from retention.

Uses CPU-only radix inspection to estimate suffix work. Only the selected
request may invoke admission's eviction/restore path. Aging bounds queue
preference; no request is preempted and long prefills continue making progress.
"""
import time
from dataclasses import replace

from sharar.engine.scheduler import Scheduler


def prefix_length(cache, ids):
    if cache is None or cache.flushing:
        return 0
    node, position, best = cache.root, 0, 0
    while position < len(ids):
        child = node.children.get(ids[position])
        if child is None or ids[position:position + len(child.key)] != child.key:
            break
        position += len(child.key)
        if child.snapshot_slot is not None:
            best = position
        node = child
    return best


class AgentScheduler(Scheduler):
    def __init__(self, executor, cfg=None, on_progress=None, *, short_tokens=512, max_wait_s=2.0):
        super().__init__(executor, cfg, on_progress)
        self.short_tokens, self.max_wait_s = short_tokens, max_wait_s
        self.resume_bypasses = self.short_prefill_steps = 0

    def _admit(self):
        while self.queue and len(self.prefilling) + len(self.running) < self.cfg.max_running:
            now = time.perf_counter()
            estimates = {}
            for req in self.queue:
                prefix = prefix_length(self.ex.cache, req.prefill_ids[:-1])
                remaining = req.prefill_len - prefix
                short = prefix > 0 and remaining <= self.short_tokens
                aged = now - req.t_arrival >= self.max_wait_s
                estimates[req] = (0 if aged else 1 if short else 2, remaining)
            ranked = sorted(self.queue, key=lambda req: (estimates[req][0], req.t_arrival))
            req = ranked[0]
            reserve = self.ex.admission_reserve(req, self.cfg.reserve_tokens)
            # If the preferred request needs reclamation, admit a request that
            # fits free capacity first. Avoid speculative eviction while probing
            # every queue entry; that would destroy the locality being valued.
            if (estimates[req][0] != 0 and self.ex.state_pool.free_count and
                    self.ex.kv_pool.free_count < estimates[req][1] + reserve):
                fits = [r for r in ranked if self.ex.kv_pool.free_count >= estimates[r][1]
                        + self.ex.admission_reserve(r, self.cfg.reserve_tokens)]
                if fits:
                    req = fits[0]
                    reserve = self.ex.admission_reserve(req, self.cfg.reserve_tokens)
            if not self.ex.can_admit(req.prefill_ids, reserve):
                break
            if req is not self.queue[0]:
                self.resume_bypasses += 1
            self.ex.admit(req.prompt_ids, req.params, req)
            self.queue.remove(req)
            self.prefilling.append(req)
            self.metrics.admitted += 1

    def _prefill_step(self):
        now = time.perf_counter()
        def short(req):
            return req.prefix_len > 0 and req.prefill_len - req.prefill_pos <= self.short_tokens
        self.prefilling.sort(key=lambda req: (0 if now - req.t_arrival >= self.max_wait_s else 1 if short(req) else 2,
                                              req.t_arrival))
        cfg = self.cfg
        if any(short(req) for req in self.prefilling):
            self.cfg = replace(cfg, chunk_size=min(cfg.chunk_size, self.short_tokens))
            self.short_prefill_steps += 1
        try:
            super()._prefill_step()
        finally:
            self.cfg = cfg

    def stats(self):
        return {**super().stats(), 'resume_bypasses': self.resume_bypasses,
                'short_prefill_steps': self.short_prefill_steps}
