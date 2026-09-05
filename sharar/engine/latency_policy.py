"""Bounded feedback policy for prefill work between visible decode bursts.

This is a latency target, not a hard real-time guarantee: GPU work cannot be
preempted. Keep observations bounded and never starve a queued prompt.
"""
from collections import deque


class LatencyPolicy:
    def __init__(self, target_ms):
        self.target_ms = target_ms
        self.prefills = deque(maxlen=32)
        self.decodes = deque(maxlen=32)

    def prefill_limit(self, configured):
        # Start with one 64-token recurrent chunk until this process observes
        # its own costs. Use the slowest recent rate, plus the decode cost.
        if not self.prefills:
            return min(configured, 64)
        rate = max(ms / tokens for tokens, ms in self.prefills)
        decode = max(self.decodes, default=0)
        room = max(0, self.target_ms - decode)
        limit = max(64, int(room / max(rate, 1e-6)) // 64 * 64)
        # Bounded experiments: never exceed 512 active-step prefill tokens.
        return min(configured, limit, 512)

    def observe_prefill(self, tokens, milliseconds):
        if tokens > 0:
            self.prefills.append((tokens, milliseconds))

    def observe_decode(self, milliseconds):
        self.decodes.append(milliseconds)
