"""Stage-level allocator peaks and 20 ms host MemAvailable samples.

Host deltas include other processes and reclaimable mappings. They are sampled
observations, not an attribution of physical traffic or a hard peak guarantee.
"""
from contextlib import contextmanager
import threading
import time

import torch

from sharar.memory.ledger import mem_available_bytes


@contextmanager
def measure_stage(report, name):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = mem_available_bytes()
    samples = [(0., start)]
    stop = threading.Event()
    began = time.monotonic()
    def sample():
        while not stop.wait(.02):
            samples.append((time.monotonic() - began, mem_available_bytes()))
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        yield
        torch.cuda.synchronize()
    finally:
        stop.set()
        thread.join()
        samples.append((time.monotonic() - began, mem_available_bytes()))
        report[name] = {'elapsed_s': time.monotonic() - began, 'start_available_bytes': start,
                        'minimum_available_bytes': min(m for _, m in samples),
                        'net_peak_available_drop_bytes': start - min(m for _, m in samples),
                        'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                        'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                        'cuda_allocated_bytes': torch.cuda.memory_allocated(),
                        'cuda_reserved_bytes': torch.cuda.memory_reserved(), 'host_samples': samples}
