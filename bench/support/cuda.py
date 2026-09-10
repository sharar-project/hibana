"""Real-operand hashing and CUDA graph timings for matrix experiments."""
import hashlib
import statistics
import torch
from sharar.kernels import ops

def tensor_hash(x):
    return hashlib.sha256(x.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def capture_graph(fn):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    return graph, output


def time_graph(graph, flush=None, samples=9, repeats=10):
    values = []
    for _ in range(samples):
        if flush is not None:
            flush.add_(1)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(1 if flush is not None else repeats):
            graph.replay()
        b.record()
        b.synchronize()
        values.append(a.elapsed_time(b) / (1 if flush is not None else repeats))
    return {'samples_ms': values, 'median_ms': statistics.median(values)}


def gemm(q, sf, lin, backend=None):
    return ops._mm_fp4(q, lin.weight.T, sf, lin.weight_scale_swizzled.T,
                       lin.alpha, torch.bfloat16,
                       backend=backend or ops.fp4_backend(len(q), lin.out_features, lin.in_features))
