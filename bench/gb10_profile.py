"""Unprofiled round timing plus CUDA/CPU traces at fixed GB10 shapes.

Kernel interval union measures device activity, not physical DRAM traffic.
Profiler timings are diagnostic and kept separate from HTTP latency samples.
"""

from collections import Counter
from contextlib import contextmanager
import gzip
import json
from pathlib import Path
import statistics
import time


def interval_union(intervals):
    total, end = 0.0, float('-inf')
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def trace_summary(trace):
    events = trace['traceEvents']
    spans = [e for e in events if e.get('name') == 'sharar.decode_step' and e.get('ph') == 'X']
    if not spans:
        raise ValueError('missing decode scopes in trace')
    start = min(e['ts'] for e in spans)
    stop = max(e['ts'] + e['dur'] for e in spans)
    kernels = [e for e in events if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')
               and e.get('ph') == 'X' and e['ts'] < stop and e['ts'] + e['dur'] > start]
    intervals = [(max(start, e['ts']), min(stop, e['ts'] + e['dur'])) for e in kernels]
    wall = stop - start
    busy = interval_union(intervals)
    runtime = [e for e in events if e.get('cat') == 'cuda_runtime' and e.get('ph') == 'X'
               and start <= e['ts'] <= stop]
    names, times = Counter(), Counter()
    for e in kernels:
        names[e['name']] += 1
        times[e['name']] += e['dur']
    return {'profiled_span_ms': wall / 1000, 'gpu_activity_union_ms': busy / 1000,
            'device_idle_fraction': (wall - busy) / wall, 'gpu_event_count': len(kernels),
            'runtime_sync_calls': {name: sum(e['name'] == name for e in runtime)
                                   for name in sorted({e['name'] for e in runtime if 'Synchronize' in e['name']})},
            'top_kernels': [{'name': name, 'sum_ms': us / 1000, 'calls': names[name]}
                            for name, us in times.most_common(15)],
            'physical_dram_bytes': None,
            'limitation': 'Instrumented queue gaps include Python and profiler overhead. Kernel duration sums may overlap; union is used for activity. No DRAM hardware counters.'}


def kernel_breakdown(trace):
    """Attribute kernels through CUPTI launch correlation and nested CPU scopes.

CUDA graph kernels share their replay launch correlation, so their delayed GPU
timestamps must not be compared directly with CPU method-entry timestamps.
"""
    events = trace['traceEvents']
    scopes = [e for e in events if e.get('ph') == 'X' and e.get('name', '').startswith('sharar.')]
    launches = {e['args']['correlation']: e for e in events
                if e.get('cat') in ('cuda_runtime', 'cuda_driver') and 'correlation' in e.get('args', {})}
    totals, counts = Counter(), Counter()
    all_ms = 0.
    for event in events:
        if event.get('cat') != 'kernel' or event.get('ph') != 'X':
            continue
        launch = launches.get(event.get('args', {}).get('correlation'))
        parents = [scope for scope in scopes if launch and scope.get('tid') == launch.get('tid')
                   and scope['ts'] <= launch['ts'] < scope['ts'] + scope['dur']]
        owner = min(parents, key=lambda e: e['dur'])['name'] if parents else 'unattributed'
        name = event['name'].lower()
        if 'blockscaled' in name or 'block_scaled' in name:
            family = 'nvfp4_gemm'
        elif 'nvjet' in name or 'gemm' in name:
            family = 'other_gemm'
        elif 'prefillwith' in name or 'attention' in name:
            family = 'attention'
        else:
            family = 'other'
        key = (owner, family)
        totals[key] += event['dur'] / 1000
        counts[key] += 1
        all_ms += event['dur'] / 1000
    return {'kernel_sum_ms': all_ms,
            'groups': [{'owner': owner, 'family': family, 'kernel_ms': value, 'calls': counts[owner, family]}
                       for (owner, family), value in totals.most_common()],
            'limitation': 'Diagnostic kernel-duration sums, not additive phase spans or measured DRAM traffic. Other GEMM includes FP8 and small BF16 projections.'}


@contextmanager
def phase_scopes(ex):
    import torch
    originals = []
    if ex.spec is not None:
        for obj, method, label in ((ex.spec.proposer, 'propose', 'sharar.propose'),
                                   (ex.spec, '_accept', 'sharar.accept'),
                                   (ex.spec.verify_graphs, 'run', 'sharar.verify')):
            if obj is None:
                continue
            original = getattr(obj, method)
            originals.append((obj, method, original))
            def wrapped(*args, original=original, label=label, **kwargs):
                with torch.profiler.record_function(label):
                    return original(*args, **kwargs)
            setattr(obj, method, wrapped)
    try:
        yield
    finally:
        for obj, method, original in originals:
            setattr(obj, method, original)


def run(ex, tok, output):
    import torch
    from sharar.engine.sampler import SamplingParams
    from sharar.parity.prompts import PARAGRAPH
    from sharar.memory.ledger import mem_available_bytes
    output = Path(output)
    output.mkdir()
    ids = tok.encode(PARAGRAPH * 256)
    spec, cache = ex.spec, ex.cache
    ex.flush_cache()
    ex.cache = None
    rows = []
    try:
        for mode in ('plain', spec.proposer.requirements.name):
            ex.spec = spec if mode != 'plain' else None
            for length, batch in ((128, 1), (128, 4), (8192, 1), (8192, 4)):
                owned = []
                def prepare():
                    reqs = []
                    for _ in range(batch):
                        r = ex.admit(ids[:length], SamplingParams(max_new_tokens=128, ignore_eos=True))
                        owned.append(r)
                        ex.prefill(r)
                        reqs.append(r)
                    return reqs
                def release(reqs):
                    for req in reqs:
                        req.cacheable = False
                        ex.release(req)
                try:
                    unprofiled = []
                    # Independent contexts, identical token prefix and number
                    # of rounds. Emitted IDs/counts expose work differences.
                    for _ in range(3):
                        reqs = prepare()
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        for _ in range(4):
                            ex.decode_step(reqs)
                        torch.cuda.synchronize()
                        unprofiled.append({'ms_per_step': (time.perf_counter() - started) * 1000 / 4,
                                           'emitted_ids': [r.output_ids[1:] for r in reqs]})
                        release(reqs)
                    reqs = prepare()
                    torch.cuda.synchronize()
                    with phase_scopes(ex), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                                             torch.profiler.ProfilerActivity.CUDA]) as prof:
                        for _ in range(4):
                            with torch.profiler.record_function('sharar.decode_step'):
                                ex.decode_step(reqs)
                                # End scope after device work, including the
                                # state commit queued after host acceptance.
                                torch.cuda.synchronize()
                    path = output / f'{mode}-t{length}-b{batch}.json'
                    prof.export_chrome_trace(str(path))
                    trace = json.loads(path.read_text())
                    summary = trace_summary(trace)
                    if not summary['gpu_event_count']:
                        raise RuntimeError('CUDA trace is empty; cannot infer device headroom')
                    with gzip.open(str(path) + '.gz', 'wt') as out:
                        json.dump(trace, out)
                    path.unlink()
                    release(reqs)
                    row = {'mode': mode, 'input_tokens': length, 'batch': batch, 'steps': 4,
                           'unprofiled': unprofiled, 'median_unprofiled_ms_per_step': statistics.median(s['ms_per_step'] for s in unprofiled),
                           'trace': summary, 'available_gib': mem_available_bytes() / 2**30,
                           'allocated_gib': torch.cuda.memory_allocated() / 2**30,
                           'verify_ssm_shape': list(spec.inter_ssm_all.shape) if spec.inter_ssm_all is not None else None,
                           'verify_state_strategy': spec.state_strategy,
                           'verify_buffer_bytes': spec.bytes,
                           'verify_ssm_bytes': spec.inter_ssm_all.numel() * spec.inter_ssm_all.element_size() if spec.inter_ssm_all is not None else 0,
                           'verify_conv_bytes': spec.conv_windows_all.numel() * spec.conv_windows_all.element_size()}
                    rows.append(row)
                    (output / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
                    print(json.dumps({k: row[k] for k in ('mode', 'input_tokens', 'batch', 'median_unprofiled_ms_per_step')}), flush=True)
                    ex.ledger.check_floor()
                finally:
                    release(owned)
    finally:
        ex.spec, ex.cache = spec, cache
    return rows
