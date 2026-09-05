"""Compare per-forward GDN planning with frozen pre-change source.

The reference directory contains legacy_{gdn,chunk,chunk_o,chunk_delta_h}.py
exported from d319896. Only the old prefill core/helpers are substituted; all
other runtime code and allocations remain the same. Timings exclude profiling.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

import torch

from bench import local_workloads as workloads


def load_reference(root):
    modules = {}
    for name in ('chunk_o', 'chunk_delta_h', 'chunk', 'gdn'):
        spec = importlib.util.spec_from_file_location('legacy_' + name, root / ('legacy_' + name + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    modules['chunk'].chunk_fwd_o = modules['chunk_o'].chunk_fwd_o
    modules['chunk'].chunk_gated_delta_rule_fwd_h = modules['chunk_delta_h'].chunk_gated_delta_rule_fwd_h
    modules['gdn'].chunk_gated_delta_rule = modules['chunk'].chunk_gated_delta_rule
    return modules['gdn'].GatedDeltaNet._prefill_core


@contextmanager
def implementation(legacy, mode, captures=None):
    from sharar.model import gdn, qwen38
    original, planner = gdn.GatedDeltaNet._prefill_core, qwen38.prepare_gdn_prefill
    selected = legacy if mode == 'legacy' else original
    def digest(tensor):
        return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy()).hexdigest()
    def core(self, qkv, b, a, ctx):
        out = selected(self, qkv, b, a, ctx)
        if captures is not None:
            slots = (ctx.seq_slots.tolist() if ctx.batched else [ctx.slot]) + ctx.checkpoint_slots
            captures.append({'layer': self.layer_id, 'core': digest(out),
                             'conv': digest(ctx.state_pool.conv[self.layer_id][slots]),
                             'ssm': digest(ctx.state_pool.ssm[self.layer_id][slots])})
        return out
    gdn.GatedDeltaNet._prefill_core = core
    if mode == 'legacy':
        qwen38.prepare_gdn_prefill = lambda *args: None
    try:
        yield
    finally:
        gdn.GatedDeltaNet._prefill_core, qwen38.prepare_gdn_prefill = original, planner


@torch.inference_mode()
def main():
    from bench.phase7c import http_server
    from bench.gb10_profile import trace_summary
    from sharar import config
    from sharar.engine.executor import Executor
    from sharar.engine.sampler import SamplingParams
    from sharar.model.qwen38 import Qwen38
    from sharar.tokenizer import SnapshotTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    legacy = load_reference(args.reference)
    frozen = workloads.materialize(workloads.DEFAULT_MANIFEST, config.snapshot_dir())
    tok = SnapshotTokenizer(config.snapshot_dir())
    ex = Executor(Qwen38(config.snapshot_dir()), num_pages=35000, num_slots=12)
    ex.enable_spec(7, proposer='dflash2', state_strategy='reconstruct', log=print)
    ex.capture_graphs(log=print)
    ex.warmup(log=print)
    report = {'baseline': 'd319896329a12035cc5f41aae63b06bd9c2763da', 'target': config.MODEL_REVISION,
              'reference_sha256': {p.name: workloads.digest(p.read_bytes()) for p in args.reference.glob('legacy_*.py')},
              'exact': [], 'http': [], 'traces': {}, 'frozen': frozen}
    def save():
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    # Fixed grouping and checkpoint history; hash every GDN core and all owned
    # conv/SSM states outside timing, including intermediate checkpoints.
    ids = tok.encode('A local inference engine processes requests. ' * 128)
    for lengths, prefix in [([1], 0), ([65], 0), ([129], 0), ([3, 65, 129], 0), ([129], 128), ([65, 129], 128)]:
        snapshots = {}
        for mode in ('legacy', 'plan'):
            ex.flush_cache()
            spec, ex.spec = ex.spec, None
            reqs = [ex.admit(ids[:n + prefix], SamplingParams(max_new_tokens=8)) for n in lengths]
            try:
                with implementation(legacy, mode):
                    if prefix:
                        ex.prefill_batch([(r, prefix) for r in reqs])
                captures = []
                with implementation(legacy, mode, captures):
                    ex.prefill_batch(list(zip(reqs, lengths)))
                snapshots[mode] = {'layers': captures, 'tokens': [r.output_ids[:] for r in reqs]}
            finally:
                for r in reqs:
                    r.cacheable = False
                    ex.release(r)
                ex.flush_cache()
                ex.spec = spec
        row = {'lengths': lengths, 'prefix': prefix, 'equal': snapshots['legacy'] == snapshots['plan'], 'snapshots': snapshots}
        report['exact'].append(row)
        save()
        assert row['equal'], row
    selected = [(c, cache) for c in frozen['cases'] if c['id'] in ('chat', 'coding', 'session')
                for cache in (('cold', 'session') if c['id'] == 'session' else ('cold',))]
    with http_server(ex, tok, config.snapshot_dir(), {'profile': 'prefill-plan-v1'}) as url:
        for case, cache in selected:
            for block in range(-1, 3):
                for position, mode in enumerate(('legacy', 'plan', 'plan', 'legacy')):
                    workloads.await_idle(url)
                    ex.flush_cache()
                    if cache == 'session':
                        prime = workloads.stream_request(url, case, 'prime', ids=case['prime_ids'],
                                     sampling={'max_new_tokens': 1, 'temperature': 0})
                        assert not prime['error']
                        workloads.await_idle(url)
                    with implementation(legacy, mode):
                        row = workloads.stream_request(url, case, f'{case["id"]}-{cache}-{block}-{position}')
                        workloads.await_idle(url)
                    row.update(mode=mode, cache=cache, block=block, warmup=block < 0,
                               text=tok.decode([t for t in row['output_ids'] if t not in (248046, 248044)]))
                    row['quality'] = workloads.quality(case, row['text'])
                    report['http'].append(row)
                    save()
                    assert not row['error'] and row['quality']['passed'] is not False, row
    # Instrument one warmed ordinary prefill to count eliminated host waits.
    for mode in ('legacy', 'plan'):
        ex.flush_cache()
        req = ex.admit(ids[:129], SamplingParams(max_new_tokens=1))
        try:
            torch.cuda.synchronize()
            with implementation(legacy, mode), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                                                 torch.profiler.ProfilerActivity.CUDA]) as prof:
                with torch.profiler.record_function('sharar.decode_step'):
                    ex.prefill_batch([(req, 129)])
                    torch.cuda.synchronize()
            path = args.output / (mode + '-trace.json')
            prof.export_chrome_trace(str(path))
            trace = json.loads(path.read_text())
            report['traces'][mode] = trace_summary(trace)
            report['traces'][mode]['stream_syncs'] = sum(e.get('name') == 'cudaStreamSynchronize' for e in trace['traceEvents'])
        finally:
            ex.release(req)
    report['summary'] = {}
    for case, cache in selected:
        samples = [r for r in report['http'] if r['case'] == case['id'] and r['cache'] == cache and not r['warmup']]
        values = {mode: {k: statistics.median(r[k] for r in samples if r['mode'] == mode)
                         for k in ('ttft_s', 'completion_s')} for mode in ('legacy', 'plan')}
        values['output_ids_equal'] = len({r['output_sha256'] for r in samples}) == 1
        report['summary'][case['id'] + '-' + cache] = values
    report['completed'] = True
    save()
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
