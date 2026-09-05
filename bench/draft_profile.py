"""Attribute external-draft cost on GB10, with separate uninstrumented timing.

Run in the exclusive GPU harness. CUDA event spans include device queue gaps;
they are not physical DRAM counters. Raw outputs make work changes visible.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import gzip
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

import torch

from sharar import config
from sharar.engine.executor import Executor
from sharar.engine.sampler import SamplingParams
from sharar.model.qwen38 import Qwen38
from sharar.parity.prompts import PARAGRAPH
from sharar.tokenizer import SnapshotTokenizer

from gb10_profile import trace_summary
import local_workloads
from memory_trace import measure_stage
from phase7c import http_server


@contextmanager
def measurements(ex, records):
    """Instrument only the diagnostic pass; never the speed comparison."""
    p, spec = ex.spec.proposer, ex.spec
    methods = [(p, 'propose', 'propose'), (p.graphs, 'run', 'draft_graph'),
               (spec, '_accept', 'accept'), (spec.verify_graphs, 'run', 'verify'),
               (p.model, 'project_features', 'feature_projection'),
               (p.model, 'context_kv', 'context_kv'),
               (ex.state, 'commit', 'commit')]
    if spec.replay is not None:
        methods.append((spec.replay, 'run', 'reconstruct'))
    originals = []
    for obj, method, label in methods:
        original = getattr(obj, method)
        originals.append((obj, method, original))
        def wrapped(*args, original=original, label=label, **kwargs):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            started = time.perf_counter()
            with torch.profiler.record_function('sharar.' + label):
                begin.record()
                result = original(*args, **kwargs)
                end.record()
            records[label].append((begin, end, (time.perf_counter() - started) * 1000))
            return result
        setattr(obj, method, wrapped)
    try:
        yield
    finally:
        for obj, method, original in originals:
            setattr(obj, method, original)


def graph_cost(fn, repetitions=20):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    samples = []
    for _ in range(3):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repetitions):
            graph.replay()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop) / repetitions)
    return {'samples_ms': samples, 'median_ms': statistics.median(samples)}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--name', choices=('dflash2', 'dspark'), default='dflash2')
    parser.add_argument('--workloads', action='store_true')
    parser.add_argument('--compare-proposer', type=Path,
                        help='alternate candidate/control/control/candidate fixed-work timing against this frozen source')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(config.__file__).resolve().parents[1]
    report = {'target': config.MODEL_REVISION, 'name': args.name, 'cells': [],
              'sources': {str(p.relative_to(root)): local_workloads.digest(p.read_bytes())
                          for folder in ('sharar', 'bench', 'tests') for p in (root / folder).rglob('*.py')},
              'torch': torch.__version__, 'device': torch.cuda.get_device_name(),
              'limitation': 'Three timing repetitions; event spans include queue gaps; no physical DRAM counters.'}
    def save():
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    try:
        with measure_stage(report, 'target_load'):
            target = Qwen38(config.snapshot_dir())
        torch.cuda.empty_cache()
        ex = Executor(target, num_pages=35000, num_slots=12)
        with measure_stage(report, 'draft_load'):
            ex.enable_spec(7, proposer=args.name, state_strategy='reconstruct', log=print)
        with measure_stage(report, 'graphs'):
            ex.capture_graphs(log=print)
        tok = SnapshotTokenizer(config.snapshot_dir())
        ids = tok.encode(PARAGRAPH * 256)
        cache, ex.cache = ex.cache, None
        current = ex.spec.proposer.propose
        control = None
        if args.compare_proposer:
            spec = importlib.util.spec_from_file_location('sharar_profile_control', args.compare_proposer)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            control = module.ParallelProposer.propose.__get__(ex.spec.proposer)
            report['control_proposer_sha256'] = local_workloads.digest(args.compare_proposer.read_bytes())
        for length, batch in ((128, 1), (128, 4), (8192, 1), (8192, 4)):
            def prepare():
                reqs = [ex.admit(ids[:length], SamplingParams(max_new_tokens=256, ignore_eos=True)) for _ in range(batch)]
                for req in reqs:
                    ex.prefill(req)
                return reqs
            def release(reqs):
                for req in reqs:
                    req.cacheable = False
                    ex.release(req)
            if control is not None:
                # Same warmed model/graphs, fresh requests and no prefix cache.
                # Reversed order limits drift; three ABBA repetitions are a pilot.
                compared = []
                for repeat in range(4):
                    for mode in ('candidate', 'control', 'control', 'candidate'):
                        ex.spec.proposer.propose = current if mode == 'candidate' else control
                        reqs = prepare()
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        for _ in range(8):
                            ex.decode_step(reqs)
                        torch.cuda.synchronize()
                        sample = {'mode': mode, 'repeat': repeat,
                                  'ms_per_round': (time.perf_counter() - started) * 1000 / 8,
                                  'outputs': [r.output_ids for r in reqs]}
                        if repeat:
                            compared.append(sample)
                        release(reqs)
                assert all(s['outputs'] == compared[0]['outputs'] for s in compared)
                row = {'context': length, 'batch': batch, 'samples': compared,
                       'medians_ms': {mode: statistics.median(s['ms_per_round'] for s in compared if s['mode'] == mode)
                                      for mode in ('control', 'candidate')}, 'outputs_equal': True}
                report['cells'].append(row)
                save()
                print(json.dumps({k: row[k] for k in ('context', 'batch', 'medians_ms', 'outputs_equal')}), flush=True)
                ex.ledger.check_floor()
                continue
            samples = []
            for repeat in range(4):
                reqs = prepare()
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(8):
                    ex.decode_step(reqs)
                torch.cuda.synchronize()
                sample = {'ms_per_round': (time.perf_counter() - start) * 1000 / 8,
                          'outputs': [r.output_ids for r in reqs]}
                if repeat:
                    samples.append(sample)
                release(reqs)
            reqs = prepare()
            events = defaultdict(list)
            with measurements(ex, events):
                for _ in range(8):
                    ex.decode_step(reqs)
            torch.cuda.synchronize()
            phases = {label: {'calls_per_round': len(rows) / 8,
                              'event_ms_per_round': sum(a.elapsed_time(b) for a, b, _ in rows) / 8,
                              'host_ms_per_round': sum(ms for _, _, ms in rows) / 8}
                      for label, rows in events.items()}
            release(reqs)
            p = ex.spec.proposer
            st = p.graphs.static[batch]
            hidden = st['hidden'].clone()
            components = {'vocab_projection': graph_cost(lambda: target.logits(hidden.reshape(-1, 5120)))}
            if args.name == 'dflash2':
                logits = target.logits(hidden.reshape(-1, 5120)).view(batch, 7, -1)
                components['topk'] = graph_cost(lambda: logits.topk(p.model.cfg['dflash_config']['selector_top_k'], dim=-1))
                components['selector_total'] = graph_cost(lambda: p.model.selector(hidden, st['tokens'][:, 0]))
            reqs = prepare()
            with measurements(ex, defaultdict(list)), torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(2):
                    with torch.profiler.record_function('sharar.decode_step'):
                        ex.decode_step(reqs)
                        torch.cuda.synchronize()
            path = args.output / f't{length}-b{batch}.json'
            prof.export_chrome_trace(str(path))
            trace = json.loads(path.read_text())
            with gzip.open(str(path) + '.gz', 'wt') as out:
                json.dump(trace, out)
            path.unlink()
            release(reqs)
            row = {'context': length, 'batch': batch, 'samples': samples,
                   'median_ms_per_round': statistics.median(s['ms_per_round'] for s in samples),
                   'phases': phases, 'components': components, 'trace': trace_summary(trace),
                   'allocated_gib': torch.cuda.memory_allocated() / 2**30}
            report['cells'].append(row)
            save()
            print(json.dumps({k: row[k] for k in ('context', 'batch', 'median_ms_per_round', 'phases', 'components')}), flush=True)
            ex.ledger.check_floor()
        ex.cache = cache
        ex.spec.proposer.propose = current
        if args.workloads:
            frozen = local_workloads.materialize(local_workloads.DEFAULT_MANIFEST, config.snapshot_dir())
            with http_server(ex, tok, config.snapshot_dir(), {'experiment': 'draft-profile'}) as url:
                report['workloads'] = local_workloads.run(url, frozen, tok, args.output / 'workloads')
            assert not report['workloads']['request_failures'] and not report['workloads']['automatic_failures']
        report['passed'] = True
    finally:
        save()


if __name__ == '__main__':
    main()
