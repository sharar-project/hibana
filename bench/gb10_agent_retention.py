"""Paired cold controls and interleaved checkpoint-retention experiments.

Run each variant from a frozen source in a fresh guarded GB10 process. Warm
probes are compared with an immediate-hit oracle at the same prefill partition;
retain all output IDs, fallback counts, feature bytes and total durations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from bench.gb10_bs4_engine import Recorder, native_batch, warm_feedback_shapes, work_summary
from bench.gb10_family_eval import clear, identity, save, sha_file


def batch(ex, tasks, count, *, retain=True):
    import torch
    from sharar.engine.sampler import SamplingParams
    proposer = ex.spec.proposer
    hits, fallbacks = proposer.cache_hits, proposer.cache_fallbacks
    before = ex.cache_stats()
    reqs = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    with Recorder(ex) as record:
        try:
            for i, task in enumerate(tasks):
                req = ex.admit(task['input_ids'], SamplingParams(max_new_tokens=count, ignore_eos=True))
                req.request_id = i + 1
                reqs.append(req)
            for req in reqs:
                while not req.prefill_done:
                    ex.prefill_batch([(req, min(2048, req.prefill_len - req.prefill_pos))])
            torch.cuda.synchronize()
            prefilled = time.perf_counter()
            ready = [proposer.ready(req) for req in reqs]
            while active := [r for r in reqs if not r.finished]:
                ex.decode_step(active)
            torch.cuda.synchronize()
            finished = time.perf_counter()
            outputs = [{'id': task['id'], 'output_ids': req.output_ids[:], 'cached_tokens': req.prefix_len,
                        'spec_rounds': req.spec_rounds, 'spec_accepted': req.spec_accepted,
                        'finish_reason': req.finish_reason} for task, req in zip(tasks, reqs)]
        finally:
            for req in reqs:
                req.cacheable = retain
                ex.release(req)
            torch.cuda.synchronize()
        released = time.perf_counter()
    assert all(len(r['output_ids']) == count and r['finish_reason'] == 'length' for r in outputs)
    return {'outputs': outputs, 'ready': ready, 'draft_cache_hits': proposer.cache_hits - hits,
            'draft_fallbacks': proposer.cache_fallbacks - fallbacks, 'before': before, 'after': ex.cache_stats(),
            'feature_bytes': proposer.feature_cache.bytes, 'feature_cap': proposer.feature_cache.capacity,
            'allocated_bytes': torch.cuda.memory_allocated(), 'prefill_s': prefilled - started,
            'decode_s': finished - prefilled, 'completion_s': finished - started,
            'release_s': released - finished, 'total_s': released - started,
            'decode_tps': len(tasks) * (count - 1) / (finished - prefilled) if count > 1 else None,
            'aggregate_tps': len(tasks) * count / (released - started),
            'work': work_summary(record.events), 'events': record.events}


def compare(probe, oracle):
    return {'same_outputs': [r['output_ids'] for r in probe['outputs']] == [r['output_ids'] for r in oracle['outputs']],
            'same_emitted_work': [e['emitted'] for e in probe['events'] if e['kind'] == 'decode']
                                 == [e['emitted'] for e in oracle['events'] if e['kind'] == 'decode'],
            'same_prefix': [r['cached_tokens'] for r in probe['outputs']] == [r['cached_tokens'] for r in oracle['outputs']]}


def main():
    import torch
    from sharar import config
    from sharar.engine.executor import Executor
    from sharar.model.qwen38 import Qwen38
    from sharar.proposers.parallel import FeatureCache
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--large-lru', action='store_true')
    ap.add_argument('--cold-tokens', type=int, default=1000)
    ap.add_argument('--warm-tokens', type=int, default=256)
    ap.add_argument('--families', nargs='+', default=['ring_buffer', 'deadline_queue', 'transactional_outbox',
                                                   'queue_capacity', 'build_graph', 'stream_parser'])
    args = ap.parse_args()
    tasks = {t['id']: t for t in json.loads(args.inputs.read_text())['tasks']}
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'completed': False, 'label': args.label, 'identity': identity(args),
              'input_sha256': sha_file(args.inputs), 'driver_sha256': sha_file(Path(__file__)),
              'cold_tokens': args.cold_tokens, 'warm_tokens': args.warm_tokens,
              'families': args.families, 'cells': []}
    save(args.output / 'report.json', report)
    def record(kind, name, result):
        row = {'kind': kind, 'name': name, **result}
        report['cells'].append(row)
        save(args.output / 'report.json', report)
        print(json.dumps({'kind': kind, 'name': name, **{k: v for k, v in result.items()
                         if k in ('decode_tps', 'total_s', 'feature_bytes', 'draft_fallbacks', 'comparison')}}), flush=True)
    with torch.inference_mode():
        ex = Executor(Qwen38(config.snapshot_dir(), log=print), num_pages=60000, num_slots=24, max_batch=4, log=print)
        ex.enable_spec(15, proposer='dflash2', state_strategy='reconstruct', draft_precision='fp8', log=print)
        p = ex.spec.proposer
        if args.large_lru:
            assert type(p.feature_cache) is FeatureCache, 'large-LRU control must use baseline runtime'
            p.feature_cache = FeatureCache((ex.state_pool.num_slots - 1) * p.window * 5120 * 2)
        report['feature_cache'] = {'class': type(p.feature_cache).__name__, 'capacity': p.feature_cache.capacity}
        ex.capture_graphs(log=print)
        ex.warmup(log=print)
        warm_feedback_shapes(ex)
        native_batch(ex, [tasks['benchmark_code']] * 4, 64)
        for category in ('code', 'design', 'reasoning'):
            clear(ex)
            record('cold_bs1', category, batch(ex, [tasks[f'benchmark_{category}']], args.cold_tokens, retain=False))
        clear(ex)
        record('cold_bs4', 'families', batch(ex, [tasks[f] for f in args.families[:4]], args.cold_tokens, retain=False))
        # Every family has its own independently primed immediate-hit oracle.
        for index, family in enumerate(args.families):
            task = tasks[family]
            clear(ex)
            seed = batch(ex, [task], 1)
            oracle = batch(ex, [task], args.warm_tokens, retain=False)
            record('immediate', family, {'seed': seed, **oracle})
            clear(ex)
            seed = batch(ex, [task], 1)
            others = [tasks[args.families[(index + j) % len(args.families)]] for j in (1, 2)]
            intervening = [batch(ex, [other], 1) for other in others]
            probe = batch(ex, [task], args.warm_tokens, retain=False)
            probe['comparison'] = compare(probe, oracle)
            record('interleaved', family, {'seed': seed, 'intervening': intervening, **probe})
            if type(p.feature_cache).__name__ == 'CheckpointFeatureCache':
                assert not probe['draft_fallbacks'] and all(probe['comparison'].values()), family
        clear(ex)
        report['after_flush'] = {'feature_bytes': p.feature_cache.bytes, 'cache': ex.cache_stats()}
        if type(p.feature_cache).__name__ == 'CheckpointFeatureCache':
            assert not p.feature_cache.bytes and not p.feature_cache.entries
        report['completed'] = True
        save(args.output / 'report.json', report)


if __name__ == '__main__':
    main()
