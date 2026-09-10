"""Paired cold controls and interleaved checkpoint-retention experiments.

Run each variant from a frozen source in a fresh guarded GB10 process. Warm
probes are compared with an immediate-hit oracle at the same prefill partition;
retain all output IDs, fallback counts, feature bytes and total durations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.support.recording import native_batch, warm_feedback_shapes
from bench.support.evidence import clear, identity, save, sha_file

from bench.support.retention import batch


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
