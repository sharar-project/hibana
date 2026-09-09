"""Frozen-prefix reuse with intervening requests and bounded draft retention.

This diagnoses target-cache hits that lose speculation. Cache-warm results
remain separate from the cold 10K/1K benchmark. No serving policy is changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from sharar import config
from sharar.engine.executor import Executor
from sharar.engine.sampler import SamplingParams
from sharar.model.qwen38 import Qwen38
from sharar.proposers.parallel import FeatureCache


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def run(ex, ids, output, *, retain):
    proposer = ex.spec.proposer
    hits, fallbacks = proposer.cache_hits, proposer.cache_fallbacks
    # The lookup is read-only and untimed; it does not refresh feature LRU.
    match = ex.cache.match(ids[:-1], touch=False)
    handle = match.node.auxiliary.get(proposer.cache_identity) if match.node else None
    feature_present = handle in proposer.feature_cache.entries
    torch.cuda.synchronize()
    started = time.perf_counter()
    req = ex.admit(ids, SamplingParams(max_new_tokens=output, ignore_eos=True))
    admitted = time.perf_counter()
    completed = False
    try:
        while not req.prefill_done:
            ex.prefill_batch([(req, min(2048, req.prefill_len - req.prefill_pos))])
        torch.cuda.synchronize()
        prefilled = time.perf_counter()
        ready = proposer.ready(req)
        emitted = []
        while not req.finished:
            before = len(req.output_ids)
            ex.decode_step([req])
            emitted.append(len(req.output_ids) - before)
        torch.cuda.synchronize()
        finished = time.perf_counter()
        result = {
            'input_tokens': len(ids), 'output_tokens': len(req.output_ids),
            'raw_prefix_match': match.raw_length, 'cached_tokens': req.prefix_len,
            'draft_features_present_before_admit': feature_present,
            'draft_ready_after_prefill': ready,
            'draft_cache_hits': proposer.cache_hits - hits,
            'draft_cache_fallbacks': proposer.cache_fallbacks - fallbacks,
            'spec_rounds': req.spec_rounds, 'spec_emitted': req.spec_emitted,
            'output_ids': req.output_ids[:], 'emitted_per_step': emitted,
            'admit_s': admitted - started, 'ttft_s': prefilled - started,
            'decode_s': finished - prefilled, 'total_s': finished - started,
            'decode_tps': (output - 1) / (finished - prefilled) if output > 1 else None,
            'e2e_tps': output / (finished - started),
        }
        completed = True
    finally:
        req.cacheable = retain and completed
        ex.release(req)
    result['cache_after_release'] = ex.cache.stats()
    result['feature_bytes_after_release'] = proposer.feature_cache.bytes
    return result


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--output-tokens', type=int, default=32)
    ap.add_argument('--warm-control', action='store_true', help='Record a full immediate cache-hit control before timed interleaving')
    ap.add_argument('--order', choices=('native', 'retained'), nargs='+', default=['native', 'retained', 'retained', 'native'])
    ap.add_argument('--expect-speculation', action='store_true', help='Require every timed probe to retain draft readiness and match its control')
    ap.add_argument('--gaps', type=int, choices=(0, 1, 2), nargs='+', default=[0, 1, 2])
    args = ap.parse_args()
    if args.output_tokens < 2 or args.output_tokens > 1000:
        ap.error('--output-tokens must be between 2 and 1000')
    if args.expect_speculation and not args.warm_control:
        ap.error('--expect-speculation requires --warm-control')
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = json.loads(args.inputs.read_text())
    ex = Executor(Qwen38(config.snapshot_dir(), log=print), num_pages=60000,
                  num_slots=24, max_batch=4, log=print)
    ex.enable_spec(15, proposer='dflash2', state_strategy='reconstruct', draft_precision='fp8', log=print)
    ex.capture_graphs(log=print)
    ex.warmup(log=print)
    proposer = ex.spec.proposer
    native_capacity = proposer.feature_cache.capacity
    capacities = {'native': native_capacity,
                  'retained': (ex.state_pool.num_slots - 1) * proposer.window * 5120 * 2}
    report = {
        'completed': False,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'runtime_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(Path('sharar').rglob('*.py'))},
        'input_sha256': hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
        'feature_capacity_bytes': capacities, 'output_tokens': args.output_tokens,
        'protocol': 'Fresh cache per trial; code seed 10k/1, 0/1/2 intervening 10k/1 requests, repeated code; native or larger feature budget.',
        'order': args.order, 'expect_speculation': args.expect_speculation,
        'cells': [],
    }
    save(args.output / 'report.json', report)
    control = None
    if args.warm_control:
        ex.flush_cache()
        proposer.feature_cache = FeatureCache(native_capacity)
        seed = run(ex, frozen['code'][0], 1, retain=True)
        control = run(ex, frozen['code'][0], args.output_tokens, retain=False)
        assert control['draft_ready_after_prefill'] and control['draft_cache_fallbacks'] == 0
        report['warm_control'] = {'seed': seed, 'probe': control, 'excluded_from_timings': True}
        save(args.output / 'report.json', report)
    seeds = {}
    for repeat, mode in enumerate(args.order):
        for gap in args.gaps:
            ex.flush_cache()
            proposer.feature_cache = FeatureCache(capacities[mode])
            assert ex.kv_pool.free_count == 59999 and ex.state_pool.free_count == 23
            cell = {'repeat': repeat, 'mode': mode, 'gap': gap, 'seeds': []}
            for case in ('code', 'design', 'reasoning')[:gap + 1]:
                result = run(ex, frozen[case][0], 1, retain=True)
                seeds.setdefault(case, result['output_ids'])
                assert result['output_ids'] == seeds[case], f'{case} seed changed'
                cell['seeds'].append({'case': case, **result})
            cell['probe'] = run(ex, frozen['code'][0], args.output_tokens, retain=False)
            if control is not None:
                p = cell['probe']
                p['matches_immediate_speculative_control'] = p['output_ids'] == control['output_ids']
                p['matches_immediate_emitted_work'] = p['emitted_per_step'] == control['emitted_per_step']
            report['cells'].append(cell)
            save(args.output / 'report.json', report)
            if args.expect_speculation:
                assert cell['probe']['draft_ready_after_prefill'] and cell['probe']['draft_cache_fallbacks'] == 0
            if control is not None and (mode == 'retained' or args.expect_speculation):
                assert cell['probe']['matches_immediate_speculative_control']
                assert cell['probe']['matches_immediate_emitted_work']
            print(json.dumps({'repeat': repeat, 'mode': mode, 'gap': gap,
                              **{k: v for k, v in cell['probe'].items()
                                 if k not in ('output_ids', 'emitted_per_step', 'cache_after_release')}}), flush=True)
    ex.flush_cache()
    assert ex.kv_pool.free_count == 59999 and ex.state_pool.free_count == 23
    assert len(proposer.free_slots) == ex.max_batch
    report['completed'] = True
    save(args.output / 'report.json', report)


if __name__ == '__main__':
    main()
