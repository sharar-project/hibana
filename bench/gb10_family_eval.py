"""Paired cold 10K/1K evaluation over the frozen 21-task set in the native executor.

Three benchmark prompts cannot decide 3-10% changes: a changed continuation
swings acceptance by tens of percent per prompt, and the benchmark code prompt
emits <|im_end|> at output index 98 and continues under ignore_eos for the
remaining 902 tokens. This driver runs every task of the frozen causal-selector
split (18 synthetic families plus the 3 benchmark prompts) through the engine's
own prefill/decode loop, records exact outputs and per-round work, and
`compare` reports paired per-task log-ratios with bootstrap intervals,
decomposed into round cost and accepted tokens per round.

Run `run` inside an exclusive GPU window (production and the memory guard
stopped and restored by the host harness). `compare` is CPU-only.

    python3 -m bench.gb10_family_eval run --inputs /proof/task-inputs.json \
        --reference /proof/reference-ids.json --output /proof/native-1 --label native
    python3 -m bench.gb10_family_eval compare --baseline /proof/native-1 \
        --candidate /proof/cand-1 /proof/cand-2 --baseline-repeat /proof/native-2

A candidate is a Python hook `module:function` called as `fn(stage, executor,
args)` at stages 'model' (after weights load, executor is None), 'spec' (after
enable_spec, before graph capture) and 'ready' (after warmup). It may patch
kernels, dispatch, proposer settings or graphs; the report records the hook
name and the source hashes so the comparison is reproducible.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
from pathlib import Path

from bench.support.evidence import sha_file, save, clear, identity

EOS_IDS = (248046, 248044)  # <|im_end|>, <|endoftext|>
SPLITS = ('train', 'validation', 'test', 'benchmark')


# ---------------------------------------------------------------- metrics (pure)

def first_eos(output_ids: list[int]) -> int | None:
    for i, t in enumerate(output_ids):
        if t in EOS_IDS:
            return i
    return None


def pre_eos_rounds(emitted_per_step: list[int], first: int | None) -> int | None:
    """Decode steps needed to emit output index `first` (the first EOS)."""
    if first is None:
        return None
    produced = 1  # the prefill token is output index 0
    for step, n in enumerate(emitted_per_step):
        produced += n
        if produced > first:
            return step + 1
    return len(emitted_per_step)


def task_metrics(result: dict) -> dict:
    """Derived per-task numbers from a raw run record."""
    n = len(result['output_ids'])
    rounds = result['spec_rounds']
    decode_s = result['decode_s']
    steps = result['emitted_per_step']
    first = first_eos(result['output_ids'])
    out = {
        'decode_tps': (n - 1) / decode_s if n > 1 and decode_s > 0 else None,
        'e2e_tps': n / result['total_s'] if result['total_s'] > 0 else None,
        'tok_per_round': (n - 1) / rounds if rounds else None,
        'round_ms': 1000 * decode_s / rounds if rounds else None,
        'first_eos': first,
        'eos_count': sum(1 for t in result['output_ids'] if t in EOS_IDS),
        'plain_steps': len(steps) - rounds,
    }
    pre = pre_eos_rounds(steps, first)
    out['pre_eos_rounds'] = pre
    out['pre_eos_tok_per_round'] = (first / pre) if pre else None
    return out


def _bootstrap_ci(values: list[float], resamples: int = 10000, seed: int = 0) -> tuple[float, float]:
    import numpy as np
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (math.nan, math.nan)
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(resamples, arr.size))
    means = arr[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def paired_stats(rows: list[dict], key: str) -> dict:
    """Mean paired log-ratio of `key` (candidate/baseline) with a bootstrap 95% CI, as ratios."""
    logs = [math.log(r['candidate'][key] / r['baseline'][key]) for r in rows
            if r['candidate'].get(key) and r['baseline'].get(key)]
    if not logs:
        return {'n': 0}
    mean = sum(logs) / len(logs)
    lo, hi = _bootstrap_ci(logs)
    sd = math.sqrt(sum((x - mean) ** 2 for x in logs) / max(1, len(logs) - 1))
    return {'n': len(logs), 'ratio': math.exp(mean), 'ci95': [math.exp(lo), math.exp(hi)],
            'log_sd': sd, 'min': math.exp(min(logs)), 'max': math.exp(max(logs))}


def first_difference(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def _index(report: dict) -> dict:
    table = {}
    for row in report['tasks']:
        table.setdefault(row['id'], []).append(row)
    return table


def _average(rows: list[dict]) -> dict:
    keys = ('decode_tps', 'e2e_tps', 'tok_per_round', 'round_ms')
    out = {k: sum(r['metrics'][k] for r in rows) / len(rows) for k in keys}
    out['ttft_s'] = sum(r['ttft_s'] for r in rows) / len(rows)
    out['output_ids'] = rows[0]['output_ids']
    out['split'] = rows[0]['split']
    out['category'] = rows[0]['category']
    out['metrics'] = rows[0]['metrics']
    return out


def compare_reports(baseline: dict, candidates: list[dict], baseline_repeat: dict | None = None) -> dict:
    """Pair every task present in the baseline with the candidate runs (repeat-averaged)."""
    base = _index(baseline)
    cand_rows = {}
    for report in candidates:
        for tid, rows in _index(report).items():
            cand_rows.setdefault(tid, []).extend(rows)
    paired = []
    for tid, rows in base.items():
        if tid not in cand_rows:
            continue
        b, c = _average(rows), _average(cand_rows[tid])
        paired.append({'id': tid, 'split': b['split'], 'category': b['category'], 'baseline': b, 'candidate': c,
                       'same_output': b['output_ids'] == c['output_ids'],
                       'first_difference': first_difference(b['output_ids'], c['output_ids']),
                       'decode_ratio': c['decode_tps'] / b['decode_tps'],
                       'e2e_ratio': c['e2e_tps'] / b['e2e_tps'],
                       'round_ms_ratio': c['round_ms'] / b['round_ms'],
                       'tok_per_round_ratio': c['tok_per_round'] / b['tok_per_round']})
    groups = {'families': [r for r in paired if r['split'] != 'benchmark'],
              'benchmarks': [r for r in paired if r['split'] == 'benchmark'],
              'all': paired}
    summary = {}
    for name, rows in groups.items():
        summary[name] = {
            'decode_tps': paired_stats(rows, 'decode_tps'),
            'e2e_tps': paired_stats(rows, 'e2e_tps'),
            'round_ms': paired_stats(rows, 'round_ms'),
            'tok_per_round': paired_stats(rows, 'tok_per_round'),
            'ttft_s': paired_stats(rows, 'ttft_s'),
            'same_output_count': sum(1 for r in rows if r['same_output']),
            'count': len(rows),
        }
    noise = None
    if baseline_repeat is not None:
        rep = _index(baseline_repeat)
        pairs = [{'baseline': _average(base[t]), 'candidate': _average(rep[t])} for t in base if t in rep]
        noise = {'decode_tps': paired_stats(pairs, 'decode_tps'), 'round_ms': paired_stats(pairs, 'round_ms'),
                 'tok_per_round': paired_stats(pairs, 'tok_per_round'),
                 'same_output_count': sum(1 for p in pairs if p['baseline']['output_ids'] == p['candidate']['output_ids']),
                 'count': len(pairs)}
    tasks = []
    for r in paired:
        row = {k: v for k, v in r.items() if k not in ('baseline', 'candidate')}
        for side in ('baseline', 'candidate'):
            for k in ('decode_tps', 'tok_per_round', 'round_ms', 'ttft_s'):
                row[f'{side}_{k}'] = r[side][k]
        tasks.append(row)
    return {'summary': summary, 'noise_floor': noise, 'tasks': tasks}


def format_comparison(comparison: dict) -> str:
    lines = []
    for name in ('families', 'benchmarks', 'all'):
        s = comparison['summary'][name]
        if not s['count']:
            continue
        d, r, t = s['decode_tps'], s['round_ms'], s['tok_per_round']
        lines.append(f"{name:10s} n={s['count']:2d}  decode x{d['ratio']:.4f} [{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}]  "
                     f"round_ms x{r['ratio']:.4f} [{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}]  "
                     f"tok/round x{t['ratio']:.4f} [{t['ci95'][0]:.4f}, {t['ci95'][1]:.4f}]  "
                     f"same output {s['same_output_count']}/{s['count']}")
    if comparison['noise_floor']:
        nf = comparison['noise_floor']
        lines.append(f"noise      n={nf['count']:2d}  decode x{nf['decode_tps']['ratio']:.4f} "
                     f"[{nf['decode_tps']['ci95'][0]:.4f}, {nf['decode_tps']['ci95'][1]:.4f}]  "
                     f"round_ms x{nf['round_ms']['ratio']:.4f}  same output {nf['same_output_count']}/{nf['count']} (baseline vs its repeat)")
    lines.append(f"{'task':22s} {'split':10s} {'dec':>7s} {'round':>7s} {'tok/r':>7s} {'same':>5s} {'diff@':>6s}")
    for r in comparison['tasks']:
        lines.append(f"{r['id']:22s} {r['split']:10s} {r['decode_ratio']:7.4f} {r['round_ms_ratio']:7.4f} "
                     f"{r['tok_per_round_ratio']:7.4f} {str(r['same_output']):>5s} {str(r['first_difference']):>6s}")
    return '\n'.join(lines)


# ------------------------------------------------------------------ GPU driver

def select_tasks(frozen: dict, spec: str) -> list[dict]:
    wanted = [s.strip() for s in spec.split(',') if s.strip()]
    if not wanted or wanted == ['all']:
        return list(frozen['tasks'])
    out = [t for t in frozen['tasks'] if t['split'] in wanted or t['id'] in wanted]
    missing = [w for w in wanted if w not in SPLITS and w not in {t['id'] for t in out}]
    if missing:
        raise ValueError(f'unknown tasks: {missing}')
    return out


def load_hook(spec: str | None):
    if not spec:
        return None
    module, _, name = spec.partition(':')
    if not module or not name:
        raise ValueError('candidate hook must be module:function')
    return getattr(importlib.import_module(module), name)




def run_one(ex, ids: list[int], count: int, chunk: int) -> dict:
    import torch
    from sharar.engine.sampler import SamplingParams

    proposer = ex.spec.proposer
    hits, fallbacks = proposer.cache_hits, proposer.cache_fallbacks
    torch.cuda.synchronize()
    start = time.perf_counter()
    req = ex.admit(ids, SamplingParams(max_new_tokens=count, ignore_eos=True))
    try:
        while not req.prefill_done:
            ex.prefill_batch([(req, min(chunk, req.prefill_len - req.prefill_pos))])
        torch.cuda.synchronize()
        prefilled = time.perf_counter()
        ready = ex.spec.ready(req)
        emitted = []
        while not req.finished:
            before = len(req.output_ids)
            ex.decode_step([req])
            emitted.append(len(req.output_ids) - before)
        torch.cuda.synchronize()
        end = time.perf_counter()
        result = {
            'output_ids': req.output_ids[:], 'emitted_per_step': emitted, 'spec_rounds': req.spec_rounds,
            'spec_proposed': req.spec_proposed, 'spec_accepted': req.spec_accepted,
            'cached_tokens': req.prefix_len, 'draft_ready': ready,
            'draft_cache_hits': proposer.cache_hits - hits, 'draft_fallbacks': proposer.cache_fallbacks - fallbacks,
            'ttft_s': prefilled - start, 'decode_s': end - prefilled, 'total_s': end - start,
        }
    finally:
        req.cacheable = False
        ex.release(req)
    torch.cuda.synchronize()
    result['metrics'] = task_metrics(result)
    return result


def cmd_run(args) -> int:
    import torch
    from sharar import config
    from sharar.engine.executor import Executor
    from sharar.model.qwen38 import Qwen38
    from sharar.proposers.parallel import PINS

    frozen = json.loads(args.inputs.read_text())
    tasks = select_tasks(frozen, args.tasks)
    references = json.loads(args.reference.read_text()) if args.reference else {}
    args.output.mkdir(parents=True, exist_ok=False)
    hook = load_hook(args.candidate)
    report = {'schema': 1, 'completed': False, 'label': args.label, 'candidate': args.candidate,
              'protocol': {'inputs_sha256': sha_file(args.inputs), 'inputs_version': frozen.get('version'),
                           'output_tokens': args.output_tokens, 'repeats': args.repeats, 'chunk': args.chunk,
                           'proposer': args.proposer, 'depth': args.depth, 'draft_precision': args.draft_precision,
                           'num_pages': args.num_pages, 'num_slots': args.num_slots, 'max_batch': 4,
                           'target_revision': config.MODEL_REVISION, 'draft': PINS.get(args.proposer),
                           'greedy': True, 'ignore_eos': True, 'cold_prefix_cache': True,
                           'tasks': [t['id'] for t in tasks]},
              'identity': identity(args), 'tasks': []}
    save(args.output / 'report.json', report)

    model = Qwen38(config.snapshot_dir(), log=print)
    if hook:
        hook('model', None, args)
    ex = Executor(model, num_pages=args.num_pages, num_slots=args.num_slots, max_batch=4, log=print)
    ex.enable_spec(args.depth, proposer=args.proposer, state_strategy='reconstruct', draft_precision=args.draft_precision, log=print)
    if hook:
        hook('spec', ex, args)
    ex.capture_graphs(log=print)
    ex.warmup(log=print)
    if hook:
        hook('ready', ex, args)
    report['allocated_bytes'] = torch.cuda.memory_allocated()
    report['reserved_bytes'] = torch.cuda.memory_reserved()

    warm = next((t for t in frozen['tasks'] if t['id'] == args.warmup_task), None) or tasks[0]
    clear(ex)
    warmup = run_one(ex, warm['input_ids'], args.output_tokens, args.chunk)
    report['warmup'] = {'id': warm['id'], **warmup}
    save(args.output / 'report.json', report)
    print(json.dumps({'warmup': warm['id'], 'decode_tps': warmup['metrics']['decode_tps'], 'rounds': warmup['spec_rounds']}), flush=True)

    ok = True
    for repeat in range(args.repeats):
        for task in tasks:
            clear(ex)
            result = run_one(ex, task['input_ids'], args.output_tokens, args.chunk)
            row = {'id': task['id'], 'split': task['split'], 'category': task['category'], 'repeat': repeat, **result}
            reference = references.get(task['category']) if task['split'] == 'benchmark' else None
            if reference is not None:
                row['matches_reference'] = result['output_ids'] == reference
            row['pools_recovered'] = (ex.kv_pool.free_count == ex.kv_pool.num_pages - 1
                                      and ex.state_pool.free_count == ex.state_pool.num_slots - 1)
            row['ok'] = (row['cached_tokens'] == 0 and row['draft_fallbacks'] == 0 and row['draft_cache_hits'] == 0 and row['draft_ready']
                         and len(row['output_ids']) == args.output_tokens
                         and sum(row['emitted_per_step']) == args.output_tokens - 1 and row['pools_recovered'])
            ok &= row['ok']
            report['tasks'].append(row)
            save(args.output / 'report.json', report)
            m = row['metrics']
            print(json.dumps({'task': task['id'], 'split': task['split'], 'repeat': repeat, 'rounds': row['spec_rounds'],
                              'decode_tps': round(m['decode_tps'], 3), 'tok_per_round': round(m['tok_per_round'], 4),
                              'round_ms': round(m['round_ms'], 3), 'ttft_s': round(row['ttft_s'], 3),
                              'first_eos': m['first_eos'], 'ok': row['ok'],
                              'matches_reference': row.get('matches_reference')}), flush=True)
    clear(ex)
    report['completed'] = True
    report['all_ok'] = ok
    save(args.output / 'report.json', report)
    return 0 if ok else 1


def cmd_compare(args) -> int:
    baseline = json.loads((args.baseline / 'report.json').read_text())
    candidates = [json.loads((p / 'report.json').read_text()) for p in args.candidate]
    repeat = json.loads((args.baseline_repeat / 'report.json').read_text()) if args.baseline_repeat else None
    for report in (baseline, *candidates, *([repeat] if repeat else [])):
        if not report.get('completed'):
            raise ValueError(f"incomplete report: {report.get('label')}")
        if report['protocol']['inputs_sha256'] != baseline['protocol']['inputs_sha256'] or \
           report['protocol']['output_tokens'] != baseline['protocol']['output_tokens']:
            raise ValueError('reports use different inputs or token budgets')
    comparison = compare_reports(baseline, candidates, repeat)
    comparison['labels'] = {'baseline': baseline['label'], 'candidates': [c['label'] for c in candidates],
                            'baseline_repeat': repeat['label'] if repeat else None}
    print(format_comparison(comparison))
    if args.output:
        save(args.output, comparison)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='stage', required=True)
    run = sub.add_parser('run', help='generate every selected task once per repeat (GPU)')
    run.add_argument('--inputs', type=Path, required=True, help='frozen task-inputs.json (causal-selector-v1)')
    run.add_argument('--reference', type=Path, help='reference-ids.json for the benchmark prompts (native mode should match)')
    run.add_argument('--output', type=Path, required=True, help='new directory')
    run.add_argument('--label', required=True)
    run.add_argument('--candidate', help='hook module:function applied at stages model/spec/ready')
    run.add_argument('--tasks', default='all', help='comma list of task ids and/or splits (train,validation,test,benchmark)')
    run.add_argument('--output-tokens', type=int, default=1000)
    run.add_argument('--repeats', type=int, default=1)
    run.add_argument('--chunk', type=int, default=2048)
    run.add_argument('--proposer', default='dflash2')
    run.add_argument('--depth', type=int, default=15)
    run.add_argument('--draft-precision', choices=('bf16', 'fp8'), default='fp8')
    run.add_argument('--num-pages', type=int, default=60000)
    run.add_argument('--num-slots', type=int, default=24)
    run.add_argument('--warmup-task', default='benchmark_code')
    cmp_ = sub.add_parser('compare', help='paired statistics between completed reports (CPU)')
    cmp_.add_argument('--baseline', type=Path, required=True)
    cmp_.add_argument('--candidate', type=Path, nargs='+', required=True)
    cmp_.add_argument('--baseline-repeat', type=Path, help='second baseline run for the noise floor')
    cmp_.add_argument('--output', type=Path)
    args = ap.parse_args(argv)
    if args.stage == 'run':
        if args.output_tokens < 2 or args.repeats < 1:
            ap.error('need at least two output tokens and one repeat')
        return cmd_run(args)
    return cmd_compare(args)


if __name__ == '__main__':
    sys.exit(main())
