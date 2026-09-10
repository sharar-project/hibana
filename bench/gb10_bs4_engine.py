"""Cold BS1/BS4 verification-width study, with complete-round work accounting.

Run each label in a fresh process inside an exclusive, guarded GPU window.
All variants capture depths 3/7/15 and keep the FP8 DFlash2 drafter at k=15.
Inputs and four-request groups are frozen externally before the first run.
Round durations are host submission/completion intervals, not isolated GPU
phase timings; only the whole decode interval includes a final synchronize.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.support.evidence import identity, save, sha_file

from bench.support.recording import native_batch


def make_runtime(depth, log=print):
    import torch
    from sharar import config
    from sharar.engine.executor import Executor
    from sharar.model.qwen38 import Qwen38
    torch.manual_seed(42)
    ex = Executor(Qwen38(config.snapshot_dir(), log=log), num_pages=60000, num_slots=24, max_batch=4, log=log)
    ex.enable_spec(15, proposer='dflash2', state_strategy='reconstruct', draft_precision='fp8', log=log)
    ex.spec.verify_depths = (3, 7, 15)
    ex.capture_graphs(log=log)
    ex.spec.verify_depth = depth
    ex.warmup(log=log)
    return ex


def run(args):
    import torch
    tasks = {t['id']: t for t in json.loads(args.inputs.read_text())['tasks']}
    groups = json.loads(args.groups.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'completed': False, 'label': args.label, 'depth': args.depth,
              'protocol': {'input_sha256': sha_file(args.inputs), 'groups_sha256': sha_file(args.groups),
                           'output_tokens': args.output_tokens, 'draft_depth': 15, 'draft_precision': 'fp8',
                           'capture_depths': [3, 7, 15], 'cold': True, 'ignore_eos': True},
              'identity': identity(args), 'cells': []}
    report['identity']['source_sha256']['bench/gb10_bs4_engine.py'] = sha_file(Path(__file__))
    save(args.output / 'report.json', report)
    with torch.inference_mode():
        ex = make_runtime(args.depth)
        # Short stopping/state/pool preflight at the selected width; repeat
        # verifies reproducibility at the same physical shape, not across widths.
        preflight = [native_batch(ex, [tasks[t] for t in groups[0]['tasks']], 31, check_state=True, short=True)
                     for _ in range(2)]
        assert [r['output_ids'] for r in preflight[0]['tasks']] == [r['output_ids'] for r in preflight[1]['tasks']]
        save(args.output / 'preflight.json', preflight)
        report['allocated_bytes'] = torch.cuda.memory_allocated()
        warm = native_batch(ex, [tasks['benchmark_code']] * 4, 128)
        save(args.output / 'warmup.json', warm)
        cells = [{'id': f'bs1_{category}', 'kind': 'bs1', 'tasks': [f'benchmark_{category}']}
                 for category in ('code', 'design', 'reasoning')]
        cells += [{'kind': 'bs4', **g} for g in groups]
        for cell in cells:
            result = native_batch(ex, [tasks[t] for t in cell['tasks']], args.output_tokens)
            report['cells'].append({k: v for k, v in cell.items() if k != 'tasks'} | result)
            save(args.output / 'report.json', report)
            print(json.dumps({'cell': cell['id'], 'depth': args.depth, 'decode_tps': result['decode_tps'],
                              'decode_s': result['decode_s'], 'work': result['work'], 'ok': result['ok']}), flush=True)
        report['completed'] = True
        save(args.output / 'report.json', report)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs', type=Path, required=True)
    ap.add_argument('--groups', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--depth', type=int, choices=[3, 7, 15], required=True)
    ap.add_argument('--output-tokens', type=int, default=1000)
    args = ap.parse_args()
    if args.output_tokens < 2:
        ap.error('output-tokens must be at least 2')
    run(args)


if __name__ == '__main__':
    main()
