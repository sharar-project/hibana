"""Fixed-history DFlash2 failure diagnosis; deliberately outside timing runs.

Record candidate support at the actual first rejection and compare an FP32
selector on the same hidden states. Agreement with a recorded continuation is
diagnostic only: it is not a measured acceptance rate for another engine run.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from bench.gb10_optimize import configure, prepare, release, selector_fp32
from sharar import config
from sharar.engine.executor import Executor
from sharar.kernels.draft_selection import greedy_lattice
from sharar.model.qwen38 import Qwen38


def prefix_matches(left, right):
    return next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--inputs', type=Path, default=Path('/proof/frozen/inputs.json'))
    ap.add_argument('--baseline-source', type=Path, default=Path('/proof/frozen/parallel_draft_baseline.py'))
    args = ap.parse_args()
    args.output.mkdir(exist_ok=False)
    report = {'completed': False, 'protocol': {'rounds_observed_per_case': 96, 'output_tokens': 1000,
              'cases': ['code', 'design', 'reasoning'], 'candidate_topk': 16,
              'counterfactual_is_fixed_history_only': True}, 'cases': []}
    def save():
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    configure('control', args.baseline_source)
    inputs = json.loads(args.inputs.read_text())
    ex = Executor(Qwen38(config.snapshot_dir()), num_pages=60000, num_slots=24, max_batch=4)
    ex.enable_spec(15, proposer='dflash2', state_strategy='reconstruct')
    ex.capture_graphs()
    ex.warmup()
    model, graphs = ex.spec.proposer.model, ex.spec.proposer.graphs
    original = graphs.run
    for case in report['protocol']['cases']:
        reqs = prepare(ex, inputs[case][:1])
        req, rows = reqs[0], []
        def record(tokens, positions, pages, histories):
            result = original(tokens, positions, pages, histories)
            if len(rows) < 96:
                hidden, candidates, scores, _ = result
                candidates32, scores32 = selector_fp32(model, hidden, tokens[:, 0])
                assert torch.equal(candidates, candidates32)
                rows.append({'output_offset': len(req.output_ids),
                             'candidates': candidates[0].tolist(),
                             'baseline_path': greedy_lattice(candidates, scores)[0].tolist(),
                             'fp32_path': greedy_lattice(candidates32, scores32)[0].tolist()})
            return result
        graphs.run = record
        try:
            while not req.finished:
                before = len(req.output_ids)
                ex.decode_step(reqs)
                if rows and rows[-1]['output_offset'] == before:
                    rows[-1]['emitted'] = req.output_ids[before:]
        finally:
            graphs.run = original
        counts = Counter()
        for row in rows:
            future = req.output_ids[row['output_offset']:row['output_offset'] + 15]
            row['baseline_prefix_matches'] = prefix_matches(row['baseline_path'], future)
            row['fp32_prefix_matches'] = prefix_matches(row['fp32_path'], future)
            accepted = len(row['emitted']) - 1
            if accepted < 15:
                correct = row['emitted'][-1]
                choices = row['candidates'][accepted]
                row['first_rejection_target'] = correct
                row['first_rejection_candidate_rank'] = choices.index(correct) + 1 if correct in choices else None
                counts['rejections'] += 1
                counts['target_in_top16' if correct in choices else 'target_missing_top16'] += 1
            counts['rounds'] += 1
            counts['baseline_fixed_history_matches'] += row['baseline_prefix_matches']
            counts['fp32_fixed_history_matches'] += row['fp32_prefix_matches']
            delta = row['fp32_prefix_matches'] - row['baseline_prefix_matches']
            counts['fp32_better' if delta > 0 else 'fp32_worse' if delta < 0 else 'fp32_same'] += 1
        report['cases'].append({'case': case, 'summary': dict(counts), 'rows': rows,
                                'output_ids': req.output_ids[:], 'total_verify_rounds': req.spec_rounds})
        release(ex, reqs)
        save()
        print(json.dumps({'case': case, **counts}), flush=True)
    report['completed'] = True
    save()


if __name__ == '__main__':
    main()
