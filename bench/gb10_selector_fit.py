"""Small residual selector feasibility screen on fixed native histories.

Train code/design only, evaluate reasoning once after the fixed training run.
Labels use future native tokens; inference features use only draft scores and
candidate ranks. No counterfactual result is reported as measured generation.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import torch

from bench.gb10_acceptance_dataset import prefix_matches


def features(scores, position, previous, depth):
    """Eight scalar features per candidate; supports arbitrary leading axes."""
    gap = (scores.float() - scores.float().amax(-1, keepdim=True)) / 16
    rank = torch.arange(scores.shape[-1], device=scores.device).expand_as(gap) / max(1, scores.shape[-1] - 1)
    pos = position / max(1, depth - 1)
    prev = previous / max(1, scores.shape[-1] - 1)
    return torch.stack([gap, rank, gap * pos, rank * pos, gap * prev, rank * prev,
                        gap.square(), gap * rank], -1)


def training_rows(capture):
    depth = capture['depth']
    xs, base, labels, weights = [], [], [], []
    for row in capture['rows']:
        if row['right_censored']:
            continue
        future = capture['output_ids'][row['output_offset']:]
        previous = 0
        for j, candidates in enumerate(row['candidates']):
            if future[j] not in candidates:
                break
            scores = torch.tensor(row['scores'][j][previous])
            xs.append(features(scores, j, previous, depth))
            base.append(scores - scores.max())
            labels.append(candidates.index(future[j]))
            weights.append(.9 ** j)
            previous = labels[-1]
    return torch.stack(xs), torch.stack(base), torch.tensor(labels), torch.tensor(weights)


def evaluate(capture, weights):
    native, candidate, ceiling, details = 0, 0, 0, []
    for row in capture['rows']:
        if row['right_censored']:
            continue
        previous, proposed = 0, []
        for j, choices in enumerate(row['candidates']):
            scores = torch.tensor(row['scores'][j][previous])
            adjusted = scores.float() + features(scores, j, previous, capture['depth']) @ weights
            previous = int(adjusted.argmax())
            proposed.append(choices[previous])
        future = capture['output_ids'][row['output_offset']:]
        a, b, c = row['native_prefix'] + 1, prefix_matches(proposed, future) + 1, row['support_prefix'] + 1
        native += a
        candidate += b
        ceiling += c
        details.append({'output_offset': row['output_offset'], 'native_emitted': a,
                        'candidate_emitted': b, 'oracle_emitted': c})
    return {'rounds': len(details), 'native_emitted_sum': native, 'candidate_emitted_sum': candidate,
            'oracle_emitted_sum': ceiling, 'fixed_history_improvement_pct': 100 * (candidate / native - 1),
            'details': details}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--captures', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    captures, hashes = {}, {}
    for case in ('code', 'design', 'reasoning'):
        path = args.captures / f'{case}-candidates.json.gz'
        hashes[case] = hashlib.sha256(path.read_bytes()).hexdigest()
        with gzip.open(path, 'rt') as f:
            captures[case] = json.load(f)
    data = [training_rows(captures[case]) for case in ('code', 'design')]
    x, base, labels, importance = [torch.cat([row[i] for row in data]) for i in range(4)]
    weights = torch.zeros(8, requires_grad=True)
    opt = torch.optim.Adam([weights], lr=.03)
    losses = []
    for step in range(200):
        opt.zero_grad()
        logits = base + x @ weights
        ce = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
        loss = (ce * importance).sum() / importance.sum() + .01 * weights.square().sum()
        loss.backward()
        opt.step()
        if step % 20 == 0 or step == 199:
            losses.append({'step': step, 'loss': float(loss.detach())})
    fitted = weights.detach()
    evaluation = {case: evaluate(capture, fitted) for case, capture in captures.items()}
    report = {'completed': True, 'training_cases': ['code', 'design'], 'heldout_cases': ['reasoning'],
              'training_rows': len(labels), 'feature_count': 8, 'steps': 200, 'learning_rate': .03,
              'l2': .01, 'position_discount': .9, 'seed': 42, 'capture_sha256': hashes,
              'weights': fitted.tolist(), 'losses': losses, 'evaluation': evaluation,
              'passes_fixed_history_screen': evaluation['reasoning']['fixed_history_improvement_pct'] >= 10,
              'limitations': ['One held-out prompt; no broad generalization claim.',
                              'Fixed native histories; timings and native candidate quality not measured.',
                              'Eight-feature ranker tests cheap selector feasibility, not the ceiling of backbone or selector training.']}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'evaluation'}, indent=2))
    print(json.dumps({case: {k: v for k, v in row.items() if k != 'details'} for case, row in evaluation.items()}, indent=2))


if __name__ == '__main__':
    main()
