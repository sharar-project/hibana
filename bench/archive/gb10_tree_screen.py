"""Current-draft tree coverage, evaluated only after draft-only selection.

These are counterfactual prefixes on native histories, never measured tree TPS.
"""
import argparse
import gzip
import hashlib
import heapq
import json
import math
from pathlib import Path

from bench.gb10_acceptance_dataset import walk
from bench.tree_study import select_tree, tree_coverage


def probability_tree(candidates, scores, target_rows):
    depth = len(candidates)
    width = len(candidates[0]) if depth else 0
    if (not width or len(scores) != depth
            or any(len(c) != width or len(set(c)) != width for c in candidates)
            or any(len(s) != width or any(len(r) != width for r in s) for s in scores)
            or any(not math.isfinite(v) for s in scores for r in s for v in r)):
        raise ValueError('expected a finite rectangular conditional lattice')
    if type(target_rows) is not int or not 1 <= target_rows <= 1 + sum(width ** d for d in range(1, depth + 1)):
        raise ValueError('invalid tree budget')
    probabilities = []
    for rows in scores:
        normalized = []
        for row in rows:
            peak = max(row)
            norm = peak + math.log(sum(math.exp(x - peak) for x in row))
            normalized.append([x - norm for x in row])
        probabilities.append(normalized)
    retained, frontier, nodes = {(): 0}, [], []
    def expand(prefix, mass):
        if len(prefix) == depth:
            return
        previous = prefix[-1] if prefix else 0
        for rank, value in enumerate(probabilities[len(prefix)][previous]):
            heapq.heappush(frontier, (-mass - value, prefix + (rank,)))
    expand((), 0.)
    for _ in range(target_rows - 1):
        negative_mass, prefix = heapq.heappop(frontier)
        node = {'id': len(nodes) + 1, 'parent': retained[prefix[:-1]], 'depth': len(prefix),
                'rank': prefix[-1], 'token': candidates[len(prefix) - 1][prefix[-1]], 'log_mass': -negative_mass}
        nodes.append(node)
        retained[prefix] = node['id']
        expand(prefix, -negative_mass)
    return {'target_rows': target_rows, 'greedy': walk(candidates, scores), 'nodes': nodes}


def totals(rows):
    baseline, candidate = (sum(r[key] for r in rows) for key in ('baseline', 'candidate'))
    return {'baseline': baseline, 'candidate': candidate,
            'ratio': candidate / baseline if baseline else None,
            'regressing_histories': sum(r['candidate'] < r['baseline'] for r in rows)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--captures', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--policy', choices=('protected', 'probability'), required=True)
    ap.add_argument('--budgets', type=int, nargs='+', required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = {'policy': args.policy, 'budgets': args.budgets,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'cells': []}
    for case in ('code', 'design', 'reasoning'):
        raw = (args.captures / f'{case}-candidates.json.gz').read_bytes()
        data = json.loads(gzip.decompress(raw))
        for budget in args.budgets:
            measured = []
            for row in data['rows']:
                offset, depth = row['output_offset'], data['depth']
                future = data['output_ids'][offset:offset + depth + 1]
                if len(future) != depth + 1:
                    continue
                tree = (select_tree(row['candidates'], row['scores'], budget - depth - 1)
                        if args.policy == 'protected' else probability_tree(row['candidates'], row['scores'], budget))
                assert tree['greedy'] == row['proposal_ids']
                baseline = next((i + 1 for i, (x, y) in enumerate(zip(tree['greedy'], future)) if x != y), depth + 1)
                assert baseline == len(row['emitted_ids'])
                measured.append({'offset': offset, 'baseline': baseline, 'candidate': tree_coverage(tree, future)})
            report['cells'].append({'case': case, 'budget': budget,
                                    'capture_sha256': hashlib.sha256(raw).hexdigest(),
                                    'all': totals(measured), 'late': totals([r for r in measured if r['offset'] >= 512]),
                                    'rows': measured})
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
