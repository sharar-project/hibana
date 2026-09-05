"""Fit an explicit pilot policy from retained GB10 costs and acceptance traces.

Only the accepted prefix and first rejected token are observations. Later draft
tokens are censored. Calibration describes current-round confidence; serving
uses the previous round as a predictor, which needs a separate workload gate.
This tool never enables or promotes its result.
"""
import argparse
import json
from pathlib import Path


def fit(report):
    if report['name'] != 'dspark' or not report.get('passed'):
        raise ValueError('a passing DSpark measurement is required')
    calibration, observations = {}, {}
    for sampled, name in ((False, 'greedy'), (True, 'sampled')):
        bins = [[0, 0] for _ in range(10)]
        for row in report['acceptance_trace']:
            if row['sampled'] != sampled:
                continue
            for j, confidence in enumerate(row['confidence'][:min(row['accepted'] + 1, row['verified_depth'])]):
                slot = min(9, max(0, int(confidence * 10)))
                bins[slot][0] += int(j < row['accepted'])
                bins[slot][1] += 1
        if not sum(n for _, n in bins):
            continue  # unobserved modes keep the fixed width
        # Jeffreys smoothing is explicit. Empty bins use the pooled estimate.
        pooled = (sum(a for a, _ in bins) + .5) / (sum(n for _, n in bins) + 1)
        calibration[name] = [(a + .5) / (n + 1) if n else pooled for a, n in bins]
        observations[name] = bins
    costs = {}
    for row in report['width_costs']:
        costs.setdefault(str(row['batch']), {})[str(row['depth'])] = row['round_ms']
    return dict(hardware='GB10', target_revision=report['target'],
                draft_revision=report['draft_pin'][1], state_strategy='reconstruct',
                round_ms=costs, calibration=calibration, observations=observations,
                status='pilot; independent workload gate required',
                cost_contexts=sorted({row['context'] for row in report['width_costs']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    with args.output.open('x') as stream:
        json.dump(fit(json.loads(args.report.read_text())), stream, indent=2)
        stream.write('\n')
