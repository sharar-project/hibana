"""Fail closed on missing samples, output drift or a 7B timing regression."""

import argparse
import json
import math
from pathlib import Path
import statistics


def compare(before, after, limit=0.05):
    expected = {(mode, length, batch) for mode in ('plain', 'mtp') for length in (128, 2048, 8192) for batch in (1, 4)}
    def index(report):
        rows = report['rows']
        keyed = {(r['mode'], r['input_tokens'], r['batch']): r for r in rows}
        if set(keyed) != expected or len(rows) != len(expected):
            raise ValueError('expected all 12 unique plain/MTP cells')
        for row in rows:
            if len(row['samples']) != report['profile']['repetitions'] or len(row['samples']) < 3:
                raise ValueError('missing repeated measurements')
            for sample in row['samples']:
                for key in ('decode_tps', 'emitted_tps', 'available_gib'):
                    if not math.isfinite(sample[key]) or sample[key] <= 0:
                        raise ValueError('invalid measurement')
                if sample['available_gib'] < 16:
                    raise ValueError('host floor violated')
        return keyed
    if before['profile'] != after['profile']:
        raise ValueError('profiles differ')
    baseline, candidate = index(before), index(after)
    results = []
    for key in sorted(expected):
        b, c = baseline[key]['samples'], candidate[key]['samples']
        hashes = {s['output_sha256'] for s in b + c}
        ratios = {metric: statistics.median(s[metric] for s in c) / statistics.median(s[metric] for s in b)
                  for metric in ('decode_tps', 'emitted_tps')}
        results.append({'mode': key[0], 'input_tokens': key[1], 'batch': key[2],
                        'identical_outputs': len(hashes) == 1, 'ratios': ratios,
                        'pass': len(hashes) == 1 and min(ratios.values()) >= 1 - limit})
    return {'pass': all(r['pass'] for r in results), 'max_regression_fraction': limit, 'rows': results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    args = parser.parse_args()
    result = compare(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['pass'] else 1)
