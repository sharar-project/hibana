"""Legacy exact-ID AIPerf entry point; prefer ``sharar benchmark``."""
import argparse
import json
from pathlib import Path

from sharar.benchmark_aiperf import MODEL, REVISION, audit_raw, digest, payload_for, trial

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True)
    ap.add_argument('--aiperf', type=Path, required=True)
    ap.add_argument('--inputs', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--cases', nargs='+', default=['code', 'design', 'reasoning'])
    ap.add_argument('--repeats', type=int, default=3)
    args = ap.parse_args()
    frozen = json.loads(args.inputs.read_text())
    args.output.mkdir(exist_ok=True)
    results = []
    for case in args.cases:
        ids = frozen[case][0]
        assert len(ids) == 10000 and all(type(token) is int for token in ids)
        for rep in range(-1, args.repeats):
            result = trial(args.url, args.aiperf, ids, args.output / f'{case}-{rep+1}', rep < 0, case, rep)
            results.append({k: v for k, v in result.items() if k not in ('output_ids', 'text')})
            (args.output / 'results.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
