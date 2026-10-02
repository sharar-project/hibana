"""Independent-startup verification strategy comparison on frozen 7C work."""
import argparse
import importlib.metadata
import json
from pathlib import Path

import torch

from sharar import config
from sharar.engine.executor import Executor
from sharar.model.qwen38 import Qwen38
from sharar.tokenizer import SnapshotTokenizer

import gb10_profile
from support.evals import evals_module

local_workloads = evals_module('evals.speed.local')  # the frozen local-use HTTP client (sharar-eval)
from memory_trace import measure_stage
from phase7c import http_server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--strategy', choices=('snapshots', 'reconstruct'), required=True)
    parser.add_argument('--profiles-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = config.snapshot_dir()
    config.check()
    tok = SnapshotTokenizer(snapshot)
    frozen = local_workloads.materialize(local_workloads.DEFAULT_MANIFEST, snapshot)
    report = {'strategy': args.strategy, 'model': config.MODEL_ID, 'revision': config.MODEL_REVISION,
              'packages': {p: importlib.metadata.version(p) for p in ('torch', 'flashinfer-python', 'triton', 'uvicorn')}}
    root = Path(config.__file__).resolve().parents[1]
    report['sources'] = {str(p.relative_to(root)): local_workloads.digest(p.read_bytes())
                         for pattern in ('sharar/**/*.py', 'bench/**/*.py', 'bench/workloads/*')
                         for p in root.glob(pattern) if p.is_file()}
    try:
        with measure_stage(report, 'load'):
            model = Qwen38(snapshot)
        torch.cuda.empty_cache()
        with measure_stage(report, 'pools'):
            ex = Executor(model, num_pages=35000, num_slots=16, log=print)
        with measure_stage(report, 'spec'):
            ex.enable_spec(3, log=print, state_strategy=args.strategy)
        torch.cuda.empty_cache()
        with measure_stage(report, 'graphs'):
            ex.capture_graphs(log=print)
        with measure_stage(report, 'warmup'):
            ex.warmup(log=print)
        torch.cuda.empty_cache()
        report['verify_bytes'] = ex.spec.bytes
        spec = ex.spec
        report['workloads'] = {}
        for mode in (() if args.profiles_only else ('plain', 'mtp')):
            ex.spec = spec if mode == 'mtp' else None
            ex.flush_cache()
            spec.reset_stats()
            with http_server(ex, tok, snapshot, {'strategy': args.strategy, 'mode': mode}) as url:
                report['workloads'][mode] = local_workloads.run(url, frozen, tok, args.output / mode)
            ex.ledger.check_floor()
        ex.spec = spec
        report['draft_cache'] = {'hits': spec.proposer.cache_hits, 'fallbacks': spec.proposer.cache_fallbacks}
        gb10_profile.run(ex, tok, args.output / 'profiles')
        if any(s['automatic_failures'] or s['request_failures'] for s in report['workloads'].values()):
            raise RuntimeError('frozen workload gate failed')
    finally:
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
