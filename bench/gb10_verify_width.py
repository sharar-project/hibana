"""Current 16/32/64-row verifier cost, with shared immutable history.

Multiple flat paths are a cost proxy, not a tree implementation. Record
cross-shape argmax changes explicitly. No proposed speedup is reported as TPS.
"""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from bench.gb10_matrix import digest, save
from bench.gb10_optimize import prepare, release
from bench.two_path_study import history_hashes
from sharar import config
from sharar.engine.executor import Executor
from sharar.model.qwen38 import Qwen38


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = json.loads(args.inputs.read_text())
    ex = Executor(Qwen38(config.snapshot_dir(), log=print), num_pages=60000, num_slots=24, max_batch=4, log=print)
    ex.enable_spec(15, proposer='dflash2', state_strategy='reconstruct', draft_precision='fp8', log=print)
    ex.capture_graphs(log=print)
    ex.warmup(log=print)
    report = {'completed': False, 'source_sha256': digest(Path(__file__)), 'inputs_sha256': digest(args.inputs),
              'protocol': 'Code context, 3 continuation positions, 2 warmups + 7 alternating samples, 3 replays/sample. Duplicated flat path with private writable tails.',
              'cells': []}
    save(args.output / 'report.json', report)
    for position in (32, 384, 768):
        reqs = prepare(ex, frozen['code'][:1])
        req = reqs[0]
        extras = []
        try:
            while len(req.output_ids) < position:
                ex.decode_step(reqs)
            batch = ex.spec.prepare(reqs)
            assert ex.spec.steps == 16
            tail_pages = [batch.pages]
            for _ in range(3):
                pages, ids = ex.kv_pool.alloc_with_ids(16)
                extras.extend(ids)
                tail_pages.append(pages)
            assert len(set(extras)) == 48 and set(extras).isdisjoint(req.pages.tolist() + req.reservation.ids)
            before = history_hashes(ex, req)
            samples, targets = {b: [] for b in (1, 2, 4)}, {}
            for repeat in range(-2, 7):
                order = (1, 2, 4) if repeat % 2 else (4, 2, 1)
                for b in order:
                    inputs = (batch.tokens.repeat(b), batch.positions.repeat(b),
                              torch.cat(tail_pages[:b]), batch.slots.repeat(b),
                              [torch.cat([req.pages, tail]) for tail in tail_pages[:b]])
                    torch.cuda.synchronize()
                    a, z = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    started = time.perf_counter()
                    a.record()
                    for _ in range(3):
                        result = ex.spec.verify_graphs.run(*inputs)
                    z.record()
                    z.synchronize()
                    elapsed = (time.perf_counter() - started) * 1000 / 3
                    target = result[1].cpu().tolist()
                    if b in targets:
                        assert targets[b] == target
                    targets[b] = target
                    if repeat >= 0:
                        samples[b].append({'wall_ms': elapsed, 'event_ms': a.elapsed_time(z) / 3})
            assert before == history_hashes(ex, req)
            cell = {'requested_position': position, 'actual_position': len(req.output_ids),
                    'samples': samples, 'target_ids': targets, 'immutable_history_hashes': before,
                    'history_unchanged': True, 'median_wall_ms': {b: statistics.median(x['wall_ms'] for x in rows) for b, rows in samples.items()},
                    'argmax_changes_first_path': {b: sum(x != y for x, y in zip(targets[1], targets[b][:16])) for b in (2, 4)}}
            report['cells'].append(cell)
            save(args.output / 'report.json', report)
            print({k: cell[k] for k in ('requested_position', 'median_wall_ms', 'argmax_changes_first_path')}, flush=True)
        finally:
            if extras:
                ex.kv_pool.free_ids(extras)
            release(ex, reqs)
    report['completed'] = True
    save(args.output / 'report.json', report)


if __name__ == '__main__':
    main()
