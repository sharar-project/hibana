"""Attribute real DFlash2 generation on frozen agent inputs, outside HTTP timing.

Use native proposals and acceptance. Phase event spans overlap; kernel duration
sums are not DRAM counters. Instrumented traces never enter benchmark TPS.
"""
from collections import Counter, defaultdict
import argparse
import gzip
import json
from pathlib import Path
import statistics
import time

import torch

from bench.agent_workloads import sha
from bench.agent_coverage import select_branch, score_branch
from bench.gb10_profile import trace_summary
from bench.verify_cost import phases, summarize_phases
from sharar import config
from sharar.engine.executor import Executor
from sharar.engine.sampler import SamplingParams
from sharar.model.qwen38 import Qwen38


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    frozen = json.loads(args.inputs.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'frozen_sha256': sha(frozen), 'cells': [], 'completed': False,
              'protocol': {'unprofiled_rounds': 32, 'phase_rounds': 8, 'trace_rounds': 4, 'shadow_rounds': 16,
                           'native_proposals_and_acceptance': True, 'output_budget': 512,
                           'timing': 'Direct executor; 32 uninstrumented rounds, then eight event-instrumented and four profiler-instrumented rounds on the same growing continuation. No forced counts or replacement tokens.',
                           'physical_dram_bytes': None}}

    def save():
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    ex = Executor(Qwen38(config.snapshot_dir()), num_pages=35000, num_slots=12)
    ex.enable_spec(7, proposer='dflash2', state_strategy='reconstruct', log=print)
    ex.capture_graphs(log=print)
    ex.warmup(log=print)
    for cell in frozen['fixed']:
        for batch in (1, 4):
            ex.flush_cache()
            ids = [cell['input_ids']] if batch == 1 else cell['branches']
            if batch == 4:
                prime = ex.admit(cell['shared_ids'], SamplingParams(max_new_tokens=1, ignore_eos=True))
                ex.prefill(prime)
                ex.release(prime)
            reqs = []
            try:
                for tokens in ids:
                    reqs.append(ex.admit(tokens, SamplingParams(max_new_tokens=512, ignore_eos=True)))
                while any(r.prefill_pos < r.prefill_len for r in reqs):
                    ex.prefill_batch([(r, min(2048, r.prefill_len - r.prefill_pos)) for r in reqs if r.prefill_pos < r.prefill_len])
                assert all(ex.spec.ready(r) for r in reqs)
                row = {'context': cell['length'], 'batch': batch, 'input_ids': ids,
                       'cached_tokens': [r.prefix_len for r in reqs], 'unprofiled': []}
                for step in range(32):
                    before = [len(r.output_ids) for r in reqs]
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    ex.decode_step(reqs)
                    torch.cuda.synchronize()
                    row['unprofiled'].append({'round': step, 'wall_ms': (time.perf_counter() - start) * 1000,
                                              'emitted': [len(r.output_ids) - n for r, n in zip(reqs, before)]})
                row['unprofiled_total_ms'] = sum(r['wall_ms'] for r in row['unprofiled'])
                row['unprofiled_tokens'] = sum(sum(r['emitted']) for r in row['unprofiled'])
                row['median_round_ms'] = statistics.median(r['wall_ms'] for r in row['unprofiled'])
                records = defaultdict(list)
                with phases(ex, records):
                    for _ in range(8):
                        ex.decode_step(reqs)
                torch.cuda.synchronize()
                row['phases'] = summarize_phases(records)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                    for _ in range(4):
                        with torch.profiler.record_function('sharar.decode_step'):
                            ex.decode_step(reqs)
                            torch.cuda.synchronize()
                tracepath = args.output / f't{cell["length"]}-b{batch}.json'
                prof.export_chrome_trace(str(tracepath))
                trace = json.loads(tracepath.read_text())
                row['trace'] = trace_summary(trace)
                assert row['trace']['gpu_event_count'] > 0
                totals, counts = Counter(), Counter()
                for event in trace['traceEvents']:
                    if event.get('cat') == 'kernel' and event.get('ph') == 'X':
                        totals[event['name']] += event['dur'] / 1000
                        counts[event['name']] += 1
                row['all_kernel_totals'] = [{'name': name, 'ms': ms, 'calls': counts[name]} for name, ms in totals.most_common()]
                with gzip.open(str(tracepath) + '.gz', 'wt') as out:
                    json.dump(trace, out)
                tracepath.unlink()
                # Diagnostic copies happen only after all cost measurements.
                # This predicts no actual tree outputs and modifies no proposals.
                native = ex.spec.proposer.graphs.run
                lattice = {}
                def capture(*a, **kw):
                    result = native(*a, **kw)
                    lattice['candidates'] = result[1].cpu().tolist()
                    lattice['scores'] = result[2].cpu().tolist()
                    lattice['branches'] = [select_branch(c, s) for c, s in zip(lattice['candidates'], lattice['scores'])]
                    return result
                row['coverage'] = []
                ex.spec.proposer.graphs.run = capture
                try:
                    for _ in range(16):
                        before = [len(r.output_ids) for r in reqs]
                        ex.decode_step(reqs)
                        for i, req in enumerate(reqs):
                            emitted = req.output_ids[before[i]:]
                            branch = lattice['branches'][i]
                            row['coverage'].append({'row': i, 'branch': branch, 'emitted': emitted,
                                                    **score_branch(branch, emitted),
                                                    'candidates': lattice['candidates'][i], 'scores': lattice['scores'][i]})
                finally:
                    ex.spec.proposer.graphs.run = native
                row['output_ids'] = [r.output_ids[:] for r in reqs]
                row['spec_rounds'] = [r.spec_rounds for r in reqs]
                row['cuda_allocated_bytes'] = torch.cuda.memory_allocated()
                row['cuda_reserved_bytes'] = torch.cuda.memory_reserved()
                report['cells'].append(row)
                save()
                print(json.dumps({'context': cell['length'], 'batch': batch, 'round_ms': row['median_round_ms'],
                                  'tokens_per_round_per_request': row['unprofiled_tokens'] / 32 / batch,
                                  'phases': row['phases']}), flush=True)
            finally:
                for req in reqs:
                    ex.release(req)
    ex.flush_cache()
    assert ex.kv_pool.free_count == 34999 and ex.state_pool.free_count == 11
    report['completed'] = True
    save()


if __name__ == '__main__':
    main()
