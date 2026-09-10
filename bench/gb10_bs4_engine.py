"""Cold BS1/BS4 verification-width study, with complete-round work accounting.

Run each label in a fresh process inside an exclusive, guarded GPU window.
All variants capture depths 3/7/15 and keep the FP8 DFlash2 drafter at k=15.
Inputs and four-request groups are frozen externally before the first run.
Round durations are host submission/completion intervals, not isolated GPU
phase timings; only the whole decode interval includes a final synchronize.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from bench.gb10_family_eval import clear, identity, save, sha_file


def work_summary(events):
    decode = [r for r in events if r['kind'] == 'decode']
    prefill = [r for r in events if r['kind'] == 'prefill']
    sizes = Counter()
    for r in decode:
        sizes[r['batch']] += r['seconds']
    seconds = sum(sizes.values())
    physical = sum(r['physical_rows'] for r in decode)
    emitted = sum(sum(r['emitted']) for r in decode)
    return {
        'decode_rounds': len(decode), 'decode_host_s': seconds,
        'prefill_forwards': len(prefill), 'prefill_host_s': sum(r['seconds'] for r in prefill),
        'decode_s_by_batch': dict(sorted(sizes.items())),
        'rounds_by_batch': dict(sorted(Counter(r['batch'] for r in decode).items())),
        'prefill_forwards_by_batch': dict(sorted(Counter(r['batch'] for r in prefill).items())),
        'full_batch_time_fraction': sizes[4] / seconds if seconds else None,
        'physical_verify_rows': physical, 'emitted_decode_tokens': emitted,
        'padded_rows': sum(r['padded_rows'] for r in decode),
        'uncommitted_real_rows': sum(r['real_rows'] - sum(r['emitted']) for r in decode),
        'committed_row_fraction': emitted / physical if physical else None,
        'full_batch_tps': (sum(sum(r['emitted']) for r in decode if r['batch'] == 4) / sizes[4]
                           if sizes[4] else None),
    }


class Recorder:
    """Observe real forwards without adding device synchronization to timing."""
    def __init__(self, ex, *, check_state=False):
        self.ex, self.check_state = ex, check_state
        self.events = []
        self.requests = {}
        self.decode, self.prefill = ex.decode_step, ex.prefill_batch
        self.started = time.perf_counter()

    def __enter__(self):
        self.ex.decode_step, self.ex.prefill_batch = self._decode, self._prefill
        return self

    def __exit__(self, *exc):
        self.ex.decode_step, self.ex.prefill_batch = self.decode, self.prefill

    def _decode(self, reqs):
        ex = self.ex
        depth = getattr(ex.spec, 'verify_depth', ex.spec.k)
        graph = ex.spec.width_graphs[depth]
        before = [len(r.output_ids) for r in reqs]
        rounds = [r.spec_rounds for r in reqs]
        accepted = [r.spec_accepted for r in reqs]
        start = time.perf_counter()
        result = self.decode(reqs)
        end = time.perf_counter()
        physical = graph.pick(len(reqs)) * (depth + 1)
        event = {'kind': 'decode', 'at_s': start - self.started, 'seconds': end - start,
                 'batch': len(reqs), 'request_ids': [r.request_id for r in reqs],
                 'depth': depth, 'physical_rows': physical, 'real_rows': len(reqs) * (depth + 1),
                 'padded_rows': physical - len(reqs) * (depth + 1),
                 'emitted': [len(r.output_ids) - n for r, n in zip(reqs, before)],
                 'accepted': [r.spec_accepted - n for r, n in zip(reqs, accepted)],
                 'spec_rounds': [r.spec_rounds - n for r, n in zip(reqs, rounds)]}
        self.events.append(event)
        for r in reqs:
            self.requests[r.request_id] = r
            if self.check_state:
                assert r.committed_len == r.context_len - 1 == len(r.pages)
                assert r.proposer_state.valid and r.proposer_state.end == r.committed_len
                assert r.proposer_state.pending == (None if r.finished else r.output_ids[-1])
        return result

    def _prefill(self, chunks):
        start = time.perf_counter()
        result = self.prefill(chunks)
        self.events.append({'kind': 'prefill', 'at_s': start - self.started,
                            'seconds': time.perf_counter() - start, 'batch': len(chunks),
                            'request_ids': [r.request_id for r, _ in chunks],
                            'tokens': [n for _, n in chunks]})
        for r, _ in chunks:
            self.requests[r.request_id] = r
        return result


def native_batch(ex, tasks, count, *, check_state=False, short=False):
    import torch
    from sharar.engine.sampler import SamplingParams
    clear(ex)
    proposer = ex.spec.proposer
    hits, fallbacks = proposer.cache_hits, proposer.cache_fallbacks
    reqs = []
    torch.cuda.synchronize()
    start = time.perf_counter()
    with Recorder(ex, check_state=check_state) as record:
        try:
            for i, task in enumerate(tasks):
                ids = task['input_ids'][-128:] if short else task['input_ids']
                req = ex.admit(ids, SamplingParams(max_new_tokens=count + i if short else count, ignore_eos=True))
                req.request_id = i + 1
                reqs.append(req)
            # Serial, identical chunk boundaries for every width. The separate
            # HTTP study measures actual scheduling and batched prefill.
            for req in reqs:
                while not req.prefill_done:
                    ex.prefill_batch([(req, min(2048, req.prefill_len - req.prefill_pos))])
            torch.cuda.synchronize()
            prefilled = time.perf_counter()
            ready = all(ex.spec.ready(r) for r in reqs)
            while active := [r for r in reqs if not r.finished]:
                ex.decode_step(active)
            torch.cuda.synchronize()
            end = time.perf_counter()
            outputs = [{'id': t['id'], 'request_id': r.request_id, 'output_ids': r.output_ids[:],
                        'finish_reason': r.finish_reason, 'spec_rounds': r.spec_rounds,
                        'spec_accepted': r.spec_accepted, 'cached_tokens': r.prefix_len,
                        'ttft_s': r.t_first_token - start, 'total_s': r.t_finish - start}
                       for t, r in zip(tasks, reqs)]
        finally:
            for req in reqs:
                req.cacheable = False
                ex.release(req)
    clear(ex)
    n = sum(len(r['output_ids']) for r in outputs)
    result = {'tasks': outputs, 'events': record.events, 'work': work_summary(record.events),
              'decode_s': end - prefilled, 'prefill_s': prefilled - start, 'total_s': end - start,
              'decode_tps': (n - len(reqs)) / (end - prefilled), 'e2e_tps': n / (end - start),
              'draft_cache_hits': proposer.cache_hits - hits, 'draft_fallbacks': proposer.cache_fallbacks - fallbacks,
              'draft_ready': ready, 'pools_recovered': True}
    result['ok'] = (ready and not result['draft_cache_hits'] and not result['draft_fallbacks']
                    and all(len(r['output_ids']) == (count + i if short else count) for i, r in enumerate(outputs))
                    and all(n == 1 for e in record.events if e['kind'] == 'decode' for n in e['spec_rounds']))
    assert result['ok'], {k: v for k, v in result.items() if k not in ('tasks', 'events')}
    return result


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


def warm_feedback_shapes(ex):
    """Warm every BS4 accepted-row extent, including strided metadata.

    Use only the reserved draft dummy slot. This is benchmark preparation:
    first-use compilation and steady serving are reported separately.
    """
    import torch
    p = ex.spec.proposer
    maximum = ex.max_batch * (p.k + 1)
    assert maximum <= p.pitch
    features = {lid: torch.zeros((maximum, 5120), dtype=torch.bfloat16, device=ex.device)
                for lid in p.model.features}
    tape = torch.arange(maximum, device=ex.device)
    positions, _, pages = torch.stack((tape, tape, tape), dim=1).unbind(1)
    for count in range(1, maximum + 1):
        hidden = p.model.project_features({lid: value[:count] for lid, value in features.items()})
        p.model.context_kv(hidden, positions[:count], p.caches, pages[:count])
    torch.cuda.synchronize()


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
