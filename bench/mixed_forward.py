"""Frozen real HTTP comparison of opt-in mixed target execution on GB10.

Run under the exclusive GPU harness. The same process retains the extra plan
workspace in every mode. HTTP timing has no CUDA events or profiler callbacks.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
from pathlib import Path
import statistics
import threading
import time

import torch

from bench import local_workloads as workloads

MANIFEST = Path(__file__).parent / 'workloads/mixed_v1.json'
METRICS = ('anchor_completion_s', 'arrival_completion_s', 'arrival_ttft_s',
           'anchor_max_gap_s', 'arrival_max_gap_s')


@contextmanager
def mixed_groups(ex, cases):
    """Record actual membership, including failed attempts, in both modes."""
    names = {tuple(c['input_ids']): c['id'] for c in cases}
    observations, original = [], ex.mixed_prefill
    def member(req):
        return {'request_id': req.request_id, 'case': names.get(tuple(req.prompt_ids)),
                'prefill_pos': req.prefill_pos, 'output_tokens': len(req.output_ids)}
    def observed(chunks, verifying):
        row = {'prefill': [{**member(r), 'rows': n} for r, n in chunks],
               'verify': [member(r) for r in verifying]}
        row['mixed'] = original(chunks, verifying)
        observations.append(row)
        return row['mixed']
    ex.mixed_prefill = observed
    try:
        yield observations
    finally:
        ex.mixed_prefill = original


def selected_case(url, case, frozen, tok, rid, observations):
    by_id = {c['id']: c for c in frozen['cases']}
    anchor, arrival = by_id[case['anchor']], by_id[case['arrival']]
    workloads.await_idle(url)
    workloads.json_request(url + '/flush_cache', {})
    if case['cache'] == 'session':
        prime = workloads.stream_request(url, arrival, rid + '-prime', ids=arrival['prime_ids'],
                                         sampling={'max_new_tokens': 1, 'temperature': 0})
        if prime['error']:
            raise RuntimeError('session priming failed')
        workloads.await_idle(url)
    before = workloads.json_request(url + '/get_server_info')['scheduler']['mixed_steps']
    group_start = len(observations)
    torch.cuda.reset_peak_memory_stats()
    with ThreadPoolExecutor(max_workers=4) as pool:
        first = [threading.Event() for _ in range(case['anchor_batch'])]
        anchors = [pool.submit(workloads.stream_request, url, anchor, f'{rid}-anchor-{i}',
                              on_first_token=event.set) for i, event in enumerate(first)]
        for event in first:
            if not event.wait(30):
                raise RuntimeError('anchor did not produce a first token')
        newcomer = pool.submit(workloads.stream_request, url, arrival, rid + '-arrival')
        rows = [future.result() for future in anchors] + [newcomer.result()]
    after = workloads.await_idle(url)['mixed_steps']
    peak = {'allocated_bytes': torch.cuda.max_memory_allocated(),
            'reserved_bytes': torch.cuda.max_memory_reserved()}
    groups = observations[group_start:]
    for row, definition in zip(rows, [anchor] * len(anchors) + [arrival]):
        row['text'] = tok.decode([t for t in row['output_ids'] if t not in (248046, 248044)])
        row['quality'] = workloads.quality(definition, row['text'])
    cached = rows[-1]['meta'].get('cached_tokens', 0)
    cache_error = workloads.cache_state_error(arrival, case['cache'], cached,
                                              frozen['manifest']['cache_checkpoint_alignment'])
    return {'requests': rows, 'mixed_steps': after - before, 'groups': groups,
            'arrival_mixed': any(g['mixed'] and any(r['case'] == arrival['id'] for r in g['prefill']) for g in groups),
            'cache_error': cache_error, 'cuda_peak': peak,
            'anchor_completion_s': statistics.mean(row['completion_s'] for row in rows[:-1]),
            'arrival_completion_s': rows[-1]['completion_s'], 'arrival_ttft_s': rows[-1]['ttft_s'],
            'anchor_max_gap_s': max((gap for row in rows[:-1] for gap in row['delivery_gaps_s']), default=0.),
            'arrival_max_gap_s': max(rows[-1]['delivery_gaps_s'], default=0.),
            'automatic_passed': not cache_error and all(not row['error'] and row['quality']['passed'] is not False for row in rows),
            'rubric_review_pending': any(row['quality']['passed'] is None for row in rows)}


def summarize(rows):
    result = {}
    for case in sorted({r['case'] for r in rows}):
        selected = [r for r in rows if r['case'] == case and not r['warmup'] and not r['ablation']]
        modes = {}
        for mode in ('off', 'full'):
            samples = [r for r in selected if r['mode'] == mode]
            if not samples:
                continue
            modes[mode] = {'samples': len(samples), 'automatic_passed': all(r['automatic_passed'] for r in samples),
                           'eligible': all(r['mixed_steps'] > 0 and r['arrival_mixed'] for r in samples) if mode != 'off' else True,
                           **{metric: statistics.median(r[metric] for r in samples)
                              if all(r[metric] is not None for r in samples) else None for metric in METRICS}}
        cell = {'modes': modes, 'rubric_review_pending': any(r['rubric_review_pending'] for r in selected)}
        if len(modes) == 2 and all(m['automatic_passed'] and m['eligible'] for m in modes.values()):
            cell['anchor_reduction'] = 1 - modes['full']['anchor_completion_s'] / modes['off']['anchor_completion_s']
            cell['arrival_regression'] = modes['full']['arrival_completion_s'] / modes['off']['arrival_completion_s'] - 1
            cell['arrival_ttft_regression'] = modes['full']['arrival_ttft_s'] / modes['off']['arrival_ttft_s'] - 1
            for metric in ('anchor_max_gap_s', 'arrival_max_gap_s'):
                baseline = modes['off'][metric]
                cell[metric + '_regression'] = modes['full'][metric] / baseline - 1 if baseline else None
        result[case] = cell
    return result


@torch.inference_mode()
def main():
    from sharar import config
    from sharar.engine.executor import Executor
    from sharar.model.qwen38 import Qwen38
    from sharar.tokenizer import SnapshotTokenizer
    from phase7c import http_server
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protected', action='store_true', help='also run the full local-v1 protection suite in off/full modes')
    parser.add_argument('--pilot', action='store_true', help='one warmup and one measured ABBA block; never a promotion result')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(MANIFEST.read_text())
    frozen = workloads.materialize(workloads.DEFAULT_MANIFEST, config.snapshot_dir())
    assert manifest['target_revision'] == config.MODEL_REVISION
    (args.output / 'manifest.json').write_text(json.dumps({'experiment': manifest, 'workloads': frozen}, indent=2) + '\n')
    root = Path(config.__file__).resolve().parents[1]
    report = {'pilot': args.pilot, 'target': config.MODEL_REVISION, 'device': torch.cuda.get_device_name(),
              'torch': torch.__version__, 'sources': {str(p.relative_to(root)): workloads.digest(p.read_bytes())
                  for folder in ('sharar', 'bench', 'tests') for p in (root / folder).rglob('*.py')},
              'physical_dram_bytes': None, 'selected': [], 'protected': {}}
    def save():
        report['summary'] = summarize(report['selected'])
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    ex = Executor(Qwen38(config.snapshot_dir()), num_pages=35000, num_slots=12)
    ex.enable_spec(7, proposer='dflash2', state_strategy='reconstruct', log=print)
    ex.enable_mixed('full')
    ex.capture_graphs(log=print)
    ex.warmup(log=print)
    tok = SnapshotTokenizer(config.snapshot_dir())
    report['optional_workspace_bytes_in_all_modes'] = ex._mixed_verify_ws.numel()
    description = {'profile': manifest['version'], 'graphs': True, 'proposer': 'dflash2', 'k': 7,
                   'mixed_forward': 'controlled by benchmark while idle'}
    with mixed_groups(ex, frozen['cases']) as observations, http_server(ex, tok, config.snapshot_dir(), description) as url:
        for case in manifest['selected']:
            blocks = 1 if args.pilot else manifest['repetitions']
            for block in range(-manifest['warmup_blocks'], blocks):
                for position, mode in enumerate(manifest['order']):
                    workloads.await_idle(url)
                    ex.mixed_mode = mode
                    row = selected_case(url, case, frozen, tok, f'{case["id"]}-{block}-{position}-{mode}', observations)
                    row.update(case=case['id'], block=block, position=position, mode=mode, warmup=block < 0, ablation=False)
                    report['selected'].append(row)
                    save()
                    print(json.dumps({k: row[k] for k in ('case', 'block', 'mode', 'mixed_steps', 'anchor_completion_s',
                                                         'arrival_completion_s', 'automatic_passed')}), flush=True)
            for mode in ('separate', 'mlp'):
                for repeat in range(1 if args.pilot else 3):
                    workloads.await_idle(url)
                    ex.mixed_mode = mode
                    row = selected_case(url, case, frozen, tok, f'{case["id"]}-ablation-{mode}-{repeat}', observations)
                    row.update(case=case['id'], block=repeat, mode=mode, warmup=False, ablation=True)
                    report['selected'].append(row)
                    save()
            ex.ledger.check_floor()
        if args.protected:
            for mode in ('off', 'full'):
                workloads.await_idle(url)
                ex.mixed_mode = mode
                report['protected'][mode] = workloads.run(url, frozen, tok, args.output / ('protected-' + mode))
                save()
    report['completed'] = True
    save()


if __name__ == '__main__':
    main()
