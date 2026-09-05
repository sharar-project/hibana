"""Engine-neutral native HTTP benchmark with frozen token inputs.

Run each server exclusively under the host memory guard. This client requires
cumulative output_ids (SGLang's default); it never synthesizes per-token arrival
times inside speculative bursts. The CLI can freeze inputs or measure a URL.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import time
from urllib.error import HTTPError

from bench import local_workloads as workloads

MANIFEST = Path(__file__).parent / 'workloads/competitive_v1.json'


def sampling_params(case, profile):
    params = dict(case['sampling'])
    if profile.startswith('sglang') and 'seed' in params:
        params['sampling_seed'] = params.pop('seed')
    return params


def materialize(snapshot):
    from sharar.tokenizer import SnapshotTokenizer
    tok = SnapshotTokenizer(snapshot)
    spec = json.loads(MANIFEST.read_text())
    local = workloads.materialize(workloads.DEFAULT_MANIFEST, snapshot)
    filler = tok.encode('The quick brown fox jumps over the lazy dog while a local inference server processes a deterministic text benchmark. ')
    fixed = []
    for length in spec['fixed_work']['prompt_lengths']:
        for batch in spec['fixed_work']['concurrency']:
            cases = []
            for i in range(batch):
                prefix = tok.encode(f'Unique benchmark request {length}-{batch}-{i}. Continue processing text: ')
                ids = (prefix + filler * ((length // len(filler)) + 1))[:length]
                cases.append({'id': f'fixed-{length}-b{batch}-{i}', 'input_ids': ids,
                              'input_sha256': workloads.digest(json.dumps(ids).encode()),
                              'sampling': {'max_new_tokens': spec['fixed_work']['output_tokens'],
                                           'temperature': 0, 'ignore_eos': True}})
            fixed.append({'id': f'fixed-{length}-b{batch}', 'cases': cases})
    return {'spec': spec, 'spec_sha256': workloads.digest(MANIFEST.read_bytes()), 'local': local, 'fixed': fixed}


def flush(url):
    # A finished HTTP response may precede the scheduler's final cleanup.
    for _ in range(100):
        try:
            reply = workloads.json_request(url + '/flush_cache', {})
        except HTTPError as exc:
            if exc.code != 400:
                raise
            reply = exc.read().decode()
        # Sharar immediately invalidates future matches; existing readers keep
        # their leases. SGLang refuses a flush until its active work drains.
        if isinstance(reply, str) and reply.startswith(('Cache flushed', 'Cache invalidated;')):
            return
        time.sleep(.05)
    raise RuntimeError(f'cache flush failed: {reply!r}')


def summarize(samples):
    result = {}
    for cell in sorted({r['cell'] for r in samples}):
        rows = [r for r in samples if r['cell'] == cell and not r['warmup']]
        if not rows:
            continue
        requests = [r for row in rows for r in row['requests']]
        ok = all(row['valid'] for row in rows)
        values = {'samples': len(rows), 'valid': ok, 'errors': [r['error'] for r in requests if r['error']],
                  'output_tokens': [len(r['output_ids']) for r in requests],
                  'cached_tokens': [r['meta'].get('cached_tokens') for r in requests],
                  'automatic_quality': all(r.get('quality', {}).get('passed') is not False for r in requests),
                  'rubric_pending': any(r.get('quality', {}).get('passed') is None for r in requests if 'quality' in r)}
        if ok:
            decode = [r['decode_tok_s'] for r in requests if r['decode_tok_s'] is not None]
            values.update(aggregate_tok_s=statistics.median(row['aggregate_tok_s'] for row in rows),
                          completion_s=statistics.median(r['completion_s'] for r in requests),
                          ttft_s=statistics.median(r['ttft_s'] for r in requests),
                          decode_tok_s=statistics.median(decode) if decode else None)
        result[cell] = values
    return result


def run(url, frozen, tok, output, profile):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {'profile': profile, 'frozen_sha256': workloads.digest(json.dumps(frozen, sort_keys=True).encode()),
              'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'server_info': workloads.json_request(url + '/get_server_info'), 'samples': []}
    (output / 'inputs.json').write_text(json.dumps(frozen, indent=2) + '\n')
    def save():
        report['summary'] = summarize(report['samples'])
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    def measure(cell, cases, block, *, fixed=False, cache='cold'):
        flush(url)
        if cache == 'session':
            prime = workloads.stream_request(url, cases[0], 'prime', ids=cases[0]['prime_ids'],
                                             sampling={'max_new_tokens': 1, 'temperature': 0})
            if prime['error']:
                raise RuntimeError(prime['error'])
        deadline = time.perf_counter() + .05
        with ThreadPoolExecutor(max_workers=len(cases)) as pool:
            futures = [pool.submit(workloads.stream_request, url, case, f'{cell}-{block}-{i}', deadline=deadline,
                                   sampling=sampling_params(case, profile))
                       for i, case in enumerate(cases)]
            rows = [f.result() for f in futures]
        for row, case in zip(rows, cases):
            row['wire_sampling_params'] = sampling_params(case, profile)
            events = row['events']
            if row['meta'].get('completion_tokens') != len(row['output_ids']):
                row['error'] = 'server token count disagrees with cumulative output IDs'
            if fixed and len(row['output_ids']) != frozen['spec']['fixed_work']['output_tokens']:
                row['error'] = 'fixed output budget was not fulfilled'
            if cache == 'cold' and len(cases) == 1 and row['meta'].get('cached_tokens', 0) != 0:
                row['error'] = 'cold request unexpectedly reused a cached prefix'
            if cache == 'session' and row['meta'].get('cached_tokens', 0) <= 0:
                row['error'] = 'primed session did not hit the prefix cache'
            elapsed = events[-1]['t_s'] - events[0]['t_s'] if events else 0
            row['decode_tok_s'] = (len(row['output_ids']) - events[0]['tokens']) / elapsed if elapsed > 0 else None
            row['text'] = tok.decode([t for t in row['output_ids'] if t not in (248046, 248044)])
            if not fixed:
                row['quality'] = workloads.quality(case, row['text'])
        makespan = max(r['completion_s'] + r['arrival_lateness_s'] for r in rows)
        result = {'cell': cell, 'block': block, 'warmup': block < 0, 'requests': rows, 'makespan_s': makespan,
                  'aggregate_tok_s': sum(len(r['output_ids']) for r in rows) / makespan,
                  'valid': all(not r['error'] for r in rows)}
        report['samples'].append(result)
        save()
        print(json.dumps({k: result[k] for k in ('cell', 'block', 'aggregate_tok_s', 'valid')}), flush=True)
    policy = frozen['spec']['fixed_work']
    for cell in frozen['fixed']:
        for block in range(-policy['warmups'], policy['repetitions']):
            measure(cell['id'], cell['cases'], block, fixed=True)
    policy = frozen['spec']['natural']
    for case in frozen['local']['cases']:
        if case['id'] not in policy['cases']:
            continue
        for cache in (('cold', 'session') if case['id'] == 'session' else ('cold',)):
            for block in range(-policy['warmups'], policy['repetitions']):
                measure(case['id'] + '-' + cache, [case], block, cache=cache)
    report['completed'] = True
    report['final_server_info'] = workloads.json_request(url + '/get_server_info')
    save()
    return report


def main():
    from sharar import config
    from sharar.tokenizer import SnapshotTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--url')
    parser.add_argument('--profile')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.freeze:
        args.freeze.write_text(json.dumps(materialize(config.snapshot_dir()), indent=2) + '\n')
    else:
        if not all((args.inputs, args.url, args.output, args.profile)):
            parser.error('measurement requires --inputs, --url, --output and --profile')
        run(args.url, json.loads(args.inputs.read_text()), SnapshotTokenizer(config.snapshot_dir()), args.output, args.profile)


if __name__ == '__main__':
    main()
