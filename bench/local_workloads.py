"""Frozen local-use requests over real streaming HTTP; no GPU imports.

The manifest owns prompts/policy. This client owns arrival and stream timing,
and writes every sample (including failures) before evaluating the run.
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import statistics
import time
import urllib.request
from sharar.benchmark_workloads import frozen_source

DEFAULT_MANIFEST = Path(__file__).parent / 'workloads/local_v1.json'


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def materialize(path, snapshot):
    from sharar.api.chat_template import ChatTemplate
    from sharar.tokenizer import SnapshotTokenizer
    path, snapshot = Path(path), Path(snapshot)
    raw = path.read_bytes()
    manifest = json.loads(raw)
    fixtures = {}
    for name, info in manifest['fixtures'].items():
        fixture = (path.parent / info['path']).resolve()
        if not fixture.is_relative_to(path.parent.resolve()):
            raise ValueError('fixture escapes the workload directory')
        # Preserve the built-in manifest's bytes/checksum with one packaged
        # historical fixture. External manifests still own their local files.
        packaged = path.resolve() == DEFAULT_MANIFEST.resolve() and info['path'] == 'request_state_7aebbe6.txt'
        data = frozen_source().encode() if packaged else fixture.read_bytes()
        if digest(data) != info['sha256']:
            raise ValueError(f'fixture checksum changed: {name}')
        fixtures[name] = data.decode()
    template, tok = ChatTemplate(snapshot), SnapshotTokenizer(snapshot)
    cases = []
    for case in manifest['cases']:
        def encode(messages, generation=True):
            messages = [dict(m) for m in messages]
            for m in messages:
                for name, text in fixtures.items():
                    m['content'] = m['content'].replace('{{' + name + '}}', text)
            return tok.encode(template.render(messages, enable_thinking=case['enable_thinking'],
                                             tools=case.get('tools'), add_generation_prompt=generation))
        ids = encode(case['messages'])
        cases.append({**case, 'input_ids': ids, 'input_sha256': digest(json.dumps(ids).encode()),
                      'prime_ids': encode(case['seed_messages'], False) if 'seed_messages' in case else None})
    return {'manifest': manifest, 'manifest_sha256': digest(raw), 'cases': cases,
            'tokenizer_sha256': digest((snapshot / 'tokenizer.json').read_bytes()),
            'template_sha256': digest((snapshot / 'chat_template.jinja').read_bytes())}


def quality(case, text):
    from sharar.api.parsers.reasoning import ReasoningParser
    from sharar.api.parsers.tools import parse_tool_calls
    parser = ReasoningParser(case['enable_thinking'])
    parser.feed(text)
    parser.finish()
    text = parser.content.strip()
    rule = case['quality']
    kind = rule['kind']
    try:
        if kind == 'manual':
            return {'passed': None if text else False, 'rubric': rule['rubric'], 'nonempty': bool(text)}
        if kind == 'json':
            passed = json.loads(text) == rule['expected']
        elif kind == 'exact':
            passed = text == rule['expected']
        elif kind == 'tool':
            _, calls = parse_tool_calls(text, case.get('tools'))
            passed = (len(calls) == 1 and calls[0]['function']['name'] == rule['name']
                      and json.loads(calls[0]['function']['arguments']) == rule['arguments'])
        elif kind == 'clamp':
            # A deliberately tiny, bounded subset: no calls, attributes,
            # imports, loops or decorators. Check behavior, not just syntax.
            tree = ast.parse(text)
            allowed = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.If, ast.IfExp,
                       ast.Return, ast.Compare, ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.Eq,
                       ast.Name, ast.Load, ast.Constant)
            if (len(list(ast.walk(tree))) > 128 or any(not isinstance(n, allowed) for n in ast.walk(tree))
                    or len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef)
                    or tree.body[0].name != 'clamp' or tree.body[0].decorator_list):
                raise ValueError('code is outside the bounded clamp test subset')
            namespace = {'__builtins__': {}}
            exec(compile(tree, '<clamp-smoke>', 'exec'), namespace)
            examples = [(-4, 0, 3, 0), (9, 0, 3, 3), (2, 0, 3, 2), (0, 0, 3, 0),
                        (3, 0, 3, 3), (-2, -3, -1, -2), (5, 2, 2, 2)]
            passed = all(namespace['clamp'](x, lo, hi) == answer for x, lo, hi, answer in examples)
        else:
            raise ValueError(f'unknown quality rule: {kind}')
        return {'passed': bool(passed), 'kind': kind}
    except Exception as exc:
        return {'passed': False, 'kind': kind, 'error': str(exc)}


def json_request(url, payload=None):
    request = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=180) as response:
        raw = response.read()
    try:
        return json.loads(raw)
    except ValueError:
        return raw.decode()


def cache_state_error(case, variant, cached, alignment):
    # Matching excludes the last prompt token (which must produce logits).
    # A short repeat has no interior recurrent checkpoint to restore; its
    # terminal snapshot is too long. Keep it as an explicit expected miss.
    wants_hit = variant == 'session' or (variant == 'repeat' and len(case['input_ids']) > alignment)
    if (wants_hit and cached <= 0) or (not wants_hit and cached != 0):
        return f'wrong cache state: {variant}, cached_tokens={cached}, expected_hit={wants_hit}'
    return None


def stream_request(url, case, rid, *, ids=None, sampling=None, after_tokens=None, deadline=None, on_first_token=None):
    if deadline is not None:
        time.sleep(max(0, deadline - time.perf_counter()))
    start = time.perf_counter()
    body = {'input_ids': case['input_ids'] if ids is None else ids,
            'sampling_params': case['sampling'] if sampling is None else sampling,
            'stream': True, 'rid': rid}
    row = {'id': rid, 'case': case['id'], 'events': [], 'output_ids': [], 'meta': {},
           'arrival_lateness_s': max(0, start - deadline) if deadline is not None else 0,
           'cancelled_by_client': False, 'error': None}
    request = urllib.request.Request(url + '/generate', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            for line in response:
                if not line.startswith(b'data: '):
                    continue
                payload = line[6:].strip()
                if payload == b'[DONE]':
                    break
                event = json.loads(payload)
                if 'error' in event:
                    raise RuntimeError(event['error'])
                count = len(event['output_ids'])
                if count > len(row['output_ids']):
                    first = not row['events']
                    row['events'].append({'t_s': time.perf_counter() - start, 'tokens': count})
                    if first and on_first_token is not None:
                        on_first_token()
                row['output_ids'], row['meta'] = event['output_ids'], event['meta_info']
                if after_tokens is not None and count >= after_tokens:
                    row['cancelled_by_client'] = True
                    break
        if not row['cancelled_by_client'] and not row['meta'].get('finish_reason'):
            raise RuntimeError('stream ended without a terminal event')
    except Exception as exc:
        row['error'] = f'{type(exc).__name__}: {exc}'
    row['completion_s'] = time.perf_counter() - start
    events = row['events']
    row['ttft_s'] = events[0]['t_s'] if events else None
    # An SSE event can contain several speculative tokens. These are delivery
    # gaps between bursts, never invented timestamps within a burst.
    row['delivery_gaps_s'] = [b['t_s'] - a['t_s'] for a, b in zip(events, events[1:])]
    row['output_sha256'] = digest(json.dumps(row['output_ids']).encode())
    return row


def await_idle(url):
    for _ in range(400):
        stats = json_request(url + '/get_server_info')['scheduler']
        if not any(stats.get(k, 0) for k in ('queued', 'prefilling', 'running')):
            return stats
        time.sleep(0.05)
    raise TimeoutError('server did not drain')


def run(url, frozen, tok, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'manifest.json').write_text(json.dumps(frozen, indent=2) + '\n')
    (output / 'server.json').write_text(json.dumps(json_request(url + '/get_server_info'), indent=2) + '\n')
    manifest, cases = frozen['manifest'], frozen['cases']
    by_id = {c['id']: c for c in cases}
    rows = []
    def record(row, scenario, repetition, case, *, check=True, warmup=False):
        row.update(scenario=scenario, repetition=repetition, warmup=warmup)
        text = tok.decode([t for t in row['output_ids'] if t not in (248046, 248044)])
        row['text'] = text
        row['quality'] = quality(case, text) if check else {'passed': None, 'reason': 'partial output'}
        rows.append(row)
        with (output / 'requests.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps({k: row[k] for k in ('id', 'completion_s', 'ttft_s', 'quality', 'error')}), flush=True)
    for case in cases:
        for variant in case['cache_variants']:
            for repetition in range(-manifest['warmups'], manifest['repetitions']):
                await_idle(url)
                json_request(url + '/flush_cache', {})
                rid = f'{case["id"]}-{variant}-{repetition}'
                if variant != 'cold':
                    prime_ids = case['prime_ids'] if variant == 'session' else case['input_ids']
                    prime = stream_request(url, case, rid + '-prime', ids=prime_ids,
                                           sampling={'max_new_tokens': 1, 'temperature': 0})
                    record(prime, variant + '-prime', repetition, case, check=False, warmup=True)
                    if prime['error']:
                        raise RuntimeError('priming failed')
                    await_idle(url)
                row = stream_request(url, case, rid)
                cached = row['meta'].get('cached_tokens', 0)
                cache_error = cache_state_error(case, variant, cached, manifest['cache_checkpoint_alignment'])
                if cache_error:
                    row['error'] = cache_error
                record(row, variant, repetition, case, warmup=repetition < 0)
    for repetition in range(manifest['repetitions']):
        await_idle(url)
        json_request(url + '/flush_cache', {})
        epoch = time.perf_counter() + 0.1
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs = [(by_id[item['case']], pool.submit(stream_request, url, by_id[item['case']],
                     f'mixed-{item["case"]}-{repetition}', deadline=epoch + item['delay_s']))
                    for item in manifest['mixed_arrivals']]
            for case, job in jobs:
                record(job.result(), 'mixed', repetition, case)
    await_idle(url)
    json_request(url + '/flush_cache', {})
    case = by_id[manifest['cancellation']['case']]
    before = json_request(url + '/get_server_info')['scheduler'].get('cancelled', 0)
    row = stream_request(url, case, 'cancel', after_tokens=manifest['cancellation']['after_tokens'])
    stats = await_idle(url)
    row['server_cancelled_delta'] = stats.get('cancelled', 0) - before
    if row['server_cancelled_delta'] != 1:
        row['error'] = 'disconnect did not cancel exactly one server request'
    record(row, 'cancel', 0, case, check=False)
    for budget in manifest['short_budgets']:
        json_request(url + '/flush_cache', {})
        case = by_id['json']
        row = stream_request(url, case, f'budget-{budget}', sampling={**case['sampling'], 'max_new_tokens': budget})
        row['requested_budget'] = budget
        if len(row['output_ids']) > budget:
            row['error'] = 'output exceeded the remaining budget'
        record(row, 'short_budget', budget, case, check=False)
        await_idle(url)
    groups = {}
    for row in rows:
        if not row['warmup']:
            groups.setdefault((row['case'], row['scenario']), []).append(row)
    summary = {'groups': [], 'automatic_failures': [], 'request_failures': [],
               'manual_review_required': [], 'tail_latency_claim': False}
    for (case, scenario), samples in groups.items():
        complete = [s for s in samples if not s['error']]
        summary['groups'].append({'case': case, 'scenario': scenario, 'count': len(samples),
                                 'median_completion_s': statistics.median(s['completion_s'] for s in complete) if complete else None,
                                 'median_ttft_s': statistics.median(s['ttft_s'] for s in complete if s['ttft_s'] is not None) if any(s['ttft_s'] is not None for s in complete) else None})
    for row in rows:
        if row['error']:
            summary['request_failures'].append(row['id'])
        if row['quality'].get('passed') is False:
            summary['automatic_failures'].append(row['id'])
        if 'rubric' in row['quality']:
            summary['manual_review_required'].append(row['id'])
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    from sharar import config
    from sharar.tokenizer import SnapshotTokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    snapshot = config.snapshot_dir()
    report = run(args.url, materialize(args.manifest, snapshot), SnapshotTokenizer(snapshot), args.output)
    if report['automatic_failures'] or report['request_failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
