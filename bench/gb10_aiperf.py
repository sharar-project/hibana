"""Exact-ID, cold-prefix AIPerf comparison client (standard library only).

Run with an exclusively owned, already warmed server. Each AIPerf invocation
issues one request, allowing a confirmed cache flush between identical inputs.
Model loading, client startup, flushes and full-generation warmups are excluded
from reported AIPerf request metrics. Raw response IDs are never reconstructed
from decoded text. The client asserts counts and reports, not masks, divergence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from bench.benchmark_text_server import flush_cache, http_json

MODEL = 'RadixArk/Qwen3.8-27B-NVFP4'
REVISION = '319f741cce68d7914884900c138a1fbb70a42f30'


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def payload_for(ids, budget=1000):
    return {'model': MODEL, 'prompt': ids, 'max_tokens': budget, 'temperature': 0,
            'top_p': 1, 'top_k': -1, 'seed': 42, 'ignore_eos': True,
            'frequency_penalty': 0, 'presence_penalty': 0, 'repetition_penalty': 1,
            'stop': None, 'n': 1, 'stream': True, 'skip_special_tokens': True,
            'return_token_ids': True, 'return_spec_tokens_details': True,
            'stream_options': {'include_usage': True}}


def audit_raw(raw, expected):
    if raw.get('status') != 200 or raw.get('error'):
        raise ValueError(f'Unsuccessful response: {raw.get("status")}, {raw.get("error")}')
    if raw['payload'] != expected:
        raise ValueError('AIPerf payload differs from the frozen request')
    ids, prompt_ids, text, usages, finishes, extensions = [], [], '', [], [], []
    done = False
    for response in raw['responses']:
        for packet in response.get('packets', []):
            if packet['name'] != 'data':
                continue
            if packet['value'] == '[DONE]':
                done = True
                continue
            chunk = json.loads(packet['value'])
            if 'error' in chunk:
                raise ValueError(f'Stream error: {chunk["error"]}')
            if chunk.get('usage'):
                usages.append(chunk['usage'])
            if chunk.get('sglext'):
                extensions.append(chunk['sglext'])
            for choice in chunk.get('choices', []):
                ids.extend(choice.get('token_ids') or [])
                prompt_ids.extend(choice.get('prompt_token_ids') or [])
                text += choice.get('text') or ''
                if choice.get('finish_reason'):
                    finishes.append(choice['finish_reason'])
    if not done or not usages or finishes != ['length']:
        raise ValueError(f'Incomplete response: done={done}, usage={usages}, finish={finishes}')
    usage = usages[-1]
    if prompt_ids != expected['prompt']:
        raise ValueError('Returned prompt IDs differ from the frozen input')
    if len(ids) != expected['max_tokens'] or usage['completion_tokens'] != len(ids):
        raise ValueError('Output ID count differs from requested or reported count')
    if usage['prompt_tokens'] != len(prompt_ids) or usage['total_tokens'] != len(ids) + len(prompt_ids):
        raise ValueError('Server token usage is inconsistent')
    cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0)
    if cached != 0:
        raise ValueError(f'Unexpected cached input: {cached}')
    return {'output_ids': ids, 'text': text, 'usage': usage, 'finish_reason': finishes[0],
            'input_ids_sha256': digest(prompt_ids), 'output_ids_sha256': digest(ids),
            'text_sha256': hashlib.sha256(text.encode()).hexdigest(), 'sglext': extensions}


def trial(url, aiperf, ids, output, warmup, case, repetition):
    output.mkdir(exist_ok=False)
    payload = payload_for(ids)
    dataset = {'data': [{'session_id': case, 'payloads': [payload]}]}
    (output / 'input.json').write_text(json.dumps(dataset))
    flush_cache(url)
    before = http_json(url + '/get_server_info')
    (output / 'before.json').write_text(json.dumps(before, indent=2))
    command = [str(aiperf), 'profile', '--model', MODEL, '--endpoint-type', 'completions',
               '--url', url, '--input-file', str(output / 'input.json'),
               '--custom-dataset-type', 'inputs_json', '--streaming', '--use-server-token-count',
               '--tokenizer', MODEL, '--tokenizer-revision', REVISION,
               '--concurrency', '1', '--request-count', '1', '--random-seed', '42',
               '--request-timeout-seconds', '600', '--ui', 'none', '--no-server-metrics',
               '--no-gpu-telemetry', '--export-level', 'raw', '--export-http-trace',
               '--artifact-dir', str(output / 'aiperf')]
    (output / 'command.json').write_text(json.dumps(command, indent=2))
    with (output / 'client.log').open('w') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                       timeout=720, env={**os.environ, 'HF_HUB_OFFLINE': '1',
                                         'TOKENIZERS_PARALLELISM': 'false'})
    after = http_json(url + '/get_server_info')
    (output / 'after.json').write_text(json.dumps(after, indent=2))
    records = [json.loads(line) for line in (output / 'aiperf/profile_export_raw.jsonl').read_text().splitlines()]
    if len(records) != 1:
        raise ValueError(f'AIPerf issued {len(records)} requests, expected one')
    result = audit_raw(records[0], payload)
    measured = [json.loads(line) for line in (output / 'aiperf/profile_export.jsonl').read_text().splitlines()]
    if len(measured) != 1:
        raise ValueError('Unexpected AIPerf metrics record count')
    metrics = measured[0]['metrics']
    assert metrics['input_sequence_length']['value'] == 10000
    assert metrics['output_sequence_length']['value'] == 1000
    result.update(case=case, repetition=repetition, warmup=warmup, aiperf_metrics=metrics)
    if 'spec_stats' in after:
        result['verify_rounds'] = after['spec_stats']['rounds'] - before['spec_stats']['rounds']
    else:
        details = [ext['spec_tokens_details'] for ext in result['sglext'] if ext.get('spec_tokens_details')]
        if details:
            detail = details[-1]
            if isinstance(detail, list):
                detail = detail[0]
            result['verify_rounds'] = detail['spec_verify_ct']
    (output / 'audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ('case', 'repetition', 'warmup', 'output_ids_sha256')} |
                     {'latency_ms': metrics['request_latency']['value'],
                      'ttft_ms': metrics['time_to_first_token']['value'],
                      'rounds': result.get('verify_rounds')}), flush=True)
    return result


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
