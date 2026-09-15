"""Convenient AIPerf client for an already running Sharar or SGLang server."""
import datetime as dt
import json
import os
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys
from urllib.error import URLError
from urllib.parse import urlsplit

from sharar.benchmark_aiperf import MODEL, REVISION, digest, http_json, trial
from sharar.benchmark_workloads import LEGACY_WORKLOADS, VERSION, WORKLOADS, materialize

AIPERF_VERSION = '0.12.0'


def add_arguments(parser):
    parser.add_argument('--url', default='http://127.0.0.1:30001', help='already running server (default: Sharar on port 30001)')
    parser.add_argument('--workload', choices=(*WORKLOADS, *LEGACY_WORKLOADS, 'all'), default='code', help='code_v1 is the pre-2026-09-15 prompt (stops at output 98), kept for historical comparisons')
    parser.add_argument('--input-tokens', type=int, default=10000)
    parser.add_argument('--output-tokens', type=int, default=1000)
    parser.add_argument('--repeats', type=int, default=3, help='timed requests per workload')
    parser.add_argument('--warmups', type=int, default=1, help='excluded full-generation warmups per workload')
    parser.add_argument('--output', type=Path, help='new result directory (default: timestamped under SHARAR_RESULTS)')
    parser.add_argument('--inputs', type=Path, help='reuse frozen-inputs.json from another run')
    parser.add_argument('--baseline', type=Path, help='previous result directory to compare TPS and output hashes')
    parser.add_argument('--aiperf', help='AIPerf executable (normally discovered automatically)')


def load_config():
    """Read only benchmark paths from the repository config; never execute shell."""
    path = Path(os.environ.get('SHARAR_CONFIG_FILE', Path.cwd() / 'config/gb10.env'))
    if path.is_file():
        for line in path.read_text().splitlines():
            key, sep, value = line.partition('=')
            key = key.strip()
            if sep and key in ('SHARAR_RESULTS', 'SHARAR_HF_CACHE', 'SHARAR_AIPERF'):
                parts = shlex.split(value, comments=True)
                if len(parts) == 1:
                    os.environ.setdefault(key, parts[0])
    if cache := os.environ.get('SHARAR_HF_CACHE'):
        root = Path(cache).expanduser()
        os.environ.setdefault('HF_HUB_CACHE', str(root / 'hub' if (root / 'hub').is_dir() else root))


def results_root():
    state = Path(os.environ.get('XDG_STATE_HOME', Path.home() / '.local/state'))
    return Path(os.environ.get('SHARAR_RESULTS', state / 'sharar/gb10')).expanduser()


def resolve_aiperf(explicit=None):
    configured = explicit or os.environ.get('SHARAR_AIPERF')
    candidates = [configured] if configured else [
        'aiperf', str(Path(sys.executable).parent / 'aiperf'),
        str(Path.cwd() / '.venv-aiperf/bin/aiperf'),
        str(results_root() / 'aiperf-venv/bin/aiperf')]
    for candidate in candidates:
        if found := shutil.which(os.path.expanduser(candidate)):
            return Path(found).resolve()
    raise ValueError('AIPerf not found. Install it with: python3 -m venv .venv-aiperf && '
                     f'.venv-aiperf/bin/pip install aiperf=={AIPERF_VERSION}; or set --aiperf PATH')


def summarize(results):
    summaries = {}
    for case in dict.fromkeys(row['case'] for row in results):
        rows = [r for r in results if r['case'] == case and not r['warmup']]
        if not rows:
            continue
        latency = sum(r['latency_s'] for r in rows)
        decode_time = sum(r['latency_s'] - r['ttft_s'] for r in rows)
        count = sum(r['usage']['completion_tokens'] for r in rows)
        hashes = sorted({r['output_ids_sha256'] for r in rows})
        summaries[case] = {
            'requests': len(rows), 'input_tokens': rows[0]['usage']['prompt_tokens'],
            'output_tokens': rows[0]['usage']['completion_tokens'],
            'input_ids_sha256': rows[0]['input_ids_sha256'],
            'latency_s': latency / len(rows), 'ttft_s': statistics.fmean(r['ttft_s'] for r in rows),
            'output_tps': count / latency, 'decode_tps': (count - len(rows)) / decode_time,
            'output_ids_sha256': hashes, 'repeatable_output': len(hashes) == 1,
        }
        if all(r.get('verify_rounds', 0) > 0 for r in rows):
            rounds = sum(r['verify_rounds'] for r in rows)
            summaries[case]['emitted_tokens_per_round'] = (count - len(rows)) / rounds
    return summaries


def compare(summary, baseline):
    if baseline.get('schema') != 1 or not baseline.get('completed'):
        raise ValueError('Baseline must be a completed sharar benchmark run')
    if summary['protocol'] != baseline['protocol']:
        raise ValueError('Baseline uses a different model or benchmark protocol')
    comparison = {}
    for case, row in summary['cases'].items():
        old = baseline['cases'].get(case)
        if not old or any(row[k] != old[k] for k in ('input_ids_sha256', 'input_tokens', 'output_tokens')):
            raise ValueError(f'Baseline inputs/token budgets differ for {case}')
        comparison[case] = {
            'output_tps_change_pct': 100 * (row['output_tps'] / old['output_tps'] - 1),
            'decode_tps_change_pct': 100 * (row['decode_tps'] / old['decode_tps'] - 1),
            'same_output_ids': row['repeatable_output'] and old['repeatable_output']
                               and row['output_ids_sha256'] == old['output_ids_sha256'],
        }
    return comparison


def run(args):
    if args.repeats < 1 or args.warmups < 0 or args.input_tokens < 1 or args.output_tokens < 2:
        raise ValueError('Use repeats >= 1, warmups >= 0, input tokens >= 1 and output tokens >= 2')
    if args.input_tokens + args.output_tokens > 32768:
        raise ValueError('Input and output token budgets must fit the validated 32768-token profile')
    url = args.url.rstrip('/')
    if url.endswith('/v1'):
        url = url[:-3]
    parsed = urlsplit(url)
    if parsed.scheme not in ('http', 'https') or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError('--url must be an HTTP(S) server base URL')
    load_config()
    aiperf = resolve_aiperf(args.aiperf)
    version = subprocess.run([str(aiperf), '--version'], capture_output=True, text=True, check=True, timeout=30).stdout.strip()
    if AIPERF_VERSION not in version.split():
        raise ValueError(f'Expected AIPerf {AIPERF_VERSION}, found {version!r}; use the pinned version')
    info = http_json(url + '/get_server_info')
    cases = WORKLOADS if args.workload == 'all' else (args.workload,)

    def encode(text):
        response = http_json(url + '/v1/tokenize', payload={'model': MODEL, 'prompt': text, 'add_special_tokens': False})
        ids = response.get('tokens') if isinstance(response, dict) else None
        if not isinstance(ids, list) or not ids or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError('Server /v1/tokenize did not return token IDs')
        return ids

    frozen = json.loads(args.inputs.read_text()) if args.inputs else {
        case: [materialize(encode, case, args.input_tokens)] for case in cases}
    for case in cases:
        rows = frozen.get(case)
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], list):
            raise ValueError(f'Frozen inputs are missing a token-ID row for {case}')
        if len(rows[0]) != args.input_tokens or any(type(t) is not int or t < 0 for t in rows[0]):
            raise ValueError(f'Frozen {case} input differs from the requested token budget')
    baseline = json.loads((args.baseline / 'summary.json').read_text()) if args.baseline else None
    protocol = {'model': MODEL, 'revision': REVISION, 'aiperf_version': AIPERF_VERSION,
                'temperature': 0, 'seed': 42, 'concurrency': 1, 'ignore_eos': True, 'prefix_cache': 'cold'}
    # Reject mismatched comparisons before spending GPU time.
    if baseline:
        compare({'protocol': protocol, 'cases': {case: {
            'input_ids_sha256': digest(frozen[case][0]), 'input_tokens': args.input_tokens,
            'output_tokens': args.output_tokens, 'output_tps': 1, 'decode_tps': 1,
            'repeatable_output': False} for case in cases}}, baseline)
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output = (args.output or results_root() / 'aiperf' / stamp).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {'schema': 1, 'completed': False, 'protocol': protocol, 'workload_version': VERSION,
               'url': url, 'aiperf_executable': str(aiperf), 'aiperf_version_output': version,
               'warmups_per_case': args.warmups, 'repeats_per_case': args.repeats,
               'server_info': info, 'cases': {}}
    (output / 'frozen-inputs.json').write_text(json.dumps({case: frozen[case] for case in cases}) + '\n')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'AIPerf {AIPERF_VERSION}: {url}\n{args.input_tokens:,} in / {args.output_tokens:,} out; '
          f'greedy, concurrency 1, cold prefix\nResults: {output}', flush=True)
    results = []
    for case in cases:
        for rep in range(-args.warmups, args.repeats):
            warmup = rep < 0
            label = f'warmup {rep + args.warmups + 1}/{args.warmups}' if warmup else f'run {rep + 1}/{args.repeats}'
            print(f'  {case}: {label} ...', flush=True)
            row = trial(url, aiperf, frozen[case][0], output / f'{case}-{rep + args.warmups}',
                        warmup, case, rep, output_tokens=args.output_tokens, verbose=False)
            results.append({k: v for k, v in row.items() if k not in ('output_ids', 'text')})
            (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
            print(f'    {row["output_tps"]:.2f} TPS end-to-end; {row["decode_tps"]:.2f} TPS decode', flush=True)
    summary['cases'] = summarize(results)
    if baseline:
        summary['comparison'] = compare(summary, baseline)
    summary['completed'] = True
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('\nWorkload       TPS (end-to-end)   TPS (decode)   TTFT (s)   Latency (s)')
    for case, row in summary['cases'].items():
        print(f'{case:<14} {row["output_tps"]:>16.2f} {row["decode_tps"]:>14.2f} {row["ttft_s"]:>10.3f} {row["latency_s"]:>13.3f}')
        if not row['repeatable_output']:
            print(f'  {case}: output IDs varied across timed repetitions')
        if baseline:
            change = summary['comparison'][case]
            print(f'  vs baseline: end-to-end TPS {change["output_tps_change_pct"]:+.2f}%, '
                  f'decode TPS {change["decode_tps_change_pct"]:+.2f}%; '
                  f'output IDs {"match" if change["same_output_ids"] else "differ"}')
    print(f'Warmups excluded. End-to-end TPS includes prefill. Summary: {output / "summary.json"}')
    return 0


def benchmark_cli(args):
    try:
        return run(args)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, URLError, subprocess.SubprocessError) as exc:
        print(f'sharar benchmark: {exc}', file=sys.stderr)
        return 1
