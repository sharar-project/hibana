"""BS4 HTTP scheduler experiment over a frozen finite/replenished arrival trace.

Uses the real EngineLoop and streaming /generate endpoint. Each replenished
lane sends its next frozen request after its previous response completes.
Changing service times therefore changes arrival times: this is a closed-loop
concurrency-four workload, not a fixed open-loop arrival process.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
import http.client
import json
from pathlib import Path
import socket
import threading
import time
from urllib.parse import urlsplit

from bench.benchmark_text_server import flush_cache, http_json, percentile, run_quality_canaries
from bench.gb10_bs4_engine import Recorder, make_runtime, native_batch, warm_feedback_shapes, work_summary
from bench.gb10_family_eval import clear, identity, save, sha_file


@contextmanager
def server(ex, cfg):
    import uvicorn
    from sharar import config
    from sharar.api.chat_template import ChatTemplate
    from sharar.api.server import EngineLoop, create_app
    from sharar.engine.scheduler import Scheduler
    from sharar.tokenizer import SnapshotTokenizer
    snapshot = config.snapshot_dir()
    engine = EngineLoop(Scheduler(ex, cfg))
    engine.start()
    app = create_app(engine, SnapshotTokenizer(snapshot), ChatTemplate(snapshot),
                     {'engine': 'sharar', 'scheduler_config': asdict(cfg)})
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    url = f'http://127.0.0.1:{sock.getsockname()[1]}'
    service = uvicorn.Server(uvicorn.Config(app, log_level='warning', access_log=False))
    thread = threading.Thread(target=service.run, kwargs={'sockets': [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 30
        while not service.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError('HTTP server failed to start')
            time.sleep(.05)
        yield url, engine.sched
        await_idle(url)
    finally:
        service.should_exit = True
        thread.join(timeout=30)
        engine.stop = True
        engine.thread.join(timeout=30)
        sock.close()
        if thread.is_alive() or engine.thread.is_alive():
            raise RuntimeError('HTTP benchmark server failed to stop')


def await_idle(url):
    deadline = time.monotonic() + 30
    while True:
        info = http_json(url + '/get_server_info')['scheduler']
        if not any(info[k] for k in ('running', 'prefilling', 'queued')):
            return
        if time.monotonic() > deadline:
            raise RuntimeError('server did not become idle')
        time.sleep(.05)


def request(url, item, origin, *, allow_cached=False):
    parsed = urlsplit(url)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=300)
    body = json.dumps({'rid': item['id'], 'input_ids': item['input_ids'], 'stream': True,
                       'sampling_params': {'max_new_tokens': item['output_tokens'], 'temperature': 0,
                                           'ignore_eos': True}})
    start = time.perf_counter()
    arrivals, output, meta, error, done = [], [], {}, None, False
    try:
        connection.request('POST', '/generate', body, {'Content-Type': 'application/json'})
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f'HTTP {response.status}: {response.read().decode(errors="replace")}')
        for raw in iter(response.readline, b''):
            if not raw.startswith(b'data: '):
                continue
            payload = raw[6:].strip()
            if payload == b'[DONE]':
                done = True
                break
            value = json.loads(payload)
            current = value.get('output_ids', [])
            if len(current) > len(output):
                arrivals.append({'at_s': time.perf_counter() - start, 'tokens': len(current)})
            output, meta = current, value.get('meta_info', {})
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        connection.close()
    end = time.perf_counter()
    ttft = arrivals[0]['at_s'] if arrivals else None
    gaps = [b['at_s'] - a['at_s'] for a, b in zip(arrivals, arrivals[1:])]
    return {'id': item['id'], 'input_tokens': len(item['input_ids']), 'expected_tokens': item['output_tokens'],
            'started_at_s': start - origin, 'finished_at_s': end - origin, 'total_s': end - start,
            'ttft_s': ttft, 'max_stream_gap_s': max(gaps, default=None), 'arrivals': arrivals,
            'output_ids': output, 'meta': meta, 'error': error,
            'ok': (error is None and done and len(output) == item['output_tokens']
                   and meta.get('completion_tokens') == item['output_tokens']
                   and (allow_cached or meta.get('cached_tokens', 0) == 0))}


def run_trace(ex, url, trace):
    import torch
    await_idle(url)
    flush_cache(url)
    torch.cuda.synchronize()
    proposer = ex.spec.proposer
    hits, fallbacks = proposer.cache_hits, proposer.cache_fallbacks
    before = http_json(url + '/get_server_info')
    lanes = trace['lanes']
    barrier = threading.Barrier(len(lanes) + 1)
    origin = [0.]
    def lane(items):
        barrier.wait(timeout=30)
        return [request(url, item, origin[0]) for item in items]
    with Recorder(ex) as record, ThreadPoolExecutor(max_workers=len(lanes)) as pool:
        futures = [pool.submit(lane, items) for items in lanes]
        origin[0] = time.perf_counter()
        record.started = origin[0]
        barrier.wait(timeout=30)
        results = [r for future in futures for r in future.result()]
        elapsed = time.perf_counter() - origin[0]
        await_idle(url)
    torch.cuda.synchronize()
    after = http_json(url + '/get_server_info')
    actual = [{'request_id': r.request_id, 'output_ids': r.output_ids, 'spec_rounds': r.spec_rounds,
               'finish_reason': r.finish_reason, 'released': r.released,
               'input_ids': r.prompt_ids, 'cached_tokens': r.prefix_len} for r in record.requests.values()]
    flush_cache(url)
    clear(ex)
    work = work_summary(record.events)
    n = sum(len(r['output_ids']) for r in results)
    row = {'id': trace['id'], 'total_s': elapsed, 'aggregate_tps': n / elapsed,
           'ttft_p50_s': percentile([r['ttft_s'] for r in results if r['ttft_s'] is not None], .5),
           'ttft_p95_s': percentile([r['ttft_s'] for r in results if r['ttft_s'] is not None], .95),
           'latency_p95_s': percentile([r['total_s'] for r in results], .95),
           'requests': results, 'server_requests': actual, 'events': record.events, 'work': work,
           'before': before, 'after': after, 'draft_cache_hits': proposer.cache_hits - hits,
           'draft_fallbacks': proposer.cache_fallbacks - fallbacks, 'pools_recovered': True}
    # Match input/output pairs, retaining duplicates if a future trace uses them.
    client_pairs = sorted((tuple(item['input_ids']), tuple(r['output_ids']))
                          for items, future in zip(lanes, futures) for item, r in zip(items, future.result()))
    server_pairs = sorted((tuple(r['input_ids']), tuple(r['output_ids'])) for r in actual)
    row['ok'] = (all(r['ok'] for r in results) and client_pairs == server_pairs
                 and all(r['released'] for r in actual)
                 and not row['draft_cache_hits'] and not row['draft_fallbacks']
                 and after['scheduler']['retractions'] == before['scheduler']['retractions'])
    return row


def main():
    import torch
    from sharar import config
    from sharar.engine.scheduler import SchedulerConfig
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs', type=Path, required=True)
    ap.add_argument('--trace', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--budget', type=int, default=4096)
    ap.add_argument('--decode-steps', type=int, default=1)
    ap.add_argument('--depth', type=int, default=15, choices=[3, 7, 15])
    args = ap.parse_args()
    cfg = SchedulerConfig(prefill_budget=args.budget, decode_steps_per_chunk=args.decode_steps)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'completed': False, 'label': args.label, 'scheduler': asdict(cfg), 'depth': args.depth,
              'identity': identity(args), 'trace_sha256': sha_file(args.trace), 'cells': []}
    report['identity']['source_sha256'].update({f'bench/{name}': sha_file(Path(__file__).with_name(name))
                                               for name in ('gb10_bs4_http.py', 'gb10_bs4_engine.py')})
    save(args.output / 'report.json', report)
    with torch.inference_mode():
        ex = make_runtime(args.depth)
        warm_feedback_shapes(ex)
        tasks = json.loads(args.inputs.read_text())['tasks']
        warm = next(t for t in tasks if t['id'] == 'benchmark_code')
        native_batch(ex, [warm] * 4, 128)
        with server(ex, cfg) as (url, sched):
            traces = json.loads(args.trace.read_text())
            # Warm the actual HTTP/prefill shapes, outside the timed cells.
            small = {'id': 'warmup', 'lanes': [[{**lane[0], 'output_tokens': 32}] for lane in traces[0]['lanes']]}
            save(args.output / 'warmup.json', run_trace(ex, url, small))
            for trace in traces:
                row = run_trace(ex, url, trace)
                report['cells'].append(row)
                save(args.output / 'report.json', report)
                print(json.dumps({k: row[k] for k in ('id', 'total_s', 'aggregate_tps', 'ttft_p95_s',
                                                       'latency_p95_s', 'work', 'ok')}), flush=True)
                assert row['ok'], f"HTTP trace failed: {trace['id']}"
            report['canaries'] = run_quality_canaries(url, config.MODEL_ID, 300)
            save(args.output / 'report.json', report)
            assert all(r['passed'] for r in report['canaries']), 'natural-stop canary failed'
        clear(ex)
        report['completed'] = True
        save(args.output / 'report.json', report)


if __name__ == '__main__':
    main()
