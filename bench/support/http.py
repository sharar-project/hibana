"""Real engine HTTP fixture and audited streaming requests for benchmarks."""
from contextlib import contextmanager
from dataclasses import asdict
import http.client
import json
import socket
import threading
import time
from urllib.parse import urlsplit
from bench.benchmark_text_server import http_json

@contextmanager
def server(ex, cfg):
    import uvicorn
    from sharar import config
    from sharar.api.chat_template import ChatTemplate
    from sharar.engine.loop import EngineLoop
    from sharar.api.server import create_app
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
