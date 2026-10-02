"""Exclusive-GPU 7C pilot: cache diagnostic, real HTTP workload, GB10 traces.

Run inside the prepared GB10 environment with the serving model stopped and
the host memory guard watching this process. The source image is not deployed.
"""

import argparse
from contextlib import contextmanager
import importlib.metadata
import json
from pathlib import Path
import platform
import socket
import threading
import time

import torch

from sharar import config
from sharar.api.chat_template import ChatTemplate
from sharar.engine.loop import EngineLoop
from sharar.api.server import create_app
from sharar.engine.executor import Executor
from sharar.engine.scheduler import Scheduler, SchedulerConfig
from sharar.model.qwen38 import Qwen38
from sharar.tokenizer import SnapshotTokenizer

import gb10_profile
from support.evals import evals_module

local_workloads = evals_module('evals.speed.local')  # the frozen local-use HTTP client (sharar-eval)


@contextmanager
def http_server(ex, tok, snapshot, description, scheduler_config=None):
    import uvicorn
    engine = EngineLoop(Scheduler(ex, scheduler_config or SchedulerConfig()))
    engine.start()
    app = create_app(engine, tok, ChatTemplate(snapshot), description)
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    url = f'http://127.0.0.1:{sock.getsockname()[1]}'
    server = uvicorn.Server(uvicorn.Config(app, log_level='warning', access_log=False))
    thread = threading.Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 30
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError('HTTP server did not start')
            time.sleep(0.05)
        yield url
        local_workloads.await_idle(url)
    finally:
        server.should_exit = True
        thread.join(timeout=30)
        engine.stop = True
        engine.thread.join(timeout=30)
        sock.close()
        if thread.is_alive() or engine.thread.is_alive():
            raise RuntimeError('benchmark server did not stop')


def main():
    from sharar.parity.cache_trace import run as cache_trace
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = config.snapshot_dir()
    config.check()
    tok = SnapshotTokenizer(snapshot)
    frozen = local_workloads.materialize(local_workloads.DEFAULT_MANIFEST, snapshot)
    assert frozen['manifest']['model'] == config.MODEL_ID and frozen['manifest']['revision'] == config.MODEL_REVISION
    source_root = Path(config.__file__).resolve().parents[1]
    description = {'model': config.MODEL_ID, 'revision': config.MODEL_REVISION, 'graphs': True,
                   'kv_pages': 35000, 'state_slots': 16, 'max_batch': 4, 'max_context': 32768,
                   'platform': platform.platform(), 'torch': torch.__version__,
                   'packages': {p: importlib.metadata.version(p) for p in ('flashinfer-python', 'triton', 'uvicorn')},
                   'source_sha256': {str(p.relative_to(source_root)): local_workloads.digest(p.read_bytes())
                                     for pattern in ('sharar/**/*.py', 'bench/**/*.py', 'bench/workloads/*')
                                     for p in source_root.glob(pattern) if p.is_file()}}
    (args.output / 'environment.json').write_text(json.dumps(description, indent=2) + '\n')
    model = Qwen38(snapshot, log=print)
    torch.cuda.empty_cache()
    ex = Executor(model, num_pages=35000, num_slots=16, log=print)
    cache_trace(ex, tok.encode('hello ' * 128), args.output / 'cache-trace.json')
    ex.enable_spec(3, log=print)
    torch.cuda.empty_cache()
    ex.capture_graphs(log=print)
    ex.warmup(log=print)
    torch.cuda.empty_cache()
    spec = ex.spec
    summaries = {}
    for mode in ('plain', 'mtp'):
        ex.spec = spec if mode == 'mtp' else None
        ex.flush_cache()
        spec.reset_stats()
        with http_server(ex, tok, snapshot, {**description, 'mode': mode, 'k': 3 if mode == 'mtp' else 0}) as url:
            summaries[mode] = local_workloads.run(url, frozen, tok, args.output / mode)
        ex.ledger.check_floor()
    ex.spec = spec
    gb10_profile.run(ex, tok, args.output / 'profiles')
    (args.output / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
    if any(s['automatic_failures'] or s['request_failures'] for s in summaries.values()):
        raise SystemExit('Workload gates failed; evidence retained. Do not promote an optimization.')


if __name__ == '__main__':
    main()
