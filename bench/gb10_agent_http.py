"""Real HTTP agent-turn and hot-resume scheduling screen with frozen fixtures.

Tool delays/results are controlled fixtures; prompts for later turns contain
actual previous outputs. This measures serving behavior, not an external
agent's repository task success or real tool CPU/memory contention.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time

from bench.benchmark_text_server import http_json, percentile, run_quality_canaries
from bench.support.retention import batch
from bench.support.recording import Recorder, native_batch, warm_feedback_shapes, work_summary
from bench.support.http import server, request, await_idle
from bench.support.evidence import clear, identity, save, sha_file


def chat_suffixes(tokenizer, template, tasks, families):
    """Close the prior assistant turn and append a real user message/header.

    Reports are controlled fixtures, not outputs of tests actually executed.
    Preserve the snapshot's exact role boundaries without repeating padding.
    """
    marker = 'SHARAR_FIXTURE_ASSISTANT_PREFIX'
    assistant = [{'role': 'user', 'content': 'Original request placeholder.'},
                 {'role': 'assistant', 'content': marker}]
    prefix = template.render(assistant, add_generation_prompt=False)
    assert prefix.count(marker) == 1
    closing = prefix.split(marker, 1)[1]
    result = {}
    for family in families:
        requirement = tasks[family]['spec']
        reports = [
            'The smoke test passed for one item, but an empty-input case raised an unexpected exception. '
            'Identify the invariant this threatens and provide the corrected Python handling with a regression test.',
            'The focused empty-input regression now passes. A repeated-operation test still produces a different result '
            'on its second execution. Explain two plausible state-lifecycle causes, then give a minimal reproducer and repair.',
            'The focused lifecycle tests now pass; the full suite has not run. Propose an adversarial test for a different '
            'boundary condition, give concrete inputs and expected outcomes, and explain which invariant it protects.',
        ]
        values = []
        for index, report in enumerate(reports):
            content = f'Test report {index + 1} for {family}: {report}\nOriginal requirement: {requirement}\nContinue the implementation review.'
            rendered = template.render(assistant + [{'role': 'user', 'content': content}])
            assert rendered.startswith(prefix), 'chat template changed the preceding assistant turn'
            suffix = closing + rendered[len(prefix):]
            values.append({'text': suffix, 'input_ids': tokenizer.encode(suffix)})
        result[family] = values
    return result


def trace(ex, url, lanes, *, append, tool_ids, pause_s):
    import torch
    await_idle(url)
    p = ex.spec.proposer
    hits, fallbacks = p.cache_hits, p.cache_fallbacks
    before = http_json(url + '/get_server_info')
    barrier = threading.Barrier(len(lanes) + 1)
    origin = [0.]
    def lane(items):
        barrier.wait(timeout=30)
        rows, history = [], None
        for i, original in enumerate(items):
            item = dict(original)
            if i and append:
                time.sleep(pause_s)
                item['input_ids'] = history + item.get('suffix_ids', tool_ids)
            delay = item.get('delay_s', 0)
            if delay:
                time.sleep(delay)
            row = request(url, item, origin[0], allow_cached=True)
            row['input_ids'] = item['input_ids']
            rows.append(row)
            history = item['input_ids'] + row['output_ids']
            if not row['ok']:
                break
        return rows
    torch.cuda.synchronize()
    with Recorder(ex) as rec, ThreadPoolExecutor(max_workers=len(lanes)) as pool:
        jobs = [pool.submit(lane, items) for items in lanes]
        origin[0] = rec.started = time.perf_counter()
        barrier.wait(timeout=30)
        rows = [r for job in jobs for r in job.result()]
        elapsed = time.perf_counter() - origin[0]
        await_idle(url)
    torch.cuda.synchronize()
    after = http_json(url + '/get_server_info')
    actual = [{'input_ids': r.prompt_ids, 'output_ids': r.output_ids, 'released': r.released,
               'request_id': r.request_id, 'spec_rounds': r.spec_rounds} for r in rec.requests.values()]
    pairs = lambda values: sorted((tuple(r['input_ids']), tuple(r['output_ids'])) for r in values)
    ok = (len(rows) == sum(map(len, lanes)) and all(r['ok'] for r in rows)
          and pairs(rows) == pairs(actual) and all(r['released'] for r in actual)
          and after['scheduler']['retractions'] == before['scheduler']['retractions'])
    return {'ok': ok, 'rows': rows, 'server_requests': actual, 'total_s': elapsed,
            'aggregate_tps': sum(len(r['output_ids']) for r in rows) / elapsed,
            'ttft_p50_s': percentile([r['ttft_s'] for r in rows if r['ttft_s'] is not None], .5),
            'ttft_p95_s': percentile([r['ttft_s'] for r in rows if r['ttft_s'] is not None], .95),
            'latency_p95_s': percentile([r['total_s'] for r in rows], .95),
            'max_stream_gap_s': max((r['max_stream_gap_s'] or 0) for r in rows),
            'draft_hits': p.cache_hits - hits, 'draft_fallbacks': p.cache_fallbacks - fallbacks,
            'before': before, 'after': after, 'work': work_summary(rec.events), 'events': rec.events}


def main():
    import torch
    from sharar import config
    from sharar.engine.executor import Executor
    from sharar.engine.scheduler import SchedulerConfig
    from sharar.model.qwen38 import Qwen38
    from sharar.tokenizer import SnapshotTokenizer
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--schedule', action='store_true')
    ap.add_argument('--slots', type=int, default=24)
    ap.add_argument('--pages', type=int, default=100000)
    ap.add_argument('--chat-fixture', action='store_true')
    args = ap.parse_args()
    tasks = {t['id']: t for t in json.loads(args.inputs.read_text())['tasks']}
    families = ['ring_buffer', 'deadline_queue', 'transactional_outbox', 'queue_capacity']
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'completed': False, 'identity': identity(args), 'label': args.label,
              'input_sha256': sha_file(args.inputs), 'driver_sha256': sha_file(Path(__file__)),
              'geometry': {'num_pages': args.pages, 'num_slots': args.slots, 'max_batch': 4},
              'schedule': args.schedule, 'helper_sha256': {name: sha_file(Path(__file__).with_name(name))
                                                          for name in ('agent_schedule.py', 'gb10_bs4_http.py',
                                                                       'gb10_agent_retention.py', 'gb10_bs4_engine.py')}, 'cells': []}
    save(args.output / 'report.json', report)
    with torch.inference_mode():
        # Hold four primed histories plus two fresh long arrivals without KV
        # eviction confounding the scheduling screen. Native tests use 60K.
        ex = Executor(Qwen38(config.snapshot_dir(), log=print), num_pages=args.pages, num_slots=args.slots, max_batch=4, log=print)
        ex.enable_spec(15, proposer='dflash2', state_strategy='reconstruct', draft_precision='fp8', log=print)
        ex.capture_graphs(log=print)
        ex.warmup(log=print)
        warm_feedback_shapes(ex)
        native_batch(ex, [tasks['benchmark_code']] * 4, 64)
        tokenizer = SnapshotTokenizer(config.snapshot_dir())
        tool_ids = tokenizer.encode('\nTool result: the test suite completed. 37 tests passed, 0 failed. Continue the implementation review.\n')
        tool_ids = (tool_ids * (128 // len(tool_ids) + 1))[:128]
        report['tool_ids'] = tool_ids
        chats = None
        if args.chat_fixture:
            from sharar.api.chat_template import ChatTemplate
            chats = chat_suffixes(tokenizer, ChatTemplate(config.snapshot_dir()), tasks, families)
            report['chat_fixtures'] = chats
        if args.schedule:
            from bench.archive.agent_schedule import AgentScheduler
            import sharar.engine.scheduler as scheduler_module
            scheduler_module.Scheduler = AgentScheduler
        with server(ex, SchedulerConfig()) as (url, sched):
            # Exercise the actual cold/append/mixed HTTP shapes once before
            # timing. Keep the excluded warmup, including every output, on disk.
            for repeat in (-1, 0, 1):
                kinds = ['cold_bs4', 'agent_turns', *(['agent_chat'] if chats else []), 'mixed_arrivals']
                for kind in kinds:
                    clear(ex)
                    seeds = []
                    if kind != 'cold_bs4':
                        seeds = [batch(ex, [tasks[family]], 1) for family in families]
                    histories = [tasks[family]['input_ids'] + seed['outputs'][0]['output_ids'] + tool_ids
                                 for family, seed in zip(families, seeds)]
                    if kind == 'cold_bs4':
                        lanes = [[{'id': f'cold-control-{repeat}-{i}', 'input_ids': tasks[family]['input_ids'],
                                   'output_tokens': 32 if repeat < 0 else 256}] for i, family in enumerate(families)]
                    elif kind == 'agent_turns':
                        lanes = [[{'id': f'agent-{repeat}-{i}-{j}', 'input_ids': histories[i], 'output_tokens': 32 if repeat < 0 else 128}
                                  for j in range(3)] for i in range(4)]
                    elif kind == 'agent_chat':
                        lanes = [[{'id': f'chat-{repeat}-{i}-{j}',
                                   'input_ids': tasks[family]['input_ids'] + seeds[i]['outputs'][0]['output_ids'] + chats[family][0]['input_ids'],
                                   'suffix_ids': chats[family][j]['input_ids'], 'output_tokens': 128}
                                  for j in range(3)] for i, family in enumerate(families)]
                    else:
                        lanes = [[{'id': f'cold-{repeat}-{i}', 'input_ids': tasks[family]['input_ids'], 'output_tokens': 32 if repeat < 0 else 256}]
                                 for i, family in enumerate(('build_graph', 'stream_parser'))]
                        lanes += [[{'id': f'resume-{repeat}-{i}', 'input_ids': histories[i], 'output_tokens': 32 if repeat < 0 else 256,
                                    'delay_s': .1}] for i in range(2)]
                    row = trace(ex, url, lanes, append=kind in ('agent_turns', 'agent_chat'), tool_ids=tool_ids, pause_s=.1)
                    row.update(kind=kind, repeat=repeat, warmup=repeat < 0, seeds=seeds)
                    if kind == 'agent_chat':
                        row['output_texts'] = [tokenizer.decode(r['output_ids']) for r in row['rows']]
                    report['cells'].append(row)
                    save(args.output / 'report.json', report)
                    print(json.dumps({k: row[k] for k in ('kind','repeat','aggregate_tps','ttft_p95_s','draft_fallbacks','ok')}), flush=True)
                    assert row['ok'], kind
                    if type(ex.spec.proposer.feature_cache).__name__ == 'CheckpointFeatureCache':
                        assert not row['draft_fallbacks'], 'a retained checkpoint lost draft features'
            clear(ex)
            with ThreadPoolExecutor(max_workers=2) as pool:
                origin = time.perf_counter()
                background = [pool.submit(request, url, {'id': f'quality-background-{i}',
                              'input_ids': tasks[family]['input_ids'], 'output_tokens': 128}, origin,
                              allow_cached=True) for i, family in enumerate(('build_graph', 'stream_parser'))]
                time.sleep(.1)
                report['canaries'] = run_quality_canaries(url, config.MODEL_ID, 300)
                report['quality_background'] = [job.result() for job in background]
            await_idle(url)
            save(args.output / 'report.json', report)
            assert all(r['passed'] for r in report['canaries']), 'natural-stop canary failure'
            assert all(r['ok'] for r in report['quality_background']), 'quality background failure'
        clear(ex)
        report['completed'] = True
        save(args.output / 'report.json', report)


if __name__ == '__main__':
    main()
