"""Current-profile HTTP comparison at concurrency 1, 2 and 4.

Cold prompts, three append-only turns with controlled tool delays, and fresh
prefills beside cached turns. Actual preceding outputs feed later turns.
These synthetic fixtures measure serving, not agent task success. Run each
arm in an exclusive GPU window and keep the native/candidate repeats.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.gb10_agent_http import trace
from bench.benchmark_text_server import run_quality_canaries
from bench.support.evidence import clear, identity, save, sha_file
from bench.support.http import await_idle, server
from bench.support.recording import warm_feedback_shapes
from bench.support.retention import batch as seed_batch


def main():
    import torch
    from sharar import config
    from sharar.api.validation import set_accept_default
    from sharar.engine.executor import Executor
    from sharar.engine.scheduler import SchedulerConfig
    from sharar.model.qwen38 import Qwen38
    from sharar.tokenizer import SnapshotTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--draft', help='trained draft directory; omitted uses the pinned checkpoint')
    parser.add_argument('--candidate', help='benchmark hook module:function at spec and ready stages')
    parser.add_argument('--batches', default='1,2,4')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--output-tokens', type=int, default=256)
    args = parser.parse_args()
    batches = [int(value) for value in args.batches.split(',')]
    if not batches or any(value not in (1, 2, 4) for value in batches) or args.repeats < 1 or args.output_tokens < 16:
        parser.error('need batches from 1,2,4, positive repeats and at least 16 output tokens')
    args.output.mkdir(parents=True, exist_ok=False)
    tasks = {task['id']: task for task in json.loads(args.inputs.read_text())['tasks']}
    families = ['ring_buffer', 'deadline_queue', 'transactional_outbox', 'queue_capacity']
    report = {'completed': False, 'identity': identity(args), 'label': args.label,
              'inputs_sha256': sha_file(args.inputs), 'driver_sha256': sha_file(Path(__file__)),
              'geometry': {'kv_pages': 100000, 'state_slots': 24, 'max_batch': 4},
              'protocol': {'batches': batches, 'repeats': args.repeats, 'warmup_repeats': 1,
                           'acceptance': [1, 0.0], 'output_tokens': args.output_tokens,
                           'warm_turns': 3, 'tool_delay_seconds': .1,
                           'limitation': 'Synthetic fixed tool results; no actual tool CPU load or task-success claim'},
              'cells': []}
    save(args.output / 'report.json', report)
    set_accept_default(1, 0.0)
    with torch.inference_mode():
        model = Qwen38(config.snapshot_dir(), log=print)
        ex = Executor(model, num_pages=100000, num_slots=24, max_batch=4, log=print)
        ex.enable_spec(15, proposer='dflash2', state_strategy='reconstruct', draft_precision='fp8',
                       draft_dir=args.draft, log=print)
        from bench.gb10_family_eval import load_hook
        hook = load_hook(args.candidate)
        if hook:
            hook('spec', ex, args)
        ex.capture_graphs(log=print)
        ex.warmup(log=print)
        warm_feedback_shapes(ex)
        if hook:
            hook('ready', ex, args)
        tokenizer = SnapshotTokenizer(config.snapshot_dir())
        suffix = tokenizer.encode('\nTool result fixture: the focused test completed. Review the implementation, cover an additional boundary case, and provide the revised Python code.\n')
        suffix = (suffix * (128 // len(suffix) + 1))[:128]
        report.update(w3_target_identity=getattr(model, 'w3_identity', None),
                      draft=ex.spec.proposer.model.source, tool_suffix_ids=suffix)
        with server(ex, SchedulerConfig()) as (url, _):
            for repeat in range(-1, args.repeats):
                for concurrency in batches:
                    for kind in ('cold', 'warm', *(['mixed'] if concurrency > 1 else [])):
                        await_idle(url)
                        clear(ex)
                        chosen = families[:concurrency]
                        output = 32 if repeat < 0 else args.output_tokens
                        seeds = []
                        if kind != 'cold':
                            seeds = [seed_batch(ex, [tasks[family]], 1) for family in chosen]
                        lanes = []
                        for lane, family in enumerate(chosen):
                            prompt = tasks[family]['input_ids']
                            if kind != 'cold':
                                prompt = prompt + seeds[lane]['outputs'][0]['output_ids'] + suffix
                            if kind == 'mixed' and lane == 0:
                                prompt = tasks['build_graph']['input_ids']
                            items = [{'id': f'{args.label}-{repeat}-{concurrency}-{kind}-{lane}-{turn}',
                                      'input_ids': prompt,
                                      'output_tokens': output // 2 if kind == 'warm' else output,
                                      'delay_s': .1 if kind == 'mixed' and lane else 0}
                                     for turn in range(3 if kind == 'warm' else 1)]
                            lanes.append(items)
                        row = trace(ex, url, lanes, append=kind == 'warm', tool_ids=suffix, pause_s=.1)
                        row.update(kind=kind, concurrency=concurrency, repeat=repeat,
                                   warmup=repeat < 0, seeds=seeds)
                        report['cells'].append(row)
                        save(args.output / 'report.json', report)
                        print(json.dumps({key: row[key] for key in ('kind','concurrency','repeat','aggregate_tps','ttft_p95_s','ok')}), flush=True)
                        assert row['ok'] and not row['draft_fallbacks'], (kind, concurrency)
            clear(ex)
            report['quality_canaries'] = run_quality_canaries(url, config.MODEL_ID, 300)
            assert all(row['passed'] for row in report['quality_canaries'])
            await_idle(url)
        clear(ex)
        report['completed'] = True
        save(args.output / 'report.json', report)


if __name__ == '__main__':
    main()
