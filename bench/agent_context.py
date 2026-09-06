"""Fixed-work long-context cache and generation study (not a task-quality gate).

Run identical protocol on baseline and candidate images. Preserve output IDs:
cache boundaries can change numerical continuations, so decode rates describe
each measured continuation rather than an equal-quality speedup.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from sharar import config
from sharar.engine.executor import Executor
from sharar.engine.sampler import SamplingParams
from sharar.model.qwen38 import Qwen38
from sharar.parity.prompts import PARAGRAPH
from sharar.tokenizer import SnapshotTokenizer


PROTOCOL = {'version': 1, 'input_lengths': [8192, 16384, 32256], 'output_tokens': 512,
            'repeats': 2, 'boundary_input': 32766, 'boundary_output': 2,
            'state_slots': 12, 'kv_tokens': 35000, 'max_context': 32768,
            'proposer': 'dflash2', 'depth': 7, 'state_strategy': 'reconstruct',
            'workload': 'PARAGRAPH repetition; fixed work, ignore EOS; no task-quality claim',
            'timing': 'direct executor, per request; decode excludes prefill token and prefill time',
            'gates': 'candidate repeat recomputes <=64 tokens; exact restored target state; aligned draft features; no leaks'}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expect-retention', action='store_true')
    parser.add_argument('--scheduler-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = SnapshotTokenizer(config.snapshot_dir())
    tape = tokenizer.encode(PARAGRAPH * 1024)
    report = {'protocol': PROTOCOL, 'input_sha256': hashlib.sha256(json.dumps(tape[:32766]).encode()).hexdigest(),
              'runs': [], 'passed': False}
    def save():
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    ex = Executor(Qwen38(config.snapshot_dir()), num_pages=35000, num_slots=12)
    ex.enable_spec(7, proposer='dflash2', state_strategy='reconstruct', log=print)
    ex.capture_graphs(log=print)
    ex.warmup(log=print)
    proposer = ex.spec.proposer

    def run(ids, outputs, label, *, chunk=None, expected_prefix=None):
        hits, fallbacks = proposer.cache_hits, proposer.cache_fallbacks
        torch.cuda.synchronize()
        started = time.perf_counter()
        req = ex.admit(ids, SamplingParams(max_new_tokens=outputs, ignore_eos=True))
        if expected_prefix is not None:
            assert req.prefix_len == expected_prefix
        restored = None
        if req.match is not None:
            source = req.match.snapshot_slot
            restored = bool(torch.equal(ex.state_pool.ssm_all[:, source], ex.state_pool.ssm_all[:, req.slot])
                            and torch.equal(ex.state_pool.conv_all[:, source], ex.state_pool.conv_all[:, req.slot]))
            assert restored
        while req.prefill_pos < req.prefill_len:
            ex.prefill(req, chunk)
        torch.cuda.synchronize()
        first = time.perf_counter()
        draft_ready = proposer.ready(req)
        checkpoints = sorted(req.checkpoints)
        while not req.finished:
            ex.decode_step([req])
        torch.cuda.synchronize()
        end = time.perf_counter()
        assert req.committed_len == len(ids) + outputs - 1
        row = {'label': label, 'input_tokens': len(ids), 'output_tokens': len(req.output_ids),
               'input_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
               'cached_tokens': req.prefix_len, 'recomputed_tokens': len(ids) - req.prefix_len,
               'ttft_s': first - started, 'decode_s': end - first, 'completion_s': end - started,
               'decode_tps': (outputs - 1) / (end - first), 'aggregate_tps': outputs / (end - started),
               'spec_rounds': req.spec_rounds, 'spec_accepted': req.spec_accepted,
               'spec_proposed': req.spec_proposed, 'draft_ready': draft_ready,
               'draft_cache_hits': proposer.cache_hits - hits, 'draft_fallbacks': proposer.cache_fallbacks - fallbacks,
               'restored_target_exact': restored, 'checkpoints': checkpoints, 'output_ids': req.output_ids[:],
               'cuda_allocated_bytes': torch.cuda.memory_allocated(), 'cuda_reserved_bytes': torch.cuda.memory_reserved()}
        ex.release(req)
        report['runs'].append(row)
        save()
        print(label, 'input', len(ids), 'cached', row['cached_tokens'], 'TTFT', round(row['ttft_s'], 3),
              'decode TPS', round(row['decode_tps'], 2), 'draft', draft_ready, flush=True)
        return row

    for length in ([] if args.scheduler_only else PROTOCOL['input_lengths']):
        for repeat in range(PROTOCOL['repeats']):
            ex.flush_cache()
            run(tape[:length], 512, f'{length}-{repeat}-cold')
            row = run(tape[:length], 512, f'{length}-{repeat}-retry')
            if args.expect_retention:
                assert row['recomputed_tokens'] <= 64 and row['draft_ready'] and row['draft_fallbacks'] == 0
    for chunk in (() if args.scheduler_only else (None, 7000)):
        ex.flush_cache()
        run(tape[:32766], 2, f'boundary-{chunk}-cold', chunk=chunk)
        row = run(tape[:32766], 2, f'boundary-{chunk}-retry')
        if args.expect_retention:
            assert row['recomputed_tokens'] <= 64 and row['draft_ready']

    # Actual append-only turns include the prior generated response and tool
    # text. Inputs are saved per profile; this is lifecycle/latency evidence,
    # not an identical-token comparison between engines or revisions.
    ex.flush_cache()
    ids = tape[:8192]
    for turn in range(0 if args.scheduler_only else 4):
        row = run(ids, 512, f'agent-turn-{turn}', expected_prefix=len(ids) - 129 if turn else 0)
        if args.expect_retention:
            assert row['draft_ready'] and row['draft_fallbacks'] == 0
        ids += row['output_ids'] + tape[1000 + turn * 128:1128 + turn * 128]
    from sharar.engine.scheduler import Scheduler
    for repeat in range(2):
        ex.flush_cache()
        run(tape[:32766], 2, f'scheduler-{repeat}-seed')
        scheduler = Scheduler(ex)
        started = time.perf_counter()
        req = scheduler.submit(tape[:32766], SamplingParams(max_new_tokens=2, ignore_eos=True))
        scheduler.run_until_idle()
        row = {'label': f'scheduler-{repeat}-retry', 'input_tokens': 32766, 'output_tokens': len(req.output_ids),
               'cached_tokens': req.prefix_len, 'recomputed_tokens': 32766 - req.prefix_len,
               'ttft_s': req.t_first_token - started, 'completion_s': time.perf_counter() - started,
               'spec_rounds': req.spec_rounds, 'output_ids': req.output_ids, 'metrics': scheduler.stats()}
        assert req.released and len(req.output_ids) == 2
        if args.expect_retention:
            assert row['recomputed_tokens'] <= 64 and req.spec_rounds == 1
        report['runs'].append(row)
        save()
        print(row['label'], 'cached', row['cached_tokens'], 'TTFT', round(row['ttft_s'], 3), flush=True)
    ex.flush_cache()
    assert ex.kv_pool.free_count == 34999 and ex.state_pool.free_count == 11 and len(proposer.free_slots) == 4
    report['passed'] = True
    save()


if __name__ == '__main__':
    main()
