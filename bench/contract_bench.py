"""Fixed-profile 7B refactor check, usable against either source tree.

Run inside the prepared GB10 environment, with exclusive memory. This is a
synthetic engine regression check; Phase 7C supplies competitive/local-use gates.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from sharar import config
from sharar.engine.executor import Executor
from sharar.engine.sampler import SamplingParams
from sharar.model.qwen38 import Qwen38
from sharar.parity.prompts import PARAGRAPH
from sharar.tokenizer import SnapshotTokenizer
from sharar.memory.ledger import mem_available_bytes
from sharar.memory.radix import RadixCache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--output-tokens', type=int, default=128)
    args = parser.parse_args()
    config.check()
    model = Qwen38(config.snapshot_dir(), log=print)
    torch.cuda.empty_cache()
    ex = Executor(model, num_pages=35000, num_slots=12, use_cache=False, log=print)
    ex.enable_spec(3, log=print)
    torch.cuda.empty_cache()
    ex.capture_graphs(log=print)
    ex.warmup(log=print)
    torch.cuda.empty_cache()
    saved_spec = ex.spec
    tok = SnapshotTokenizer(config.snapshot_dir())
    all_ids = tok.encode(PARAGRAPH * 256)
    assert len(all_ids) >= 8192
    params = SamplingParams(max_new_tokens=args.output_tokens, ignore_eos=True)
    rows = []
    report = {'profile': {'model': config.MODEL_ID, 'revision': config.MODEL_REVISION,
                         'max_context': 32768, 'max_batch': 4, 'kv_pages': 35000,
                         'state_slots': 12, 'cache': False, 'graphs': True,
                         'greedy': True, 'ignore_eos': True, 'output_tokens': args.output_tokens,
                         'repetitions': args.repetitions}, 'rows': rows}
    # Preserve the adversarial untemplated prompt as diagnostic evidence on
    # both trees. It is not substituted for the stable structured canary.
    ex.spec = None
    ex.cache = RadixCache(ex.kv_pool, ex.state_pool, ex.device)
    report['repetitive_prompt_diagnostic'] = []
    repeated = tok.encode('hello ' * 128)
    for cache in ('cold', 'repeat', 'partial_hit'):
        if cache == 'partial_hit':
            ex.flush_cache()
            ex.generate(repeated[:len(repeated) // 2], SamplingParams(max_new_tokens=1))
        request = ex.generate(repeated, SamplingParams(max_new_tokens=16, ignore_eos=True))
        report['repetitive_prompt_diagnostic'].append({'cache': cache, 'prefix_len': request.prefix_len,
                                                      'output_ids': request.output_ids})
    ex.flush_cache()
    ex.cache = None
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    for mode in ('plain', 'mtp'):
        ex.spec = saved_spec if mode == 'mtp' else None
        for length in (128, 2048, 8192):
            for batch in (1, 4):
                items = [(all_ids[:length], params) for _ in range(batch)]
                samples = []
                for repetition in range(args.repetitions + 1):
                    saved_spec.reset_stats()
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    requests = ex.generate_batch(items)
                    torch.cuda.synchronize()
                    wall = time.perf_counter() - started
                    emitted = sum(len(r.output_ids) for r in requests)
                    assert emitted == batch * args.output_tokens
                    sample = {'wall_s': wall, 'decode_s': requests[0].decode_seconds,
                              'decode_tps': (emitted - batch) / requests[0].decode_seconds,
                              'emitted_tps': emitted / wall,
                              'output_sha256': hashlib.sha256(json.dumps([r.output_ids for r in requests]).encode()).hexdigest(),
                              'spec': saved_spec.stats(), 'available_gib': mem_available_bytes() / 2**30,
                              'allocated_gib': torch.cuda.memory_allocated() / 2**30,
                              'reserved_gib': torch.cuda.memory_reserved() / 2**30}
                    if repetition:
                        samples.append(sample)
                    ex.ledger.check_floor()
                row = {'mode': mode, 'input_tokens': length, 'batch': batch, 'samples': samples,
                       'median_decode_tps': statistics.median(s['decode_tps'] for s in samples),
                       'median_emitted_tps': statistics.median(s['emitted_tps'] for s in samples)}
                rows.append(row)
                args.output.write_text(json.dumps(report, indent=2) + '\n')
                print(json.dumps({key: value for key, value in row.items() if key != 'samples'}), flush=True)


if __name__ == '__main__':
    main()
