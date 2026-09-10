"""Interleaved whole-round screen for the measured small-M cuDNN dispatch fix.

Both implementations own separate verification/draft CUDA graphs. Native
acceptance and fixed counts share real proposals, logits and state commits.
Synthetic count rows measure cost only; they never imply answer throughput.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from bench.verify_cost import digest
from sharar import config
from sharar.engine.executor import Executor
from sharar.engine.sampler import SamplingParams
from sharar.engine.spec_graphs import VerifyGraphs
from sharar.kernels import ops
from sharar.model.qwen38 import Qwen38
from sharar.parity.prompts import PARAGRAPH
from sharar.proposers.parallel_graphs import ParallelGraphs
from sharar.tokenizer import SnapshotTokenizer


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'baseline': 'f6af0ef1483cac3d68d1c0c0ad10d4c7cd2a1ac8',
              'candidate': 'cuDNN threshold 32 instead of 16; all other dispatch unchanged',
              'target_revision': config.MODEL_REVISION, 'cells': [],
              'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    def save():
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    ex = Executor(Qwen38(config.snapshot_dir()), num_pages=35000, num_slots=12)
    ex.enable_spec(7, proposer='dflash2', state_strategy='reconstruct', log=print)
    original_threshold = ops.FP4_SMALL_M
    ops.FP4_SMALL_M = 16
    ex.capture_graphs(log=print)
    ex.warmup(log=print)
    spec, proposer = ex.spec, ex.spec.proposer
    graphs = {'baseline': (spec.verify_graphs, proposer.graphs)}
    ops.FP4_SMALL_M = 32
    vg, dg = VerifyGraphs(spec), ParallelGraphs(proposer)
    vg.capture(log=print)
    dg.capture(log=print)
    graphs['candidate'] = (vg, dg)
    native_accept, native_propose = spec._accept_batch, proposer.propose
    def select(mode):
        ops.FP4_SMALL_M = 16 if mode == 'baseline' else 32
        spec.verify_graphs, proposer.graphs = graphs[mode]
        spec.width_graphs = {7: spec.verify_graphs}
    ids = SnapshotTokenizer(config.snapshot_dir()).encode(PARAGRAPH * 256)
    try:
        for length in (128, 8192):
            for batch in (1, 4):
                select('baseline')
                ex.flush_cache()
                prime = ex.admit(ids[:length - 1], SamplingParams(max_new_tokens=1, ignore_eos=True))
                ex.prefill(prime)
                ex.release(prime)
                def prepare():
                    reqs = [ex.admit(ids[:length], SamplingParams(max_new_tokens=32, ignore_eos=True)) for _ in range(batch)]
                    assert all(r.prefix_len == length - 1 for r in reqs)
                    for r in reqs:
                        r.cacheable = r.cache_enabled = False
                    ex.prefill_batch([(r, 1) for r in reqs])
                    assert all(spec.ready(r) and r.committed_len == length for r in reqs)
                    return reqs
                for count in (None, 1, 4, 8):
                    seen, frozen, samples = {}, {}, []
                    def record_propose(*a, **kw):
                        proposal = native_propose(*a, **kw)
                        # Retain GPU IDs here; read/compare after timing.
                        seen['proposal'] = proposal.tokens.clone()
                        return proposal
                    def accept(b, z, features):
                        seen['target'] = z.clone()
                        if count is not None:
                            forced = torch.cat([b.proposal.tokens, torch.zeros((batch, 1), dtype=torch.int64, device=ex.device)], dim=1)
                            if count < 8:
                                forced[:, count - 1] = (b.proposal.tokens[:, count - 1] + 1) % config.VOCAB_SIZE
                            z = forced
                        return native_accept(b, z, features)
                    proposer.propose, spec._accept_batch = record_propose, accept
                    for repeat in range(-1, 3):
                        for mode in ('baseline', 'candidate', 'candidate', 'baseline'):
                            select(mode)
                            reqs = prepare()
                            anchors = [r.output_ids[-1] for r in reqs]
                            torch.cuda.synchronize()
                            started = time.perf_counter()
                            spec.round(reqs)
                            torch.cuda.synchronize()
                            elapsed = (time.perf_counter() - started) * 1000
                            observed = {'proposal': seen['proposal'].tolist(), 'target': seen['target'].tolist(),
                                        'anchors': anchors, 'output_ids': [r.output_ids for r in reqs],
                                        'committed_lengths': [r.committed_len for r in reqs]}
                            if not frozen:
                                frozen.update(observed)
                            assert observed == frozen, (length, batch, count, mode, 'work or outputs changed')
                            if count is not None:
                                assert all(r.committed_len == length + count for r in reqs)
                            samples.append({'warmup': repeat < 0, 'block': repeat, 'mode': mode, 'round_ms': elapsed})
                            for r in reqs:
                                ex.release(r)
                    medians = {mode: statistics.median(r['round_ms'] for r in samples if r['mode'] == mode and not r['warmup'])
                               for mode in graphs}
                    row = {'context': length, 'batch': batch, 'forced_count': count, 'samples': samples,
                           'medians_ms': medians, 'improvement': 1 - medians['candidate'] / medians['baseline'],
                           'observed': frozen, 'input_ids': ids[:length],
                           'work_sha256': digest([ids[:length], count, frozen])}
                    report['cells'].append(row)
                    save()
                    print(length, batch, count, medians, row['improvement'], flush=True)
                proposer.propose, spec._accept_batch = native_propose, native_accept
        report['completed'] = True
        save()
    finally:
        ops.FP4_SMALL_M = original_threshold
        proposer.propose, spec._accept_batch = native_propose, native_accept


if __name__ == '__main__':
    main()
