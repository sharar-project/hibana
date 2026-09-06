"""ABBA MTP feedback repair against frozen source, with exact real-state checks.

The reference is sharar/proposers/mtp.py exported from 56605f0. KV-only
graphs are an archived benchmark ablation; serving uses pinned feedback. Timed rounds
retain native proposals/target outputs; forced counts exercise every accepted
KV extent and are cost diagnostics, not answer throughput.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
from types import MethodType

import torch

from sharar import config
from sharar.engine.executor import Executor
from sharar.engine.sampler import SamplingParams
from sharar.model.qwen38 import Qwen38
from sharar.parity.prompts import PARAGRAPH
from sharar.tokenizer import SnapshotTokenizer


class FeedbackGraphs:
    """Target-conditioned KV correction at the original per-request row shape.

    Static inputs and intermediates are owned here and reused on the executor's
    single CUDA stream. Capture writes only the reserved dummy page zero.
    """

    def __init__(self, proposer):
        self.proposer = proposer
        self._pool = torch.cuda.graph_pool_handle()
        self._states = {}

    def capture(self, log=None):
        d = self.proposer.context.device
        for count in range(1, self.proposer.k + 2):
            inputs = (torch.zeros(count, dtype=torch.int64, device=d),
                      torch.zeros(count, config.HIDDEN_SIZE, dtype=torch.bfloat16, device=d),
                      torch.zeros(count, dtype=torch.int64, device=d),
                      torch.zeros(count, dtype=torch.int32, device=d))
            for _ in range(2):
                self.proposer.head.write_kv(*inputs)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._pool):
                self.proposer.head.write_kv(*inputs)
            self._states[count] = inputs, graph
            if log:
                log(f"  captured MTP feedback for {count} accepted inputs")

    def run(self, tokens, hidden, positions, pages):
        inputs, graph = self._states[len(tokens)]
        for dest, source in zip(inputs, (tokens, hidden, positions, pages)):
            dest.copy_(source)
        graph.replay()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    module_spec = importlib.util.spec_from_file_location('legacy_mtp', args.reference)
    legacy = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = legacy
    module_spec.loader.exec_module(legacy)
    report = {'reference_commit': '56605f0', 'reference_sha256': hashlib.sha256(args.reference.read_bytes()).hexdigest(),
              'target_revision': config.MODEL_REVISION, 'depth': 3, 'cells': [], 'completed': False}
    ex = Executor(Qwen38(config.snapshot_dir()), num_pages=35000, num_slots=12)
    ex.enable_spec(3, state_strategy='reconstruct', log=print)
    ex.capture_graphs(log=print)
    ex.warmup(log=print)
    spec, p = ex.spec, ex.spec.proposer
    modern_batch, modern_row = p.accepted_batch, p._accepted
    feedback_graphs = FeedbackGraphs(p)
    feedback_graphs.capture(log=print)
    original_write = p.head.write_kv
    def graph_feedback(*args):
        p.head.write_kv = feedback_graphs.run
        try:
            return modern_batch(*args)
        finally:
            p.head.write_kv = original_write
    original_accept = spec._accept_batch
    ids = SnapshotTokenizer(config.snapshot_dir()).encode(PARAGRAPH * 256)
    def select(mode):
        p.accepted_batch = (MethodType(legacy.MtpProposer.accepted_batch, p) if mode == 'legacy'
                            else graph_feedback if mode == 'graphs' else modern_batch)
        p._accepted = MethodType(legacy.MtpProposer._accepted, p) if mode == 'legacy' else modern_row
    try:
        for length in (128, 8192):
            for batch in (1, 4):
                ex.flush_cache()
                prime = ex.admit(ids[:length - 1], SamplingParams(max_new_tokens=1, ignore_eos=True))
                ex.prefill(prime)
                ex.release(prime)
                for count in (None, 1, 2, 3, 4):
                    reference, samples = None, []
                    def accept(b, z, features):
                        if count is not None:
                            z = torch.cat([b.proposal.tokens, torch.zeros((batch, 1), dtype=torch.int64, device=ex.device)], dim=1)
                            if count < 4:
                                z[:, count - 1] = (b.proposal.tokens[:, count - 1] + 1) % config.VOCAB_SIZE
                        return original_accept(b, z, features)
                    spec._accept_batch = accept
                    for block in range(-1, 3):
                        for mode in ('legacy', 'pinned', 'graphs', 'graphs', 'pinned', 'legacy'):
                            select(mode)
                            reqs = [ex.admit(ids[:length], SamplingParams(max_new_tokens=16, ignore_eos=True)) for _ in range(batch)]
                            assert all(r.prefix_len == length - 1 for r in reqs)
                            for r in reqs:
                                r.cacheable = False
                            ex.prefill_batch([(r, 1) for r in reqs])
                            torch.cuda.synchronize()
                            started = time.perf_counter()
                            spec.round(reqs)
                            torch.cuda.synchronize()
                            milliseconds = (time.perf_counter() - started) * 1000
                            slots = [r.slot for r in reqs]
                            pages = torch.cat([r.pages[length:r.committed_len] for r in reqs])
                            observed = {'outputs': [r.output_ids for r in reqs], 'committed': [r.committed_len for r in reqs]}
                            tensors = [ex.state_pool.ssm_all[:, slots], ex.state_pool.conv_all[:, slots],
                                       *(t[pages.long()] for t in p.head.kv.cache(0)), torch.cat([r.proposer_state.pending_hidden for r in reqs])]
                            if reference is None:
                                reference = observed, [t.clone() for t in tensors]
                            assert observed == reference[0], (length, batch, count, mode, 'outputs')
                            assert all(torch.equal(a, b) for a, b in zip(tensors, reference[1])), (length, batch, count, mode, 'state')
                            samples.append({'block': block, 'warmup': block < 0, 'mode': mode, 'round_ms': milliseconds})
                            for r in reqs:
                                ex.release(r)
                    medians = {m: statistics.median(s['round_ms'] for s in samples if s['mode'] == m and not s['warmup'])
                               for m in ('legacy', 'pinned', 'graphs')}
                    row = {'context': length, 'batch': batch, 'forced_count': count, 'medians_ms': medians,
                           'exact_state_checks': len(samples), 'samples': samples, 'observed': reference[0]}
                    report['cells'].append(row)
                    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
                    print(length, batch, count, medians, flush=True)
                    del reference, tensors
        report['completed'] = True
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    finally:
        p.accepted_batch, p._accepted, p.head.write_kv = modern_batch, modern_row, original_write
        spec._accept_batch = original_accept
        ex.flush_cache()


if __name__ == '__main__':
    main()
