"""Controlled DFlash2 round costs, separate from natural-generation speed.

Fresh requests reuse a frozen cached prefix outside timing. Every measured
round runs the real draft and target, but uses frozen proposal IDs and forced
acceptance counts. These synthetic acceptances are cost controls, not model
answers. Instrumented event spans overlap and are never added to wall timing.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from bench.gb10_profile import trace_summary
from sharar import config
from sharar.engine.executor import Executor
from sharar.engine.rejection import Proposal
from sharar.engine.sampler import SamplingParams
from sharar.model.gdn import GatedDeltaNet
from sharar.model.qwen38 import Qwen38
from sharar.parity.prompts import PARAGRAPH
from sharar.tokenizer import SnapshotTokenizer


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@contextmanager
def phases(ex, records):
    spec, proposer = ex.spec, ex.spec.proposer
    methods = [(spec, 'prepare', 'prepare_including_draft'),
               (proposer, 'propose', 'draft'), (spec.verify_graphs, 'run', 'target_verify'),
               (spec, '_accept', 'accept_including_commit_feedback'),
               (ex.state, 'commit_reconstructed', 'commit_including_reconstruction'),
               (spec.replay, 'run', 'reconstruction'),
               (proposer, 'accepted_batch', 'draft_feedback')]
    originals = []
    for obj, method, label in methods:
        original = getattr(obj, method)
        originals.append((obj, method, original))
        def wrapped(*args, original=original, label=label, **kwargs):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            started = time.perf_counter()
            with torch.profiler.record_function('sharar.' + label):
                a.record()
                result = original(*args, **kwargs)
                b.record()
            records[label].append((a, b, (time.perf_counter() - started) * 1000))
            return result
        setattr(obj, method, wrapped)
    try:
        yield
    finally:
        for obj, method, original in originals:
            setattr(obj, method, original)


def summarize_phases(records):
    return {name: {'event_ms': statistics.median(a.elapsed_time(b) for a, b, _ in rows),
                   'host_ms': statistics.median(t for _, _, t in rows), 'samples': len(rows)}
            for name, rows in records.items()}


def reference_conv(qkv, states, slots, weight, windows, steps):
    """Exact 63a9b30 convolution/window expression, isolated for cost attribution."""
    batch, dim = len(slots), qkv.shape[-1]
    hist = states[slots].transpose(1, 2)
    xs = torch.cat([hist, qkv.view(batch, steps, dim)], dim=1)
    conv = (xs[:, :steps] * weight[:, 0] + xs[:, 1:steps + 1] * weight[:, 1]
            + xs[:, 2:steps + 2] * weight[:, 2] + xs[:, 3:steps + 3] * weight[:, 3])
    conv = torch.nn.functional.silu(conv.float()).to(torch.bfloat16)
    retained = torch.stack([xs[:, s + 1:s + 4].transpose(1, 2) for s in range(steps)], dim=1)
    windows[:batch, :steps].copy_(retained)
    return conv


def graph_cost(fn, repeats=30):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    samples = []
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(repeats):
            graph.replay()
        b.record()
        b.synchronize()
        samples.append(a.elapsed_time(b) / repeats)
    return {'samples_ms': samples, 'median_ms': statistics.median(samples)}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dump-convolution', action='store_true', help='retain real per-layer inputs outside git for a kernel prototype')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    report = {'baseline': '63a9b30ba19134cb173a26869ac4a9285c4f3bdd',
              'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'target_revision': config.MODEL_REVISION, 'cells': [],
              'source_hashes': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for folder in ('sharar', 'bench') for p in (root / folder).rglob('*.py')},
              'protocol': {'contexts': [128, 8192], 'batches': [1, 4], 'counts': [1, 4, 8],
                           'warmups': 1, 'samples': 5, 'verify_steps': 8,
                           'target_kv_tokens': 35000, 'state_slots': 12,
                           'note': 'One synthetic round per fresh request. Prefix restore/prefill is outside timing; ephemeral requests suppress optional draft snapshot publication so the primer remains resident. Native draft runs; target proposal IDs and counts are frozen. No answer-quality or physical DRAM claim.'}}
    def save():
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    ex = Executor(Qwen38(config.snapshot_dir()), num_pages=35000, num_slots=12)
    ex.enable_spec(7, proposer='dflash2', state_strategy='reconstruct', log=print)
    ex.capture_graphs(log=print)
    ex.warmup(log=print)
    tok = SnapshotTokenizer(config.snapshot_dir())
    all_ids = tok.encode(PARAGRAPH * 256)
    spec, proposer = ex.spec, ex.spec.proposer
    native_propose, native_accept = proposer.propose, spec._accept_batch
    report['verify_buffer_bytes'] = spec.bytes
    for length in (128, 8192):
        for batch_size in (1, 4):
            ex.flush_cache()
            prime = ex.admit(all_ids[:length - 1], SamplingParams(max_new_tokens=1, ignore_eos=True))
            ex.prefill(prime)
            ex.release(prime)
            ids = all_ids[:length]
            def prepare():
                reqs = [ex.admit(ids, SamplingParams(max_new_tokens=32, ignore_eos=True)) for _ in range(batch_size)]
                assert all(r.prefix_len == length - 1 for r in reqs)
                # Keep the setup primer resident within the unchanged feature
                # cache budget. These one-round requests never publish cache
                # entries; suppress their optional draft snapshot creation too.
                for r in reqs:
                    r.cacheable = r.cache_enabled = False
                ex.prefill_batch([(r, 1) for r in reqs])
                assert all(spec.ready(r) and r.committed_len == length for r in reqs)
                return reqs
            def release(reqs):
                for r in reqs:
                    r.cacheable = False
                    ex.release(r)
            reqs = prepare()
            initial = spec.prepare(reqs)
            frozen_tokens = initial.proposal.tokens.clone()
            frozen_anchor = initial.tokens.view(batch_size, 8)[:, 0].tolist()
            captured = []
            original_core = GatedDeltaNet._verify_core
            def capture(self, qkv, b, a, ctx):
                captured.append((qkv.clone(), ctx.state_pool.conv[self.layer_id].clone(),
                                 ctx.seq_slots.clone(), self.conv_weight,
                                 torch.empty_like(ctx.verify_conv_windows[self.layer_id]), ctx.verify_steps))
                return original_core(self, qkv, b, a, ctx)
            GatedDeltaNet._verify_core = capture
            try:
                ctx = spec.context(initial, ex._paged_prefill)
                ex.model.forward(initial.tokens, ctx)
            finally:
                GatedDeltaNet._verify_core = original_core
            assert len(captured) == 48
            if args.dump_convolution:
                torch.save([(qkv.cpu(), state.cpu(), slots.cpu(), weight.cpu(), steps)
                            for qkv, state, slots, weight, _, steps in captured],
                           args.output / f't{length}-b{batch_size}-conv.pt')
            release(reqs)
            def convolution():
                for values in captured:
                    result = reference_conv(*values)
                return result
            conv_cost = graph_cost(convolution)
            del captured
            def fixed_propose(*args, **kwargs):
                native_propose(*args, **kwargs)
                return Proposal(frozen_tokens)
            proposer.propose = fixed_propose
            for count in (1, 4, 8):
                forced = torch.cat([frozen_tokens, torch.zeros((batch_size, 1), dtype=torch.int64, device=ex.device)], dim=1)
                if count < 8:
                    forced[:, count - 1] = (frozen_tokens[:, count - 1] + 1) % config.VOCAB_SIZE
                spec._accept_batch = lambda b, z, features: native_accept(b, forced, features)
                row = {'context': length, 'batch': batch_size, 'accepted_count': count,
                       'input_ids': ids, 'proposal_ids': frozen_tokens.tolist(), 'anchor_ids': frozen_anchor,
                       'forced_target_ids': forced.tolist(), 'convolution_48_layers': conv_cost, 'samples': []}
                for repeat in range(-1, 5):
                    reqs = prepare()
                    assert [r.output_ids[-1] for r in reqs] == frozen_anchor
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    spec.round(reqs)
                    torch.cuda.synchronize()
                    ms = (time.perf_counter() - started) * 1000
                    outputs = [r.output_ids for r in reqs]
                    assert all(r.committed_len == length + count and len(r.output_ids) == count + 1 for r in reqs)
                    row['samples'].append({'warmup': repeat < 0, 'round_ms': ms, 'output_ids': outputs})
                    release(reqs)
                records = defaultdict(list)
                for _ in range(2):
                    reqs = prepare()
                    with phases(ex, records):
                        spec.round(reqs)
                    torch.cuda.synchronize()
                    release(reqs)
                row['phases'] = summarize_phases(records)
                row['median_round_ms'] = statistics.median(r['round_ms'] for r in row['samples'] if not r['warmup'])
                row['work_sha256'] = digest([ids, frozen_anchor, row['proposal_ids'], count])
                assert all(s['output_ids'] == row['samples'][0]['output_ids'] for s in row['samples'])
                if count == 8:
                    reqs = prepare()
                    with phases(ex, defaultdict(list)), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                                                       torch.profiler.ProfilerActivity.CUDA]) as prof:
                        with torch.profiler.record_function('sharar.decode_step'):
                            spec.round(reqs)
                            torch.cuda.synchronize()
                    path = args.output / f't{length}-b{batch_size}-trace.json'
                    prof.export_chrome_trace(str(path))
                    row['trace'] = trace_summary(json.loads(path.read_text()))
                    release(reqs)
                report['cells'].append(row)
                save()
                print(json.dumps({k: row[k] for k in ('context', 'batch', 'accepted_count', 'median_round_ms', 'phases', 'convolution_48_layers')}), flush=True)
            proposer.propose, spec._accept_batch = native_propose, native_accept
            ex.flush_cache()
            ex.ledger.check_floor()
    report['completed'] = True
    save()


if __name__ == '__main__':
    main()
