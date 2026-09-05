"""Pinned publisher-oracle and real-target smoke gates for external drafts.

Reference dependencies live in /proof/oracle-python, never in the runtime.
Run only in the exclusive GB10 harness with a host memory watchdog.
"""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import sys
import statistics
import time

import torch
from safetensors.torch import load_file

from sharar import config
from sharar.engine.contracts import StepInputs
from sharar.engine.executor import Executor
from sharar.engine.sampler import SamplingParams
from sharar.model.qwen38 import Qwen38
from sharar.proposers.parallel import PINS
from sharar.tokenizer import SnapshotTokenizer

import local_workloads
from memory_trace import measure_stage
from phase7c import http_server


def oracle_model(name, snapshot):
    sys.path.insert(0, '/proof/oracle-python')
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Config, Qwen3RotaryEmbedding
    path = snapshot / 'dflash.py' if name == 'dspark' else Path('/proof/oracle-source/model.py')
    spec = importlib.util.spec_from_file_location('published_draft_oracle', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = Qwen3Config(**json.loads((snapshot / 'config.json').read_text()))
    with torch.device('meta'):
        model = (module.DFlashDraftModel if name == 'dspark' else module.DFlash2DraftModel)(cfg)
    weights = load_file(snapshot / 'model.safetensors')
    if name == 'dflash2':
        for key in ('predecessor_codebook', 'successor_codebook'):
            weights[f'candidate_selector.{key}.weight'] = weights.pop(f'candidate_selector.{key}')
    result = model.load_state_dict(weights, strict=False, assign=True)
    assert not result.missing_keys, result
    assert all(k.startswith(('markov_head.', 'confidence_head.')) for k in result.unexpected_keys), result
    model.rotary_emb = Qwen3RotaryEmbedding(cfg, device='cuda')
    return model.to(device='cuda', dtype=torch.bfloat16).eval()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', choices=PINS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workloads', action='store_true')
    parser.add_argument('--extended', action='store_true')
    parser.add_argument('--verify-policy', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'name': args.name, 'draft_pin': PINS[args.name], 'target': config.MODEL_REVISION}
    root = Path(config.__file__).resolve().parents[1]
    report['sources'] = {str(p.relative_to(root)): local_workloads.digest(p.read_bytes())
                         for pattern in ('sharar/**/*.py', 'bench/**/*.py', 'bench/workloads/*')
                         for p in root.glob(pattern) if p.is_file()}
    try:
        with measure_stage(report, 'target_load'):
            target = Qwen38(config.snapshot_dir())
        torch.cuda.empty_cache()
        ex = Executor(target, num_pages=35000, num_slots=12)
        with measure_stage(report, 'draft_load'):
            ex.enable_spec(7, proposer=args.name, state_strategy='reconstruct', log=print)
        torch.cuda.empty_cache()
        tok = SnapshotTokenizer(config.snapshot_dir())
        ids = tok.encode('The cache contains the computation of a shared prompt prefix. ' * 16)
        captured = {}
        forward = target.forward
        def capture(ids, ctx):
            out = forward(ids, ctx)
            captured.update(out)
            return out
        target.forward = capture
        req = ex.admit(ids, SamplingParams(max_new_tokens=16, ignore_eos=True))
        ex.prefill(req)
        target.forward = forward
        adapter = ex.spec.proposer
        model = adapter.model
        native_forward = model.forward
        native = {}
        def capture_draft(*args):
            output = native_forward(*args)
            native['tokens'], native['hidden'] = args[0].clone(), output.clone()
            return output
        model.forward = capture_draft
        inputs = StepInputs(torch.tensor([req.output_ids[-1]], device='cuda'),
                            torch.tensor([req.committed_len], device='cuda'),
                            torch.tensor([req.slot], dtype=torch.int32, device='cuda'))
        adapter.propose([req], None, inputs, graphs=False)
        model.forward = native_forward
        name, revision = PINS[args.name]
        snapshot = config.snapshot_dir().parents[2] / ('models--' + name.replace('/', '--')) / 'snapshots' / revision
        oracle = oracle_model(args.name, snapshot)
        features = torch.cat([captured['features'][i] for i in model.features], dim=-1).unsqueeze(0)
        expected = oracle(position_ids=torch.arange(len(ids) + adapter.noise_steps, device='cuda').unsqueeze(0),
                          noise_embedding=torch.nn.functional.embedding(native['tokens'], target.embed),
                          target_hidden=features, use_cache=False)
        actual = native['hidden']
        error = actual.float() - expected.float()
        relative = (error.square().mean() / expected.float().square().mean()).sqrt().item()
        report['oracle'] = {'relative_rms': relative, 'max_abs': error.abs().max().item(),
                            'cosine': torch.nn.functional.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0).item(),
                            'target_head_top1_agreement': (target.logits(actual.reshape(-1, 5120)).argmax(-1) == target.logits(expected.reshape(-1, 5120)).argmax(-1)).float().mean().item()}
        print(json.dumps(report['oracle']), flush=True)
        assert relative < .02, report['oracle']  # BF16 backend comparison; investigate larger drift
        del oracle, features, expected, actual, native, captured
        gc.collect()
        torch.cuda.empty_cache()
        ex.release(req)
        ex.flush_cache()
        with measure_stage(report, 'graphs'):
            ex.capture_graphs(log=print)
        records = []
        for depth, batch, temperature in ((d, b, t) for d in ex.spec.verify_depths for b, t in ((1, 0.), (2, 0.), (3, .7), (4, 0.))):
            ex.spec.verify_depth = depth
            params = SamplingParams(max_new_tokens=16, temperature=temperature, seed=31, ignore_eos=True)
            for repeat in range(2):
                requests = ex.generate_batch([(ids, params) for _ in range(batch)])
                assert all(len(r.output_ids) == 16 and r.spec_rounds > 0 and r.released for r in requests)
                records.append({'depth': depth, 'batch': batch, 'temperature': temperature, 'repeat': repeat,
                                'outputs': [r.output_ids for r in requests],
                                'cached': [r.prefix_len for r in requests],
                                'rounds': [r.spec_rounds for r in requests]})
        report['smoke'] = records
        report['cache'] = {'hits': adapter.cache_hits, 'fallbacks': adapter.cache_fallbacks}
        assert adapter.cache_hits > 0 and adapter.cache_fallbacks == 0
        ex.flush_cache()
        assert ex.kv_pool.free_count == ex.kv_pool.num_pages - 1
        assert ex.state_pool.free_count == ex.state_pool.num_slots - 1
        if args.extended:
            boundaries = []
            for length in (8192, 32766):
                long_ids = (ids * (length // len(ids) + 1))[:length]
                for graphs in (False, True):
                    ex.use_graphs = graphs
                    for temperature in (0., .7):
                        for repeat in range(2):
                            r = ex.generate(long_ids, SamplingParams(max_new_tokens=2, temperature=temperature, seed=47, ignore_eos=True))
                            assert r.released and len(r.output_ids) == 2 and r.spec_rounds == 1
                            assert r.committed_len == length + 1
                            boundaries.append(dict(context=length, graphs=graphs, temperature=temperature,
                                                   repeat=repeat, cached=r.prefix_len, outputs=r.output_ids))
                        ex.flush_cache()
            report['boundaries'] = boundaries
            ex.use_graphs = True
        ex.spec.verify_depth = ex.spec.k
        if args.verify_policy is not None:
            from sharar.engine.width_policy import WidthPolicy
            ex.spec.width_policy = WidthPolicy.load(args.verify_policy, args.name, 'reconstruct')
            report['verify_policy'] = json.loads(args.verify_policy.read_text())
        ex.spec.collect_acceptance = args.name == 'dspark'
        if args.workloads:
            frozen = local_workloads.materialize(local_workloads.DEFAULT_MANIFEST, config.snapshot_dir())
            with http_server(ex, tok, config.snapshot_dir(), {'proposer': args.name}) as url:
                report['workloads'] = local_workloads.run(url, frozen, tok, args.output / 'workloads')
            assert not report['workloads']['automatic_failures'] and not report['workloads']['request_failures']
        report['acceptance_trace'] = list(ex.spec.acceptance_trace)
        ex.spec.collect_acceptance = False
        ex.spec.width_policy = None
        # Synchronize only at the boundaries of independent timing samples.
        # Retain token work alongside costs; short context is a pilot profile.
        costs = []
        cache, ex.cache = ex.cache, None
        for batch in (1, 4):
            for depth in ex.spec.verify_depths:
                ex.spec.verify_depth = depth
                samples = []
                for repeat in range(3):
                    reqs = [ex.admit(ids, SamplingParams(max_new_tokens=128, ignore_eos=True)) for _ in range(batch)]
                    for r in reqs:
                        ex.prefill(r)
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    for _ in range(4):
                        ex.decode_step(reqs)
                    torch.cuda.synchronize()
                    samples.append({'round_ms': (time.perf_counter() - started) * 250,
                                    'output_ids': [r.output_ids for r in reqs]})
                    for r in reqs:
                        ex.release(r)
                costs.append({'batch': batch, 'depth': depth, 'context': len(ids), 'samples': samples,
                              'round_ms': statistics.median(s['round_ms'] for s in samples)})
        ex.cache = cache
        report['width_costs'] = costs
        report['passed'] = True
    finally:
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
