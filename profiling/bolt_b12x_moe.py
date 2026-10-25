"""Compare b12x IQ2_XXS routed experts with Bolt's BF16 serving kernel.

Requires optional b12x==1.5.0, or --package pointing to that unpacked wheel.
--root accepts the pinned FlashInfer experimental source instead. No serving
settings change. Native checkpoint bytes are losslessly retiled; real weights,
changed expert ids, zero-weight padding and graph replay are checked before
any timings. Eight route sets cycle the working set beyond L2. Timings cover
routed up/ReLU²/down/reduction, excluding router, latent and shared projections.

    uv run --with b12x==1.5.0 python -m profiling.bolt_b12x_moe --output /tmp/bolt-moe.json
"""
import argparse
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

import torch
import torch.nn.functional as F

from sharar.engine import config
from sharar.kernels import ggml_bf16
from sharar.launch.guard import start_thread
from sharar.model.bolt.config import BOLT
from sharar.model.bolt.model import BoltMoe
from sharar.target import NEMOTRON_BOLT
from sharar.weights.safetensors import SnapshotWeights


def others():
    text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    return [int(line) for line in text.splitlines() if line.strip().isdigit() and int(line) != os.getpid()]


def graph_time(run):
    for _ in range(3):
        run()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run()
    values = []
    for _ in range(7):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(50):
            g.replay()
        b.record()
        b.synchronize()
        values.append(a.elapsed_time(b) * 1000 / 50)
    return statistics.median(values)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, help='pinned FlashInfer source checkout, instead of installed b12x')
    ap.add_argument('--package', type=Path, help='unpacked b12x 1.5.0 package directory')
    ap.add_argument('--snapshot', type=Path)
    ap.add_argument('--layer', type=int, default=1)
    ap.add_argument('--rows', type=int, nargs='+', default=[1, 8, 32])
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if any(m <= 0 for m in args.rows):
        ap.error('--rows must contain positive token counts')
    if others() and not args.check_only:
        raise RuntimeError('timing needs an exclusive GPU')
    start_thread()
    torch.manual_seed(74 + args.layer)
    if args.root and args.package:
        ap.error('choose --root or --package')
    revision = 'b12x==1.5.0'
    package = args.package
    if args.root:
        revision = subprocess.check_output(['git', '-C', str(args.root), 'rev-parse', 'HEAD'], text=True).strip()
        if revision != 'cb93a30db500810a8be64fb45a1cc1c26eff0770':
            raise ValueError('FlashInfer source revision differs from the qualified pin')
        package = args.root / 'flashinfer/experimental/b12x'
    elif package:
        distributions = importlib.metadata.distributions(path=[str(package.parent)])
        version = next((d.version for d in distributions if d.metadata['Name'] == 'b12x'), None)
        if version != '1.5.0':
            raise ValueError('--package needs b12x 1.5.0 and its adjacent wheel metadata')
    if package:
        entry = package / ('_api.py' if args.root else '__init__.py')
        spec = importlib.util.spec_from_file_location('b12x', entry, submodule_search_locations=[str(package)])
        module = importlib.util.module_from_spec(spec)
        sys.modules['b12x'] = module
        spec.loader.exec_module(module)
    elif importlib.metadata.version('b12x') != '1.5.0':
        raise ValueError('this comparison is qualified against b12x==1.5.0')
    from b12x.moe import fused_moe as moe
    from b12x.preparation import PreparationSession, PreparedCall, require_prepared
    snap = args.snapshot or NEMOTRON_BOLT.snapshot_dir(config.current().hf_cache or Path.home() / '.cache/huggingface')
    if not 0 <= args.layer < len(BOLT.blocks) or BOLT.blocks[args.layer].kind != 'moe':
        ap.error('--layer must select a pinned Bolt MoE block')
    w = SnapshotWeights(snap)
    block = BoltMoe(w, f'language_model.backbone.layers.{args.layer}.mixer', BOLT.blocks[args.layer], 'cuda')
    w.close()
    plan = moe.plan_weights(source=moe.PackedSource(format='iq2_xxs', w13_layout='w31'),
        activation=moe.ActivationSpec(mode='a16', nonlinearity='relu2', io_dtype=torch.bfloat16),
        geometry=moe.MoEGeometry(num_experts=block.experts, hidden_size=1024, intermediate_size=block.inter))
    prepared = moe.prepare_weights(plan=plan, weights=moe.BlockQuantWeights(block.w1, block.w2, codec='iq2_xxs'))
    compact = prepared._impl.representation.value
    payload = sum(t.numel() * t.element_size() for t in (compact.w13, compact.w13_scale, compact.w2, compact.w2_scale))
    assert payload == block.w1.numel() + block.w2.numel()
    report = {'b12x_revision': revision, 'model_revision': NEMOTRON_BOLT.model_revision,
              'package_path': str(package),
              'layer': args.layer, 'geometry': [block.experts, block.inter, 1024, block.topk],
              'check_only': args.check_only, 'resident_packed_bytes': payload, 'cases': []}
    for m in args.rows:
        decl = moe.plan_execution(experts=prepared,
            capacity=moe.ExecutionCapacity(max_tokens=m, top_k=block.topk, warmup_token_counts=(m,)),
            routing=moe.RoutingSpec(deterministic_output=True))
        counts = getattr(decl, 'token_counts', (m,))
        inputs = {}
        for count in counts:
            x = torch.randn(count, 1024, device='cuda', dtype=torch.bfloat16)
            ids = (torch.arange(count * 9, device='cuda', dtype=torch.int32).view(count, 9) % block.experts)
            weights = torch.softmax(torch.randn(count, 9, device='cuda'), -1)
            inputs[count] = (x, ids, weights, torch.empty_like(x))
        def factory(count):
            x, ids, weights, out = inputs[count]
            def make(state):
                scratch = tuple(torch.empty(s.shape, dtype=s.dtype, device='cuda') for s in state.scratch.scratch_specs())
                bound = state.bind(scratch=scratch, a=x, topk_ids=ids, topk_weights=weights, output=out)
                return PreparedCall(run=bound.run, output=out, owners=(scratch, bound))
            return make
        with PreparationSession(device='cuda', autotune=False, compile_workers=1) as session:
            request = (decl.request(name=f'bolt-{m}', prepare_calls={c: factory(c) for c in counts})
                       if hasattr(decl, 'token_counts') else decl.request(name=f'bolt-{m}', prepare_call=factory(m)))
            session.prepare((request,))
            session.freeze()
            state = require_prepared(decl, 'moe.decode')
            state = state.variants[m] if hasattr(state, 'variants') else state
            scratch = tuple(torch.empty(s.shape, dtype=s.dtype, device='cuda') for s in state.scratch.scratch_specs())
            x, ids, weights, out = inputs[m]
            bound = moe.bind(decl, scratch=scratch, a=x, topk_ids=ids, topk_weights=weights, output=out)
            def candidate():
                return moe.run(binding=bound)
            def baseline(route_ids=ids):
                up = ggml_bf16.matmul(x, block.w1, 'IQ2_XXS', ids=route_ids, repeat_input=True, relu2=True)
                down = ggml_bf16.matmul(up, block.w2, 'IQ2_XXS', ids=route_ids)
                return ggml_bf16.combine(down, weights, scale=1.)
            candidate()
            expected = baseline()
            def error():
                cosine = F.cosine_similarity(out.float().flatten(), expected.float().flatten(), dim=0).item()
                rel = ((out.float() - expected.float()).norm() / expected.float().norm()).item()
                assert cosine > .999 and rel < .015, (cosine, rel)
                return {'cosine': cosine, 'relative_l2': rel}
            case = {'rows': m, **error()}
            g = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(g):
                candidate()
            ids.add_(17).remainder_(block.experts)
            g.replay()
            expected = baseline()
            case['mutated_graph'] = error()
            weights[-1].zero_()
            ids[-1].copy_(torch.arange(9, device='cuda', dtype=torch.int32))
            g.replay()
            case['padded_nonzero'] = int(torch.count_nonzero(out[-1]))
            assert case['padded_nonzero'] == 0, case
            ids[-1].fill_(block.experts)
            out.fill_(float('nan'))
            g.replay()
            case['invalid_padded_nonzero'] = int(torch.count_nonzero(out[-1]))
            case['invalid_padded_finite'] = bool(torch.isfinite(out[-1]).all())
            assert case['invalid_padded_nonzero'] == 0 and case['invalid_padded_finite'], case
            ids[-1].copy_(torch.arange(9, device='cuda', dtype=torch.int32))
            weights[-1].fill_(1. / 9)
            candidate()
            if not args.check_only:
                cycle_ids = [(ids + i * m * 9).remainder(block.experts).contiguous() for i in range(8)]
                cycle_bindings = [moe.bind(decl, scratch=scratch, a=x, topk_ids=route_ids,
                    topk_weights=weights, output=out) for route_ids in cycle_ids]
                def candidate_cycle():
                    for b in cycle_bindings:
                        moe.run(binding=b)
                def baseline_cycle():
                    for route_ids in cycle_ids:
                        baseline(route_ids)
                case['b12x_us'], case['baseline_us'] = graph_time(candidate_cycle) / 8, graph_time(baseline_cycle) / 8
                case['method'] = 'CUDA graph cycles eight route sets across more than L2; 7 rounds of 50 replays'
                case['speedup'] = case['baseline_us'] / case['b12x_us']
            case['other_gpu_pids'] = others()
            report['cases'].append(case)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(case), flush=True)


if __name__ == '__main__':
    main()
