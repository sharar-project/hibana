"""Compare b12x IQ2_XXS routed experts with Bolt's BF16 serving kernel.

Requires optional b12x==1.5.0, or --package pointing to that unpacked wheel.
--root accepts the pinned FlashInfer experimental source instead. No serving
settings change. Native checkpoint bytes are losslessly retiled; real weights,
changed expert ids, zero-weight padding and graph replay are checked before
any timings. --sibling adds a frozen checkout of the grouped FP8 IQ2 kernel.
Route sets cycle the working set beyond L2. Timings cover
routed up/ReLU²/down/reduction, excluding router, latent and shared projections.

    uv run --with b12x==1.5.0 python -m profiling.bolt_b12x_moe --output /tmp/bolt-moe.json
"""
import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import platform
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


def graph_times(runs, calls):
    graphs = {}
    for name, run in runs.items():
        for _ in range(3):
            run()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run()
        graphs[name] = g
    values = {name: [] for name in runs}
    names = list(runs)
    for round in range(9):
        for name in names[round % len(names):] + names[:round % len(names)]:
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(10):
                graphs[name].replay()
            b.record()
            b.synchronize()
            values[name].append(a.elapsed_time(b) * 1000 / (10 * calls))
    return {name: {'median_us': statistics.median(v), 'min_us': min(v), 'max_us': max(v)}
            for name, v in values.items()}


def load_sibling(root):
    """Import frozen development modules without altering serving dispatch."""
    for name in ('ggml', 'iq2_moe'):
        spec = importlib.util.spec_from_file_location(f'sharar.kernels.{name}', root / 'sharar/kernels' / f'{name}.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        setattr(sys.modules['sharar.kernels'], name, module)
    return module


def numerical_error(actual, expected):
    cosine = F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0).item()
    rel = ((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-20)).item()
    assert torch.isfinite(actual).all(), (cosine, rel)
    return {'cosine': cosine, 'relative_l2': rel}


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, help='pinned FlashInfer source checkout, instead of installed b12x')
    ap.add_argument('--package', type=Path, help='unpacked b12x 1.5.0 package directory')
    ap.add_argument('--snapshot', type=Path)
    ap.add_argument('--sibling', type=Path, help='frozen source tree containing sharar/kernels/iq2_moe.py')
    ap.add_argument('--random-routes', action='store_true')
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
    native = load_sibling(args.sibling) if args.sibling else None
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
    native_experts = native.pack_experts(block.w1, block.w2) if native else None
    compact = prepared._impl.representation.value
    payload = sum(t.numel() * t.element_size() for t in (compact.w13, compact.w13_scale, compact.w2, compact.w2_scale))
    assert payload == block.w1.numel() + block.w2.numel()
    report = {'b12x_revision': revision, 'model_revision': NEMOTRON_BOLT.model_revision,
              'package_path': str(package),
              'layer': args.layer, 'geometry': [block.experts, block.inter, 1024, block.topk],
              'check_only': args.check_only, 'resident_packed_bytes': payload, 'cases': []}
    report.update(host=platform.node(), gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if native:
        assert native_experts.bytes == payload
        report.update(sibling_root=str(args.sibling), native_revision=native._source_revision(),
                      native_python_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
                      activation_profile='b12x BF16; sibling E4M3 with per-K32 FP32 scales')
    report.update(random_routes=args.random_routes, score_scale=5., input_std=.125)
    for m in args.rows:
        decl = moe.plan_execution(experts=prepared,
            capacity=moe.ExecutionCapacity(max_tokens=m, top_k=block.topk, warmup_token_counts=(m,)),
            routing=moe.RoutingSpec(deterministic_output=True))
        counts = getattr(decl, 'token_counts', (m,))
        inputs = {}
        for count in counts:
            x = torch.randn(count, 1024, device='cuda', dtype=torch.bfloat16) * .125
            ids = (torch.stack([torch.randperm(block.experts, device='cuda')[:9] for _ in range(count)]).int()
                   if args.random_routes else torch.arange(count * 9, device='cuda', dtype=torch.int32).view(count, 9) % block.experts)
            weights = torch.softmax(torch.randn(count, 9, device='cuda'), -1) * 5.
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
            if native:
                native_workspace = native.make_workspace(native_experts, m, 9)
                native_out = torch.empty_like(x)
                def native_run(route_ids=ids):
                    return native.moe(native_experts, x, route_ids, weights, workspace=native_workspace, out=native_out)
                native_run()
                ng = torch.cuda.CUDAGraph()
                with torch.cuda.graph(ng):
                    native_run()
            candidate()
            expected = baseline()
            def error():
                e = numerical_error(out, expected)
                assert e['cosine'] > .999 and e['relative_l2'] < .015, e
                return e
            case = {'rows': m, **error()}
            if native:
                case['native_vs_bf16'] = numerical_error(native_out, expected)
                fp8_reference = native.reference(native_experts, x[:2], ids[:2], weights[:2])
                case['native_vs_fp8_reference'] = numerical_error(native_out[:2], fp8_reference)
                assert case['native_vs_fp8_reference']['relative_l2'] < .01, case
            g = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(g):
                candidate()
            ids.add_(17).remainder_(block.experts)
            g.replay()
            expected = baseline()
            case['mutated_graph'] = error()
            if native:
                ng.replay()
                case['native_mutated_graph'] = numerical_error(native_out, expected)
                fp8_reference = native.reference(native_experts, x[:2], ids[:2], weights[:2])
                case['native_mutated_vs_fp8_reference'] = numerical_error(native_out[:2], fp8_reference)
                assert case['native_mutated_vs_fp8_reference']['relative_l2'] < .01, case
            saved_ids, saved_weights = ids[-1].clone(), weights[-1].clone()
            weights[-1].zero_()
            ids[-1].copy_(torch.arange(9, device='cuda', dtype=torch.int32))
            g.replay()
            case['padded_nonzero'] = int(torch.count_nonzero(out[-1]))
            assert case['padded_nonzero'] == 0, case
            if native:
                ng.replay()
                assert torch.count_nonzero(native_out[-1]) == 0
            ids[-1].fill_(block.experts)
            out.fill_(float('nan'))
            g.replay()
            case['invalid_padded_nonzero'] = int(torch.count_nonzero(out[-1]))
            case['invalid_padded_finite'] = bool(torch.isfinite(out[-1]).all())
            assert case['invalid_padded_nonzero'] == 0 and case['invalid_padded_finite'], case
            if native:
                native_out.fill_(float('nan'))
                ng.replay()
                assert torch.isfinite(native_out[-1]).all() and torch.count_nonzero(native_out[-1]) == 0
                case['native_padding_passed'] = True
            ids[-1].copy_(saved_ids)
            weights[-1].copy_(saved_weights)
            candidate()
            if not args.check_only:
                cycle_ids, union = [], set()
                while len(cycle_ids) < 8 or (len(union) * payload // block.experts <= 64 * 2**20
                                            and len(union) < block.experts and len(cycle_ids) < 32):
                    route_ids = (ids + len(cycle_ids) * m * 9).remainder(block.experts).contiguous()
                    union.update(route_ids.flatten().tolist())
                    cycle_ids.append(route_ids)
                cycle_bindings = [moe.bind(decl, scratch=scratch, a=x, topk_ids=route_ids,
                    topk_weights=weights, output=out) for route_ids in cycle_ids]
                def candidate_cycle():
                    for b in cycle_bindings:
                        moe.run(binding=b)
                def baseline_cycle():
                    for route_ids in cycle_ids:
                        baseline(route_ids)
                runs = {'b12x': candidate_cycle, 'baseline': baseline_cycle}
                if native:
                    def native_cycle():
                        for route_ids in cycle_ids:
                            native_run(route_ids)
                    runs['native'] = native_cycle
                case['timings'] = graph_times(runs, len(cycle_ids))
                case.update({name + '_us': t['median_us'] for name, t in case['timings'].items()})
                case['method'] = 'CUDA graph route cycles above L2; rotating backend order over 9 rounds of 10 replays'
                case.update(cycle_calls=len(cycle_ids), distinct_experts=len(union),
                            active_weight_bytes=len(union) * payload // block.experts)
                case['speedup'] = case['baseline_us'] / case['b12x_us']
            case['other_gpu_pids'] = others()
            report['cases'].append(case)
            report['valid_for_speed_comparison'] = not args.check_only and not any(c['other_gpu_pids'] for c in report['cases'])
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(case), flush=True)


if __name__ == '__main__':
    main()
