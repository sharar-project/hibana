"""Real packed-input NVFP4 costs; diagnostic GPU benchmark, never a serving mode.

Capture one eager DFlash2 round after a cached prefix, then replay the exact
packed tensors and checkpoint weights in CUDA graphs. Cold samples flush a
128 MiB buffer before (outside) each timed replay. Warm means repeated address
reuse, not an assertion that a whole projection fits in L2.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import torch

from sharar import config
from sharar.engine.executor import Executor
from sharar.engine.sampler import SamplingParams
from sharar.kernels import ops
from sharar.model import mlp
from sharar.model.qwen38 import Qwen38
from sharar.parity.prompts import PARAGRAPH
from sharar.tokenizer import SnapshotTokenizer

from bench.support.cuda import tensor_hash, capture_graph, time_graph, gemm


def linears(model):
    return {**{f'l{i}.{kind}': getattr(layer.mlp, kind)
               for i, layer in enumerate(model.layers) for kind in ('gate_up', 'down')},
            'vocab': model.lm_head}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'baseline': 'f6af0ef1483cac3d68d1c0c0ad10d4c7cd2a1ac8',
              'target_revision': config.MODEL_REVISION, 'cells': [],
              'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'protocol': {'contexts': [128, 8192], 'batches': [1, 4], 'draft_k': 7,
                           'cold_flush_bytes': 128 * 2**20, 'samples': 9, 'warm_replays': 10,
                           'note': 'Component CUDA-event costs, not whole-round TPS or physical DRAM counters.'}}
    def save():
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    model = Qwen38(config.snapshot_dir())
    ex = Executor(model, num_pages=35000, num_slots=12)
    ex.enable_spec(7, proposer='dflash2', state_strategy='reconstruct', log=print)
    weights = linears(model)
    names = {id(v): k for k, v in weights.items()}
    tok = SnapshotTokenizer(config.snapshot_dir())
    ids = tok.encode(PARAGRAPH * 256)
    flush = torch.zeros(128 * 2**20, dtype=torch.uint8, device='cuda')
    native_linear, native_silu = ops.nvfp4_linear, mlp.silu_mul_nvfp4_quantize
    for length in (128, 8192):
        ex.flush_cache()
        prime = ex.admit(ids[:length - 1], SamplingParams(max_new_tokens=1, ignore_eos=True))
        ex.prefill(prime)
        ex.release(prime)
        for batch in (1, 4):
            reqs = [ex.admit(ids[:length], SamplingParams(max_new_tokens=32, ignore_eos=True)) for _ in range(batch)]
            assert all(r.prefix_len == length - 1 for r in reqs)
            for r in reqs:
                r.cacheable = r.cache_enabled = False
            ex.prefill_batch([(r, 1) for r in reqs])
            packed, packing = [], []
            def record_linear(x, lin, prequantized=None):
                if prequantized is None:
                    # Clone producer output before any later in-place use.
                    owned = x.clone()
                    packing.append((ops.nvfp4_quantize, owned, lin.input_scale_inv))
                    q, sf = ops.nvfp4_quantize(x, lin.input_scale_inv)
                else:
                    q, sf = prequantized
                out = gemm(q, sf, lin)
                packed.append((names[id(lin)], q.clone(), sf.clone(), out.clone()))
                return out
            def record_silu(x, inv):
                packing.append((native_silu, x.clone(), inv))
                return native_silu(x, inv)
            ops.nvfp4_linear, mlp.silu_mul_nvfp4_quantize = record_linear, record_silu
            try:
                ex.spec.round(reqs)
            finally:
                ops.nvfp4_linear, mlp.silu_mul_nvfp4_quantize = native_linear, native_silu
            torch.cuda.synchronize()
            assert len(packed) == len(packing) == 130
            label = f't{length}-b{batch}'
            torch.save([(name, q.cpu(), sf.cpu(), out.cpu()) for name, q, sf, out in packed],
                       args.output / (label + '-packed.pt'))
            row = {'context': length, 'batch': batch, 'calls': [], 'groups': {}, 'backends': []}
            for name, q, sf, out in packed:
                lin = weights[name]
                row['calls'].append({'name': name, 'm': len(q), 'n': lin.out_features, 'k': lin.in_features,
                                     'packed_sha256': tensor_hash(q), 'scales_sha256': tensor_hash(sf),
                                     'output_sha256': tensor_hash(out),
                                     'weight_bytes': lin.weight.numel(), 'scale_bytes': lin.weight_scale_swizzled.numel()})
            groups = defaultdict(list)
            for p in packed:
                groups[p[0].split('.')[-1]].append(p)
            for name, values in groups.items():
                def run(values=values):
                    for key, q, sf, _ in values:
                        output = gemm(q, sf, weights[key])
                    return output
                g, _ = capture_graph(run)
                row['groups'][name] = {'calls': len(values), 'warm': time_graph(g), 'cold': time_graph(g, flush)}
            def pack_all():
                for fn, x, inv in packing:
                    result = fn(x, inv)
                return result
            g, _ = capture_graph(pack_all)
            row['packing_130_calls'] = {'warm': time_graph(g), 'cold': time_graph(g, flush)}
            # Early/middle/late layer plus both target/draft vocabulary calls.
            for name, q, sf, reference in packed:
                if name not in ('l0.gate_up', 'l0.down', 'l31.gate_up', 'l31.down', 'l63.gate_up', 'l63.down', 'vocab'):
                    continue
                item = {'name': name, 'm': len(q), 'results': {}}
                for backend in ('cudnn', 'b12x', 'cutlass'):
                    g, output = capture_graph(lambda: gemm(q, sf, weights[name], backend))
                    g.replay()
                    torch.cuda.synchronize()
                    delta = (output.float() - reference.float()).abs()
                    item['results'][backend] = {'warm': time_graph(g), 'cold': time_graph(g, flush),
                                                'exact': torch.equal(output, reference),
                                                'max_abs_error': delta.max().item(),
                                                'relative_l2_error': (delta.norm() / reference.float().norm().clamp_min(1e-30)).item(),
                                                'argmax_equal': torch.equal(output.argmax(-1), reference.argmax(-1))}
                row['backends'].append(item)
            row['output_ids'] = [r.output_ids for r in reqs]
            for r in reqs:
                ex.release(r)
            report['cells'].append(row)
            save()
            print(json.dumps({k: row[k] for k in ('context', 'batch', 'groups', 'packing_130_calls')}), flush=True)
            del packed, packing, groups
    report['completed'] = True
    save()


if __name__ == '__main__':
    main()
