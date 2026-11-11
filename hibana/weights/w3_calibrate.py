"""Calibrate W3 decode copies, one target layer at a time, on frozen token inputs.

Inputs: JSON object with `samples`, each with `input_ids`; keep calibration
prompts disjoint from the accuracy/family gates and drafter test prompts.
No target logits or task answers are used to fit the quantizer.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from hibana import config
from hibana.kernels import ops, w3
from hibana.weights.safetensors import SnapshotWeights
from hibana.weights.w3_calibration import quantize
from hibana.weights.w3_checkpoint import FORMAT_VERSION, model_linears, save_weight, sha256


def source_names(layer, name):
    base = f"model.language_model.layers.{layer.index}."
    suffixes = {"mlp.gate_up": ("mlp.gate_proj", "mlp.up_proj"), "mlp.down": ("mlp.down_proj",),
                "mixer.in_qkv": ("linear_attn.in_proj_qkv",), "mixer.in_z": ("linear_attn.in_proj_z",),
                "mixer.out": ("linear_attn.out_proj",), "mixer.qkv": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
                "mixer.o": ("self_attn.o_proj",)}
    return [base + suffix for suffix in suffixes[name]]


def source_weight(reader, prefixes, device):
    parts = []
    for prefix in prefixes:
        weight = reader.get(prefix + ".weight").to(device).float()
        scale = reader.get(prefix + ".weight_scale_inv").to(device).float()
        if scale.shape != (weight.shape[0] // 128, weight.shape[1] // 128):
            raise ValueError(f"unexpected FP8 source scale shape: {prefix}")
        weight = (weight.reshape(weight.shape[0] // 128, 128, weight.shape[1] // 128, 128)
                  * scale[:, None, :, None]).reshape_as(weight)
        parts.append(weight)
        reader.close()
    return parts[0] if len(parts) == 1 else torch.cat(parts)


class InputCovariances:
    """x^T x of each named projection's input, from `ops.fp8_linear` / `ops.nvfp4_linear` calls. `served_fp8`: an FP8
    projection's BF16 input counts as the FP8 activation it reads (quantized at its static scale), as its 3-bit copy
    will. While `holdout` is set, calls add no covariance: `held_out_rows` evenly spaced rows of each call are kept in
    `held` (BF16), the probes of a held-out sample."""

    held_out_rows = 64

    def __init__(self, linears, served_fp8: bool = False):
        self.names = {id(linear): name for name, linear in linears}
        self.matrices, self.counts, self.probes = {}, {}, {}
        self.originals = {}
        self.served_fp8, self.holdout, self.held = served_fp8, False, {}

    def add(self, name, x):
        x = x.reshape(-1, x.shape[-1]).float()
        if self.holdout:
            self.held.setdefault(name, []).append(x[::max(1, len(x) // self.held_out_rows)][:self.held_out_rows].to(torch.bfloat16))
            return
        if name not in self.matrices:
            self.matrices[name] = torch.zeros((x.shape[-1], x.shape[-1]), device=x.device)
            self.counts[name] = 0
            self.probes[name] = []
        self.matrices[name].addmm_(x.T, x)
        self.counts[name] += len(x)
        # Uniformly spaced input rows for a separate projection-error diagnostic.
        if sum(len(p) for p in self.probes[name]) < 256:
            self.probes[name].append(x[::max(1, len(x) // 4)][:4].clone())

    def __enter__(self):
        for kind in ("fp8_linear", "nvfp4_linear"):
            original = getattr(ops, kind)
            self.originals[kind] = original
            def wrapped(x, linear, *args, kind=kind, original=original, **kwargs):
                name = self.names.get(id(linear))
                if name is not None:
                    if kind == "fp8_linear" and kwargs.get("x_q") is not None:
                        value = kwargs['x_q'].float() * linear.input_scale.float()
                    elif kind == "fp8_linear" and self.served_fp8:
                        from hibana.kernels.fp8_quant import quant_fp8
                        value = quant_fp8(x.reshape(-1, x.shape[-1]), linear.input_scale).float() * linear.input_scale.float()
                    elif x is not None:
                        value = x
                    else:
                        raise ValueError(f"calibration input for {name} was already packed")
                    self.add(name, value)
                return original(x, linear, *args, **kwargs)
            setattr(ops, kind, wrapped)
        return self

    def __exit__(self, *args):
        for name, original in self.originals.items():
            setattr(ops, name, original)


@torch.inference_mode()
def run(args):
    from hibana.engine.context import scratch_prefill_context
    from hibana.model.qwen38 import Qwen38
    if w3.enabled():
        raise ValueError("calibration must start with SHARAR_W3_DECODE=0")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'manifest.json').exists():
        raise ValueError("refusing to overwrite a finished calibration")
    frozen = json.loads(args.inputs.read_text())
    samples = [s['input_ids'] for s in frozen['samples']]
    if not samples or any(not ids or len(ids) > 2048 for ids in samples):
        raise ValueError("calibration samples must have between 1 and 2048 tokens")
    source_cfg = config.load_model_config(args.source)
    quant_cfg = source_cfg['quantization_config']
    if quant_cfg['quant_method'] != 'fp8' or quant_cfg['weight_block_size'] != [128, 128]:
        raise ValueError("calibration source must be the block-128 FP8 release")
    model = Qwen38(config.snapshot_dir(), fused_silu_quant=False)
    reader = SnapshotWeights(args.source)
    # Keep only one layer's hidden/residual streams and input covariances live.
    hidden = [torch.nn.functional.embedding(torch.tensor(ids, device=model.device), model.embed) for ids in samples]
    residual = [None] * len(samples)
    manifest = {'format': FORMAT_VERSION, 'target_revision': config.MODEL_REVISION,
                'source_revision': args.source.name, 'source_config_sha256': sha256(args.source / 'config.json'),
                'inputs_sha256': sha256(args.inputs), 'samples': len(samples), 'tokens': sum(map(len, samples)),
                'damping': args.damping, 'rows_per_chunk': args.rows_per_chunk, 'cuda_graphs': not args.no_graphs,
                'weights': {}, 'reports': {}, 'prefill_fused_silu_quant': False}
    progress_path = args.output / 'progress.json'
    if args.resume and progress_path.exists():
        previous = json.loads(progress_path.read_text())
        identity_keys = ('format', 'target_revision', 'source_revision', 'source_config_sha256', 'inputs_sha256',
                         'damping', 'rows_per_chunk', 'cuda_graphs')
        if any(previous[key] != manifest[key] for key in identity_keys):
            raise ValueError('resume inputs do not match the unfinished calibration')
        for entry in previous['weights'].values():
            if sha256(args.output / entry['file']) != entry['sha256']:
                raise ValueError('unfinished calibration weight checksum mismatch')
        manifest.update(weights=previous['weights'], reports=previous['reports'])
    elif progress_path.exists():
        raise ValueError('unfinished calibration exists; use --resume or a new output directory')
    linears = list(model_linears(model))
    started = time.perf_counter()
    for layer in model.layers:
        current = [(name, linear) for name, linear in linears if name.startswith(f'layers.{layer.index}.')]
        with InputCovariances(current) as collector:
            for i, ids in enumerate(samples):
                ctx = scratch_prefill_context(len(ids), model.device)
                try:
                    hidden[i], residual[i] = layer.forward(hidden[i], residual[i], ctx)
                    # FlashInfer's planner reuses pinned staging storage. Finish
                    # this sequence before replanning for a different length.
                    torch.cuda.synchronize()
                except Exception:
                    print(f'activation collection failed: layer={layer.index} sample={i} tokens={len(ids)}', flush=True)
                    raise
        print(f'layer {layer.index}: collected {sum(map(len, samples))} activation rows', flush=True)
        for name, linear in current:
            if name in manifest['weights']:
                collector.matrices.pop(name)
                collector.probes.pop(name)
                continue
            short = name.split('.', 2)[2]
            weight = source_weight(reader, source_names(layer, short), model.device)
            covariance = collector.matrices.pop(name) / collector.counts[name]
            probe = torch.cat(collector.probes.pop(name))
            reference = probe @ weight.T
            source_rtn = w3.pack(weight)
            source_error = float((probe @ w3.dequantize(source_rtn).T - reference).square().mean() / reference.square().mean())
            del source_rtn
            quantized, report = quantize(weight, covariance, damping=args.damping, capture=not args.no_graphs,
                                        rows_per_chunk=args.rows_per_chunk)
            error = float((probe @ w3.dequantize(quantized).T - reference).square().mean() / reference.square().mean())
            report.update({'projection_relative_mse': error, 'source_rtn_relative_mse': source_error,
                           'activation_rows': collector.counts[name]})
            manifest['weights'][name] = save_weight(args.output / (name + '.safetensors'), quantized)
            manifest['reports'][name] = report
            print(f'  {name}: error {source_error:.6f} -> {error:.6f}; elapsed {time.perf_counter()-started:.0f}s', flush=True)
            del weight, covariance, probe, reference, quantized
        temporary = args.output / 'progress.tmp'
        temporary.write_text(json.dumps(manifest, indent=2) + '\n')
        temporary.replace(progress_path)
        if args.layers and layer.index + 1 >= args.layers:
            break
    manifest['seconds'] = time.perf_counter() - started
    if len(manifest['weights']) == len(linears):
        temporary = args.output / 'manifest.tmp'
        temporary.write_text(json.dumps(manifest, indent=2) + '\n')
        temporary.replace(args.output / 'manifest.json')
    else:
        print('partial calibration: progress.json retained; no serving manifest published', flush=True)
    reader.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--damping', type=float, default=.01)
    parser.add_argument('--rows-per-chunk', type=int, default=1024)
    parser.add_argument('--layers', type=int, default=0, help='development screen only; incomplete artifacts cannot be served')
    parser.add_argument('--no-graphs', action='store_true')
    parser.add_argument('--resume', action='store_true', help='resume matching, checksummed completed layers')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
