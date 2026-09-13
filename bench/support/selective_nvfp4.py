"""Benchmark-only selective target requantization; no serving configuration.

Retain the existing FP8 activation boundary, then dequantize, balance and
pack NVFP4. Weights come from the loaded FP8 control, not original BF16.
The extra activation work is included in all candidate timings.
"""
from dataclasses import dataclass
from pathlib import Path

import torch

from sharar.kernels import ops
from sharar.weights.modelopt import Fp8Linear, Nvfp4Linear


@dataclass
class PackedProjection:
    packed: Nvfp4Linear
    input_scale: torch.Tensor
    smooth: torch.Tensor

    @property
    def out_features(self):
        return self.packed.out_features

    def __call__(self, x=None, x_q=None):
        if x_q is None:
            x_q = ops.quant_fp8(x.reshape(-1, x.shape[-1]), self.input_scale)
        # Keep the explicit BF16 reconstruction boundary used in calibration.
        restored = (x_q.float() * self.input_scale).bfloat16()
        balanced = (restored.float() / self.smooth).bfloat16()
        return ops.nvfp4_linear(balanced, self.packed)

    def save(self, path):
        torch.save({'packed': {k: v.cpu() if isinstance(v, torch.Tensor) else v
                              for k, v in vars(self.packed).items()},
                    'input_scale': self.input_scale.cpu(), 'smooth': self.smooth.cpu()}, path)

    @classmethod
    def load(cls, path, device):
        row = torch.load(path, map_location=device, weights_only=True)
        return cls(Nvfp4Linear(**row['packed']), row['input_scale'], row['smooth'])


def projections(model):
    return {f'l{i}.{name}': (layer.mixer, name, value)
            for i, layer in enumerate(model.layers)
            for name, value in vars(layer.mixer).items() if isinstance(value, Fp8Linear)}


class Override:
    """Temporary dispatch for individual interventions on a frozen control."""
    def __init__(self, model):
        self.original = ops.fp8_linear
        self.names = {id(value): name for name, (_, _, value) in projections(model).items()}
        self.active = {}

    def __enter__(self):
        def dispatch(x, lin, x_q=None):
            selected = self.active.get(self.names.get(id(lin)))
            return selected(x, x_q) if selected is not None else self.original(x, lin, x_q=x_q)
        ops.fp8_linear = dispatch
        return self

    def __exit__(self, *exc):
        ops.fp8_linear = self.original


def install(model, directory: Path, names):
    """Install before graph capture; release replaced FP8 weights.

    Only selected target objects get the new type; existing draft FP8 linears
    and unselected target linears still use the original implementation.
    """
    found = projections(model)
    if len(names) != len(set(names)) or set(names) - found.keys():
        raise ValueError('duplicate or unknown selective projection')
    for name in names:
        owner, attr, old = found[name]
        new = PackedProjection.load(directory / (name + '.pt'), model.device)
        if (new.packed.out_features, new.packed.in_features) != tuple(old.weight.shape):
            raise ValueError(f'projection shape differs: {name}')
        if not torch.equal(new.input_scale, old.input_scale):
            raise ValueError(f'FP8 activation scale differs: {name}')
        setattr(owner, attr, new)
    original = ops.fp8_linear

    def dispatch(x, lin, x_q=None):
        return lin(x, x_q) if isinstance(lin, PackedProjection) else original(x, lin, x_q=x_q)

    ops.fp8_linear = dispatch
    model.cache_identity = ('selective-nvfp4-experiment', tuple(names))
    return original


def error(actual, expected):
    a, b = actual.float(), expected.float()
    return {'nrmse': ((a - b).square().sum() / b.square().sum().clamp_min(1e-30)).sqrt().item(),
            'max_abs': (a - b).abs().max().item(), 'finite': bool(torch.isfinite(a).all())}


def logit_error(actual, expected, batch):
    a, b = actual.float(), expected.float()
    al, bl = a.log_softmax(-1), b.log_softmax(-1)
    kl = (bl.exp() * (bl - al)).sum(-1)
    same = a.argmax(-1) == b.argmax(-1)
    return {'finite': bool(torch.isfinite(a).all()), 'mean_kl': kl.mean().item(),
            'max_row_kl': kl.max().item(), 'argmax_agreement': same.float().mean().item(),
            'first_argmax_agreement': same.reshape(batch, 16)[:, 0].float().mean().item(),
            'logit_nrmse': error(a, b)['nrmse']}


def assignments(rows, gates):
    selected = {}
    for label, gate in gates.items():
        eligible = []
        for row in rows:
            if set(row.get('ablations', {})) != {'1', '4'}:
                continue
            if not all(v['finite'] and v['nrmse'] <= gate['local_nrmse'] for v in row['validation'].values()):
                continue
            if not all(v['finite'] and v['mean_kl'] <= gate['mean_kl']
                       and v['argmax_agreement'] >= gate['argmax_agreement']
                       and v['first_argmax_agreement'] >= gate['first_argmax_agreement']
                       for v in row['ablations'].values()):
                continue
            savings = [v['control']['median_ms'] - v['candidate']['median_ms'] for v in row['timing'].values()]
            if min(savings) <= 0:
                continue
            eligible.append((sum(savings), row['name']))
        selected[label] = [name for _, name in sorted(eligible, reverse=True)[:gate['max_projections']]]
    return selected
