"""Prospective speculation costs, evaluated before optional target cache sizing."""
from dataclasses import dataclass
import json
import struct

from sharar import config

GIB = 2**30


@dataclass(frozen=True)
class SpecConfig:
    name: str = 'mtp'
    depth: int = 3
    state_strategy: str = 'snapshots'

    def __post_init__(self):
        limit = 7 if self.name in ('dspark', 'dflash2') else 4
        if self.name not in ('mtp', 'ngram', 'dspark', 'dflash2') or type(self.depth) is not int or not 1 <= self.depth <= limit:
            raise ValueError('unsupported proposer/depth')
        if self.state_strategy not in ('snapshots', 'reconstruct'):
            raise ValueError('unsupported verify state strategy')


def tensor_bytes(snapshot, prefix=''):
    """Safetensors headers only: no tensor materialization or backend import."""
    total, largest = 0, 0
    for shard in snapshot.glob('*.safetensors'):
        with shard.open('rb') as stream:
            header = json.loads(stream.read(struct.unpack('<Q', stream.read(8))[0]))
        for name, value in header.items():
            if name == '__metadata__' or not name.startswith(prefix):
                continue
            left, right = value['data_offsets']
            total += right - left
            largest = max(largest, right - left)
    if not total:
        raise ValueError('empty draft checkpoint budget')
    return total, largest


def verify_bytes(batch, steps, strategy):
    ssm = 48 * batch * 48 * 128 * 128 * 4
    conv = 48 * batch * steps * (2 * 16 * 128 + 48 * 128) * 3 * 2
    if strategy == 'snapshots':
        return ssm * steps + conv
    compact = 48 * batch * steps * ((16 * 128 + 48 * 128) * 2 + 48 * 2 * 4)
    return compact + conv


def budget(spec, batch, context):
    """Return fixed future bytes and extra bytes per target KV page.

    Header-derived weights, exact persistent tensor shapes, plus explicit
    bounded load/graph/activation reserves. Reserves are not claimed allocations.
    """
    fixed = verify_bytes(batch, spec.depth + 1, spec.state_strategy)
    page_bytes = 0
    if spec.name == 'mtp':
        weights, largest = tensor_bytes(config.snapshot_dir(), 'mtp.')
        fixed += weights + 2 * largest + 128 * 2**20
        page_bytes = 2 * config.NUM_KV_HEADS * config.HEAD_DIM * 2
    elif spec.name in ('dspark', 'dflash2'):
        from sharar.proposers.parallel import PINS
        name, revision = PINS[spec.name]
        snapshot = config.snapshot_dir().parents[2] / ('models--' + name.replace('/', '--')) / 'snapshots' / revision
        weights, largest = tensor_bytes(snapshot)
        cfg = json.loads((snapshot / 'config.json').read_text())
        window = cfg['sliding_window'] or context
        # Active rings (+dummy), cached features, and one gather/clone temporary.
        fixed += (2 * batch + 2) * window * 5120 * 2
        fixed += 5 * (batch + 1) * (window + spec.depth + 1) * 2 * 8 * 128 * 2
        fixed += weights + 2 * largest + 128 * 2**20
    # Graph/workspace growth and allocator slack kept separate from live tensors.
    fixed += 2 * GIB
    return fixed, page_bytes
