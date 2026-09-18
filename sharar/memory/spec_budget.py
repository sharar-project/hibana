"""Prospective speculation costs, evaluated before optional target cache sizing."""
from dataclasses import dataclass
import json
import struct

from sharar.target import DEFAULT_TARGET

GIB = 2**30


@dataclass(frozen=True)
class SpecConfig:
    name: str = 'mtp'
    depth: int = 3
    state_strategy: str = 'snapshots'
    draft_precision: str = 'bf16'
    draft_dir: str | None = None  # a trained DFlash2/DSpark checkpoint directory instead of the pinned snapshot

    def __post_init__(self):
        from sharar.proposers.factory import MAX_DEPTHS
        if type(self.depth) is not int or not 1 <= self.depth <= MAX_DEPTHS.get(self.name, 0):
            raise ValueError('unsupported proposer/depth')
        if self.state_strategy not in ('snapshots', 'reconstruct'):
            raise ValueError('unsupported verify state strategy')
        if self.draft_dir is not None and self.name not in ('dspark', 'dflash2'):
            raise ValueError('a draft directory applies to parallel drafts only')
        if self.draft_precision not in ('bf16', 'fp8') or (self.draft_precision == 'fp8' and self.name not in ('dspark', 'dflash2')):
            raise ValueError('FP8 draft precision requires a parallel draft')


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


def verify_bytes(batch, steps, strategy, target=None):
    t = target or DEFAULT_TARGET
    layers = len(t.gdn_layer_ids)
    ssm = layers * batch * t.linear_num_value_heads * t.linear_value_head_dim * t.linear_key_head_dim * 4
    conv = layers * batch * steps * t.conv_dim * t.conv_state_len * 2
    if strategy == 'snapshots':
        return ssm * steps + conv
    compact = layers * batch * steps * ((t.linear_num_key_heads * t.linear_key_head_dim + t.linear_num_value_heads * t.linear_value_head_dim) * 2
                                        + t.linear_num_value_heads * 2 * 4)
    return compact + conv


def budget(spec, batch, context, *, cache_slots=96, target=None):
    """Return fixed future bytes and extra bytes per target KV page.

    Header-derived weights, exact persistent tensor shapes, plus explicit
    bounded load/graph/activation reserves. Reserves are not claimed allocations.
    """
    t = target or DEFAULT_TARGET
    fixed = verify_bytes(batch, spec.depth + 1, spec.state_strategy, t)
    page_bytes = 0
    if spec.name == 'mtp':
        weights, largest = tensor_bytes(t.snapshot_dir(), 'mtp.')
        fixed += weights + 2 * largest + 128 * 2**20
        page_bytes = 2 * t.num_kv_heads * t.head_dim * 2
    elif spec.name in ('dspark', 'dflash2'):
        from sharar.proposers.parallel import PINS
        name, revision = PINS[spec.name]
        snapshot = t.snapshot_dir().parents[2] / ('models--' + name.replace('/', '--')) / 'snapshots' / revision
        weights, largest = tensor_bytes(snapshot)
        cfg = json.loads((snapshot / 'config.json').read_text())
        window = cfg['sliding_window'] or context
        # Active rings (+dummy), one feature window per usable recurrent slot,
        # and two temporaries while gathering and replacing a retained window.
        # Automatic pool sizing caps recurrent slots at 96; explicit pools pass
        # their actual count before allocation. This is a prospective cap.
        fixed += (batch + cache_slots + 2) * window * 5120 * 2
        fixed += 5 * (batch + 1) * (window + spec.depth + 1) * 2 * 8 * 128 * 2
        fixed += weights + 2 * largest + 128 * 2**20
    # Graph/workspace growth and allocator slack kept separate from live tensors.
    fixed += 2 * GIB
    return fixed, page_bytes
