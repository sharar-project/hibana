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
    draft_dir: str | None = None  # a trained DFlash2 checkpoint directory instead of the pinned snapshot

    def __post_init__(self):
        from sharar.proposers.factory import MAX_DEPTHS
        if type(self.depth) is not int or not 1 <= self.depth <= MAX_DEPTHS.get(self.name, 0):
            raise ValueError('unsupported proposer/depth')
        if self.state_strategy not in ('snapshots', 'reconstruct'):
            raise ValueError('unsupported verify state strategy')
        if self.draft_dir is not None and self.name != 'dflash2':
            raise ValueError('a draft directory applies to parallel drafts only')
        if self.draft_precision not in ('bf16', 'fp8') or (self.draft_precision == 'fp8' and self.name != 'dflash2'):
            raise ValueError('FP8 draft precision requires a parallel draft')


def tensor_bytes(snapshot, prefix=''):
    """Safetensors headers only: no tensor materialization or backend import."""
    total, largest = 0, 0
    shards = sorted(snapshot.glob('*.safetensors'))
    if not shards:
        raise FileNotFoundError(f'no *.safetensors under {snapshot}: download the pinned checkpoint into the model cache')
    for shard in shards:
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


def budget(spec, batch, *, cache_slots=96, target=None):
    """Return fixed future bytes and extra bytes per target KV page.

    Header-derived weights, exact persistent tensor shapes, plus explicit
    bounded load/graph/activation reserves. Reserves are not claimed allocations.
    """
    t = target or DEFAULT_TARGET
    fixed = verify_bytes(batch, spec.depth + 1, spec.state_strategy, t)
    page_bytes = 0
    if spec.name == 'mtp':
        weights, largest = tensor_bytes(t.snapshot_dir(), 'mtp.')
        if t.family == 'flashnext':
            # The expert tensors (most of the bytes) are stored FP8 and loaded one expert at a time.
            weights, largest = weights // 2, 4 * 2**20
        fixed += weights + 2 * largest + 128 * 2**20
        page_bytes = 2 * t.num_kv_heads * t.head_dim * 2
    elif spec.name == 'dflash2':
        from sharar.proposers.factory import draft_snapshot
        snapshot = draft_snapshot(spec.name, spec.draft_dir, t)  # the checkpoint served: pinned, or --draft DIR
        weights, largest = tensor_bytes(snapshot)
        cfg = json.loads((snapshot / 'config.json').read_text())
        window = cfg['sliding_window']
        # Active rings (+dummy), one feature window per usable recurrent slot,
        # and two temporaries while gathering and replacing a retained window.
        # Automatic pool sizing passes the most slots its plan gives; explicit
        # pools pass their actual count before allocation. This is a prospective cap.
        fixed += (batch + cache_slots + 2) * window * cfg['hidden_size'] * 2
        fixed += cfg['num_hidden_layers'] * (batch + 1) * (window + spec.depth + 1) * 2 * cfg['num_key_value_heads'] * cfg['head_dim'] * 2
        fixed += weights + 2 * largest + 128 * 2**20 * max(1, -(-batch // 32))  # the draft attention's workspace
    # CUDA graphs, workspaces and cached activation blocks are counted once, by `ledger.after_plan_bytes`: one verify
    # depth's graphs; a measured depth captures more
    return fixed + depth_policy_bytes(spec, batch, t), page_bytes


def depth_policy_bytes(spec, batch, target) -> int:
    """The verify graphs a measured depth (SHARAR_SPEC_POLICY=budget, engine/depth.RowBudget) keeps beyond one depth's:
    the logits and final-norm rows of every other captured depth (the graphs share their pool, workspace and page index).
    Packed graphs (the 27B's DFlash2) share one set of outputs per batch bucket: nothing beyond one depth's. Flash-Next,
    throughput, depths 1-4: 1.06 GiB of logits counted, 1.11 GiB measured."""
    from sharar.engine.config import current
    from sharar.engine.depth import BUDGET_DEPTHS
    from sharar.engine.graphs import capture_sizes
    if current().spec_policy != 'budget' or (spec.name == 'dflash2' and target.family != 'flashnext'):
        return 0
    rows = sum(capture_sizes(batch)) * sum(d + 1 for d in BUDGET_DEPTHS if d < spec.depth)
    return rows * (target.vocab_size + 2 * target.hidden_size) * 2
