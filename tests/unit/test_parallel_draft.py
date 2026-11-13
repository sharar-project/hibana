import json
from types import SimpleNamespace

import pytest
import torch

from hibana.model.parallel_draft import convolve, ParallelDraftModel
from hibana.proposers.parallel import FeatureCache
from hibana.cache.spec_budget import verify_bytes


def test_convolution_resets_at_each_block_and_uses_both_dynamic_taps():
    torch.manual_seed(12)
    x = torch.randn(2, 4, 32)
    delta = torch.randn(2, 4, 2, 2)
    base = torch.randn(2, 32)
    expected = torch.zeros_like(x)
    for b in range(2):
        for t in range(4):
            for c in range(32):
                for tap in range(2):
                    if t >= tap:
                        expected[b, t, c] += (base[tap, c] + delta[b, t, tap, c // 16]) * x[b, t - tap, c]
    torch.testing.assert_close(convolve(x, delta, base), expected, rtol=1e-6, atol=1e-6)


def test_feature_cache_bounds_memory_and_invalidates_evicted_handles():
    cache = FeatureCache(64)
    a = cache.put(0, 4, torch.ones(4, 2))
    b = cache.put(4, 8, torch.ones(4, 2) * 2)
    assert cache.bytes == 64
    assert cache.get(a) is not None  # a becomes most recent
    c = cache.put(8, 12, torch.ones(4, 2) * 3)
    assert cache.bytes == 64 and cache.get(b) is None
    assert cache.get(a)[1] == 4 and cache.get(c)[1] == 12
    assert cache.put(0, 100, torch.ones(100)) is None


def test_reconstruction_budget_includes_all_retained_inputs_and_conv():
    full = verify_bytes(4, 4, 'snapshots')
    compact = verify_bytes(4, 4, 'reconstruct')
    assert full == 2.25 * 2**30 + 45 * 2**20
    assert compact == 45 * 2**20 + 48 * 4 * 4 * ((16 * 128 + 48 * 128) * 2 + 48 * 2 * 4)
    assert compact < .03 * full


def test_selector_lattice_conditions_on_anchor_then_previous_candidate():
    torch.manual_seed(71)
    hidden = torch.randn(2, 3, 5120)
    logits = torch.randn(6, 32)
    weights = {'candidate_selector.hidden_projection.weight': torch.randn(4, 5120),
               'candidate_selector.predecessor_codebook': torch.randn(32, 4),
               'candidate_selector.successor_codebook': torch.randn(32, 4)}
    from hibana.proposers.draft_head import TargetHead
    model = SimpleNamespace(target=SimpleNamespace(logits=lambda x: logits), draft_head=TargetHead(lambda x: logits), weights=weights,
                            cfg={'dflash_config': {'selector_top_k': 4}}, hidden=hidden.shape[-1])
    anchor = torch.tensor([2, 9])
    candidates, scores = ParallelDraftModel.selector(model, hidden, anchor)
    gate = hidden @ weights['candidate_selector.hidden_projection.weight'].T
    for b in range(2):
        for t in range(3):
            for i in range(4):
                previous = anchor[b] if t == 0 else candidates[b, t - 1, i]
                for j in range(4):
                    token = candidates[b, t, j]
                    pair = (weights['candidate_selector.predecessor_codebook'][previous]
                            * gate[b, t] * weights['candidate_selector.successor_codebook'][token]).sum()
                    torch.testing.assert_close(scores[b, t, i, j], logits[b * 3 + t, token] + pair, rtol=1e-5, atol=1e-4)


def test_prospective_draft_budget_reduces_optional_cache_before_allocation(monkeypatch):
    from hibana.cache import ledger
    monkeypatch.setattr(ledger, 'mem_available_bytes', lambda: 64 * 2**30)
    book = ledger.Ledger()
    control = book.plan(state_slot_target=1000)
    draft = book.plan(future_bytes=10 * 2**30, extra_page_bytes=4096, state_slot_target=1000)
    assert draft.kv_tokens < control.kv_tokens and draft.state_slots < control.state_slots
    assert draft.after_pools_bytes == control.after_pools_bytes + 10 * 2**30


def test_a_draft_takes_its_mask_token_and_taps_from_its_checkpoint(tmp_path):
    # Flash-Next's DFlash2 masks with 248077 (the 27B's with 248070); the selector projects the draft's rows with the
    # target's lm_head alone (Flash-Next's mixer reads hyper streams)
    from safetensors.torch import save_file
    hidden, heads, kv, head_dim, inter, vocab = 32, 2, 1, 16, 64, 50
    shapes = {'fc.weight': (hidden, 5 * hidden), 'hidden_norm.weight': (hidden,), 'norm.weight': (hidden,),
              'layers.0.self_attn.q_proj.weight': (heads * head_dim, hidden), 'layers.0.self_attn.k_proj.weight': (kv * head_dim, hidden),
              'layers.0.self_attn.v_proj.weight': (kv * head_dim, hidden), 'layers.0.self_attn.o_proj.weight': (hidden, heads * head_dim),
              'layers.0.mlp.gate_proj.weight': (inter, hidden), 'layers.0.mlp.up_proj.weight': (inter, hidden),
              'layers.0.mlp.down_proj.weight': (hidden, inter)}
    save_file({name: torch.zeros(shape, dtype=torch.bfloat16) for name, shape in shapes.items()}, str(tmp_path / 'model.safetensors'))
    (tmp_path / 'config.json').write_text(json.dumps({
        'architectures': ['DFlash2DraftModel'], 'hidden_size': hidden, 'num_attention_heads': heads, 'num_key_value_heads': kv,
        'head_dim': head_dim, 'num_hidden_layers': 1, 'sliding_window': 2048, 'rope_parameters': {'rope_theta': 10000, 'rope_type': 'default'},
        'dflash_config': {'block_size': 8, 'mask_token_id': 248077, 'target_layer_ids': [5, 15, 25, 35, 45], 'selector_top_k': 4}}))
    rows = torch.randn(3, vocab)

    def mixer(x):
        raise AssertionError('the draft rows are not hyper streams')
    target = SimpleNamespace(embed=torch.zeros(vocab, hidden, dtype=torch.bfloat16), device=torch.device('cpu'),
                             logits=mixer, head_logits=lambda x: rows)
    model = ParallelDraftModel(tmp_path, target, 64)
    assert (model.mask_token, model.features) == (248077, (5, 15, 25, 35, 45))
    assert model.draft_head.logits(torch.zeros(3, hidden)) is rows
    target.embed = torch.zeros(vocab, 2 * hidden)
    with pytest.raises(ValueError, match='embedding width'):
        ParallelDraftModel(tmp_path, target, 64)


class _DraftModel:
    """A stand-in drafter: records its calls; hidden rows carry the token and position, the selector fixed candidates."""
    mask_token, window, width = 5, 8, 4

    def __init__(self, log):
        self.log = log

    def forward(self, tokens, positions, pages, caches, attention):
        self.log.append(('forward', tokens.clone(), positions.clone(), pages.clone()))
        b, s = tokens.shape
        return (tokens[..., None].float() + positions.view(b, s, 1).float() * 0.01).expand(b, s, 8).clone()

    def selector(self, hidden, anchor):
        b, steps = hidden.shape[:2]
        scores = torch.randn(b, steps, self.width, self.width, generator=torch.Generator().manual_seed(b * 100 + steps))
        return torch.arange(b * steps * self.width).view(b, steps, self.width) + 1000, scores


class _Planner:
    def __init__(self, log):
        self.log = log

    def paged_prefill(self, wrapper, qo, indptr, indices, last, causal, guard):
        self.log.append(('plan', id(wrapper), qo.clone(), indptr.clone(), indices.clone()))


def _proposer(k, max_batch=4):
    from hibana.engine.contracts import ProposerContext
    from hibana.proposers.parallel import ParallelProposer
    p, log = ParallelProposer.__new__(ParallelProposer), []
    p.model, p.k, p.name, p.window = _DraftModel(log), k, 'dflash2', _DraftModel.window
    p.pitch = p.window + k + 1
    p.context = ProposerContext(torch.device('cpu'), max_batch, 128)
    p.caches, p.attention, p.graphs, p.workspace, p.planner = [], object(), None, None, _Planner(log)
    return p, log


def test_a_round_drafts_the_anchor_and_its_depths_masked_rows():
    # short drafts: a round at depth d drafts one block of d + 1 rows into the first d + 1 noise slots of each request's
    # slot (never past its pitch); depth k is the whole block, as without a depth
    from hibana.engine.contracts import StepInputs
    from hibana.engine.sampler import SamplingParams
    from hibana.proposers.parallel import DraftState
    k, ends = 9, [3, 11, 20]

    def requests(n):
        return [SimpleNamespace(proposer_state=DraftState(i + 1, end=ends[i]), committed_len=ends[i],
                                params=SamplingParams(temperature=0.7, seed=i), generator=torch.Generator().manual_seed(i)) for i in range(n)]
    for n in (1, 3):
        inputs = StepInputs(torch.tensor([7] * n), torch.tensor(ends[:n]), torch.zeros(n, dtype=torch.int32))
        whole, whole_log = _proposer(k)
        at_k, at_k_log = _proposer(k)
        a = whole.propose(requests(n), None, inputs, graphs=False)
        b = at_k.propose(requests(n), None, inputs, graphs=False, depth=k)
        fields = lambda e: e[2:] if e[0] == 'plan' else e[1:]  # noqa: E731  (a plan's second field is its wrapper's id)
        assert [e[0] for e in whole_log] == [e[0] for e in at_k_log]
        assert all(torch.equal(x, y) for ea, eb in zip(whole_log, at_k_log) for x, y in zip(fields(ea), fields(eb)))  # drafted tokens too
        assert torch.equal(a.tokens, b.tokens) and torch.equal(a.probabilities, b.probabilities)
        for d in (1, 3, 4, 7):
            p, log = _proposer(k)
            out = p.propose(requests(n), None, inputs, graphs=False, depth=d)
            (_, _, qo, _, _), (_, tokens, positions, pages) = log
            assert out.tokens.shape == (n, d) and out.probabilities.shape == out.support.shape == (n, d, _DraftModel.width)
            assert tokens.shape == (n, d + 1) and torch.equal(tokens[:, 0], inputs.tokens) and (tokens[:, 1:] == _DraftModel.mask_token).all()
            assert torch.equal(qo, torch.arange(0, (n + 1) * (d + 1), d + 1, dtype=torch.int32))
            for i in range(n):
                base = (i + 1) * p.pitch
                assert torch.equal(pages[i * (d + 1):(i + 1) * (d + 1)], torch.arange(base + p.window, base + p.window + d + 1))
                assert torch.equal(positions[i * (d + 1):(i + 1) * (d + 1)], ends[i] + torch.arange(d + 1))


def test_draft_graphs_are_captured_per_bucket_and_depth(monkeypatch):
    # the verify graphs' (depth, buckets) pairs: each pair replays its own wrapper, planned and padded at its block
    import sys
    import types
    from hibana.engine import graph_set
    from hibana.kernels import draft_selection
    from hibana.proposers.parallel_graphs import ParallelGraphs

    class Wrapper:
        def __init__(self, workspace, layout, **buffers):
            self.buffers = buffers

    class Graph:
        def __init__(self, body):
            self.body = body

        def replay(self):
            self.body()

    def capture(pool, body, **kw):
        body()
        return Graph(body)
    monkeypatch.setitem(sys.modules, 'flashinfer', types.SimpleNamespace(BatchPrefillWithPagedKVCacheWrapper=Wrapper))
    monkeypatch.setattr(torch.cuda, 'graph_pool_handle', lambda: None)
    monkeypatch.setattr(graph_set, 'capture', capture)
    monkeypatch.setattr(draft_selection, 'greedy_lattice', lambda candidates, scores: None)
    p, log = _proposer(9, max_batch=8)
    table = {9: (1,), 7: (2, 4), 4: (8,)}
    graphs = ParallelGraphs(p, table)
    graphs.capture()
    assert set(graphs.static) == {(1, 9), (2, 7), (4, 7), (8, 4)}
    for batch, d in ((3, 7), (5, 4), (1, 9), (2, 7)):
        s, size = d + 1, graph_set.pick(table[d], batch)
        log.clear()
        hidden, candidates, _ = graphs.run(torch.full((batch, s), 5), torch.arange(batch * s), torch.arange(batch * s) + 100,
                                           [torch.arange(3) for _ in range(batch)])
        (_, wrapper, qo, _, _), (_, tokens, _, pages) = log
        assert wrapper == id(graphs.static[size, d]['wrapper'])
        assert torch.equal(qo, torch.arange(0, (size + 1) * s, s, dtype=torch.int32)) and tokens.shape == (size, s)
        assert hidden.shape == (batch, d, 8) and candidates.shape == (batch, d, _DraftModel.width)
        assert size == batch or int(pages[batch * s:].max()) < (size - batch) * s  # padded rows: the dummy slot's pages
    with pytest.raises(ValueError, match='uncaptured draft depth 3'):
        graphs.run(torch.full((1, 4), 5), torch.arange(4), torch.arange(4), [torch.arange(3)])
