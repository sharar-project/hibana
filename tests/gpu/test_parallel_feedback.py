"""Exercise graph reuse, changing accepted lengths, private slots and ring wrap."""
from types import SimpleNamespace
import torch

from sharar.proposers.parallel_feedback import FeedbackGraphs


@torch.inference_mode()
def test_feedback_graphs_match_eager_and_keep_inputs_alive():
    from sharar.model.parallel_draft import rotate_and_cache
    torch.manual_seed(714)
    layers, window, pitch = (1, 3), 32, 48
    angles = torch.randn(12000, 64, device='cuda')
    cos, sin = angles.cos(), angles.sin()
    def context(hidden, positions, caches, pages):
        k = hidden[:, :1024].reshape(-1, 8, 128)
        v = hidden[:, 1024:2048].reshape(-1, 8, 128)
        for cache in caches:
            rotate_and_cache(k, v, positions, cos, sin, cache, pages)
    def model():
        return SimpleNamespace(features=layers, project_features=lambda f: f[1] + f[3], context_kv=context, hidden=5120, feature_width=5120)
    def proposer():
        return SimpleNamespace(k=15, window=window, pitch=pitch, context=SimpleNamespace(device=torch.device('cuda')),
            model=model(), features=torch.zeros(5, window, 5120, dtype=torch.bfloat16, device='cuda'),
            caches=[tuple(torch.zeros(5*pitch, 1, 8, 128, dtype=torch.bfloat16, device='cuda') for _ in range(2)) for _ in range(2)],
            _save=lambda req, end: None)
    p, ref = proposer(), proposer()
    graphs = FeedbackGraphs(p)
    graphs.capture()
    # Captures use only the reserved dummy slot; reset it for whole-arena comparisons.
    p.features.zero_()
    for pair in p.caches:
        for cache in pair: cache.zero_()
    for index, count in enumerate((16, 1, 3, 8, 2, 15, 4, 16, 1)):
        slot, start = index % 4 + 1, 10015 + index * 27
        state = SimpleNamespace(slot=slot, end=start, pending=-1)
        req = SimpleNamespace(committed_len=start+count, proposer_state=state, private={}, finished=index==8)
        values = {lid:torch.randn(1, 16, 5120, dtype=torch.bfloat16, device='cuda') for lid in layers}
        emitted = list(range(200, 200+count))
        graphs.run(req, emitted, SimpleNamespace(layers=values), 0)
        positions = torch.arange(start, start+count, device='cuda')
        hidden = ref.model.project_features({lid:x[0,:count] for lid,x in values.items()})
        ref.features[slot, positions % window] = hidden
        ref.model.context_kv(hidden, positions, ref.caches, slot*pitch + positions % window)
        assert torch.equal(p.features, ref.features)
        assert all(torch.equal(a,b) for ac,bc in zip(p.caches,ref.caches) for a,b in zip(ac,bc))
        assert state.end == start+count and state.pending == (None if req.finished else emitted[-1])
        # Churn allocations after each capture/replay to expose dangling input storage.
        scratch = [torch.ones(2048, device='cuda') * i for i in range(40)]
        del scratch
