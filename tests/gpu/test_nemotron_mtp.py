"""Nemotron's MTP draft (`sharar/model/nemotron/mtp.py`) and its draft heads over the checkpoint's FP8 lm_head.

- Draft heads over an `Fp8Linear` (Nemotron's head, a static input scale beside the weight scale): the 2/3-bit copy
  and the rescored shortlist are bitwise what they are over a `TrunkLinear` (Flash-Next's) holding the same FP8 matrix
  and scale, so Flash-Next's heads are unchanged by the generalization.
- On the pinned checkpoint (skipped without it; the MTP head, embedding and FP8 lm_head, about 3.5 GB): the FP8
  projections stay close to the checkpoint's BF16 ones, and a replayed draft graph is bitwise the eager step for the
  served settings (FP8 projections, 2-bit head).
"""

from types import SimpleNamespace

import pytest
import torch


def _heads(n: int, k: int, seed: int = 0):
    """The same FP8 matrix as a `TrunkLinear` and as an `Fp8Linear` (weight scale [] instead of [1, 1])."""
    from sharar.model.flashnext.linear import TrunkLinear
    from sharar.weights.modelopt import Fp8Linear
    torch.manual_seed(seed)
    w = torch.randn(n, k, device="cuda")
    w = (w / w.norm(dim=-1, keepdim=True) * 8).to(torch.bfloat16)
    trunk = TrunkLinear.from_bf16(w, "fp8")
    fp8 = Fp8Linear(weight=trunk.weight, weight_scale=trunk.scale.reshape(()), input_scale=torch.tensor(0.05, device="cuda"))
    return trunk, fp8, w


@pytest.mark.parametrize("fmt", ["w3", "w2"])
def test_low_bit_copy_of_an_fp8linear_is_the_trunklinears(fmt):
    from sharar.proposers.draft_head import pack_head
    trunk, fp8, _ = _heads(2048, 4096)
    a, b = pack_head(trunk, fmt, rows_per_chunk=512), pack_head(fp8, fmt, rows_per_chunk=512)
    assert torch.equal(a.planes, b.planes) and torch.equal(a.scales, b.scales) and torch.equal(a.row_scale, b.row_scale)


@pytest.mark.parametrize("fmt,rescore", [("w2", 16), ("w3", 8), ("w2", 0)])
def test_shortlist_over_an_fp8linear_picks_as_over_a_trunklinear(fmt, rescore):
    from sharar.proposers.draft_head import ShortlistHead
    trunk, fp8, _ = _heads(4096, 4096, seed=1)
    x = torch.randn(4, 4096, device="cuda").to(torch.bfloat16)
    over_trunk, over_fp8 = ShortlistHead(trunk, fmt=fmt, rescore=rescore), ShortlistHead(fp8, fmt=fmt, rescore=rescore)
    assert torch.equal(over_trunk.argmax(x), over_fp8.argmax(x))
    for (v0, i0), (v1, i1) in [(over_trunk.topk(x, 4), over_fp8.topk(x, 4))]:
        assert torch.equal(i0, i1) and torch.equal(v0, v1)  # topk values carry the weight scale either way


# ---- on the pinned checkpoint -------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def snapshot():
    from sharar.target import NEMOTRON
    try:
        return NEMOTRON.snapshot_dir()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def parts(snapshot):
    """The embedding and FP8 lm_head the draft shares with the target, without the trunk."""
    from sharar.model.nemotron.config import PREFIX
    from sharar.target import NEMOTRON
    from sharar.weights.modelopt import load_fp8_linear
    from sharar.weights.safetensors import SnapshotWeights
    w = SnapshotWeights(snapshot)
    model = SimpleNamespace(target=NEMOTRON, embed=w.get(f"{PREFIX}.backbone.embeddings.weight").cuda(),
                            lm_head=load_fp8_linear(w, [f"{PREFIX}.lm_head"], torch.device("cuda")))
    w.close()
    return model


def _head(snapshot, model, bf16: bool = False):
    """The served head, or (`bf16`) the same head with the checkpoint's BF16 eh_proj and attention q|k|v and o as
    plain BF16 matmuls in place of its FP8 ones."""
    from sharar.model.flashnext.linear import TrunkLinear
    from sharar.model.nemotron.mtp import PREFIX, NemotronMtpHead
    from sharar.weights.safetensors import SnapshotWeights
    w = SnapshotWeights(snapshot)
    dev = torch.device("cuda")
    try:
        head = NemotronMtpHead(w, dev, model, 4 * 700 + 64, padding_rows=4)
        if bf16:
            p = f"{PREFIX}.0.mixer"
            head.eh = TrunkLinear.from_bf16(w.get(f"{PREFIX}.0.eh_proj.weight").to(dev), "bf16")
            head.qkv = TrunkLinear.load(w, [f"{p}.q_proj.weight", f"{p}.k_proj.weight", f"{p}.v_proj.weight"], dev, "bf16")
            head.o = TrunkLinear.load(w, [f"{p}.o_proj.weight"], dev, "bf16")
        return head
    finally:
        w.close()


def test_fp8_projections_stay_close_to_the_checkpoints(snapshot, parts):
    """Draft-only: the d rows of one step through FP8 eh_proj and attention against the checkpoint's BF16, over the
    same draft KV. Random tokens and rows route and attend arbitrarily, which amplifies the rounding (u keeps cosine
    0.9993, d 0.985 measured); on recorded target rows the FP8 head's tokens per round matched the BF16 head's."""
    torch.manual_seed(4)
    rows = {}
    for bf16 in (True, False):
        head = _head(snapshot, parts, bf16)
        gen = torch.Generator().manual_seed(5)
        n = 300
        toks = torch.randint(0, 131072, (n,), generator=gen).cuda()
        hid = torch.randn(n, 4096, generator=gen).to(torch.bfloat16).cuda()
        pages = torch.arange(1, n + 1, dtype=torch.int32, device="cuda")
        head.write_kv(toks[:-1], hid[:-1], torch.arange(1, n, device="cuda"), pages[:-1])
        rows[bf16] = head.forward(toks[-1:], hid[-1:], torch.tensor([n], device="cuda"), pages[-1:], [1], [pages])
        del head
        torch.cuda.empty_cache()
    base, *others = rows.values()
    for d in others:
        assert torch.nn.functional.cosine_similarity(d.float(), base.float(), dim=-1).min() > 0.97


def fill(head, b: int, context: int, gen: torch.Generator, scale: float):
    """Draft KV for `b` sequences of `context` tokens on disjoint pages; returns each sequence's pages."""
    device = head.device
    pages = []
    for i in range(b):
        p = torch.arange(1 + i * (context + 16), 1 + i * (context + 16) + context + 16, dtype=torch.int32, device=device)
        toks = torch.randint(0, head.lm_head.out_features, (context - 1,), generator=gen).to(device)
        hid = (torch.randn(context - 1, head.hidden_width, generator=gen) * scale).to(torch.bfloat16).to(device)
        head.write_kv(toks, hid, torch.arange(1, context, dtype=torch.int64, device=device), p[1:context])
        pages.append(p)
    return pages


def chain(proposer, b: int, context: int, pages, tokens, hidden, steps: int, graphs: bool):
    """`steps` chained draft steps for b sequences at position `context`; returns [(hidden, token)] per step."""
    out = []
    positions = torch.full((b,), context, dtype=torch.int64, device=tokens.device)
    for j in range(steps):
        kv = [torch.cat([p[1:context], p[context:context + j + 1]]) for p in pages]
        write = torch.stack([p[context + j] for p in pages]).contiguous()
        hidden, tokens, _, _ = proposer._step(tokens, hidden, positions + j, write, kv, graphs)
        out.append((hidden.clone(), tokens.clone()))
    return out


@pytest.mark.parametrize("fmt", ["w2"])
def test_draft_graph_replays_the_eager_step(snapshot, parts, fmt):
    """Three chained steps at batch 1 and 4, served settings: bitwise."""
    from sharar.proposers.mtp import MtpProposer
    from sharar.proposers.draft_head import RESCORE
    head = _head(snapshot, parts)
    head.set_draft_head(None, fmt, RESCORE[fmt])
    proposer = MtpProposer(head, SimpleNamespace(device=torch.device("cuda"), max_batch=4, max_context=664), 3)
    proposer.capture_graphs()
    for b in (1, 4):  # whole buckets: a padded row moves the per-call FP8 activation scale
        gen = torch.Generator().manual_seed(10 + b)
        pages = fill(head, b, 600, gen, 1.0)
        tokens = torch.randint(0, 131072, (b,), generator=gen).cuda()
        hidden = torch.randn(b, 4096, generator=gen).to(torch.bfloat16).cuda()
        eager = chain(proposer, b, 600, pages, tokens, hidden, 3, graphs=False)
        graphed = chain(proposer, b, 600, pages, tokens, hidden, 3, graphs=True)
        for (h0, t0), (h1, t1) in zip(eager, graphed):
            assert torch.equal(t0, t1) and torch.equal(h0, h1)
