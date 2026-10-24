"""The Nemotron-3.5-Super plain-torch reference (`sharar/model/nemotron/reference.py`) against Transformers' own
`nemotron_h` modules carrying the same dequantised weights: one block of each kind (Mamba-2, attention, latent MoE)
and the MTP head (which Transformers does not build; it is assembled here from its blocks as vLLM composes it), on
random and real inputs, in FP32 on both sides. Also the SSD scan forms against Transformers' torch scan.

Needs the pinned snapshot in the model cache (skipped otherwise); reads blocks 0-7, the head and MTP, about 4 GiB, and
briefly an FP32 copy of one layer's 512 experts (11 GiB) for the Transformers MoE block."""

import gc
import math

import pytest
import torch

from sharar.model.nemotron import reference as R

MAMBA, MOE, ATTENTION = 0, 1, 7


@pytest.fixture(scope="module")
def ref():
    try:
        snap = R.find_snapshot()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    model = R.NemotronReference(snap, dtype=torch.float32, blocks=range(ATTENTION + 1), expert_cache_gib=4.0)
    yield model
    del model
    gc.collect()
    torch.cuda.empty_cache()


def hf_config(snap):
    from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
    cfg = NemotronHConfig(**R.read_json(snap / "config.json")["llm_config"])
    cfg._attn_implementation = "eager"
    return cfg


def hf_block(ref, layer: int, params: R.Block, as_layer: int | None = None):
    """Transformers' NemotronHBlock for trunk index `as_layer` (default `layer`) holding `params`, FP32, on the GPU."""
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHBlock
    cfg = hf_config(ref.snap)
    with torch.device("meta"):
        block = NemotronHBlock(cfg, as_layer if as_layer is not None else layer)
    block = block.to_empty(device="cuda").float().eval()
    p, sd = params.mixer, {"norm.weight": params.norm}
    if params.kind == "mamba":
        sd |= {"mixer.in_proj.weight": p.in_proj.weight(), "mixer.conv1d.weight": p.conv_w[:, None, :],
               "mixer.conv1d.bias": p.conv_b, "mixer.A_log": p.A_log, "mixer.D": p.D, "mixer.dt_bias": p.dt_bias,
               "mixer.norm.weight": p.norm_w, "mixer.out_proj.weight": p.out_proj.weight()}
        block.mixer.time_step_limit = ref.cfg.time_step_limit  # config.json's (0, inf), not Transformers' (1e-3, inf)
    elif params.kind == "attention":
        sd |= {f"mixer.{n}_proj.weight": getattr(p, n).weight() for n in "qkvo"}
    else:
        ups, downs = [], []
        for e in range(ref.cfg.experts):
            up, down = R.load_linear(ref.w, f"{p.prefix}.experts.{e}.up_proj", "cuda"), \
                R.load_linear(ref.w, f"{p.prefix}.experts.{e}.down_proj", "cuda")
            ups.append(up.weight())
            downs.append(down.weight())
        sd |= {"mixer.experts.up_proj": torch.stack(ups), "mixer.experts.down_proj": torch.stack(downs),
               "mixer.gate.weight": p.gate_w, "mixer.gate.e_score_correction_bias": p.bias,
               "mixer.fc1_latent_proj.weight": p.fc1.weight(), "mixer.fc2_latent_proj.weight": p.fc2.weight(),
               "mixer.shared_experts.up_proj.weight": p.shared_up.weight(),
               "mixer.shared_experts.down_proj.weight": p.shared_down.weight()}
        del ups, downs
    missing = set(block.state_dict()) - set(sd)
    assert not missing, missing
    block.load_state_dict({k: v.float() for k, v in sd.items()})
    return block


def hf_run(block, x):
    t = x.shape[0]
    if block.block_type == "full_attention":
        mask = torch.full((t, t), -math.inf, device=x.device).triu(1)[None, None]
        return block(x[None], attention_mask=mask)[0]
    return block(x[None])[0]


def compare(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, 0).item(), (a - b).abs().max().item() / b.abs().max().item()


def real_inputs(ref):
    """The residual stream entering blocks 0..7 for a short real text."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(ref.snap / "tokenizer.json"))
    ids = torch.tensor(tok.encode("The quick brown fox jumps over the lazy dog. In Python, a list comprehension "
                                  "such as [x * x for x in range(10)] builds the squares of 0 through 9.").ids)
    x = ref.embed_ids(ids)
    inputs = []
    with torch.inference_mode():
        for i in range(ATTENTION + 1):
            inputs.append(x)
            x = ref.run_block(i, x)
    return inputs


@pytest.mark.parametrize("layer", [MAMBA, ATTENTION, MOE])
def test_block_matches_transformers(ref, layer):
    torch.manual_seed(layer)
    cases = {"random": torch.randn(77, ref.cfg.hidden, device="cuda"), "real": real_inputs(ref)[layer]}
    block = hf_block(ref, layer, ref.blocks[layer])
    with torch.inference_mode():
        for name, x in cases.items():
            cos, rel = compare(ref.run_block(layer, x) - x, hf_run(block, x) - x)
            print(f"block {layer} {ref.cfg.block_types[layer]} {name}: cos {cos:.8f} max|d|/max {rel:.2e}")
            assert cos > 0.99999 and rel < 1e-3, (name, cos, rel)
    del block
    gc.collect()
    torch.cuda.empty_cache()


def test_mtp_matches_transformers_blocks(ref):
    """MTP = eh_proj([enorm(e), hnorm(h)]) -> attention block -> MoE block -> final_layernorm, from HF modules."""
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHRMSNorm
    m, eps = ref.mtp, ref.cfg.eps
    attn = hf_block(ref, -1, m.attn, as_layer=ATTENTION)
    moe = hf_block(ref, -1, m.moe, as_layer=MOE)
    norms = []
    for w in (m.enorm, m.hnorm, m.final_norm):
        n = NemotronHRMSNorm(ref.cfg.hidden, eps).cuda()
        n.weight.data.copy_(w.float())
        norms.append(n)
    eh = torch.nn.Linear(2 * ref.cfg.hidden, ref.cfg.hidden, bias=False).cuda()
    eh.weight.data.copy_(m.eh_proj.weight())
    torch.manual_seed(0)
    h = torch.randn(40, ref.cfg.hidden, device="cuda")
    ids = torch.randint(0, ref.cfg.vocab, (40,), device="cuda")
    with torch.inference_mode():
        got = ref.mtp_forward(h, ids)
        x = eh(torch.cat([norms[0](ref.embed[ids].float()), norms[1](h)], -1))
        want = norms[2](hf_run(moe, hf_run(attn, x)))
        cos, rel = compare(got["final"], want)
        print(f"mtp: cos {cos:.8f} max|d|/max {rel:.2e}")
        assert cos > 0.99999 and rel < 1e-3
        # stepwise through the MTP KV cache equals the whole-sequence pass
        state = R.State()
        steps = torch.cat([ref.mtp_forward(h[i:i + 1], ids[i:i + 1], state)["final"] for i in range(8)])
        assert compare(steps, got["final"][:8])[0] > 0.99999
    del attn, moe
    gc.collect()
    torch.cuda.empty_cache()


def test_scan_forms_match_transformers():
    from transformers.models.nemotron_h.modeling_nemotron_h import mamba2_chunk_scan
    torch.manual_seed(0)
    t, h, p, g, n = 300, 8, 16, 2, 32
    x = torch.randn(t, h, p, device="cuda")
    B, C = torch.randn(t, g, n, device="cuda"), torch.randn(t, g, n, device="cuda")
    dt, A, D = torch.rand(t, h, device="cuda") * 0.5, -torch.rand(h, device="cuda") * 3, torch.randn(h, device="cuda")
    s0 = torch.randn(h, p, n, device="cuda")
    y_seq, f_seq = R.ssd_sequential(x, dt, A, B, C, D, s0)
    y_chk, f_chk = R.ssd_chunked(x, dt, A, B, C, D, 64, s0)
    y_hf, f_hf = mamba2_chunk_scan(x[None], dt[None], A, B[None], C[None], 64, D=D, initial_states=s0[None],
                                   return_final_states=True)
    ya, fa = R.ssd_chunked(x[:130], dt[:130], A, B[:130], C[:130], D, 64, s0)
    yb, fb = R.ssd_chunked(x[130:], dt[130:], A, B[130:], C[130:], D, 64, fa)
    scale = y_seq.abs().max()
    for y, f in ((y_chk, f_chk), (y_hf[0], f_hf[0]), (torch.cat([ya, yb]), fb)):
        assert (y - y_seq).abs().max() < 1e-5 * scale and (f - f_seq).abs().max() < 1e-4
