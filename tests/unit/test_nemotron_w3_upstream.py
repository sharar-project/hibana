"""The asymmetric (upstream) calibration's pairing and target: CPU only."""
from types import SimpleNamespace

import torch

from sharar.weights.nemotron_w3_calibrate import UpstreamCapture, asymmetric_errors


def test_upstream_capture_pairs_chunked_calls_with_the_whole_sample_rows():
    lin = SimpleNamespace()
    cap = UpstreamCapture([("p", lin)], {"p"}, probe_stride=4)
    scale = torch.tensor([0.5])
    x = torch.randn(200, 16)
    xt = x + 0.1 * torch.randn(200, 16)
    q, qt = x.to(torch.float8_e4m3fn), xt.to(torch.float8_e4m3fn)
    cap.start_sample()
    cap.add("p", q, scale)  # the whole-sample FP8 call
    cap.switch()
    for r in range(0, 200, 64):  # the 3-bit model's 64-row calls
        cap.add("p", qt[r:r + 64], scale)
    cap.finish_sample()
    xf, xs = q.float() * .5, qt.float() * .5
    assert torch.allclose(cap.h["p"], xs.T @ xs, rtol=1e-5, atol=1e-4)
    assert torch.allclose(cap.q["p"], xf.T @ xs, rtol=1e-5, atol=1e-4)
    assert cap.rows["p"] == 200
    cap.holdout = True
    cap.start_sample()
    cap.add("p", q, scale)
    cap.switch()
    for r in range(0, 200, 64):
        cap.add("p", qt[r:r + 64], scale)
    probes, probes_fp = torch.cat(cap.probes["p"]), torch.cat(cap.probes_fp["p"])
    assert probes.shape == (50, 16) and torch.equal(probes_fp.float(), xf[::4].bfloat16().float())


def test_asymmetric_error_is_zero_for_the_exact_model():
    w = torch.randn(8, 16)
    x = torch.randn(32, 16)
    assert asymmetric_errors(w, w, x.bfloat16(), x.bfloat16()) == 0
    assert asymmetric_errors(w, w, (x + 1).bfloat16(), x.bfloat16()) > 0
