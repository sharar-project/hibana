"""The MTP draft's confidence per drafted position (#32): the draft head's top-1 probability and margin,
and the trace round records that carry it (SHARAR_TRACE_CONFIDENCE, the input for adaptive depth #43)."""
import torch

from sharar.engine.config import EngineConfig
from sharar.engine.request import Request
from sharar.engine.sampler import SamplingParams
from sharar.engine.spec import SpecDecoder
from sharar.proposers.draft_head import TargetHead, confidence


def test_draft_confidence_is_the_heads_softmax_top1_and_margin():
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0], [0.0, 0.0, 5.0, 0.0]])
    head = TargetHead(lambda x: x)
    token, conf = head.argmax_confidence(logits)
    p = torch.softmax(logits, -1)
    assert token.tolist() == head.argmax(logits).tolist() == [0, 2]
    assert torch.allclose(conf[:, 0], p.max(-1).values)
    assert torch.allclose(conf[0, 1], p[0, 0] - p[0, 1])
    assert torch.allclose(confidence(torch.zeros(3, 5)), torch.tensor([[0.2, 0.0]] * 3))


def test_round_records_get_confidence_only_when_a_request_records_rounds():
    reqs = [Request(prompt_ids=[1], params=SamplingParams(), request_id=i) for i in range(2)]
    conf = torch.tensor([[[0.9, 0.5], [0.4, 0.1]], [[1.0, 1.0], [0.5, 0.0]]])
    assert SpecDecoder._confidence_rows(reqs, conf) is None  # nobody records traces
    assert SpecDecoder._confidence_rows(reqs, None) is None  # the proposer did not report it
    reqs[1].trace_rounds = []
    rows = SpecDecoder._confidence_rows(reqs, conf)
    assert torch.allclose(torch.tensor(rows), conf)


def test_setting_is_off_by_default():
    assert EngineConfig.from_env({}).trace_confidence is False
    assert EngineConfig.from_env({"SHARAR_TRACE_CONFIDENCE": "1"}).trace_confidence is True
