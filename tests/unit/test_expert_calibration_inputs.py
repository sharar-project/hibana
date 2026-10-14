"""Flash-Next expert calibration: resuming a capture, ShareGPT prompts for reasoning traces."""
import json
from types import SimpleNamespace

import pytest

from sharar.weights import flashnext_expert_inputs as inputs
from sharar.weights.flashnext_w3_experts import resume_point


def test_resume_point_counts_whole_samples():
    assert resume_point([3, 4, 5], 0) == 0
    assert resume_point([3, 4, 5], 7) == 2
    assert resume_point([3, 4, 5], 12) == 3
    with pytest.raises(SystemExit):
        resume_point([3, 4, 5], 8)  # the capture ended inside the third sample


def test_sharegpt_prompts_split_english_and_other(tmp_path):
    rows = [{"id": f"en{i}", "conversations": [{"from": "human", "value": f"Explain topic number {i} in detail please."}]} for i in range(5)]
    rows += [{"id": f"x{i}", "conversations": [{"from": "human", "value": f"Объясни, пожалуйста, тему номер {i} подробно."}]} for i in range(3)]
    rows += [{"id": "short", "conversations": [{"from": "human", "value": "hi"}]},
             {"id": "gpt-first", "conversations": [{"from": "gpt", "value": "Hello there, how can I help you today?"}]}]
    src = tmp_path / "sharegpt.json"
    src.write_text(json.dumps(rows))
    out = tmp_path / "prompts.json"
    inputs.prompts_sharegpt(SimpleNamespace(sharegpt=src, english=2, other=10, seed=0, output=out))
    items = json.loads(out.read_text())["items"]
    assert sorted(it["source"] for it in items) == ["sharegpt/en"] * 2 + ["sharegpt/other"] * 3
    assert all(it["id"] not in ("short", "gpt-first") for it in items)
