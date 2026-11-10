"""Calibration samples (`weights/calibration_inputs.py`): eval filtering, rendering and packing, the captures' loader;
resuming a Flash-Next expert capture."""
import json
from types import SimpleNamespace

import pytest

from sharar.weights import calibration_inputs as inputs
from sharar.weights.flashnext_w3_experts import resume_point

GSM8K_ITEM = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four."


def test_resume_point_counts_whole_samples():
    assert resume_point([3, 4, 5], 0) == 0
    assert resume_point([3, 4, 5], 7) == 2
    assert resume_point([3, 4, 5], 12) == 3
    with pytest.raises(SystemExit):
        resume_point([3, 4, 5], 8)  # the capture ended inside the third sample


def _evals(root):
    for name, fields in inputs.EVALS.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        row = {field: "" for field in fields}
        if name == "accuracy/gsm8k_test.jsonl":
            row["question"] = GSM8K_ITEM
        if name == "hard/mmlupro.jsonl":
            row["question"] = "Which organ produces insulin?"
        (root / name).write_text(json.dumps(row) + "\n")
    (root / inputs.MGSM).mkdir(parents=True)
    (root / inputs.MGSM / "mgsm_de.tsv").write_text("Wie viele Eier legen die Enten von Janet jeden Tag, wenn sie drei davon isst?\t5\n")
    return inputs.EvalFilter(root)


def test_eval_filter_drops_shared_spans_and_short_questions(tmp_path):
    evals = _evals(tmp_path)
    assert f"Please solve: {GSM8K_ITEM.upper()} Show your work." in evals  # a 13-word span, case and punctuation aside
    assert "which organ produces INSULIN?" in evals  # a short eval question, whole
    assert "Which organ produces insulin in fish, and why?" not in evals
    assert "Janet's ducks lay 16 eggs per day. How many in a week?" not in evals  # fewer than 13 shared words
    assert "Wie viele Eier legen die Enten von Janet jeden Tag, wenn sie drei davon isst? Erkläre." in evals
    (tmp_path / "hard/ifeval.jsonl").unlink()
    with pytest.raises(SystemExit, match="ifeval"):
        inputs.EvalFilter(tmp_path)


def test_first_turn_skips_a_system_turn():
    assert inputs.first_turn([{"from": "system", "value": "Be nice."}, {"from": "human", "value": " Hi there "}]) == "Hi there"
    assert inputs.first_turn([{"from": "gpt", "value": "Hello"}]) is None
    assert inputs.first_turn([]) is None


@pytest.fixture
def target(tmp_path, monkeypatch):
    from tokenizers import Tokenizer, models, pre_tokenizers
    snap = tmp_path / "snap"
    snap.mkdir()
    words = "<|im_start|> <|im_end|> <think> </think> user assistant q a r long word".split()
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(["[UNK]"] + words)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.save(str(snap / "tokenizer.json"))
    (snap / "chat_template.jinja").write_text(
        "{% for m in messages %}<|im_start|> {{ m.role }} {% if m.reasoning_content %}<think> {{ m.reasoning_content }} </think> "
        "{% endif %}{{ m.content }} <|im_end|> {% endfor %}")
    t = SimpleNamespace(name="toy", model_revision="rev", template_defaults=(), snapshot_dir=lambda: snap)
    from sharar import target as targets
    monkeypatch.setattr(targets, "get", lambda name: t)
    return t


def test_build_packs_short_turns_and_cuts_long_ones(tmp_path, target):
    items = [{"source": "perfectblend/x", "prompt": "q", "reasoning": "r", "answer": "a"},
             {"source": "mmlu-val/law", "prompt": "q", "reasoning": "", "answer": "a"},
             {"source": "perfectblend/y", "prompt": "q", "reasoning": " ".join(["long"] * 40), "answer": "a"},
             {"source": "perfectblend/z", "prompt": "q", "reasoning": "", "answer": ""}]  # empty: left out
    answers = tmp_path / "answers.json"
    answers.write_text(json.dumps({"items": items}))
    out = tmp_path / "samples.json"
    inputs.build(SimpleNamespace(target="toy", answers=answers, output=out, max_tokens=24))
    data = json.loads(out.read_text())
    assert [len(s["input_ids"]) for s in data["samples"]] == [11 + 8, 24]  # two short turns packed whole, then a cut one
    assert data["samples"][0]["source"] == "mmlu-val,perfectblend/x"
    assert inputs.load_samples(out, target) == [s["input_ids"] for s in data["samples"]]
    assert inputs.load_samples(out, target, limit=1) == [data["samples"][0]["input_ids"]]
    with pytest.raises(SystemExit, match="between 1 and 20"):
        inputs.load_samples(out, target, max_tokens=20)
    with pytest.raises(SystemExit, match="for other"):
        inputs.load_samples(out, SimpleNamespace(name="other", snapshot_dir=target.snapshot_dir))
    (target.snapshot_dir() / "tokenizer.json").write_text((target.snapshot_dir() / "tokenizer.json").read_text() + " ")
    with pytest.raises(SystemExit, match="another toy tokenizer"):
        inputs.load_samples(out, target)
