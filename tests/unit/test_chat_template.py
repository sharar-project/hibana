import pytest

from sharar.api.chat_template import ChatTemplate
from sharar.target import NEMOTRON, NEMOTRON_BOLT
from sharar.tokenizer import SnapshotTokenizer


def test_render_no_thinking_ends_with_generation_prompt(snapshot):
    tmpl = ChatTemplate(snapshot)
    text = tmpl.render([{"role": "user", "content": "hi"}], enable_thinking=False)
    assert text.startswith("<|im_start|>user")
    # With thinking disabled the template closes an empty think block itself.
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    tok = SnapshotTokenizer(snapshot)
    ids = tok.encode(text)
    assert tok.decode(ids) == text


@pytest.mark.parametrize("target", [NEMOTRON, NEMOTRON_BOLT], ids=lambda t: t.name)
def test_nemotron_keeps_earlier_reasoning_so_a_new_user_turn_appends(target):
    try:
        snap = target.snapshot_dir()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    tmpl = ChatTemplate(snap, target.template_defaults)
    first = [{"role": "user", "content": "read the file"}]
    calls = [{"id": "c1", "type": "function", "function": {"name": "Read", "arguments": {"path": "a.py"}}}]
    turn = [{"role": "assistant", "content": "", "reasoning_content": "\nI should read it.", "tool_calls": calls},
            {"role": "tool", "tool_call_id": "c1", "content": "x = 1"},
            {"role": "assistant", "content": "It sets x.", "reasoning_content": "\nThe file sets x."}]
    second = [{"role": "user", "content": "now explain y"}]
    # what the prefix cache holds after the first task: its prompts and the model's own outputs
    held = tmpl.render(first + turn, add_generation_prompt=False, enable_thinking=True)
    assert tmpl.render(first + turn + second, enable_thinking=True).startswith(held)
    assert "I should read it." in held and "The file sets x." in held
    # a request can still ask for the template's own default
    dropped = tmpl.render(first + turn + second, enable_thinking=True, truncate_history_thinking=True)
    assert "The file sets x." not in dropped and not dropped.startswith(held)
    # without reasoning in the history (most OpenAI clients drop it) the prompt is the template default's
    plain = [{k: v for k, v in m.items() if k != "reasoning_content"} for m in first + turn + second]
    assert tmpl.render(plain, enable_thinking=True) == tmpl.render(plain, enable_thinking=True, truncate_history_thinking=True)
