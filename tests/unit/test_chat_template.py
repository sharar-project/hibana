from sharar.api.chat_template import ChatTemplate
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
