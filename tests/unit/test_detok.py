"""Incremental detokenizer: bounded decode windows, UTF-8 hold-back and stop strings."""

from hibana.api.detok import Detokenizer


class ByteTokenizer:
    """Token ids are bytes; decoding joins them as UTF-8 with replacement, like a
    byte-level BPE whose multi-byte characters straddle token boundaries."""

    def __init__(self):
        self.calls: list[int] = []

    def decode(self, ids, skip_special_tokens=False):
        self.calls.append(len(ids))
        return bytes(ids).decode("utf-8", errors="replace")


def feed(detok, ids):
    out, seen = [], []
    for i in range(1, len(ids) + 1):
        out.append(detok.update(ids[:i]))
        seen.append(detok.text)
    return out, seen


def test_ascii_streams_one_character_per_token():
    tok = ByteTokenizer()
    d = Detokenizer(tok)
    deltas, _ = feed(d, list(b"hello"))
    assert deltas == ["h", "e", "l", "l", "o"]
    assert d.text == "hello"


def test_incomplete_utf8_is_held_back_then_emitted_whole():
    d = Detokenizer(ByteTokenizer())
    deltas, _ = feed(d, list("aé b".encode()))
    assert deltas == ["a", "", "é", " ", "b"]
    assert d.text == "aé b"


def test_repeated_update_without_new_tokens_emits_nothing():
    d = Detokenizer(ByteTokenizer())
    assert d.update(list(b"ab")) == "ab"
    assert d.update(list(b"ab")) == ""
    assert d.text == "ab"


def test_stop_string_cuts_text_even_across_a_token_boundary():
    d = Detokenizer(ByteTokenizer(), stop=["END"])
    deltas, _ = feed(d, list(b"xyEN"))
    assert d.text == "xyEN" and not d.stopped
    assert d.update(list(b"xyENDzzz")) == ""
    assert d.stopped and d.text == "xy"
    assert d.update(list(b"xyENDzzzq")) == ""


def test_stop_string_inside_one_chunk():
    d = Detokenizer(ByteTokenizer(), stop=["</s>"])
    assert d.update(list(b"answer</s>trailing")) == "answer"
    assert d.stopped


def test_decode_window_is_bounded_by_the_unstable_tail():
    tok = ByteTokenizer()
    d = Detokenizer(tok)
    ids = list(b"a" * 4000)
    feed(d, ids)
    assert d.text == "a" * 4000
    # Two decodes per update (emitted part and window), each two tokens at most
    # once the stream is steady: never the whole output.
    assert max(tok.calls) <= 2


def test_whole_output_matches_a_single_decode():
    tok = ByteTokenizer()
    text = "naïve café — 日本語 test\n" * 20
    ids = list(text.encode())
    d = Detokenizer(tok)
    deltas, _ = feed(d, ids)
    assert "".join(deltas) == text == d.text
