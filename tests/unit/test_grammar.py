"""Constrained output (#16): grammar masks for sampling and for a verify block of drafted tokens, and acceptance on
masked logits. A small plain-text vocabulary stands in for Qwen's."""
from types import SimpleNamespace

import pytest
import torch

from sharar.engine.grammar import GrammarState, Grammars, apply_bitmask, host_bitmask, sample_masks
from sharar.engine.request import Request
from sharar.engine.sampler import Sampler, SamplingParams
from sharar.engine.spec import SpecDecoder

PIECES = ['{"', '"', '":', ':', ',', '}', ' ', '\n', 'city', 'days', 'Paris', 'a', 'b', '1', '2', '3',
          '<tool_call>', '</tool_call>', '<function=', '<parameter=', '</parameter>', '</function>', '>', 'get_weather',
          '<think>', '</think>', 'hmm', '<|im_end|>']
ID = {p: i for i, p in enumerate(PIECES)}
STOP = ID['<|im_end|>']
V = 64  # the model's vocabulary is larger than the tokenizer's, as Qwen's
SCHEMA = {'type': 'object', 'properties': {'city': {'type': 'string'}, 'days': {'type': 'integer'}}, 'required': ['city', 'days']}
TOOLS = [{'type': 'function', 'function': {'name': 'get_weather', 'parameters': {
    'type': 'object', 'properties': {'city': {'type': 'string'}}}}}]


@pytest.fixture(scope='module')
def grammars():
    return Grammars(PIECES, V, [STOP], ID['</think>'], byte_level=False)


def ids(*pieces):
    return [ID[p] for p in pieces]


def allowed(bitmask, row=0):
    return {i for i in range(V) if (int(bitmask[row, i // 32]) >> (i % 32)) & 1}


JSON = ids('{"', 'city', '":', ' ', '"', 'Paris', '"', ',', ' ', '"', 'days', '":', ' ', '2', '}')


def test_a_schema_allows_leading_whitespace_then_its_json_then_only_stop(grammars):
    state = GrammarState(grammars.json_schema(SCHEMA, armed=True))
    mask = host_bitmask(1, V)
    state.fill(mask, 0)
    assert ID['{"'] in allowed(mask) and ID['\n'] in allowed(mask) and ID['Paris'] not in allowed(mask) and STOP not in allowed(mask)
    assert all(state.accept(t) for t in ids('\n', '\n') + JSON)
    state.fill(mask, 0)
    assert allowed(mask) == {STOP}


def test_thinking_is_free_until_think_end(grammars):
    state = GrammarState(grammars.json_object(armed=False))
    mask = host_bitmask(1, V)
    state.fill(mask, 0)
    assert allowed(mask) == set(range(V))
    assert state.accept(ID['hmm']) and state.accept(ID['</think>'])
    state.fill(mask, 0)
    assert ID['hmm'] not in allowed(mask) and ID['{"'] in allowed(mask)


def test_draft_walk_masks_each_row_by_the_drafted_prefix_and_restores_the_state(grammars):
    state = GrammarState(grammars.json_schema(SCHEMA, armed=True))
    drafted = ids('{"', 'city', 'Paris', '":')  # 'Paris' is not allowed after '{"city'
    mask = host_bitmask(5, V)
    state.fill_draft(drafted, mask, 0)
    assert ID['{"'] in allowed(mask, 0)
    assert allowed(mask, 1) == {ID['city']}  # properties in the schema's order
    assert ID['":'] in allowed(mask, 2) and ID['Paris'] not in allowed(mask, 2)
    assert allowed(mask, 3) == allowed(mask, 4) == set(range(V))  # past the forbidden draft: never reached
    again = host_bitmask(1, V)
    state.fill(again, 0)
    assert allowed(again) == allowed(mask, 0)  # the walk took nothing back from the request's state


def test_forced_tool_call_grammar(grammars):
    state = GrammarState(grammars.tool_call(TOOLS, 'get_weather', armed=True))
    call = ids('<tool_call>', '\n', '<function=', 'get_weather', '>', '\n', '<parameter=', 'city', '>', '\n', 'Paris',
               '\n', '</parameter>', '\n', '</function>', '\n', '</tool_call>')
    mask = host_bitmask(1, V)
    state.fill(mask, 0)
    assert ID['<tool_call>'] in allowed(mask) and ID['Paris'] not in allowed(mask)
    assert all(state.accept(t) for t in call)
    state.fill(mask, 0)
    assert allowed(mask) == {STOP}
    with pytest.raises(ValueError, match='not among the tools'):
        grammars.tool_call(TOOLS, 'other', armed=True)


def test_sampler_picks_the_best_allowed_token(grammars):
    req = Request(prompt_ids=[1], params=SamplingParams(constraint=grammars.json_schema(SCHEMA, armed=True)))
    plain = Request(prompt_ids=[1], params=SamplingParams())
    logits = torch.zeros(2, V)
    logits[:, ID['Paris']] = 10.0  # the model wants 'Paris', the grammar wants '{"' or whitespace
    logits[:, ID['{"']] = 5.0
    out = Sampler(torch.device('cpu')).sample(logits, [req.params, plain.params], [None, None], sample_masks([req, plain], V))
    assert out.tolist() == [ID['{"'], ID['Paris']]
    assert sample_masks([plain], V) is None


def test_verify_rows_are_masked_so_acceptance_stops_at_a_forbidden_draft(grammars):
    constrained = Request(prompt_ids=[1], params=SamplingParams(constraint=grammars.json_schema(SCHEMA, armed=True)))
    free = Request(prompt_ids=[1], params=SamplingParams())
    drafted = [ids('{"', 'city', 'Paris'), ids('a', 'b', '1')]
    batch = SimpleNamespace(requests=[constrained, free], proposal=SimpleNamespace(tokens=torch.tensor(drafted)))
    logits = torch.zeros(8, V)
    for row, tok in enumerate(drafted[0] + [ID['":']] + drafted[1] + [ID['2']]):
        logits[row, tok] = 10.0  # the target agrees with every drafted token
    logits[2, ID['":']] = 5.0  # and ranks the grammar's continuation second where it drafted 'Paris'
    holder = SimpleNamespace()
    assert SpecDecoder._constrain(holder, batch, logits) is True
    z = logits.argmax(-1).view(2, 4)
    assert z[0].tolist()[:2] == ids('{"', 'city') and z[0, 2] == ID['":']  # 'Paris' is masked: correction '":'
    assert z[1].tolist() == ids('a', 'b', '1', '2')  # the unconstrained request is untouched
    assert SpecDecoder._constrain(holder, SimpleNamespace(requests=[free], proposal=batch.proposal), logits) is False


def test_push_advances_the_grammar_and_finishes_on_a_forbidden_token(grammars):
    req = Request(prompt_ids=[1], params=SamplingParams(max_new_tokens=50, constraint=grammars.json_schema(SCHEMA, armed=True)))
    req.push(ID['{"'])
    assert not req.finished
    req.push(ID['Paris'])
    assert req.finished and req.finish_reason == 'grammar'


def test_apply_bitmask_matches_the_bit_layout():
    mask = torch.zeros(1, 2, dtype=torch.int32)
    for tok in (0, 7, 8, 31, 33):
        mask[0, tok // 32] |= (1 << (tok % 32)) if tok % 32 != 31 else -(1 << 31)
    logits = torch.zeros(1, 40)
    apply_bitmask(logits, mask)
    assert torch.isfinite(logits[0]).nonzero().flatten().tolist() == [0, 7, 8, 31, 33]


def test_a_drafted_stop_token_ends_the_walk(grammars):
    """A draft that completes the JSON and then proposes the stop token: rows after it are never reached, and the
    walk must not ask the finished grammar for another mask (it raised in the engine, 2026-10-05)."""
    state = GrammarState(grammars.json_schema(SCHEMA, armed=True))
    assert all(state.accept(t) for t in JSON[:-1])
    drafted = [ID['}'], STOP, ID['a'], ID['b']]
    mask = host_bitmask(5, V)
    state.fill_draft(drafted, mask, 0)
    assert ID['}'] in allowed(mask, 0) and allowed(mask, 1) == {STOP}
    assert allowed(mask, 2) == allowed(mask, 3) == allowed(mask, 4) == set(range(V))
    assert state.accept(ID['}']) and state.accept(STOP)  # the walk left the state where it was
    finished = host_bitmask(1, V)
    state.fill(finished, 0)  # a finished grammar does not raise
    assert allowed(finished) == set(range(V))


def test_whitespace_is_bounded_before_and_inside_the_json(grammars):
    """Unbounded whitespace let a sampled request drift into it until max_tokens (2026-10-05)."""
    state = GrammarState(grammars.json_schema(SCHEMA, armed=True))
    assert all(state.accept(ID['\n']) for _ in range(4))
    mask = host_bitmask(1, V)
    state.fill(mask, 0)
    assert ID['\n'] not in allowed(mask) and ID[' '] not in allowed(mask) and ID['{"'] in allowed(mask)
    assert state.accept(ID['{"']) and state.accept(ID['city']) and state.accept(ID['":'])
    taken = 0
    while state.accept(ID[' ']):
        taken += 1
        assert taken <= Grammars.MAX_WS
    assert taken == Grammars.MAX_WS


def test_json_object_takes_any_object(grammars):
    state = GrammarState(grammars.json_object(armed=True))
    assert all(state.accept(t) for t in ids('{"', 'a', '":', ' ', '1', ',', ' ', '"', 'b', '":', ' ', '2', '}', '<|im_end|>'))
