"""The fixed prompt set: token-ID sequences shared by the engine's CLI checks and tests.

Random-ID prompts exercise the kernels at each length independent of tokenizer
behavior; text prompts have realistic activation statistics; the canaries are
the bench harness's quality prompts rendered through the chat template with
thinking disabled, exactly as the harness sends them.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from sharar import config
from sharar.api.chat_template import ChatTemplate
from sharar.tokenizer import SnapshotTokenizer

# Layer dumps are captured for prompts up to this length; longer prompts are
# checked through top-logprobs over HTTP only.
DUMP_MAX_TOKENS = 2048

PARAGRAPH = (
    "The workshop kept its ledgers in numbered boxes along the north wall. Each "
    "morning the apprentice counted them, wrote the total in chalk on the slate, "
    "and compared it with the previous day's figure before anyone else arrived. "
)

NEEDLE = "SPARK-NEEDLE-7291"
FILLER = "A quiet workshop contains ordinary numbered boxes. " * 600
CANARIES = {
    "canary_exact": "Reply with exactly SHARAR_OK and nothing else.",
    "canary_arith": "What is 17 + 25? Reply with only the integer.",
    "canary_json": (
        'Return only valid JSON with keys "device" and "model". '
        'Their values must be "GB10" and "Qwen3.8-27B".'
    ),
    "canary_needle": (
        FILLER
        + f"The unique access code is {NEEDLE}. "
        + FILLER
        + "What is the unique access code? Reply with only the code."
    ),
}


def random_ids(n: int, seed: int, first: int = 0, vocab: int = 200_000) -> list[int]:
    """`n` seeded ids of ordinary text: from `first` (`Target.first_text_token`) up to the vocabulary and below the
    Qwen targets' special-token range (200,000 and up)."""
    rng = random.Random(seed)
    return [rng.randrange(first, min(vocab, 200_000)) for _ in range(n)]


def build_prompts(snap: Path) -> dict[str, list[int]]:
    tok = SnapshotTokenizer(snap)
    tmpl = ChatTemplate(snap)
    prompts: dict[str, list[int]] = {
        "rand8": random_ids(8, 8),
        "rand128": random_ids(128, 128),
        "rand2048": random_ids(2048, 2048),
        "rand8192": random_ids(8192, 8192),
    }
    text_ids = tok.encode(PARAGRAPH * 40)
    prompts["text1024"] = text_ids[:1024]
    for name, content in CANARIES.items():
        rendered = tmpl.render([{"role": "user", "content": content}], enable_thinking=False)
        prompts[name] = tok.encode(rendered)
    return prompts


def write_prompts(path: Path) -> dict[str, list[int]]:
    snap = config.snapshot_dir()
    prompts = build_prompts(snap)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prompts))
    return prompts


def read_prompts(path: Path) -> dict[str, list[int]]:
    return {k: list(map(int, v)) for k, v in json.loads(path.read_text()).items()}
