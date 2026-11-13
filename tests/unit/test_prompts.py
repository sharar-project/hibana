from hibana.engine.prompts import random_ids


def test_random_ids_are_deterministic_and_in_range():
    a = random_ids(64, 7)
    b = random_ids(64, 7)
    assert a == b
    assert all(0 <= t < 200_000 for t in a)
