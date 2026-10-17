"""The frozen 10K benchmark prompts the in-process benchmarks use (family inputs, check_distribution).
sharar-eval's evals/speed/workloads.py is a byte-identical copy and pins the same digests."""
import hashlib
import json

import pytest

from sharar.benchmark_workloads import materialize

SOURCE_SHA256 = "534d0490fb307754cea397cbf0ef816dc4abd2c2ff9e72f73c39db1ce1ba276d"  # request_state_7aebbe6.txt


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


@pytest.mark.parametrize('case,expected', [
    ('code', '2286736c0fbe18746221e5f483170178626dfe6b1f72649d02a5f4e021943050'),  # gb10-coding-v2, tool-free prompt
    ('code_v1', '04d6e0023e3b6b63d412de5a8bf49f6ee7efe68ef7061da013ec126426b4e195'),  # the tool-call prompt
    ('design', '0402378ae156b5906548dcd7bbacf6484421643b6386284e8b8375727df975b1'),
    ('reasoning', 'fced3b9d8b1cafb6d661e1f604d171681751e555530b97b0ad6b4ef66f79ed40'),
])
def test_builtin_workloads_match_frozen_aiperf_baseline(snapshot, case, expected):
    from sharar.tokenizer import SnapshotTokenizer

    ids = materialize(SnapshotTokenizer(snapshot).encode, case, 10000)
    assert len(ids) == 10000
    assert digest(ids) == expected


def test_packaged_source_fixture_stays_frozen():
    from sharar.benchmark_workloads import frozen_source
    assert hashlib.sha256(frozen_source().encode()).hexdigest() == SOURCE_SHA256
