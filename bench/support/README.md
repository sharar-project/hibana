# Shared benchmark support

These modules are infrastructure shared by measurement drivers. They do not
select serving policies or execute benchmarks at import time.

| Module | Responsibility |
|---|---|
| `evidence.py` | Source hashes, result writing and cache/pool cleanup (`gb10_family_eval`) |
| `target_features.py` | Target hidden features for drafter references (`dflash2_reference`) |

Keep numerical experiments in their drivers. Finished experiments and their helpers are on the
[archive branch](https://github.com/abdelfattah-lab/sharar/tree/archive/pre-cleanup-2026-09-26).
