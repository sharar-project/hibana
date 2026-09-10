# Shared benchmark support

These modules are infrastructure shared by measurement drivers. They do not
select serving policies or execute benchmarks at import time.

| Module | Responsibility |
|---|---|
| `evidence.py` | Source hashes, result writing and cache/pool cleanup |
| `recording.py` | Forward recording, physical/emitted work and cold native batches |
| `retention.py` | Native retained-batch measurement |
| `http.py` | Temporary real-engine HTTP server and audited request streams |
| `cuda.py` | Matrix operand hashes and CUDA graph timing |

CPU tests import recording/HTTP helpers directly. Source identity records all
benchmark Python modules so moved helpers remain covered by run provenance.
Keep numerical experiments in their drivers or [archive](../archive/README.md).
