import pytest


@pytest.fixture(autouse=True)
def _cache_invariants(monkeypatch):
    """Every scheduler step in the unit tests ends with the prefix cache's invariants checked (`StateManager.check`:
    one lock per running request, the lock counts, the evictable totals, slot and page ownership, no private snapshot
    between steps, no dead KV), whenever the executor under the
    scheduler has a real StateManager over a radix cache."""
    from hibana.cache.request_state import StateManager
    from hibana.engine.scheduler import Scheduler

    step = Scheduler.step

    def checked(self):
        step(self)
        state = getattr(self.ex, "state", None)
        if isinstance(state, StateManager) and state.cache is not None:
            state.check()

    monkeypatch.setattr(Scheduler, "step", checked)
