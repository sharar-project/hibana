import threading
import time

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")


def test_wait_flag_blocks_until_host_sets_it_inside_a_graph():
    from sharar.kernels.wait_flag import wait_flag
    flag = torch.zeros(1, dtype=torch.int32, pin_memory=True)
    error = torch.zeros(1, dtype=torch.int32, device="cuda")
    x = torch.ones(1024, device="cuda")
    wait_flag(flag, 0, error)  # warm the module outside capture
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        wait_flag(flag, 1, error, timeout_ms=5000)
        y = x * 2
    done = torch.cuda.Event()
    t0 = time.perf_counter()
    g.replay()
    done.record()
    assert not done.query()  # the graph is waiting on the host

    def release():
        time.sleep(0.2)
        flag[0] = 1
    threading.Thread(target=release).start()
    done.synchronize()
    assert 0.18 < time.perf_counter() - t0 < 2.0 and error.item() == 0 and torch.equal(y, x * 2)
    flag[0] = 0
    with torch.cuda.graph(torch.cuda.CUDAGraph()) if False else torch.no_grad():
        wait_flag(flag, 5, error, timeout_ms=50)  # never satisfied: times out and flags the error
    torch.cuda.synchronize()
    assert error.item() == 1
