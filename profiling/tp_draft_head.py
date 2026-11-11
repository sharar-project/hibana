"""The vocabulary-parallel draft head (`proposers/draft_head.py:VocabParallelHead`) against the whole-vocabulary
heads it replaces, on two Sparks with a random Flash-Next-sized lm_head: the same picks, captured in a CUDA graph,
and the time per draft step.

    hibana dev tp-draft-head      (both ranks; rank 0 prints)
"""
import argparse
import time

import torch

from hibana import distributed
from hibana.model.flashnext.linear import TrunkLinear
from hibana.proposers.draft_head import ShortlistHead, TargetHead, VocabParallelHead
from hibana.weights.sharding import ColumnParallel

VOCAB, HIDDEN = 248_320, 2560


def graphed(fn, x):
    """fn(x) captured once and replayed: what the MTP draft graphs do."""
    out = fn(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn(x)
    return graph, out


def timed(graph, iters=200) -> float:
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        graph.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-address", required=True)
    parser.add_argument("--tp-rank", type=int, required=True)
    args = parser.parse_args()
    distributed.init_distributed(args.tp_rank, 2, args.tp_address)
    primary = args.tp_rank == 0
    torch.manual_seed(0)
    weight = (torch.randn(VOCAB, HIDDEN, device="cuda") * 0.02).to(torch.bfloat16)
    whole = TrunkLinear.from_bf16(weight, "fp8")
    part = TrunkLinear.from_bf16(weight, "fp8", ColumnParallel((VOCAB // 2,)))
    del weight
    heads = {
        "fp8": (TargetHead(whole), VocabParallelHead(part, "fp8")),
        "w3, top 8 rescored": (ShortlistHead(whole, None, "w3", 8), VocabParallelHead(part, "w3", 8)),
    }
    for name, (reference, parallel) in heads.items():
        for rows in (1, 4, 5, 16, 20):
            x = torch.randn(rows, HIDDEN, device="cuda", dtype=torch.bfloat16)
            same = torch.equal(reference.argmax(x), parallel.argmax(x))
            _, conf_ref = (reference.argmax_confidence(x) if hasattr(reference, "argmax_confidence")
                           else (None, None))
            _, conf = parallel.argmax_confidence(x)
            conf_err = (conf - conf_ref).abs().max().item() if conf_ref is not None else float("nan")
            g_ref, out_ref = graphed(reference.argmax, x)
            g_par, out_par = graphed(parallel.argmax, x)
            x.copy_(torch.randn_like(x))
            g_ref.replay(); g_par.replay()
            same_graphed = torch.equal(out_ref, out_par)
            t_ref, t_par = timed(g_ref), timed(g_par)
            if primary:
                print(f"{name:20s} rows {rows:2d}: same picks {same} (graphed {same_graphed}), confidence err {conf_err:.1e}; "
                      f"whole head {t_ref:7.1f} us, vocabulary-parallel {t_par:7.1f} us")
    distributed.close_distributed()


if __name__ == "__main__":
    main()
