"""Fixed-shape target verification graphs shared by every proposer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sharar.engine import graph_set
from sharar.engine.context import ForwardContext
from sharar.engine.depth import RowBudget
import dataclasses

from sharar.engine.graphs import capture_sizes, check_context_capacity, graph_variants

if TYPE_CHECKING:
    import flashinfer

def _prefill_wrapper(ws: torch.Tensor, st: dict) -> flashinfer.BatchPrefillWithPagedKVCacheWrapper:
    import flashinfer

    return flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws, "NHD", use_cuda_graph=True,
        qo_indptr_buf=st["qo_indptr"], paged_kv_indptr_buf=st["kv_indptr"],
        paged_kv_indices_buf=st["kv_indices"], paged_kv_last_page_len_buf=st["last_page_len"],
    )


def _verify_attention(ex, st: dict, b: int, rows: int):
    """The 27B's verification of more than 8 rows a sequence over an NVFP4 cache runs kernels/csrc/verify_attn.cu
    (16 rows: 1.3-1.5x FlashInfer's fa2 at 64-128K, 1.9x for eight requests); up to 8 rows fa2 is faster. Splits of
    the context: about 128 blocks over the batch's KV heads (12 for one request, 2 from sixteen)."""
    from sharar.engine.config import current
    pool = ex.kv_pool
    if (current().verify_attention != "auto" or ex.extras is not None or rows <= 8
            or not getattr(pool, "nvfp4", False)
            or (pool.num_kv_heads, pool.head_dim) != (4, 256) or ex.target.num_attention_heads != 24):
        return None
    splits = max(2, min(12, -(-128 // (b * pool.num_kv_heads))))
    return st["qo_indptr"], st["kv_indptr"], st["kv_indices"], splits


class VerifyGraphs:
    def __init__(self, spec, depth=None, sizes=None):
        self.spec = spec
        self.ex = spec.ex
        self.device = spec.device
        depth = spec.k if depth is None else depth
        self.s_len = depth + 1
        self.max_context = spec.ex.max_context + depth
        self.sizes = capture_sizes(spec.ex.max_batch) if sizes is None else tuple(sizes)  # a depth per batch bucket captures its buckets only
        # A policy that captures many depths (engine/depth.RowBudget) shares one pool and one FlashInfer workspace across
        # them: one depth's graph replays at a time and every replay plans first.
        budget = isinstance(spec.depth_policy, RowBudget)
        shared = getattr(spec, "verify_shared", None) if budget else None
        if shared is None:
            # FlashInfer's split-KV scratch grows with the verified rows: 388 MB at 128 requests x 5 rows
            shared = (torch.cuda.graph_pool_handle(),
                      torch.empty(256 * 1024 * 1024 * max(1, -(-spec.ex.max_batch // 64)), dtype=torch.uint8, device=self.device))
            if budget:
                spec.verify_shared = shared
        self._pool, self._ws = shared
        # and one page index per batch bucket (b x the deepest depth's context: 0.25 GiB a depth at 128 requests)
        self._kv_indices = getattr(spec, "verify_kv_indices", None) if budget else None
        if budget and self._kv_indices is None:
            self._kv_indices = spec.verify_kv_indices = {}
        self.variants = graph_variants(spec.ex)
        self._st: dict[int, dict] = {}
        self._graphs: dict[tuple[int, str], torch.cuda.CUDAGraph] = {}

    def _static(self, b: int) -> dict:
        d, s = self.device, self.s_len
        st = {
            "tokens": torch.zeros(b * s, dtype=torch.int64, device=d),
            "positions": torch.zeros(b * s, dtype=torch.int64, device=d),
            "pages": torch.zeros(b * s, dtype=torch.int32, device=d),
            "slots": torch.zeros(b, dtype=torch.int32, device=d),
            "qo_indptr": torch.arange(0, b * s + 1, s, dtype=torch.int32, device=d),
            "kv_indptr": torch.zeros(b + 1, dtype=torch.int32, device=d),
            "kv_indices": self._page_index(b),
            "last_page_len": torch.ones(b, dtype=torch.int32, device=d),
            "side": self.ex.extras.buffers(b * s) if self.ex.extras is not None else None,  # the model's static side inputs
        }
        st["wrapper"] = _prefill_wrapper(self._ws, st)
        st["ctx"] = ForwardContext(
            feature_request=self.spec.proposer.requirements.features,
            mode="verify", kv_pool=self.ex.kv_pool, state_pool=self.ex.state_pool, positions=st["positions"],
            pages=st["pages"], seq_len=b * s, cu_seqlens=st["qo_indptr"], seq_slots=st["slots"],
            seq_prefix_lens=[0] * b, seq_lens=[s] * b, verify_steps=s,
            verify_scratch=torch.arange(b, dtype=torch.int32, device=d), inter_ssm=self.spec.inter_ssm,
            verify_replay=self.spec.replay.layer_inputs if self.spec.replay is not None else {},
            verify_conv_windows=self.spec.conv_windows, verify_aux_windows=self.spec.aux_windows,
            paged_prefill_wrapper=st["wrapper"], extras=st["side"],
            verify_attention=_verify_attention(self.ex, st, b, self.s_len),
        )
        if "selecting" in self.variants:
            st["row_seq"] = torch.arange(b * s, device=d) // s
            st["ctx_sparse"] = dataclasses.replace(st["ctx"], extras=self.ex.extras.selecting(
                st["side"], st["kv_indices"], st["kv_indptr"], st["row_seq"], self.max_context))
        return st

    def _page_index(self, b: int) -> torch.Tensor:
        """The bucket's page index buffer: its own, or the one every depth's graph of bucket b shares."""
        if self._kv_indices is None:
            return torch.zeros(b * self.max_context, dtype=torch.int32, device=self.device)
        if b not in self._kv_indices:
            self._kv_indices[b] = torch.zeros(b * (self.ex.max_context + self.spec.k), dtype=torch.int32, device=self.device)
        return self._kv_indices[b]

    def _plan(self, st: dict, kv_all: list[torch.Tensor]) -> None:
        check_context_capacity([p.shape[0] for p in kv_all], self.max_context)
        b, s = len(kv_all), self.s_len
        # Host index pointers: plan() reads them on the CPU (a device tensor would sync).
        indptr = torch.tensor([0] + list(torch.tensor([p.shape[0] for p in kv_all]).cumsum(0).tolist()), dtype=torch.int32)
        self.ex.attention.paged_prefill(st["wrapper"], torch.arange(0, b * s + 1, s, dtype=torch.int32), indptr, torch.cat(kv_all),
                                        torch.ones(b, dtype=torch.int32), causal=True, guard=False, page_size=1)  # graph-bound buffers

    def capture(self, log=None) -> None:
        for b in self.sizes:
            st = self._static(b)
            self._st[b] = st
            # dummy plan: each row attends to page 0 plus its own rows (pages 0)
            self._plan(st, [torch.zeros(self.s_len + 1, dtype=torch.int32, device=self.device) for _ in range(b)])

            def warm():
                self.ex.model.logits(self.ex.model.forward(st["tokens"], st["ctx"])["norm"]).argmax(-1)

            def body():
                out = self.ex.model.forward(st["tokens"], st["ctx"])
                st["norm"] = out["norm"]
                st["features"] = out.get("features")
                logits = self.ex.model.logits(out["norm"])
                st["argmax"] = logits.argmax(-1)
                st["logits"] = logits  # materialized by the graph anyway; traces read the target's top-k from it
            self._graphs[b, "dense"] = graph_set.capture(self._pool, body, warmup=warm)
            if "selecting" in self.variants:  # the selecting variant writes the same outputs (device acceptance reads them)
                self._graphs[b, "selecting"] = graph_set.capture(self._pool, lambda: self._sparse_body(st))
            if log:
                log(f"  captured {'dense and selecting ' if 'selecting' in self.variants else ''}verify graph for batch {b}")

    @torch.inference_mode()
    def time_replay(self, b: int, context: int, rounds: int = 7) -> float:
        """Median ms of the dense verify graph for bucket b, each row attending `context` cached tokens (page 0).
        The rows are random tokens: a mixture of experts routes them across its experts as served traffic does,
        where identical rows would all read the same few."""
        st = self._st[b]
        vocab = self.ex.target.vocab_size
        st["tokens"].copy_(torch.randint(0, vocab, st["tokens"].shape, device=self.device,
                                         generator=torch.Generator(device=self.device).manual_seed(b)))
        self._plan(st, [torch.zeros(context + self.s_len, dtype=torch.int32, device=self.device) for _ in range(b)])
        return graph_set.time_replay(self._graphs[b, "dense"], rounds)

    @torch.inference_mode()  # the dense capture's outputs are inference tensors
    def _sparse_body(self, st: dict) -> None:
        out = self.ex.model.forward(st["tokens"], st["ctx_sparse"])
        st["norm"].copy_(out["norm"])
        for lid, rows in (out.get("features") or {}).items():
            st["features"][lid].copy_(rows)
        logits = self.ex.model.logits(out["norm"])
        st["logits"].copy_(logits)
        st["argmax"].copy_(logits.argmax(-1))

    def pick(self, b: int) -> int:
        return graph_set.pick(self.sizes, b)

    def run(self, tokens: torch.Tensor, positions: torch.Tensor, pages: torch.Tensor, slots: torch.Tensor, kv_all: list[torch.Tensor], *, sampled=False,
            side=None, variant: str = "dense"):
        """Returns (norm rows [b*s, H], argmax or logits [b*s(, V)], ctx, features,
        logits [b*s, V]) for the real rows; padded rows use slot 0, attend to page 0 and write the pool's padding pages.
        The logits view is overwritten by the next replay. `side` is (requests,
        drafts, seqs, draft columns or None) for the model's side inputs; `variant` the graph to replay
        ("selecting": some row past Flash-Next's indexer budget)."""
        b = slots.shape[0]
        size = self.pick(b)
        st = self._st[size]
        s = self.s_len
        n = b * s
        st["tokens"][:n].copy_(tokens)
        st["positions"][:n].copy_(positions)
        st["pages"][:n].copy_(pages)
        st["slots"][:b].copy_(slots)
        if size > b:
            st["tokens"][n:].zero_()
            st["positions"][n:].zero_()
            st["pages"][n:].copy_(self.ex.kv_pool.pad_pages(size * s - n))
            st["slots"][b:].zero_()
        pad_kv = [torch.zeros(s + 1, dtype=torch.int32, device=self.device) for _ in range(size - b)]
        # Plan before the side inputs: they may wait for the drafted tokens, the plan needs only the pages,
        # so the host plans while the draft graphs still run.
        self._plan(st, list(kv_all) + pad_kv)
        if st["side"] is not None:
            self.ex.extras.fill(st["side"], side[0], side[1], side[2], n, columns=side[3])
        self._graphs[size, variant].replay()
        layers = {lid: value[:n] for lid, value in st["features"].items()} if st["features"] else None
        return st["norm"][:n], st["logits" if sampled else "argmax"][:n], st["ctx"], layers, st["logits"][:n]


class RaggedVerifyGraphs:
    """Verification of sequences of their own lengths, packed: one graph per (batch bucket, rows), for the depth policy
    that gives each request its own depth (engine/depth.RowBudget, packed, with the drafts' acceptance).

    A round's sequences split the graph's rows exactly, 1 to `steps` each: a request's verified rows (its pending token
    and drafts), then padding rows on its own reserved pages (after its verified rows, so causal attention keeps them
    from its outputs; never committed), and the bucket's padded requests on padding pages. GDN and attention read the
    per-sequence offsets (cu_seqlens = qo_indptr, rows_seq / rows_pos). The outputs are unpacked to the [batch, width]
    rows acceptance, commit and feedback read: row j of request i is its packed row j (rows past a request's length
    are filler its acceptance limit never reads)."""

    def __init__(self, spec, budgets: dict[int, tuple[int, ...]]):
        self.spec, self.ex, self.device = spec, spec.ex, spec.device
        self.steps = spec.steps  # the widest sequence: the proposer's k + 1
        self.max_context = spec.ex.max_context + spec.k
        self.budgets = {b: tuple(sorted(rows)) for b, rows in budgets.items()}
        self.sizes = tuple(sorted(budgets))
        self.variants = ("dense",)
        self._pool = torch.cuda.graph_pool_handle()
        self._ws = torch.empty(256 * 1024 * 1024 * max(1, -(-spec.ex.max_batch // 64)), dtype=torch.uint8, device=self.device)
        self._st: dict[tuple[int, int], dict] = {}
        self._graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        # A bucket's graphs (one per row total) replay one at a time and their outputs are read before the next replay,
        # so they share the bucket's page index and unpacked outputs: the logits alone are b x steps x the vocabulary
        # (eleven copies: 2.5 GiB at 16 requests on the 27B)
        self._kv_indices: dict[int, torch.Tensor] = {}
        self._outputs: dict[int, dict] = {}

    def _bucket_outputs(self, b: int, out: dict, logits: torch.Tensor) -> dict:
        """Bucket b's unpacked outputs, allocated by the first warm-up run (outside capture) in the forward's shapes."""
        if b not in self._outputs:
            n = b * self.steps
            rows = lambda t: torch.empty((n, *t.shape[1:]), dtype=t.dtype, device=t.device)  # noqa: E731
            self._outputs[b] = {"norm": rows(out["norm"]), "logits": rows(logits),
                                "argmax": torch.empty(n, dtype=torch.int64, device=logits.device),
                                "features": {lid: rows(f) for lid, f in (out.get("features") or {}).items()}}
        return self._outputs[b]

    def _static(self, b: int, rows: int) -> dict:
        d, s = self.device, self.steps
        st = {
            "tokens": torch.zeros(rows, dtype=torch.int64, device=d),
            "positions": torch.zeros(rows, dtype=torch.int64, device=d),
            "pages": torch.zeros(rows, dtype=torch.int32, device=d),
            "slots": torch.zeros(b, dtype=torch.int32, device=d),
            "qo_indptr": torch.zeros(b + 1, dtype=torch.int32, device=d),
            "kv_indptr": torch.zeros(b + 1, dtype=torch.int32, device=d),
            "kv_indices": self._kv_indices.setdefault(b, torch.zeros(b * self.max_context, dtype=torch.int32, device=d)),
            "last_page_len": torch.ones(b, dtype=torch.int32, device=d),
            "rows_seq": torch.zeros(rows, dtype=torch.int32, device=d),
            "rows_pos": torch.zeros(rows, dtype=torch.int32, device=d),
            "unpack": torch.zeros(b * s, dtype=torch.int64, device=d),
        }
        st["wrapper"] = _prefill_wrapper(self._ws, st)
        st["ctx"] = ForwardContext(
            feature_request=self.spec.proposer.requirements.features,
            mode="verify", kv_pool=self.ex.kv_pool, state_pool=self.ex.state_pool, positions=st["positions"],
            pages=st["pages"], seq_len=rows, cu_seqlens=st["qo_indptr"], seq_slots=st["slots"],
            seq_prefix_lens=[0] * b, seq_lens=[s] * b, verify_steps=s,
            verify_scratch=torch.arange(b, dtype=torch.int32, device=d), inter_ssm=self.spec.inter_ssm,
            verify_replay=self.spec.replay.layer_inputs if self.spec.replay is not None else {},
            verify_conv_windows=self.spec.conv_windows, verify_aux_windows=self.spec.aux_windows,
            paged_prefill_wrapper=st["wrapper"], verify_rows_seq=st["rows_seq"], verify_rows_pos=st["rows_pos"],
            verify_attention=_verify_attention(self.ex, st, b, s),  # sequences of up to `steps` rows
        )
        return st

    def _layout(self, st: dict, lengths: list[int], width: int) -> None:
        """The round's offsets, row maps and unpack index (host-built, one upload each)."""
        offsets = [0]
        for n in lengths:
            offsets.append(offsets[-1] + n)
        seq = [i for i, n in enumerate(lengths) for _ in range(n)]
        pos = [j for n in lengths for j in range(n)]
        unpack = [offsets[i] + min(j, n - 1) for i, n in enumerate(lengths) for j in range(width)]
        st["qo_indptr"].copy_(torch.tensor(offsets, dtype=torch.int32), non_blocking=True)
        st["rows_seq"].copy_(torch.tensor(seq, dtype=torch.int32), non_blocking=True)
        st["rows_pos"].copy_(torch.tensor(pos, dtype=torch.int32), non_blocking=True)
        st["unpack"][:len(unpack)].copy_(torch.tensor(unpack, dtype=torch.int64), non_blocking=True)
        st["offsets"] = offsets

    def _plan(self, st: dict, lengths: list[int], kv_all: list[torch.Tensor]) -> None:
        check_context_capacity([p.shape[0] for p in kv_all], self.max_context)
        qo = torch.tensor(st["offsets"], dtype=torch.int32)
        indptr = torch.tensor([0] + list(torch.tensor([p.shape[0] for p in kv_all]).cumsum(0).tolist()), dtype=torch.int32)
        self.ex.attention.paged_prefill(st["wrapper"], qo, indptr, torch.cat(kv_all), torch.ones(len(lengths), dtype=torch.int32),
                                        causal=True, guard=False, page_size=1)

    @staticmethod
    def even(rows: int, b: int, steps: int) -> list[int]:
        """`rows` split as evenly as possible over b sequences of 1 to `steps` rows."""
        base, extra = divmod(rows, b)
        return [min(steps, base + (i < extra)) for i in range(b)]

    def capture(self, log=None) -> None:
        for b in reversed(self.sizes):  # the largest first: smaller graphs fit in the blocks they freed (one pool)
            for rows in reversed(self.budgets[b]):
                st = self._static(b, rows)
                lengths = self.even(rows, b, self.steps)
                self._layout(st, lengths, self.steps)
                self._plan(st, lengths, [torch.zeros(n + 1, dtype=torch.int32, device=self.device) for n in lengths])

                def body(st=st, b=b):
                    out = self.ex.model.forward(st["tokens"], st["ctx"])
                    logits = self.ex.model.logits(out["norm"])
                    o = self._bucket_outputs(b, out, logits)
                    st["norm"] = torch.index_select(out["norm"], 0, st["unpack"], out=o["norm"])
                    st["features"] = {lid: torch.index_select(f, 0, st["unpack"], out=o["features"][lid])
                                      for lid, f in (out.get("features") or {}).items()}
                    st["logits"] = torch.index_select(logits, 0, st["unpack"], out=o["logits"])
                    st["argmax"] = torch.argmax(st["logits"], -1, out=o["argmax"])
                self._graphs[b, rows] = graph_set.capture(self._pool, body)
                self._st[b, rows] = st
            if log:
                log(f"  captured ragged verify graphs for batch {b}: rows {', '.join(map(str, self.budgets[b]))}")

    @torch.inference_mode()
    def time_replay(self, b: int, rows: int, context: int, rounds: int = 7) -> float:
        """Median ms of the (b, rows) graph, sequences as even as the rows allow, each attending `context` cached
        tokens (page 0), random tokens."""
        st = self._st[b, rows]
        lengths = self.even(rows, b, self.steps)
        self._layout(st, lengths, self.steps)
        self._plan(st, lengths, [torch.zeros(context + n, dtype=torch.int32, device=self.device) for n in lengths])
        st["tokens"].copy_(torch.randint(0, self.ex.target.vocab_size, st["tokens"].shape, device=self.device,
                                         generator=torch.Generator(device=self.device).manual_seed(b * 1000 + rows)))
        return graph_set.time_replay(self._graphs[b, rows], rounds)

    def run(self, bucket: int, rows: int, lengths: list[int], width: int, tokens, positions, pages, slots, kv_all):
        """Replay the (bucket, rows) graph. lengths: every sequence's rows (the requests', then the bucket's padded
        ones), summing to `rows`; tokens/positions/pages [rows] packed; slots [bucket]; kv_all one page list per
        sequence. Returns (norm, argmax, features, logits) of the first len(kv_all real) requests unpacked to rows of
        `width`, views the next replay overwrites."""
        st = self._st[bucket, rows]
        st["tokens"].copy_(tokens)
        st["positions"].copy_(positions)
        st["pages"].copy_(pages)
        st["slots"].copy_(slots)
        self._layout(st, lengths, width)
        self._plan(st, lengths, kv_all)
        self._graphs[bucket, rows].replay()
        return st
