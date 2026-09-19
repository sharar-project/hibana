"""Device-resident acceptance for all-greedy verify batches.

After the verify graph, the eager round used to download the argmax rows,
decide acceptance on the host and only then launch the accepted-state
reconstruction, the convolution copies and the draft feedback, one small
launch at a time; the GPU idled while the host worked. Here one captured
graph per batch bucket does all of that from the verify graph's static
outputs: `accept.accept_block` decides the counts (exact or relaxed rule,
budget clamp, stop tokens), `AcceptedState.run` reconstructs the committed
recurrent state for exactly those counts, the convolution windows are
committed, and the proposer's `feedback_device` projects and scatters the
accepted rows into its rings. The host then reads one packed result through a
pinned buffer and an event, and only publishes: pages, committed lengths,
emitted tokens, traces. Ownership stays with StateManager
(`authorize_reconstruction` before the replay, `publish_reconstructed` after)
and with the proposer (`accepted_prepare` / `accepted_publish`).

Padded rows of a bucket carry budget 0 and ring slot 0: they emit nothing,
reconstruct nothing into the dummy recurrent slot and write the dummy ring.
"""

from __future__ import annotations

import os
import time

import torch

from sharar.engine.accept import MAX_STOPS, accept_block

ENABLED = os.environ.get("SHARAR_DEVICE_ACCEPT", "0") == "1"  # opt-in: exact, +2% at batch one, no serving gain at batch four yet (roadmap item 3)
META = 3 + MAX_STOPS  # budget, start, ring slot, stop ids


class AcceptGraphs:
    def __init__(self, spec, verify_graphs):
        self.spec, self.vg = spec, verify_graphs
        self.ex = spec.ex
        self.device = spec.device
        self.s = verify_graphs.s_len
        self.k = self.s - 1
        self._pool = torch.cuda.graph_pool_handle()
        self._st: dict[int, dict] = {}
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}

    @staticmethod
    def supported(spec) -> bool:
        p = spec.proposer
        return (spec.replay is not None and spec.device.type == "cuda"
                and all(hasattr(p, name) for name in ("accepted_prepare", "feedback_device", "accepted_publish"))
                and not p.requirements.features.final_norm)

    def eligible(self, requests) -> bool:
        return all(r.params.greedy and (r.params.ignore_eos or len(r.params.stop_token_ids) <= MAX_STOPS) for r in requests)

    def _static(self, b: int) -> dict:
        d, k, s = self.device, self.k, self.s
        pin = d.type == "cuda"
        return {
            "y": torch.zeros((b, k), dtype=torch.int64, device=d),
            "relax": torch.zeros((2, b), dtype=torch.float32, device=d),
            "meta": torch.full((b, META), -1, dtype=torch.int64, device=d),
            "host_relax": torch.zeros((2, b), dtype=torch.float32, pin_memory=pin),
            "host_meta": torch.full((b, META), -1, dtype=torch.int64, pin_memory=pin),
            # y [k] | z [s] | accepted | count | flags [k]
            "packet": torch.zeros((b, 2 * k + s + 2), dtype=torch.int64, device=d),
            "host_packet": torch.zeros((b, 2 * k + s + 2), dtype=torch.int64, pin_memory=pin),
            "event": torch.cuda.Event(),
        }

    def _body(self, b: int, st: dict) -> None:
        vst = self.vg._st[b]
        k, s = self.k, self.s
        z = vst["argmax"][:b * s].view(b, s)
        logits = vst["logits"][:b * s].view(b, s, -1)[:, :k]
        meta = st["meta"]
        accepted, counts, flags = accept_block(st["y"], z, logits, st["relax"], meta[:, 0], meta[:, 3:])
        pool = self.ex.state_pool
        slots = vst["slots"][:b]
        self.spec.replay.run(pool, slots, counts.to(torch.int32))
        rows = torch.arange(b, device=self.device)
        pool.conv_all[:, slots.long()] = self.spec.conv_windows_all[:, rows, (counts - 1).clamp(min=0)]
        for name, win in self.spec.aux_windows.items():
            pool.aux[name][slots.long()] = win[rows, (counts - 1).clamp(min=0)]
        self.spec.proposer.feedback_device(vst["features"], meta[:, 1], meta[:, 2], counts, s)
        st["packet"].copy_(torch.cat([st["y"], z, accepted.unsqueeze(1), counts.unsqueeze(1), flags.to(torch.int64)], dim=1))

    def capture(self, log=None) -> None:
        for b in self.vg.sizes:
            st = self._static(b)
            self._st[b] = st
            for _ in range(2):
                self._body(b, st)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self._pool):
                self._body(b, st)
            self._graphs[b] = g
            torch.cuda.synchronize()
        if log:
            log(f"  captured device accept graphs: {len(self._graphs)}")

    def round(self, batch, y: torch.Tensor, topk, t0: float, t1: float) -> None:
        """Replace SpecDecoder._accept for an eligible batch: replay, one wait, publish."""
        spec, ex = self.spec, self.ex
        reqs = batch.requests
        b = len(reqs)
        size = self.vg.pick(b)
        st = self._st[size]
        k, s = self.k, self.s
        # Authorization and validation before any in-place write, as the host path does.
        ex.state.authorize_reconstruction(reqs, spec.replay, spec.conv_windows_all, verified_steps=s)
        starts, ring_slots = spec.proposer.accepted_prepare(reqs)
        any_relaxed = any(r.params.relaxed for r in reqs)
        host_meta, host_relax = st["host_meta"], st["host_relax"]
        host_meta.fill_(-1)
        host_relax.zero_()
        for i, r in enumerate(reqs):
            host_meta[i, 0] = r.params.max_new_tokens - len(r.output_ids)
            host_meta[i, 1] = starts[i]
            host_meta[i, 2] = ring_slots[i]
            if not r.params.ignore_eos:
                for m, tok in enumerate(r.params.stop_token_ids):
                    host_meta[i, 3 + m] = tok
            if r.params.relaxed:
                host_relax[0, i], host_relax[1, i] = float(r.params.accept_top_k), float(r.params.accept_margin)
            elif any_relaxed:
                host_relax[0, i] = 1.0  # the host rule's exact row inside a relaxed batch
        if size > b:
            host_meta[b:, :3] = 0
        st["meta"].copy_(host_meta, non_blocking=True)
        st["relax"].copy_(host_relax, non_blocking=True)
        st["y"][:b].copy_(y[:, :k])
        if size > b:
            st["y"][b:].zero_()
        try:
            self._graphs[size].replay()
            st["host_packet"].copy_(st["packet"], non_blocking=True)
            st["event"].record()
            t2 = time.perf_counter()
            st["event"].synchronize()
            packet = st["host_packet"][:b].tolist()
            counts = [row[k + s + 1] for row in packet]
            if any(not 1 <= c <= s for c in counts):
                raise RuntimeError(f"device acceptance produced counts {counts} outside [1, {s}]")
            ex.state.publish_reconstructed(reqs, counts)
        except BaseException as exc:
            ex.state.poison(reqs, exc)
        outputs = []
        for i, r in enumerate(reqs):
            row = packet[i]
            y_l, z_l, a, count, flags = row[:k], row[k:k + s], row[k + s], row[k + s + 1], row[k + s + 2:]
            emitted = (y_l[:a] + [z_l[a]])[:count]
            assert len(emitted) == count and not r.finished
            outputs.append(emitted)
            if r.trace_rounds is not None:
                record = {"a": a, "y": y_l, "z": z_l, "c": z_l[a]}
                used = sum(flags)
                if used:
                    record["rx"] = [j for j, f in enumerate(flags[:count]) if f]
                r.trace_rounds.append(record)
            for tok in emitted:
                r.push(int(tok))
            spec.proposed += spec.k
            spec.accepted += a
            spec.emitted += count
            relaxed_here = min(sum(flags), count)
            spec.relaxed_accepts += relaxed_here
            r.spec_relaxed += relaxed_here
            r.spec_rounds += 1
            r.spec_proposed += spec.k
            r.spec_accepted += a
            r.spec_emitted += count
        spec.proposer.accepted_publish(reqs, outputs)
        for i, (r, emitted) in enumerate(zip(reqs, outputs)):
            if r.trace_features is not None:
                spec._trace_features(r, r.committed_len - len(emitted), r.committed_len)
            if r.trace_topk is not None and topk is not None:
                count = len(emitted)
                r.trace_topk.append(topk[0][i, :count], topk[1][i, :count], r.committed_len - count)
        spec.rounds += 1
        t3 = time.perf_counter()
        spec.draft_seconds += t1 - t0
        spec.verify_seconds += t2 - t1
        spec.accept_seconds += t3 - t2
