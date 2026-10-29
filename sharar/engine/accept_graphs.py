"""Device-resident acceptance for all-greedy verify batches.

After the verify graph, the host path downloads the argmax rows, decides
acceptance on the host and only then launches the accepted-state
reconstruction, the convolution copies and the draft feedback, one small
launch at a time; the GPU idles while the host works. Here one captured
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
reconstruct nothing into the dummy recurrent slot, leave its convolution
window as it was and write the dummy ring.

Sampled batches of point-mass drafts (MTP) replay two graphs per bucket: a decision graph (the vocabulary
top-k and kernels/sampled_verify.verify_candidates with Philox draws, which write the sampled target row: the drafts
up to the accepted count, then the correction) and the commit graph above on that row, exact rule only. Between
them the host reads whether every row's law fit its candidates; if one did not, nothing has been written yet and
the round takes the host path.
"""

from __future__ import annotations

import time

import torch

from sharar.engine import graph_set
from sharar.engine.accept import MAX_STOPS, accept_block

def enabled() -> bool:
    """SHARAR_DEVICE_ACCEPT."""
    from sharar.engine.config import current
    return current().device_accept  # opt-in: exact, +2% at batch one, no serving gain at batch four yet (roadmap item 3)
META = 3 + MAX_STOPS  # budget, start, ring slot, stop ids
SAMPLED_CANDIDATES, SAMPLED_TOP_K = 128, 64  # sampled rows: the vocabulary top-k the decision graph keeps, the top_k it serves


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
        self._decide: dict[int, torch.cuda.CUDAGraph] = {}  # sampled batches: the decision, then the commit graph
        self._commit: dict[int, torch.cuda.CUDAGraph] = {}
        self.sampled = spec.device.type == "cuda" and spec.proposer.requirements.name == "mtp"  # point-mass drafts

    @staticmethod
    def supported(spec) -> bool:
        p = spec.proposer
        return spec.replay is not None and spec.device.type == "cuda" and p.requirements.capabilities.device_feedback

    def eligible(self, requests) -> bool:
        return all(r.params.greedy and (r.params.ignore_eos or len(r.params.stop_token_ids) <= MAX_STOPS) for r in requests)

    def eligible_sampled(self, requests, proposal) -> bool:
        return (self.sampled and proposal.probabilities is None and proposal.support is None
                and all(not r.params.greedy and 0 < r.params.top_k <= SAMPLED_TOP_K and r.params.constraint is None
                        and (r.params.ignore_eos or len(r.params.stop_token_ids) <= MAX_STOPS) for r in requests))

    def _static(self, b: int) -> dict:
        d, k, s = self.device, self.k, self.s
        pin = d.type == "cuda"
        meta = torch.full((b, META), -1, dtype=torch.int64, device=d)
        # rows as a padded row reads them (budget, start and ring slot 0, no stop ids) until a round fills them: the
        # capture's eager warm-up runs the feedback on them, and a start or ring slot of -1 scatters the draft's K/V
        # before the start of its cache
        meta[:, :3] = 0
        return {
            "y": torch.zeros((b, k), dtype=torch.int64, device=d),
            "relax": torch.zeros((2, b), dtype=torch.float32, device=d),
            "meta": meta,
            "host_relax": torch.zeros((2, b), dtype=torch.float32, pin_memory=pin),
            "host_meta": torch.full((b, META), -1, dtype=torch.int64, pin_memory=pin),
            # y [k] | z [s] | accepted | count | flags [k]
            "packet": torch.zeros((b, 2 * k + s + 2), dtype=torch.int64, device=d),
            "host_packet": torch.zeros((b, 2 * k + s + 2), dtype=torch.int64, pin_memory=pin),
            "event": torch.cuda.Event(),
            # sampled batches: per-row sampling parameters and Philox (seed, offset), the decision's outputs
            "params": torch.zeros((b, 8), dtype=torch.float32, device=d),
            "rng": torch.zeros((b, 2), dtype=torch.int64, device=d),
            "host_params": torch.zeros((b, 8), dtype=torch.float32, pin_memory=pin),
            "host_rng": torch.zeros((b, 2), dtype=torch.int64, pin_memory=pin),
            "z_s": torch.zeros((b, s), dtype=torch.int64, device=d),
            "flags_s": torch.zeros((b, k), dtype=torch.int64, device=d),
            "fit": torch.zeros(b, dtype=torch.int32, device=d),
            "host_fit": torch.zeros(b, dtype=torch.int32, pin_memory=pin),
            "fit_event": torch.cuda.Event(),
        }

    def _decide_body(self, b: int, st: dict) -> None:
        """Sampled rows: the decision on the candidates, written as the target row the commit graph accepts on."""
        from sharar.kernels.sampled_verify import verify_candidates
        vst = self.vg._st[b]
        s = self.s
        logits = vst["logits"][:b * s].view(b, s, -1)
        values, index = logits.topk(SAMPLED_CANDIDATES, dim=-1)
        y = st["y"]
        accepted, correction, flags, fit = verify_candidates(values, index, y, st["params"], st["rng"])
        offsets = torch.arange(s, device=self.device).unsqueeze(0)
        drafted = torch.cat([y, y[:, -1:]], dim=1)
        st["z_s"].copy_(torch.where(offsets < accepted.unsqueeze(1), drafted, correction.unsqueeze(1)))
        st["flags_s"].copy_(flags)
        st["fit"].copy_(fit)

    def _body(self, b: int, st: dict, sampled: bool = False) -> None:
        vst = self.vg._st[b]
        k, s = self.k, self.s
        meta = st["meta"]
        if sampled:  # the decision graph's target row: the exact rule recovers its accepted count
            z = st["z_s"]
            accepted, counts, _ = accept_block(st["y"], z, None, st["relax"], meta[:, 0], meta[:, 3:])
            flags = st["flags_s"].bool()
        else:
            z = vst["argmax"][:b * s].view(b, s)
            logits = vst["logits"][:b * s].view(b, s, -1)[:, :k]
            accepted, counts, flags = accept_block(st["y"], z, logits, st["relax"], meta[:, 0], meta[:, 3:])
        pool = self.ex.state_pool
        slots = vst["slots"][:b]
        self.spec.replay.run(pool, slots, counts.to(torch.int32))
        rows = torch.arange(b, device=self.device)
        # padded rows (count 0) share the dummy slot: they write back what it holds, as the host path leaves it,
        # so the padded verify rows that read it do not change with what earlier rounds padded (Flash-Next's
        # per-tensor activation scale carries the padded rows into the real ones)
        live, last, at = counts > 0, (counts - 1).clamp(min=0), slots.long()
        conv = self.spec.conv_windows_all[:, rows, last]
        pool.conv_all[:, at] = torch.where(live.view(1, b, *[1] * (conv.dim() - 2)), conv, pool.conv_all[:, at])
        for name, win in self.spec.aux_windows.items():
            aux = win[rows, last]
            pool.aux[name][at] = torch.where(live.view(b, *[1] * (aux.dim() - 1)), aux, pool.aux[name][at])
        n = b * s  # the verify rows and inputs, for a proposer fed the final-normalized rows (MTP)
        self.spec.proposer.feedback_device(vst["features"], meta[:, 1], meta[:, 2], counts, s,
                                           norm=vst["norm"][:n], tokens=vst["tokens"][:n], pages=vst["pages"][:n])
        st["packet"].copy_(torch.cat([st["y"], z, accepted.unsqueeze(1), counts.unsqueeze(1), flags.to(torch.int64)], dim=1))

    def capture(self, log=None) -> None:
        for b in self.vg.sizes:
            st = self._static(b)
            self._st[b] = st
            self._graphs[b] = graph_set.capture(self._pool, lambda: self._body(b, st))
            if self.sampled:
                st["params"][:, :3] = 1.0  # a valid law for every row while capturing (temperature, top_k, top_p)
                self._decide[b] = graph_set.capture(self._pool, lambda: self._decide_body(b, st))
                self._commit[b] = graph_set.capture(self._pool, lambda: self._body(b, st, sampled=True))
        if log:
            log(f"  captured device accept graphs: {len(self._graphs)}" + (f" (+{len(self._decide)} sampled)" if self.sampled else ""))

    def round(self, batch, y: torch.Tensor, t0: float, t1: float) -> bool:
        """Replace SpecDecoder._accept for an eligible batch: replay, one wait, publish. A sampled batch first
        replays its decision graph and reads whether every row fit its candidates; False (nothing written yet)
        sends the round to the host path."""
        spec, ex = self.spec, self.ex
        reqs = batch.requests
        b = len(reqs)
        size = self.vg.pick(b)
        st = self._st[size]
        k, s = self.k, self.s
        sampled = batch.sampled
        if sampled:
            from sharar.engine import sampling_rng
            hp, hr = st["host_params"], st["host_rng"]
            hp.zero_()
            hp[:, :3] = 1.0  # padded rows: a valid law, never read
            hr.zero_()
            for i, (r, o) in enumerate(zip(reqs, sampling_rng.offsets(reqs, sampling_rng.VERIFY))):
                p = r.params
                hp[i, 0], hp[i, 1], hp[i, 2] = max(p.temperature, 1e-6), float(p.top_k), p.top_p
                if p.relaxed:
                    hp[i, 3], hp[i, 4] = float(p.accept_top_k), float(p.accept_margin)
                hr[i, 0], hr[i, 1] = r.rng_seed, o
            st["params"].copy_(hp, non_blocking=True)
            st["rng"].copy_(hr, non_blocking=True)
            st["y"][:b].copy_(y[:, :k])
            if size > b:
                st["y"][b:].zero_()
            self._decide[size].replay()
            st["host_fit"].copy_(st["fit"], non_blocking=True)
            st["fit_event"].record()
            st["fit_event"].synchronize()
            if not bool(st["host_fit"][:b].all()):
                return False  # a law wider than the candidates: the host path, nothing written
        observe = getattr(spec.depth_policy, "observe", None)  # adaptive depth learns each request's rate
        # Authorization and validation before any in-place write, as the host path does.
        ex.state.authorize_reconstruction(reqs, spec.replay, spec.conv_windows_all, verified_steps=s)
        starts, ring_slots = spec.proposer.accepted_prepare(reqs)
        any_relaxed = not sampled and any(r.params.relaxed for r in reqs)
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
        if not sampled:
            st["y"][:b].copy_(y[:, :k])
            if size > b:
                st["y"][b:].zero_()
        try:
            (self._commit if sampled else self._graphs)[size].replay()
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
        conf = spec._confidence_rows(reqs, batch.proposal.confidence)
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
                if conf is not None:
                    record["p1"] = [round(c[0], 5) for c in conf[i]]
                    record["m"] = [round(c[1], 5) for c in conf[i]]
                r.trace_rounds.append(record)
            for tok in emitted:
                r.push(int(tok))
            spec.proposed += k
            spec.accepted += a
            spec.emitted += count
            if observe is not None:
                observe(r, k, a)
            relaxed_here = min(sum(flags), count)
            spec.relaxed_accepts += relaxed_here
            r.spec_relaxed += relaxed_here
            r.spec_rounds += 1
            r.spec_proposed += k
            r.spec_accepted += a
            r.spec_emitted += count
        spec.proposer.accepted_publish(reqs, outputs)
        spec.rounds += 1
        t3 = time.perf_counter()
        spec.draft_seconds += t1 - t0
        spec.verify_seconds += t2 - t1
        spec.accept_seconds += t3 - t2
        return True
