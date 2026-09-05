"""Select captured verify widths from measured costs and calibrated confidence.

Uses the previous round's confidence so width selection adds no GPU-to-host
barrier between drafting and verification. New requests start at full width.
Profiles are tied to the target/draft/state strategy; no shipped guessed curves.
"""
import json
import math

from sharar import config


class WidthPolicy:
    def __init__(self, profile, proposer, state_strategy):
        from sharar.proposers.parallel import PINS
        if (profile['target_revision'] != config.MODEL_REVISION
                or profile['draft_revision'] != PINS[proposer][1]
                or profile['state_strategy'] != state_strategy or profile['hardware'] != 'GB10'):
            raise ValueError('verify policy does not match the active hardware/model profile')
        self.costs = profile['round_ms']
        self.calibration = profile['calibration']
        for row in self.costs.values():
            if any(not math.isfinite(v) or v <= 0 for v in row.values()):
                raise ValueError('verify costs must be finite and positive')
        for row in self.calibration.values():
            if not row or any(not math.isfinite(v) or not 0 <= v <= 1 for v in row):
                raise ValueError('calibration must contain probabilities')

    @classmethod
    def load(cls, path, proposer, strategy):
        with open(path) as stream:
            return cls(json.load(stream), proposer, strategy)

    def choose(self, requests, depths):
        batch = next((n for n in sorted(map(int, self.costs)) if n >= len(requests)), None)
        if batch is None:
            return max(depths)
        histories = [getattr(r.proposer_state, 'confidence', None) for r in requests]
        if any(c is None for c in histories):
            return max(depths)
        scores = {}
        for depth in depths:
            cost = self.costs[str(batch)].get(str(depth))
            if cost is None:
                continue
            useful = 0.
            for req, confidence in zip(requests, histories):
                calibration = self.calibration.get('greedy' if req.params.greedy else 'sampled')
                if calibration is None:
                    return max(depths)
                remaining = req.params.max_new_tokens - len(req.output_ids)
                expected, product = 1., 1.
                for c in confidence[:min(depth, remaining - 1)]:
                    product *= calibration[min(len(calibration) - 1, int(c * len(calibration)))]
                    expected += product
                useful += expected
            scores[depth] = useful / cost
        return max(scores, key=scores.get) if scores else max(depths)
