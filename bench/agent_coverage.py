"""Offline eight-row branch screen; not a tree verifier or a speed claim.

Spend one of the seven draft nodes on one alternative to the first six chain
nodes. Select that alternative using only the draft lattice before verification.
The native target continuation reveals whether the first correction hits it.
Such a hit could provide one additional token if a correct tree verifier existed;
an eight-token native round loses one token from shortening the chain. This is
one-round accounting, not a rollout or a measured performance improvement.
"""
import math


def select_branch(candidates, scores):
    previous, prefix_mass = 0, 1.0
    alternatives = []
    for depth in range(6):
        logits = scores[depth][previous]
        order = sorted(range(len(logits)), key=lambda i: (-logits[i], i))
        best, alternate = order[:2]
        maximum = logits[best]
        weights = [math.exp(value - maximum) for value in logits]
        total = sum(weights)
        alternatives.append({'depth': depth, 'token': candidates[depth][alternate],
                             'draft_path_mass': prefix_mass * weights[alternate] / total})
        prefix_mass *= weights[best] / total
        previous = best
    return max(alternatives, key=lambda item: item['draft_path_mass'])


def score_branch(branch, emitted):
    if not 1 <= len(emitted) <= 8:
        raise ValueError('expected one native eight-row round')
    # The last emitted token is a correction/bonus. If fewer than eight were
    # emitted, its index is the first failed draft position.
    hit = len(emitted) < 8 and branch['depth'] == len(emitted) - 1 and branch['token'] == emitted[-1]
    lost = len(emitted) == 8
    return {'native_tokens': len(emitted), 'alternative_hit': hit, 'full_chain_loss': lost,
            'potential_token_delta': int(hit) - int(lost)}
