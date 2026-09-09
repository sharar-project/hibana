"""Full-generation candidate captures and CPU-only fixed-history diagnostics.

Future output IDs are used only for evaluation/training labels. They never
enter serving. Oracle support is not an achievable policy or measured TPS.
"""
from collections import Counter
from contextlib import AbstractContextManager


def prefix_matches(left, right):
    return next((i for i, (a, b) in enumerate(zip(left, right)) if a != b),
                min(len(left), len(right)))


def walk(candidates, scores):
    previous, result = 0, []
    for choices, lattice in zip(candidates, scores):
        values = lattice[previous]
        previous = max(range(len(values)), key=values.__getitem__)
        result.append(choices[previous])
    return result


def summarize_capture(rows, output_ids, depth):
    complete = []
    for row in rows:
        offset = row['output_offset']
        future = output_ids[offset:offset + depth + 1]
        # Exclude terminal windows with insufficient future for a bonus token.
        # Their short output budget must not lower the estimated support ceiling.
        row['right_censored'] = len(future) < depth + 1
        row['native_prefix'] = prefix_matches(row['proposal_ids'], future)
        support = 0
        for choices, token in zip(row['candidates'], future):
            if token not in choices:
                break
            support += 1
        row['support_prefix'] = support
        row['native_matches_recorded_emit'] = (
            row['emitted_ids'] == future[:len(row['emitted_ids'])]
            and min(row['native_prefix'] + 1, len(output_ids) - offset) == len(row['emitted_ids']))
        if not row['native_matches_recorded_emit']:
            raise ValueError('candidate capture does not reproduce native emitted prefix')
        if not row['right_censored']:
            complete.append(row)
    def summary(selected):
        n = len(selected)
        missing = sum(r['support_prefix'] == r['native_prefix'] < depth for r in selected)
        rejected = sum(r['native_prefix'] < depth for r in selected)
        return {'rounds': n,
                'mean_native_emitted': sum(r['native_prefix'] + 1 for r in selected) / n if n else None,
                'mean_oracle_emitted': sum(r['support_prefix'] + 1 for r in selected) / n if n else None,
                'native_emitted_histogram': dict(sorted(Counter(r['native_prefix'] + 1 for r in selected).items())),
                'first_rejection_missing_support': missing, 'rejected_rounds': rejected}
    return {'all': summary(complete), 'censored_rounds': len(rows) - len(complete),
            'thirds': {name: summary([r for r in complete if low <= r['output_offset'] / len(output_ids) < high])
                       for name, low, high in [('early', 0, 1/3), ('middle', 1/3, 2/3), ('late', 2/3, 1.01)]}}


class CandidateCapture(AbstractContextManager):
    """B1 diagnostic: clone graph outputs on stream; download after generation."""
    def __init__(self, ex, req):
        self.ex, self.req, self.rows = ex, req, []

    def __enter__(self):
        self.graphs = self.ex.spec.proposer.graphs
        self.original_graph = self.graphs.run
        self.original_prepare = self.ex.spec.prepare
        def graph(*args, **kwargs):
            result = self.original_graph(*args, **kwargs)
            hidden, candidates, scores, _ = result
            self.rows.append({'output_offset': len(self.req.output_ids),
                              'candidates': candidates[0].clone(), 'scores': scores[0].clone()})
            return result
        def prepare(*args, **kwargs):
            batch = self.original_prepare(*args, **kwargs)
            self.rows[-1]['proposal_ids'] = batch.proposal.tokens[0].clone()
            return batch
        self.graphs.run, self.ex.spec.prepare = graph, prepare
        return self

    def __exit__(self, *exc):
        self.graphs.run, self.ex.spec.prepare = self.original_graph, self.original_prepare

    def export(self):
        result = []
        for i, row in enumerate(self.rows):
            out = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in row.items()}
            end = self.rows[i+1]['output_offset'] if i+1 < len(self.rows) else len(self.req.output_ids)
            out['emitted_ids'] = self.req.output_ids[out['output_offset']:end]
            result.append(out)
        return result
