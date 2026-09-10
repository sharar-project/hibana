"""Native cached-batch measurement with explicit ownership and completion work."""
import time
from bench.support.recording import Recorder, work_summary

def batch(ex, tasks, count, *, retain=True):
    import torch
    from sharar.engine.sampler import SamplingParams
    proposer = ex.spec.proposer
    hits, fallbacks = proposer.cache_hits, proposer.cache_fallbacks
    before = ex.cache_stats()
    reqs = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    with Recorder(ex) as record:
        try:
            for i, task in enumerate(tasks):
                req = ex.admit(task['input_ids'], SamplingParams(max_new_tokens=count, ignore_eos=True))
                req.request_id = i + 1
                reqs.append(req)
            for req in reqs:
                while not req.prefill_done:
                    ex.prefill_batch([(req, min(2048, req.prefill_len - req.prefill_pos))])
            torch.cuda.synchronize()
            prefilled = time.perf_counter()
            ready = [proposer.ready(req) for req in reqs]
            while active := [r for r in reqs if not r.finished]:
                ex.decode_step(active)
            torch.cuda.synchronize()
            finished = time.perf_counter()
            outputs = [{'id': task['id'], 'output_ids': req.output_ids[:], 'cached_tokens': req.prefix_len,
                        'spec_rounds': req.spec_rounds, 'spec_accepted': req.spec_accepted,
                        'finish_reason': req.finish_reason} for task, req in zip(tasks, reqs)]
        finally:
            for req in reqs:
                req.cacheable = retain
                ex.release(req)
            torch.cuda.synchronize()
        released = time.perf_counter()
    assert all(len(r['output_ids']) == count and r['finish_reason'] == 'length' for r in outputs)
    return {'outputs': outputs, 'ready': ready, 'draft_cache_hits': proposer.cache_hits - hits,
            'draft_fallbacks': proposer.cache_fallbacks - fallbacks, 'before': before, 'after': ex.cache_stats(),
            'feature_bytes': proposer.feature_cache.bytes, 'feature_cap': proposer.feature_cache.capacity,
            'allocated_bytes': torch.cuda.memory_allocated(), 'prefill_s': prefilled - started,
            'decode_s': finished - prefilled, 'completion_s': finished - started,
            'release_s': released - finished, 'total_s': released - started,
            'decode_tps': len(tasks) * (count - 1) / (finished - prefilled) if count > 1 else None,
            'aggregate_tps': len(tasks) * count / (released - started),
            'work': work_summary(record.events), 'events': record.events}
