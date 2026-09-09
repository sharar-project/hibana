"""Measure speculative proposal agreement on a reference engine's histories.

The reference JSON contains input_ids, output_ids, and verification rounds
(position, emitted, target_top1). It must come from a greedy, uncached request
with the same target/draft checkpoint revisions and sixteen verification rows.

This is teacher forcing, not ordinary generation: Sharar computes its own
target state/features but commits the reference continuation. Timings from
this diagnostic must never be reported as generation performance.
"""
import argparse
import hashlib
import json
from pathlib import Path


def prefix_matches(proposed, continuation):
    return next((i for i, (a, b) in enumerate(zip(proposed, continuation)) if a != b),
                min(len(proposed), len(continuation)))


def replay_candidates(target_top1, count):
    """Make the reference accept exactly count-1 proposals plus a correction.

    A fixed padding token can accidentally match the reference at rejection.
    Force a mismatch there; later filler cannot affect the committed prefix.
    """
    candidates = target_top1[:count-1]
    if count < 16:
        rejected = 0 if target_top1[count-1] != 0 else 1
        candidates += [rejected] + [0] * (15-count)
    return candidates


def validate_reference(data):
    prompt, output, rounds = data['input_ids'], data['output_ids'], data['rounds']
    if not prompt or not output or not rounds or data['verify_width'] != 16:
        raise ValueError('expected a nonempty sixteen-row reference')
    tape, position = [output[0]], len(prompt)
    for row in rounds:
        emitted, target = row['emitted'], row['target_top1']
        if (row['position'] != position or not 1 <= len(emitted) <= 16
                or len(target) != 16 or emitted != target[:len(emitted)]):
            raise ValueError('inconsistent reference verification boundary')
        position += len(emitted)
        tape.extend(emitted)
    if tape[:len(output)] != output:
        raise ValueError('reference trace does not reproduce its output IDs')
    return tape


def main():
    import torch
    from bench.gb10_optimize import configure_target, prepare, release
    from sharar import config
    from sharar.engine.executor import Executor
    from sharar.engine.rejection import Proposal
    from sharar.model.qwen38 import Qwen38
    from sharar.proposers.parallel import PINS

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--rounds', type=int, default=128)
    ap.add_argument('--draft-precision', choices=['bf16', 'fp8'], default='bf16')
    ap.add_argument('--target-residual-sum', choices=['native', 'bf16', 'fp32'], default='native')
    args = ap.parse_args()
    raw = args.reference.read_bytes()
    data = json.loads(raw)
    tape = validate_reference(data)
    if (data['target_revision'] != config.MODEL_REVISION
            or data['draft_revision'] != PINS['dflash2'][1]):
        raise ValueError('reference checkpoint revisions do not match Sharar')
    if not 1 <= args.rounds <= len(data['rounds']):
        raise ValueError('requested round count is outside the reference trace')
    prompt_length = len(data['input_ids'])
    selected = data['rounds'][:args.rounds]
    if selected[-1]['position'] - prompt_length + 16 > len(tape):
        raise ValueError('reference continuation does not cover all proposal slots')
    if args.output.exists():
        raise FileExistsError(args.output)
    report = {'completed': False, 'teacher_forced': True, 'reference_sha256': hashlib.sha256(raw).hexdigest(),
              'draft_precision': args.draft_precision, 'target_residual_sum': args.target_residual_sum,
              'target_revision': config.MODEL_REVISION,
              'draft_revision': PINS['dflash2'][1], 'rounds': [],
              'source_sha256': {str(p.relative_to(Path(__file__).parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (Path(__file__).parents[1] / 'sharar').rglob('*.py')}}
    def save():
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    save()
    configure_target(args.target_residual_sum)
    with torch.inference_mode():
        ex = Executor(Qwen38(config.snapshot_dir()), num_pages=60000, num_slots=24, max_batch=4)
        ex.enable_spec(15, proposer='dflash2', state_strategy='reconstruct', draft_precision=args.draft_precision)
        ex.capture_graphs()
        ex.warmup()
        reqs = prepare(ex, [data['input_ids']], budget=len(tape))
        req = reqs[0]
        report['native_first_token'] = req.output_ids[0]
        report['reference_first_token'] = tape[0]
        req.output_ids[0] = tape[0]
        req.proposer_state.pending = tape[0]
        accept = ex.spec._accept_batch
        try:
            for index, row in enumerate(selected):
                position, emitted = row['position'], row['emitted']
                offset = position - prompt_length
                assert req.committed_len == position and req.output_ids[-1] == tape[offset]
                batch = ex.spec.prepare(reqs)
                proposed = batch.proposal.tokens[0].tolist()
                count = len(emitted)
                candidates = replay_candidates(row['target_top1'], count)
                batch.proposal = Proposal(torch.tensor([candidates], device=ex.device, dtype=torch.int64))
                batch.tokens = torch.tensor([tape[offset]] + candidates, device=ex.device, dtype=torch.int64)
                observed = {}
                def force_reference(batch, target, features):
                    observed['native_target_top1'] = target[0, :count].tolist()
                    observed['target_agreement'] = sum(a == b for a, b in zip(observed['native_target_top1'], emitted))
                    accept(batch, torch.tensor([row['target_top1']], device=target.device, dtype=target.dtype), features)
                ex.spec._accept_batch = force_reference
                ex.spec.verify(batch)
                ex.spec._accept_batch = accept
                assert req.output_ids[-count:] == emitted and req.committed_len == position + count
                report['rounds'].append({'index': index, 'position': position, 'proposals': proposed,
                                         'sharar_prefix': prefix_matches(proposed, tape[offset+1:offset+16]),
                                         'reference_prefix': count-1, 'emitted': emitted, **observed})
                if index % 32 == 31:
                    save()
                    print(f'replayed {index+1} reference histories', flush=True)
        finally:
            ex.spec._accept_batch = accept
            release(ex, reqs)
    n = len(report['rounds'])
    report['summary'] = {
        'histories': n,
        'sharar_prefix_matches': sum(r['sharar_prefix'] for r in report['rounds']),
        'reference_prefix_matches': sum(r['reference_prefix'] for r in report['rounds']),
        'sharar_tokens_with_correction': sum(r['sharar_prefix'] + 1 for r in report['rounds']),
        'reference_tokens_with_correction': sum(r['reference_prefix'] + 1 for r in report['rounds']),
        'native_target_matches': sum(r['target_agreement'] for r in report['rounds']),
        'native_target_positions': sum(len(r['emitted']) for r in report['rounds']),
        'better_histories': sum(r['sharar_prefix'] > r['reference_prefix'] for r in report['rounds']),
        'worse_histories': sum(r['sharar_prefix'] < r['reference_prefix'] for r in report['rounds']),
        'equal_histories': sum(r['sharar_prefix'] == r['reference_prefix'] for r in report['rounds']),
    }
    report['completed'] = True
    save()
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
