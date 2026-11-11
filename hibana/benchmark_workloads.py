"""Versioned, fixed-work prompts matching the recorded GB10 10K/1K tasks.

Only the first request of each historical task is used (concurrency one).
Tokenization happens before timing; materialized IDs are saved with every run.
"""
from importlib.resources import files

WORKLOADS = ('code', 'design', 'reasoning')
LEGACY_WORKLOADS = ('code_v1',)  # the tool-call code prompt: the model answers with a tool call and stops at output 98
VERSION = 'gb10-coding-v2'

CODE_V1_QUESTION = ('Explain the ownership and rollback invariants in this code. '
                    'Then write a complete Python implementation of a transactional '
                    'range allocator with reserve, commit, rollback and release, '
                    'with a detailed explanation and tests.')
# v2: the same task, but the model is told it has no tools, so it writes the
# answer instead of emitting a tool call and ending its turn. Probed on the
# GB10: 1,000 greedy tokens, no EOS.
CODE_QUESTION = ('Answer in one message without using tools. First, explain the ownership and rollback invariants '
                 'in this code in detail. Second, write a complete, self-contained Python module implementing a '
                 'transactional range allocator (reserve, commit, rollback, release) with docstrings. Third, write '
                 'a complete pytest module that covers every method, including failure paths. Fourth, describe how '
                 'you would extend it with reference-counted shared prefixes.')


def frozen_source():
    """Historical source fixture shared by packaged and source-tree clients."""
    return files('hibana').joinpath('benchmark_data/request_state_7aebbe6.txt').read_text()


def materialize(encode, case, length):
    if case in ('code', 'code_v1', 'code_v2'):
        prefix = '<|im_start|>user\nReview task 0. Here is historical inference-engine source:\n'
        body = frozen_source()
        question = CODE_V1_QUESTION if case == 'code_v1' else CODE_QUESTION
    else:
        prefix = '<|im_start|>user\nIndependent review 0. Reference notes:\n'
        if case == 'design':
            body = '\n'.join(f'Note {i}: Requests own mutable state; cached prefixes are immutable. '
                             'A cancelled request must release its private reservations. '
                             'Cache eviction must wait for active readers. Optional caches may be reclaimed '
                             'under memory pressure. Streaming clients need bounded queues and orderly shutdown.'
                             for i in range(400))
            question = ('Write a detailed design for a local inference service with cancellation, prefix reuse, '
                        'memory admission and streaming backpressure. Explain failure cases, concrete state '
                        'transitions and tests. Include pseudocode for the ownership protocol.')
        elif case == 'reasoning':
            body = '\n'.join(f'Experiment {i}: workload A has 400 ms setup and 12 ms per operation. '
                             'Workload B has 700 ms setup and 8 ms per operation. Workers share a '
                             'single memory controller; parallel workers do not multiply its bandwidth. '
                             'A cached result avoids setup only when all input identifiers match.'
                             for i in range(400))
            question = ('Derive the break-even operation count for A and B, compute their costs for 50, 100 '
                        'and 1000 operations, and develop a detailed model that includes shared bandwidth, '
                        'cache hits and queueing. Explain the assumptions and give worked examples.')
        else:
            raise ValueError(f'Unknown workload: {case}')
    prefix_ids = encode(prefix)
    body_ids = encode(body)
    suffix_ids = encode('\n' + question + '\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
    size = length - len(prefix_ids) - len(suffix_ids)
    if size <= 0 or not body_ids:
        raise ValueError('Input token budget is too small for the workload envelope')
    return prefix_ids + (body_ids * ((size + len(body_ids) - 1) // len(body_ids)))[:size] + suffix_ids
