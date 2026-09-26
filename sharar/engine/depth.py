"""Draft depth policies (#45): how many drafted tokens a round verifies.

`SpecDecoder` asks its policy once per round; the answer must be one of the
depths whose verify graphs were captured (`SpecDecoder.verify_depths`).
`FixedDepth` is the served behavior: the proposer's k every round. Adaptive
depth (#43) plugs in here.
"""

from __future__ import annotations


class FixedDepth:
    def __init__(self, depth: int):
        if type(depth) is not int or depth < 1:
            raise ValueError("draft depth must be a positive integer")
        self.depth = depth

    def choose(self, requests) -> int:
        return self.depth
