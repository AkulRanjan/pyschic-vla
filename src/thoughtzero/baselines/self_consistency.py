"""B2 majority vote over N samples; PRIMARY baseline (SPEC.md §8.1). Owner: Person 3 (Akul).

Implements ``eval.runner.Method``.
"""

from __future__ import annotations

from thoughtzero.eval.runner import MethodResult
from thoughtzero.types import Problem


class SelfConsistency:
    name = "self_consistency"

    async def solve(self, problem: Problem) -> MethodResult:
        raise NotImplementedError("Person 3 (Akul)")
