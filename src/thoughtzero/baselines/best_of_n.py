"""B3 best-of-N picked by judge.final_correct (SPEC.md §8.1). Owner: Person 3 (Akul).

Implements ``eval.runner.Method``.
"""

from __future__ import annotations

from thoughtzero.eval.runner import MethodResult
from thoughtzero.types import Problem


class BestOfN:
    name = "best_of_n"

    async def solve(self, problem: Problem) -> MethodResult:
        raise NotImplementedError("Person 3 (Akul)")
