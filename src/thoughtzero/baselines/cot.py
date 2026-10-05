"""B1 greedy chain-of-thought (also C: 31B ceiling via another endpoint) (SPEC.md §8.1).
Owner: Person 3 (Akul).

Implements ``eval.runner.Method``.
"""

from __future__ import annotations

from thoughtzero.eval.runner import MethodResult
from thoughtzero.types import Problem


class ChainOfThought:
    name = "cot"

    async def solve(self, problem: Problem) -> MethodResult:
        raise NotImplementedError("Person 3 (Akul)")
