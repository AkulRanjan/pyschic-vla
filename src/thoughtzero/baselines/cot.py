"""B1 greedy chain-of-thought (also C: 31B ceiling via another endpoint) (SPEC.md §8.1).
Owner: Person 3 (Akul).

Implements ``eval.runner.Method``. C is the same class with a generator pointing at a
different ``base_url``/``model``; SPEC.md §8.1 requires recording where C ran, so pass
``where`` (provider, quantization) and it goes into ``MethodResult.raw``.
"""

from __future__ import annotations

from thoughtzero.data.grading import extract_answer
from thoughtzero.eval.runner import MethodResult
from thoughtzero.types import Generator, Problem


class ChainOfThought:
    name = "cot"

    def __init__(self, generator: Generator, where: str | None = None, name: str = "cot") -> None:
        self.generator = generator
        self.where = where
        self.name = name  # e.g. "ceiling_31b" for baseline C

    async def solve(self, problem: Problem) -> MethodResult:
        """One greedy (temperature 0) full solution; the answer is its last ``\\boxed{}``."""
        (solution,) = await self.generator.sample_solutions(problem.question, n=1, temperature=0.0)
        raw: dict[str, object] = {"solution": solution}
        if self.where:
            raw["where"] = self.where
        return MethodResult(answer=extract_answer(solution), raw=raw)
