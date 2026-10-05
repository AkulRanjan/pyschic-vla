"""HybridJudge: priors from one judge, values from another. Owner: Person 2 (Jagriti).

Needed for: prior-only / value-only ablations, PARTIAL/NO-GO (Jev prior + PRM value), B5.
Runs the two sub-calls concurrently, and prefers each sub-judge's
prior_only/value_only (if it has one) over its full prior_and_value, so e.g.
a JevJudge used as prior_from isn't charged for a `sound` answer it's about
to discard.
"""

from __future__ import annotations

import asyncio

from thoughtzero.types import Judge


class HybridJudge:
    def __init__(self, prior_from: Judge, value_from: Judge) -> None:
        self.prior_from = prior_from
        self.value_from = value_from

    async def prior_and_value(
        self, problem: str, steps: list[str], candidates: list[str]
    ) -> tuple[list[float], float]:
        priors, value = await asyncio.gather(
            self._priors(problem, steps, candidates), self._value(problem, steps)
        )
        return priors, value

    async def _priors(self, problem: str, steps: list[str], candidates: list[str]) -> list[float]:
        prior_only = getattr(self.prior_from, "prior_only", None)
        if prior_only is not None:
            priors: list[float] = await prior_only(problem, steps, candidates)
            return priors
        priors, _ = await self.prior_from.prior_and_value(problem, steps, candidates)
        return priors

    async def _value(self, problem: str, steps: list[str]) -> float:
        value_only = getattr(self.value_from, "value_only", None)
        if value_only is not None:
            result: float = await value_only(problem, steps)
            return result
        return await self.value_from.step_sound(problem, steps)

    async def final_correct(self, problem: str, steps: list[str]) -> float:
        return await self.value_from.final_correct(problem, steps)

    async def step_sound(self, problem: str, steps: list[str]) -> float:
        return await self._value(problem, steps)
