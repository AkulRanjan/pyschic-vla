"""HybridJudge: priors from one judge, values from another. Owner: Person 2 (Jagriti).

Needed for: prior-only / value-only ablations, PARTIAL/NO-GO (Jev prior + PRM value), B5.
"""

from __future__ import annotations

from thoughtzero.types import Judge


class HybridJudge:
    def __init__(self, prior_from: Judge, value_from: Judge) -> None:
        self.prior_from = prior_from
        self.value_from = value_from

    async def prior_and_value(
        self, problem: str, steps: list[str], candidates: list[str]
    ) -> tuple[list[float], float]:
        raise NotImplementedError("Person 2 (Jagriti)")

    async def final_correct(self, problem: str, steps: list[str]) -> float:
        raise NotImplementedError("Person 2 (Jagriti)")

    async def step_sound(self, problem: str, steps: list[str]) -> float:
        raise NotImplementedError("Person 2 (Jagriti)")
