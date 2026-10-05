"""UniformJudge / ConstantValueJudge for ablations and tests. Owner: Person 2 (Jagriti)."""

from __future__ import annotations


class UniformJudge:
    """Uniform priors, constant value 0.5."""

    async def prior_and_value(
        self, problem: str, steps: list[str], candidates: list[str]
    ) -> tuple[list[float], float]:
        raise NotImplementedError("Person 2 (Jagriti)")

    async def final_correct(self, problem: str, steps: list[str]) -> float:
        raise NotImplementedError("Person 2 (Jagriti)")

    async def step_sound(self, problem: str, steps: list[str]) -> float:
        raise NotImplementedError("Person 2 (Jagriti)")


class ConstantValueJudge:
    def __init__(self, value: float = 0.5) -> None:
        self.value = value

    async def prior_and_value(
        self, problem: str, steps: list[str], candidates: list[str]
    ) -> tuple[list[float], float]:
        raise NotImplementedError("Person 2 (Jagriti)")

    async def final_correct(self, problem: str, steps: list[str]) -> float:
        raise NotImplementedError("Person 2 (Jagriti)")

    async def step_sound(self, problem: str, steps: list[str]) -> float:
        raise NotImplementedError("Person 2 (Jagriti)")
