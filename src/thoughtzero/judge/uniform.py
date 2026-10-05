"""UniformJudge and ConstantValueJudge (spec A4.5)."""


class UniformJudge:
    async def prior_and_value(
        self, problem: str, steps: list[str], candidates: list[str]
    ) -> tuple[list[float], float]:
        n = len(candidates)
        return [1.0 / n for _ in range(n)], 0.5

    async def final_correct(self, problem: str, steps: list[str]) -> float:
        return 0.5

    async def step_sound(self, problem: str, steps: list[str]) -> float:
        return 0.5


class ConstantValueJudge:
    def __init__(self, value: float = 0.5) -> None:
        self.value = value

    async def prior_and_value(
        self, problem: str, steps: list[str], candidates: list[str]
    ) -> tuple[list[float], float]:
        n = len(candidates)
        return [1.0 / n for _ in range(n)], self.value

    async def final_correct(self, problem: str, steps: list[str]) -> float:
        return self.value

    async def step_sound(self, problem: str, steps: list[str]) -> float:
        return self.value
