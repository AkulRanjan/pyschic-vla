"""GemmaSelfJudge via next-token logprobs (baseline B4, SPEC.md §5.3). Owner: Person 2 (Jagriti)."""

from __future__ import annotations

from thoughtzero.config import JudgeCfg


class GemmaSelfJudge:
    def __init__(self, cfg: JudgeCfg) -> None:
        self.cfg = cfg

    async def prior_and_value(
        self, problem: str, steps: list[str], candidates: list[str]
    ) -> tuple[list[float], float]:
        raise NotImplementedError("Person 2 (Jagriti)")

    async def final_correct(self, problem: str, steps: list[str]) -> float:
        raise NotImplementedError("Person 2 (Jagriti)")

    async def step_sound(self, problem: str, steps: list[str]) -> float:
        raise NotImplementedError("Person 2 (Jagriti)")
