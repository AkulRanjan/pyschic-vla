"""Shared interfaces (spec B4). Owned by Person 1 on the real team repo;
this is a local copy of the exact contract so judge/ can be developed
against something real while this scaffold is disconnected from origin.
Replace with the real file once reconciled.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Problem:
    id: str
    question: str
    answer: str
    source: str
    level: int | None = None
    subject: str | None = None


@dataclass
class GenOut:
    text: str
    prompt_tokens: int
    completion_tokens: int


class Generator(Protocol):
    async def propose(self, problem: str, steps: list[str], k: int) -> list[GenOut]: ...
    async def complete(
        self, problem: str, steps: list[str], temperature: float = 0.0
    ) -> list[str]: ...
    async def sample_solutions(self, problem: str, n: int, temperature: float) -> list[str]: ...
    async def sample_completions(
        self, problem: str, steps: list[str], n: int, temperature: float
    ) -> list[list[str]]: ...


class Judge(Protocol):
    async def prior_and_value(
        self, problem: str, steps: list[str], candidates: list[str]
    ) -> tuple[list[float], float]: ...
    async def final_correct(self, problem: str, steps: list[str]) -> float: ...
    async def step_sound(self, problem: str, steps: list[str]) -> float: ...
