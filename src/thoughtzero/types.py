"""Shared interface contract (team file §B4).

STAND-IN: owned by Person 1 (scaffold, CP0). This copy mirrors §B4 verbatim so
Person 3's code can be built and tested before the scaffold lands. Replace it
with the scaffold's version; do not edit here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Problem:
    id: str  # stable, e.g. "math500/test/precalculus/807"
    question: str
    answer: str  # ground truth; ONLY used for grading, never inside search
    source: str  # "math500" | "aime2025" | ...
    level: int | None = None  # MATH difficulty 1-5 (for stratification)
    subject: str | None = None


@dataclass
class GenOut:
    text: str  # step text, stripped, WITHOUT the "Step n:" prefix
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
