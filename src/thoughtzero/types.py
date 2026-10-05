"""The shared contract every module codes against (team/<Name>.md §B4).

Owner: Person 1 (Prakhar). Changing anything here needs a PR tagged to all four.

Conventions:
- A *step* is plain text WITHOUT the ``Step n:`` prefix, stripped. Numbering is added only by
  ``llm.prompts.format_steps``.
- ``complete`` / ``sample_completions`` return only the NEW steps, never the given prefix.
- Judge priors have ``len == len(candidates)`` and sum to 1; values are floats in [0, 1].
- Ground truth (``Problem.answer``) is used for grading only and is never passed into search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


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


@runtime_checkable
class Generator(Protocol):
    async def propose(self, problem: str, steps: list[str], k: int) -> list[GenOut]:
        """k candidate next steps for the state ``(problem, steps)``."""
        ...

    async def complete(self, problem: str, steps: list[str], temperature: float = 0.0) -> list[str]:
        """Continue the solution to the end; returns the new steps only."""
        ...

    async def sample_solutions(self, problem: str, n: int, temperature: float) -> list[str]:
        """n full raw solution texts from an empty prefix."""
        ...

    async def sample_completions(
        self, problem: str, steps: list[str], n: int, temperature: float
    ) -> list[list[str]]:
        """n independent continuations of ``steps``; each is a list of new steps.

        Addition to SPEC.md §5.2 (needed by the pilot's Monte Carlo labels).
        """
        ...


@runtime_checkable
class Judge(Protocol):
    async def prior_and_value(
        self, problem: str, steps: list[str], candidates: list[str]
    ) -> tuple[list[float], float]:
        """(priors over candidates, V(state)). One call per expansion."""
        ...

    async def final_correct(self, problem: str, steps: list[str]) -> float:
        """P(final answer is correct) for a terminal state."""
        ...

    async def step_sound(self, problem: str, steps: list[str]) -> float:
        """P(every step so far is correct). Used by the pilot."""
        ...
