"""Runs any Method over a dataset and writes JSONL (SPEC.md §8.5). Owner: Person 4 (Harjas).

Methods receive the Problem but must never read ``problem.answer``; grading happens here after
``solve`` returns. Runs are resumable (skip finished ids) and shardable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from thoughtzero.config import Config
from thoughtzero.types import Problem


@dataclass
class MethodResult:
    answer: str | None
    raw: dict[str, Any] = field(default_factory=dict)


class Method(Protocol):
    name: str

    async def solve(self, problem: Problem) -> MethodResult: ...


async def run_method(method: Method, problems: list[Problem], cfg: Config) -> None:
    raise NotImplementedError("Person 4 (Harjas)")
