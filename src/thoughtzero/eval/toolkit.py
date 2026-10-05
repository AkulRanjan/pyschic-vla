"""Answer extraction / grading / step splitting used by the runner and the pilot.

Owner: Person 4 (Harjas). ``real_toolkit()`` delegates to Person 3's ``data.grading`` and
``llm.prompts`` (nobody writes their own answer parsing, B4.6). ``mock_toolkit()`` matches the
toy task in ``thoughtzero.mocks`` so tests and ``--mock`` runs work offline.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Toolkit:
    extract: Callable[[str], str | None]
    normalize: Callable[[str], str]
    grade: Callable[[str | None, str], bool]
    split: Callable[[str], list[str]]
    format: Callable[[list[str]], str]


def _extract(text: str) -> str | None:
    from thoughtzero.data.grading import extract_answer

    return extract_answer(text)


def _normalize(answer: str) -> str:
    from thoughtzero.data.grading import normalize_answer

    return normalize_answer(answer)


def _grade(pred: str | None, gold: str) -> bool:
    from thoughtzero.data.grading import is_equivalent

    return is_equivalent(pred, gold)


def _split(text: str) -> list[str]:
    from thoughtzero.llm.prompts import split_steps

    return split_steps(text)


def _format(steps: list[str]) -> str:
    from thoughtzero.llm.prompts import format_steps

    return format_steps(steps)


def real_toolkit() -> Toolkit:
    return Toolkit(_extract, _normalize, _grade, _split, _format)


_STEP_RE = re.compile(r"Step\s+\d+\s*:\s*")


def _toy_split(text: str) -> list[str]:
    return [p.strip() for p in _STEP_RE.split(text) if p.strip()]


def _toy_format(steps: list[str]) -> str:
    return "".join(f"Step {i}: {s}\n\n" for i, s in enumerate(steps, start=1))


def _toy_grade(pred: str | None, gold: str) -> bool:
    return pred is not None and pred.strip() == gold.strip()


def mock_toolkit() -> Toolkit:
    from thoughtzero.mocks import toy_extract_answer

    return Toolkit(toy_extract_answer, str.strip, _toy_grade, _toy_split, _toy_format)
