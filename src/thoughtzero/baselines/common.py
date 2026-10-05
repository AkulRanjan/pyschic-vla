"""Shared output type for the non-search baselines.

Person 4's ``Method`` protocol (``eval/runner.py``) doesn't exist yet. Each
baseline returns a ``BaselineOutput`` that a thin ``Method`` adapter can wrap
once it lands. Grading is left to the runner (``grading.is_equivalent``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BaselineOutput:
    answer: str | None  # raw extracted answer (grade with grading.is_equivalent)
    solutions: list[str]  # every solution text the method generated
    extra: dict[str, Any] = field(default_factory=dict)  # method-specific details for JSONL
