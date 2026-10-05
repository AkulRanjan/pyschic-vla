"""Metrics (SPEC.md §9). Owner: Person 4 (Harjas). Pure functions only."""

from __future__ import annotations

from collections.abc import Sequence


def accuracy(correct: Sequence[bool]) -> float:
    raise NotImplementedError("Person 4 (Harjas)")


def bootstrap_ci(
    values: Sequence[float], n: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    raise NotImplementedError("Person 4 (Harjas)")


def paired_bootstrap_diff(
    a: Sequence[bool], b: Sequence[bool], n: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    raise NotImplementedError("Person 4 (Harjas)")


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    raise NotImplementedError("Person 4 (Harjas)")


def brier(probs: Sequence[float], labels: Sequence[float]) -> float:
    raise NotImplementedError("Person 4 (Harjas)")


def ece(probs: Sequence[float], labels: Sequence[float], n_bins: int = 10) -> float:
    raise NotImplementedError("Person 4 (Harjas)")
