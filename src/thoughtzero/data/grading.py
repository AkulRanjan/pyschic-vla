"""Answer extraction and equivalence (SPEC.md §8.2). Owner: Person 3 (Akul).

Everyone uses these three functions; nobody writes their own answer parsing.
"""

from __future__ import annotations


def extract_answer(text: str) -> str | None:
    """Last ``\\boxed{...}`` (nested braces), fallback ``Final answer: X``; None if absent."""
    raise NotImplementedError("Person 3 (Akul)")


def normalize_answer(answer: str) -> str:
    """Canonical string used for grouping (value_vote, self-consistency)."""
    raise NotImplementedError("Person 3 (Akul)")


def is_equivalent(pred: str | None, gold: str) -> bool:
    """math-verify with a timeout, falling back to normalized string comparison."""
    raise NotImplementedError("Person 3 (Akul)")
