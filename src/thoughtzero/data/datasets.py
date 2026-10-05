"""Dataset loaders and deterministic subsets (SPEC.md §8.2). Owner: Person 3 (Akul).

Every loader returns ``list[Problem]`` with stable ids. Subset functions must be deterministic so
all four of us evaluate on exactly the same problems.
"""

from __future__ import annotations

from thoughtzero.types import Problem


def load_math500() -> list[Problem]:
    raise NotImplementedError("Person 3 (Akul): HuggingFaceH4/MATH-500 (VERIFY id and fields)")


def load_math_train() -> list[Problem]:
    raise NotImplementedError("Person 3 (Akul): Hendrycks MATH train split (pilot / prompt tuning)")


def load_aime(year: int) -> list[Problem]:
    raise NotImplementedError("Person 3 (Akul): AIME loader; tag post-cutoff problems")


def dev_subset(n: int = 50, seed: int = 0) -> list[Problem]:
    raise NotImplementedError("Person 3 (Akul): dev subset of MATH-500 (CP3)")


def pilot_subset(n: int = 200, seed: int = 0, split: str = "test") -> list[Problem]:
    raise NotImplementedError(
        "Person 3 (Akul): level-stratified pilot subset (SPEC.md §7.1, B11.1)"
    )


def ablation_subset(n: int = 200, seed: int = 1) -> list[Problem]:
    raise NotImplementedError("Person 3 (Akul): fixed ablation subset of MATH-500 (week 4)")
