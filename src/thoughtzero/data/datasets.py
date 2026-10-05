"""Dataset loaders and deterministic subsets (spec §8.2, team file §A4.3).

Dataset IDs and fields were checked against the HF datasets-server on
2026-10-05 (see docs/verified_apis.md). ``datasets`` is imported lazily;
tests use the ``*_from_rows`` functions with a bundled sample, not the network.

Problem IDs are stable across runs and machines:

- MATH-500:    ``math500/test/precalculus/807``  (from ``unique_id``)
- MATH train:  ``math_train/algebra/<sha1(problem)[:12]>``  (no native ID; content hash)
- AIME:        ``aime2025/I-01`` .. ``aime2025/II-15``

Subsets are chosen by sorting on ``sha256(f"{seed}:{id}")``, not by
``random``, so they are identical on every machine and Python version.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from thoughtzero.data.grading import extract_answer
from thoughtzero.types import Problem

MATH500_ID = "HuggingFaceH4/MATH-500"
MATH_TRAIN_ID = "EleutherAI/hendrycks_math"
MATH_SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)

# Gemma 4's training-data cutoff: "January 2025" per the official model card
# (ai.google.dev/gemma/docs/core/model_card_4). End of month, to be conservative.
GEMMA4_TRAINING_CUTOFF: date | None = date(2025, 1, 31)


@dataclass(frozen=True)
class AimeSource:
    hf_id: str
    split: str
    # Exam dates: AIME I and AIME II. VERIFY against the MAA calendar.
    date_i: date
    date_ii: date


AIME_SOURCES: dict[int, AimeSource] = {
    2024: AimeSource("HuggingFaceH4/aime_2024", "train", date(2024, 1, 31), date(2024, 2, 7)),
    2025: AimeSource("MathArena/aime_2025", "train", date(2025, 2, 6), date(2025, 2, 12)),
    2026: AimeSource("MathArena/aime_2026", "train", date(2026, 2, 5), date(2026, 2, 11)),
}


# --------------------------------------------------------------------------
# Row parsers (pure; tested offline)
# --------------------------------------------------------------------------


def _parse_level(level: Any) -> int | None:
    """MATH-500 has ``level`` as an int, Hendrycks MATH as ``"Level 5"`` (or ``"Level ?"``)."""
    if isinstance(level, int):
        return level
    m = re.search(r"\d+", str(level))
    return int(m.group(0)) if m else None


def math500_from_rows(rows: Iterable[Mapping[str, Any]]) -> list[Problem]:
    """Rows with fields ``problem, solution, answer, subject, level, unique_id``."""
    out = []
    for r in rows:
        uid = str(r["unique_id"]).removesuffix(".json")
        out.append(
            Problem(
                id=f"math500/{uid}",
                question=r["problem"],
                answer=r["answer"],
                source="math500",
                level=_parse_level(r["level"]),
                subject=r["subject"],
            )
        )
    return out


def math_train_from_rows(rows: Iterable[Mapping[str, Any]], subject: str) -> list[Problem]:
    """Hendrycks MATH rows (``problem, level, type, solution``); no ``answer`` field.

    The answer is the last ``\\boxed{}`` in the reference solution. Rows
    without one are skipped (they can't be graded).
    """
    out = []
    for r in rows:
        answer = extract_answer(r["solution"])
        if answer is None:
            continue
        digest = hashlib.sha1(r["problem"].encode()).hexdigest()[:12]
        out.append(
            Problem(
                id=f"math_train/{subject}/{digest}",
                question=r["problem"],
                answer=answer,
                source="math_train",
                level=_parse_level(r["level"]),
                subject=r.get("type", subject),
            )
        )
    return out


def _aime_index(row: Mapping[str, Any], position: int) -> tuple[str, int]:
    """(exam, problem number) for an AIME row, e.g. ("II", 3)."""
    url = str(row.get("url", ""))
    m = re.search(r"AIME_(I{1,2})_Problems/Problem_(\d+)", url)
    if m:
        return m.group(1), int(m.group(2))
    # MathArena: problem_idx 1-15 is AIME I, 16-30 is AIME II.
    idx = int(row.get("problem_idx", position + 1))
    return ("I", idx) if idx <= 15 else ("II", idx - 15)


def aime_from_rows(rows: Iterable[Mapping[str, Any]], year: int) -> list[Problem]:
    out = []
    for pos, r in enumerate(rows):
        exam, num = _aime_index(r, pos)
        out.append(
            Problem(
                id=f"aime{year}/{exam}-{num:02d}",
                question=r["problem"],
                answer=str(r["answer"]).strip(),
                source=f"aime{year}",
            )
        )
    return sorted(out, key=lambda p: p.id)


# --------------------------------------------------------------------------
# Loaders (network on first call; cached by `datasets` afterwards)
# --------------------------------------------------------------------------


def _load_hf(path: str, split: str, name: str | None = None, revision: str | None = None) -> Any:
    from datasets import load_dataset

    return load_dataset(path, name, split=split, revision=revision)


def load_math500(revision: str | None = None) -> list[Problem]:
    return math500_from_rows(_load_hf(MATH500_ID, "test", revision=revision))


def load_math_train(revision: str | None = None) -> list[Problem]:
    """The Hendrycks MATH train split (7,500 problems), disjoint from MATH-500."""
    out: list[Problem] = []
    for subject in MATH_SUBJECTS:
        out += math_train_from_rows(
            _load_hf(MATH_TRAIN_ID, "train", name=subject, revision=revision), subject
        )
    return out


def load_aime(year: int, revision: str | None = None) -> list[Problem]:
    if year not in AIME_SOURCES:
        raise ValueError(f"No AIME source registered for {year}; known: {sorted(AIME_SOURCES)}")
    src = AIME_SOURCES[year]
    return aime_from_rows(_load_hf(src.hf_id, src.split, revision=revision), year)


def aime_release_date(problem: Problem) -> date | None:
    """Exam date of an AIME problem, or None for non-AIME problems."""
    m = re.fullmatch(r"aime(\d{4})/(I{1,2})-\d+", problem.id)
    if not m:
        return None
    src = AIME_SOURCES[int(m.group(1))]
    return src.date_i if m.group(2) == "I" else src.date_ii


def is_post_cutoff(problem: Problem, cutoff: date | None = None) -> bool | None:
    """Whether a problem was released after Gemma 4's training cutoff (contamination check).

    None if unknown: not an AIME problem, or no cutoff configured.
    """
    cutoff = cutoff or GEMMA4_TRAINING_CUTOFF
    released = aime_release_date(problem)
    if cutoff is None or released is None:
        return None
    return released > cutoff


# --------------------------------------------------------------------------
# Deterministic subsets
# --------------------------------------------------------------------------


def _hash_key(seed: int, pid: str) -> str:
    return hashlib.sha256(f"{seed}:{pid}".encode()).hexdigest()


def sample_subset(problems: Sequence[Problem], n: int, seed: int) -> list[Problem]:
    """``n`` problems chosen by hash order; returned sorted by ID."""
    chosen = sorted(problems, key=lambda p: _hash_key(seed, p.id))[:n]
    return sorted(chosen, key=lambda p: p.id)


def stratified_subset(problems: Sequence[Problem], n: int, seed: int) -> list[Problem]:
    """``n`` problems stratified by ``level``, proportional to each level's size.

    Quotas use largest-remainder rounding, so they sum to exactly ``n``.
    Problems without a level form their own stratum. Returned sorted by ID.
    """
    if n >= len(problems):
        return sorted(problems, key=lambda p: p.id)
    strata: dict[int, list[Problem]] = defaultdict(list)
    for p in problems:
        strata[p.level if p.level is not None else -1].append(p)

    total = len(problems)
    exact = {lvl: n * len(ps) / total for lvl, ps in strata.items()}
    quota = {lvl: int(x) for lvl, x in exact.items()}
    leftover = n - sum(quota.values())
    for lvl in sorted(exact, key=lambda k: (-(exact[k] - quota[k]), k))[:leftover]:
        quota[lvl] += 1

    chosen: list[Problem] = []
    for lvl, ps in strata.items():
        chosen += sample_subset(ps, quota[lvl], seed)
    return sorted(chosen, key=lambda p: p.id)


def dev_subset(
    n: int = 50, seed: int = 0, problems: Sequence[Problem] | None = None
) -> list[Problem]:
    """The shared 50-problem MATH-500 dev set (CP3)."""
    return sample_subset(problems if problems is not None else load_math500(), n, seed)


def pilot_subset(
    n: int = 200,
    seed: int = 0,
    source: Literal["math500", "math_train"] = "math500",
    problems: Sequence[Problem] | None = None,
) -> list[Problem]:
    """The 200-problem pilot set, stratified by level (spec §7).

    ``source="math500"`` follows the spec. ``"math_train"`` is open decision
    B11.1 (avoid test leakage); switch only once the team agrees.
    """
    if problems is None:
        problems = load_math500() if source == "math500" else load_math_train()
    return stratified_subset(problems, n, seed)


def ablation_subset(
    n: int = 200, seed: int = 1, problems: Sequence[Problem] | None = None
) -> list[Problem]:
    """The week-4 ablation set from MATH-500, stratified by level."""
    return stratified_subset(problems if problems is not None else load_math500(), n, seed)
