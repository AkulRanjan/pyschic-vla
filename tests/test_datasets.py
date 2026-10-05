"""Dataset parsing and subset tests (offline: bundled sample, no network)."""

from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path

import pytest

from thoughtzero.data import datasets as ds
from thoughtzero.data.grading import extract_answer, is_equivalent
from thoughtzero.types import Problem

FIXTURE = Path(__file__).parent / "fixtures" / "math500_sample.json"


@pytest.fixture
def math500() -> list[Problem]:
    return ds.math500_from_rows(json.loads(FIXTURE.read_text()))


def test_math500_parsing(math500: list[Problem]) -> None:
    p = math500[0]
    assert p.id == "math500/test/algebra/100"
    assert p.source == "math500"
    assert p.level == 1 and p.subject == "Algebra"
    assert p.answer == "6"
    assert len({q.id for q in math500}) == len(math500)


def test_grader_agrees_with_reference_solutions() -> None:
    # The same check team file §A4.2 asks for on the real MATH-500.
    for r in json.loads(FIXTURE.read_text()):
        assert is_equivalent(extract_answer(r["solution"]), r["answer"])


def test_math_train_parsing() -> None:
    rows = [
        {
            "problem": "What is 1+1?",
            "level": "Level 2",
            "type": "Algebra",
            "solution": "It is $\\boxed{2}$.",
        },
        {
            "problem": "No box here.",
            "level": "Level ?",
            "type": "Algebra",
            "solution": "The answer is two.",
        },
    ]
    probs = ds.math_train_from_rows(rows, "algebra")
    assert len(probs) == 1  # ungradeable rows are skipped
    p = probs[0]
    assert p.id.startswith("math_train/algebra/") and len(p.id.split("/")[-1]) == 12
    assert (p.answer, p.level, p.source, p.subject) == ("2", 2, "math_train", "Algebra")
    # IDs come from content, so they don't depend on row order.
    assert ds.math_train_from_rows(rows[::-1], "algebra")[0].id == p.id


def test_parse_level() -> None:
    assert ds._parse_level(3) == 3
    assert ds._parse_level("Level 5") == 5
    assert ds._parse_level("Level ?") is None


def test_aime_parsing_matharena_and_h4() -> None:
    matharena = [{"problem_idx": i, "problem": f"p{i}", "answer": i * 7} for i in (1, 15, 16, 30)]
    ids = [p.id for p in ds.aime_from_rows(matharena, 2025)]
    assert ids == ["aime2025/I-01", "aime2025/I-15", "aime2025/II-01", "aime2025/II-15"]
    assert ds.aime_from_rows(matharena, 2025)[0].answer == "7"

    h4 = [
        {
            "id": 60,
            "problem": "q",
            "answer": "204",
            "year": "2024",
            "url": "https://artofproblemsolving.com/wiki/index.php/2024_AIME_II_Problems/Problem_4",
        }
    ]
    (p,) = ds.aime_from_rows(h4, 2024)
    assert p.id == "aime2024/II-04" and p.source == "aime2024"


def test_contamination_flags() -> None:
    p = Problem(id="aime2026/II-03", question="q", answer="1", source="aime2026")
    assert ds.aime_release_date(p) == ds.AIME_SOURCES[2026].date_ii
    assert ds.is_post_cutoff(p, cutoff=date(2025, 6, 1)) is True
    assert ds.is_post_cutoff(p, cutoff=date(2026, 6, 1)) is False
    other = Problem(id="math500/test/x/1", question="q", answer="1", source="math500")
    assert ds.is_post_cutoff(other, cutoff=date(2025, 6, 1)) is None


def test_load_aime_unknown_year() -> None:
    with pytest.raises(ValueError):
        ds.load_aime(1999)


def test_subsets_are_deterministic(math500: list[Problem]) -> None:
    a = ds.dev_subset(n=10, seed=0, problems=math500)
    b = ds.dev_subset(n=10, seed=0, problems=list(reversed(math500)))
    assert a == b  # independent of input order
    assert len(a) == 10 and a == sorted(a, key=lambda p: p.id)
    assert ds.dev_subset(n=10, seed=1, problems=math500) != a


def test_subset_larger_than_pool(math500: list[Problem]) -> None:
    assert len(ds.dev_subset(n=1000, problems=math500)) == len(math500)
    assert len(ds.pilot_subset(n=1000, problems=math500)) == len(math500)


def test_stratified_quotas_are_proportional() -> None:
    sizes = {1: 10, 2: 20, 3: 30, 4: 25, 5: 15}  # 100 problems
    pool = [
        Problem(id=f"x/{lvl}/{i}", question="q", answer="1", source="t", level=lvl)
        for lvl, k in sizes.items()
        for i in range(k)
    ]
    sub = ds.stratified_subset(pool, n=20, seed=0)
    assert len(sub) == 20
    assert Counter(p.level for p in sub) == {1: 2, 2: 4, 3: 6, 4: 5, 5: 3}
    assert sub == ds.pilot_subset(n=20, seed=0, problems=pool)


def test_stratified_largest_remainder_sums_to_n() -> None:
    pool = [
        Problem(id=f"y/{lvl}/{i}", question="q", answer="1", source="t", level=lvl)
        for lvl in (1, 2, 3)
        for i in range(7)
    ]
    for n in range(1, 21):
        assert len(ds.stratified_subset(pool, n=n, seed=3)) == n


def test_ablation_and_dev_use_different_seeds(math500: list[Problem]) -> None:
    dev = ds.dev_subset(n=20, problems=math500)
    abl = ds.ablation_subset(n=20, problems=math500)
    assert dev != abl
