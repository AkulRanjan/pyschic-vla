import pytest

from thoughtzero.search.dedupe import dedupe, jaccard, merge_priors, normalize
from thoughtzero.types import GenOut


def outs(*texts: str) -> list[GenOut]:
    return [GenOut(t, 10, 5) for t in texts]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Step 3: Let  x = 2.", "let x = 2"),
        ("  step 12) Compute\n\tthe sum ", "compute the sum"),
        ("STEP 1. Factor;", "factor"),
        ("x = 2", "x = 2"),
        ("   ", ""),
    ],
)
def test_normalize(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


def test_exact_and_prefix_duplicates_merge() -> None:
    unique, mapping = dedupe(outs("x = 2", "Step 4: X =  2.", "y = 3", "x = 2"))
    assert [u.text for u in unique] == ["x = 2", "y = 3"]
    assert mapping == [0, 0, 1, 0]


def test_first_occurrence_is_kept() -> None:
    unique, _ = dedupe(outs("Step 2: A b", "a b"))
    assert unique[0].text == "Step 2: A b"


def test_empty_candidates_dropped() -> None:
    unique, mapping = dedupe(outs("", "x = 1", "Step 2:   "))
    assert [u.text for u in unique] == ["x = 1"]
    assert mapping == [-1, 0, -1]


def test_jaccard_boundary() -> None:
    base = " ".join(f"w{i}" for i in range(19))
    near = base + " extra"  # 19 / 20 = 0.95 -> merged at 0.9
    far = " ".join(f"w{i}" for i in range(9)) + " other"  # 9 / 20 -> kept
    assert jaccard(normalize(base), normalize(near)) == pytest.approx(0.95)
    unique, mapping = dedupe(outs(base, near, far))
    assert mapping == [0, 0, 1]
    unique, mapping = dedupe(outs(base, near, far), jaccard_min=None)
    assert mapping == [0, 1, 2]


def test_single_survivor() -> None:
    unique, mapping = dedupe(outs("same", "Same.", "SAME"))
    assert len(unique) == 1 and mapping == [0, 0, 0]


def test_jaccard_edge_cases() -> None:
    assert jaccard("", "") == 1.0
    assert jaccard("a b", "c d") == 0.0


def test_merge_priors_sums_and_renormalizes() -> None:
    merged = merge_priors([0.4, 0.3, 0.2, 0.1], [0, 0, 1, -1], 2)
    assert merged == pytest.approx([0.7 / 0.9, 0.2 / 0.9])
    assert merge_priors([0.0, 0.0], [0, 1], 2) == [0.5, 0.5]
    with pytest.raises(ValueError):
        merge_priors([0.5], [0, 1], 2)
