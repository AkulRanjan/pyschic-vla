"""Grading tests (spec §8.2: at least 30 hand-written cases)."""

from __future__ import annotations

import time

import pytest

from thoughtzero.data import grading
from thoughtzero.data.grading import (
    extract_answer,
    has_final_answer,
    is_equivalent,
    normalize_answer,
)

# --------------------------------------------------------------------------
# extract_answer
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (r"So the answer is $\boxed{42}$.", "42"),
        (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}"),
        (r"\boxed{\frac{\sqrt{3}}{2}}", r"\frac{\sqrt{3}}{2}"),
        (r"First \boxed{1}, then corrected: \boxed{2}", "2"),  # two boxes: the last
        (r"\boxed{\boxed{7}}", "7"),  # nested boxes
        (r"\boxed {5}", "5"),
        (r"\fbox{x+1}", "x+1"),
        (r"\boxed 5 is the answer", "5"),
        (r"\boxed{\{1,2\}}", r"\{1,2\}"),  # escaped braces
        (r"\boxed{\begin{pmatrix} 1 \\ 2 \end{pmatrix}}", r"\begin{pmatrix} 1 \\ 2 \end{pmatrix}"),
        ("Step 3: Final answer: 12.", "12"),
        ("**Final Answer:** $\\frac{3}{4}$", r"\frac{3}{4}"),
        ("The final answer is 9", "9"),
        ("No answer stated here.", None),
        ("", None),
        (None, None),
        (r"\boxed{}", None),  # empty box
        (r"\boxed{\frac{1}{2}", None),  # unbalanced
    ],
)
def test_extract_answer(text: str | None, expected: str | None) -> None:
    assert extract_answer(text) == expected


def test_extract_prefers_box_over_final_answer_line() -> None:
    assert extract_answer("Final answer: 3\n\nActually \\boxed{4}") == "4"


def test_has_final_answer() -> None:
    assert has_final_answer(r"So we get \boxed{3}.")
    assert has_final_answer("Final answer: 3")
    assert not has_final_answer("Next, we compute 2 + 2 = 4.")
    # Must not confuse other commands that merely start with "boxed".
    assert not has_final_answer(r"\boxedfoo")


# --------------------------------------------------------------------------
# normalize_answer
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("$5$", "5"),
        (r"\dfrac{1}{2}", r"\frac{1}{2}"),
        (r"\tfrac{1}{2}", r"\frac{1}{2}"),
        (r"\frac12", r"\frac{1}{2}"),
        (r"\left( 1, 2 \right)", "(1,2)"),
        (r"90^\circ", "90"),
        (r"90^{\circ}", "90"),
        (r"5 \text{ cm}", "5"),
        (r"12\text{ inches}^2", "12"),
        (r"x = 3", "3"),
        ("7.", "7"),
        (".5", "0.5"),
        ("0.50", "0.5"),
        ("1,000", "1000"),
        ("1{,}000", "1000"),
        ("1,2", "1,2"),  # a list, not a thousands separator
        (r"\text{Monday}", "monday"),
        ("(C)", "c"),
        (r"10\%", "10"),
        ("3/4", r"\frac{3}{4}"),
        ("-3/4", r"-\frac{3}{4}"),
        (r"\sqrt2", r"\sqrt{2}"),
        ("1.5e3", "1500"),
        (r"\!1\,000", "1000"),
        (None, ""),
    ],
)
def test_normalize_answer(raw: str | None, expected: str) -> None:
    assert normalize_answer(raw) == expected


# --------------------------------------------------------------------------
# is_equivalent
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pred", "gold"),
    [
        ("0.5", r"\frac{1}{2}"),  # fraction vs decimal
        (r"\frac12", "0.5"),
        (r"\sqrt{2}/2", r"\frac{\sqrt2}{2}"),
        (r"\pi/2", r"\frac{\pi}{2}"),  # pi expressions
        (r"2\pi", r"2 \pi"),
        (r"90^\circ", "90"),  # degrees
        (r"5 \text{ cm}", "5"),  # units
        (r"50\%", "50"),  # percentages
        ("[1,3)", "[1, 3)"),  # interval
        (r"\{1,2\}", r"\{2,1\}"),  # unordered set
        ("1, 2", "2, 1"),  # multiple answers, any order
        ("Monday", r"\text{Monday}"),  # text answers
        ("-7", "-7"),  # negative numbers
        (r"1.5 \times 10^3", "1500"),  # scientific notation
        (r"1.5 \cdot 10^{-3}", "0.0015"),
        (r"\begin{pmatrix}1\\2\end{pmatrix}", r"\begin{pmatrix} 1 \\ 2 \end{pmatrix}"),
        ("x = 4", "4"),
        ("1,000", "1000"),
        (r"\$18.90", "18.9"),
        (r"2\sqrt{3}", r"\sqrt{12}"),
        (r"(-\infty, 2]", r"(-\infty,2]"),
    ],
)
def test_equivalent(pred: str, gold: str) -> None:
    assert is_equivalent(pred, gold)


@pytest.mark.parametrize(
    ("pred", "gold"),
    [
        ("[1,3)", "[1,3]"),  # open vs closed interval
        ("3", "-3"),
        ("Monday", "Tuesday"),
        (r"\begin{pmatrix}1\\2\end{pmatrix}", r"\begin{pmatrix}2\\1\end{pmatrix}"),
        ("1", "1, 2"),  # one of two required answers
        ("(1,2)", "(2,1)"),  # ordered pair
        (r"\frac{1}{3}", "0.33"),
        (None, "5"),  # None is never equivalent
        ("5", None),
        (None, None),
        ("", "5"),
    ],
)
def test_not_equivalent(pred: str | None, gold: str | None) -> None:
    assert not is_equivalent(pred, gold)


def test_end_to_end_two_boxes_takes_last() -> None:
    text = r"Step 1: Maybe \boxed{3}. Step 2: No, it is \boxed{\frac{1}{2}}."
    assert is_equivalent(extract_answer(text), "0.5")


def test_math_verify_timeout_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    def hang(pred: str, gold: str) -> bool:
        time.sleep(5)
        return True

    monkeypatch.setattr(grading, "_math_verify", hang)
    start = time.monotonic()
    assert not is_equivalent("x", "y", timeout_s=0.2)
    assert time.monotonic() - start < 2


def test_math_verify_error_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(pred: str, gold: str) -> bool:
        raise RuntimeError("sympy exploded")

    monkeypatch.setattr(grading, "_math_verify", boom)
    assert not is_equivalent("x", "y")
