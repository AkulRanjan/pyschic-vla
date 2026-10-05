"""Grader sanity check (team file §A4.2): MATH-500 reference solutions vs their ``answer`` field.

Agreement should be ~100%; every mismatch is printed for inspection.
Free (downloads the public dataset once). Usage:

    python scripts/check_grader_math500.py
"""

from __future__ import annotations

from thoughtzero.data.datasets import MATH500_ID, _load_hf
from thoughtzero.data.grading import extract_answer, is_equivalent, normalize_answer


def main() -> None:
    rows = list(_load_hf(MATH500_ID, "test"))
    mismatches = []
    for r in rows:
        pred = extract_answer(r["solution"])
        if not is_equivalent(pred, r["answer"]):
            mismatches.append((r["unique_id"], pred, r["answer"]))

    for uid, pred, gold in mismatches:
        print(f"MISMATCH {uid}\n  extracted: {pred!r} -> {normalize_answer(pred)!r}")
        print(f"  gold:      {gold!r} -> {normalize_answer(gold)!r}")
    agree = len(rows) - len(mismatches)
    print(f"\nagreement: {agree}/{len(rows)} = {agree / len(rows):.1%}")


if __name__ == "__main__":
    main()
