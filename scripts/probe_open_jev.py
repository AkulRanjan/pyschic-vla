"""Throwaway script: smoke-test the local open-jev stand-in model.

This runs com-kotobalabs/open-jev-deberta-v3-large (Apache-2.0, self-hosted,
no API key needed) and saves the raw outputs, so we can resolve the Jev
VERIFY items against real responses instead of docs/blogs. This is a
stand-in for TypeSafe's Jev while direct/OpenRouter access is pending
(see docs/verified_apis.md).

Usage:
    source .venv/bin/activate
    python scripts/probe_open_jev.py
"""

import json
from pathlib import Path

from typed_decisions.open_jev import OpenJev

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "jev"

STATE = "PROBLEM:\nWhat is 2 + 2?\n\nSOLUTION SO FAR:\nStep 1: 2 + 2 = 4"

SOUND_Q = {
    "id": "sound",
    "type": "noul",
    "instructions": (
        "Is every step in SOLUTION SO FAR mathematically correct and "
        "logically valid? Ignore whether the solution is finished."
    ),
}

CHOICE_Q_4 = {
    "id": "next",
    "type": "choice",
    "instructions": "Which candidate next step is most likely to lead to a correct final answer?",
    "options": {
        "c0": "Step 2: The answer is 4.",
        "c1": "Step 2: 4 is even, so divide by 2 to get 2.",
        "c2": "Step 2: Check: 2 + 2 = 4, confirmed.",
        "c3": "Step 2: The final answer is 5.",
    },
}

CHOICE_Q_1 = {
    "id": "next",
    "type": "choice",
    "instructions": "Which candidate next step is most likely to lead to a correct final answer?",
    "options": {"c0": "Step 2: The answer is 4."},
}


def save(name: str, obj) -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIXTURES_DIR / f"{name}.json"
    out_path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))
    print(f"saved {out_path}")


def main() -> None:
    model = OpenJev.from_pretrained("com-kotobalabs/open-jev-deberta-v3-large")

    save("noul_basic", model.decide(STATE, [SOUND_Q]))
    save("choice_4_options", model.decide(STATE, [CHOICE_Q_4]))
    save("choice_and_noul_together", model.decide(STATE, [SOUND_Q, CHOICE_Q_4]))
    try:
        save("choice_1_option", model.decide(STATE, [CHOICE_Q_1]))
    except Exception as e:
        save("choice_1_option_rejected", {"error_type": type(e).__name__, "error_message": str(e)})

    try:
        bad_q = {"id": "bad", "type": "not_a_real_type", "instructions": "x"}
        save("error_bad_question_type", model.decide(STATE, [bad_q]))
    except Exception as e:
        save("error_bad_question_type", {"error_type": type(e).__name__, "error_message": str(e)})


if __name__ == "__main__":
    main()
