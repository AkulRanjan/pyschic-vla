"""Throwaway script: record real Jev-shaped fixtures from the local
open-jev stand-in, through the actual JevClient (not the raw
typed_decisions list API), so fixtures match what JevJudge really sees.

Usage:
    source .venv/bin/activate
    python scripts/probe_open_jev.py
"""

import asyncio
import json
from pathlib import Path

from thoughtzero.config import JudgeCfg
from thoughtzero.judge.client import JevClient

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "jev"

STATE = "PROBLEM:\nWhat is 2 + 2?\n\nSOLUTION SO FAR:\nStep 1: 2 + 2 = 4"

SOUND_Q = {"sound": {"type": "noul", "instructions": "Is every step correct?"}}

CHOICE_Q_4 = {
    "next": {
        "type": "choice",
        "instructions": "Which candidate is most likely correct?",
        "options": {
            "c0": "Step 2: The answer is 4.",
            "c1": "Step 2: 4 is even, so divide by 2 to get 2.",
            "c2": "Step 2: Check: 2 + 2 = 4, confirmed.",
            "c3": "Step 2: The final answer is 5.",
        },
    }
}

CHOICE_Q_1 = {
    "next": {
        "type": "choice",
        "instructions": "Which candidate is most likely correct?",
        "options": {"c0": "Step 2: The answer is 4."},
    }
}


def save(name: str, obj) -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIXTURES_DIR / f"{name}.json"
    out_path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))
    print(f"saved {out_path}")


async def main() -> None:
    client = JevClient(JudgeCfg(transport="local_stub"))

    save("noul_basic", await client.ask(STATE, SOUND_Q))
    save("choice_4_options", await client.ask(STATE, CHOICE_Q_4))
    save("choice_and_noul_together", await client.ask(STATE, {**SOUND_Q, **CHOICE_Q_4}))

    try:
        save("choice_1_option", await client.ask(STATE, CHOICE_Q_1))
    except Exception as e:
        save("choice_1_option_rejected", {"error_type": type(e).__name__, "error_message": str(e)})

    try:
        bad_q = {"bad": {"type": "not_a_real_type", "instructions": "x"}}
        save("error_bad_question_type", await client.ask(STATE, bad_q))
    except Exception as e:
        save("error_bad_question_type", {"error_type": type(e).__name__, "error_message": str(e)})


if __name__ == "__main__":
    asyncio.run(main())
