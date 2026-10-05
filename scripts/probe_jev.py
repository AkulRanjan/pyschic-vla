"""Check a Jev route with one real request and record the response as a test fixture.

    python scripts/probe_jev.py                      # route from JEV_TRANSPORT (default openrouter)
    python scripts/probe_jev.py --transport zen      # direct | openrouter | zen | vercel | bocha
    python scripts/probe_jev.py --transport direct --record

One request with a noul and a 3-option choice over a two-step maths state, a few hundred
input tokens (~$0.00002 at $0.042/Mtok). Prints the parsed answers, the reported usage and
the latency. ``--record`` saves the raw response body to
``tests/fixtures/jev/real_<transport>.json`` (the response carries no key). Never prints keys.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from thoughtzero.config import JudgeCfg
from thoughtzero.judge.client import PROVIDERS, JevClient, JevConfigError, JevHTTPError
from thoughtzero.llm.prompts import NEXT_INSTRUCTION, SOUND_INSTRUCTION, judge_state

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "jev"

STATE = judge_state(
    "What is the sum of the first 5 positive odd integers?",
    ["The first 5 positive odd integers are 1, 3, 5, 7 and 9."],
)
CANDIDATES = {
    "c0": "Their sum is 1 + 3 + 5 + 7 + 9 = 25.",
    "c1": "Their sum is 1 + 3 + 5 + 7 + 9 = 30.",
    "c2": "The answer is the 5th odd number, so it is 9.",
}
QUESTIONS = {
    "sound": {"type": "noul", "instructions": SOUND_INSTRUCTION},
    "next": {"type": "choice", "instructions": NEXT_INSTRUCTION, "criteria": CANDIDATES},
}


async def probe(transport: str, record: bool) -> int:
    try:
        client = JevClient(JudgeCfg(transport=transport))  # type: ignore[arg-type]
    except JevConfigError as e:
        print(f"not configured: {e}", file=sys.stderr)
        return 2
    print(f"route={transport}  url={client.url}  model={client.model}")
    t0 = time.monotonic()
    try:
        reply = await client.ask(STATE, QUESTIONS)
    except JevHTTPError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1
    finally:
        await client.aclose()
    ms = (time.monotonic() - t0) * 1000

    print(f"answered by model={reply.model}  input_tokens={reply.input_tokens}  {ms:.0f} ms")
    print(json.dumps(reply.answers, indent=2))
    probs = reply.answers.get("next", {}).get("probabilities")
    checks = {
        "noul in [0, 1]": 0.0 <= float(reply.answers["sound"]["noul"]) <= 1.0,
        "choice has a probability for every option": bool(probs) and set(probs) == set(CANDIDATES),
        "usage.input_tokens reported": reply.input_tokens is not None,
        "prefers the correct step (c0)": bool(probs) and max(probs, key=probs.get) == "c0",
    }
    for name, ok in checks.items():
        print(f"  [{'ok' if ok else '--'}] {name}")
    if not PROVIDERS[transport].is_typesafe_model:
        print("  note: this route serves a different model than TypeSafe's jev-1.13")

    if record:
        FIXTURES.mkdir(parents=True, exist_ok=True)
        path = FIXTURES / f"real_{transport}.json"
        body = {
            "model": reply.model,
            "answers": reply.answers,
            "usage": {"input_tokens": reply.input_tokens},
            "_request": {"model": client.model, "state": STATE, "questions": QUESTIONS},
            "_recorded": time.strftime("%Y-%m-%d"),
        }
        path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
        print(f"recorded {path}")
    return 0


def main() -> int:
    load_dotenv()
    import os

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--transport",
        choices=sorted(PROVIDERS),
        default=os.environ.get("JEV_TRANSPORT") or "openrouter",
    )
    parser.add_argument("--record", action="store_true", help="save the response as a fixture")
    args = parser.parse_args()
    return asyncio.run(probe(args.transport, args.record))


if __name__ == "__main__":
    raise SystemExit(main())
