"""Throwaway script for Day 1-2 Jev research (A3.2-A3.3).

Not part of the package — exploratory only. Field names below are GUESSES
based on the spec's description ("choice" / "noul" questions) and must be
corrected once you've read the real TypeSafe docs or seen a real response.
Do not treat anything printed here as verified until it's cited in
docs/verified_apis.md with a link.

Usage:
    export TYPESAFE_API_KEY=...        # or OPENROUTER_API_KEY=...
    python scripts/probe_jev.py --transport direct --out noul_basic
    python scripts/probe_jev.py --transport openrouter --out choice_4_options

Saves the raw response (headers stripped) to tests/fixtures/jev/<out>.json
so it can be inspected and reused offline. ALWAYS check the saved file for
leaked keys/headers before committing.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "jev"

# GUESSED endpoints/shapes — overwrite once verified.
DIRECT_URL = "https://api.typesafe.ai/v1/systemone"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "typesafe/jev-1.13"


def build_request(prompt_state: str) -> dict:
    """GUESSED request shape. Replace field names once verified."""
    return {
        "model": "jev-1.13.0",
        "state": prompt_state,
        "questions": {
            "sound": {
                "type": "noul",
                "instructions": (
                    "Is every step in SOLUTION SO FAR mathematically correct "
                    "and logically valid? Ignore whether the solution is finished."
                ),
            },
        },
    }


def call_direct(payload: dict) -> httpx.Response:
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        sys.exit("TYPESAFE_API_KEY not set")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    return httpx.post(DIRECT_URL, json=payload, headers=headers, timeout=30)


def call_openrouter(payload: dict) -> httpx.Response:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY not set")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    # NOTE: OpenRouter may wrap this as a chat completion rather than the
    # typed `questions` request — this is exactly one of the VERIFY items.
    body = {"model": OPENROUTER_MODEL, **payload}
    return httpx.post(OPENROUTER_URL, json=body, headers=headers, timeout=30)


def strip_headers(resp: httpx.Response) -> dict:
    return {
        "status_code": resp.status_code,
        "body": _safe_json(resp),
        "latency_ms": resp.elapsed.total_seconds() * 1000 if resp.elapsed else None,
    }


def _safe_json(resp: httpx.Response):
    try:
        return resp.json()
    except Exception:
        return resp.text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["direct", "openrouter"], required=True)
    parser.add_argument("--out", required=True, help="fixture basename, no .json")
    parser.add_argument(
        "--state",
        default="PROBLEM:\nWhat is 2 + 2?\n\nSOLUTION SO FAR:\nStep 1: 2 + 2 = 4",
    )
    args = parser.parse_args()

    payload = build_request(args.state)

    t0 = time.monotonic()
    resp = call_direct(payload) if args.transport == "direct" else call_openrouter(payload)
    elapsed_ms = (time.monotonic() - t0) * 1000

    result = strip_headers(resp)
    result["measured_latency_ms"] = elapsed_ms
    result["transport"] = args.transport

    print(json.dumps(result, indent=2))

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIXTURES_DIR / f"{args.out}.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"\nSaved to {out_path}")
    print("!! Double-check this file for any leaked key/header before committing !!")


if __name__ == "__main__":
    main()
