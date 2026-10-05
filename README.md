# ThoughtZero

AlphaZero-style tree search over reasoning steps, judged by Jev (TypeSafe AI).
See `SPEC.md` for the full design and `Jagriti.md` (Person 2 / Judge) for this
person's work plan.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # fill in your own keys, never commit .env
```

## Status

Local scaffold only — waiting on Person 1's CP0 scaffold PR (`types.py`,
`config.py`, mocks, CI) to land. Person 2 (judge) Day 1-2 work (API access,
VERIFY research, fixtures) does not depend on it and is tracked in
`docs/verified_apis.md`.
