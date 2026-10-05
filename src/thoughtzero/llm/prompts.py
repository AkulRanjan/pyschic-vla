"""All prompt text lives here (SPEC.md §13.6). File owner: Person 3 (Akul).

The JUDGE SECTION below is edited by Person 2 (Jagriti). Any change to prompt text bumps
``PROMPT_VERSION`` (it goes into cache keys and logs).

Convention: steps are stored WITHOUT the ``Step n:`` prefix; ``format_steps`` adds numbering.
"""

from __future__ import annotations

from typing import Any

PROMPT_VERSION = "g0"


def format_steps(steps: list[str]) -> str:
    """``["a", "b"]`` -> ``"Step 1: a\\n\\nStep 2: b\\n\\n"``."""
    raise NotImplementedError("Person 3 (Akul)")


def split_steps(text: str) -> list[str]:
    """Inverse of ``format_steps``; strips ``Step n:`` prefixes."""
    raise NotImplementedError("Person 3 (Akul)")


def generator_prompt(problem: str, steps: list[str], tokenizer: Any) -> str:
    """Chat template with an open assistant turn + formatted steps + ``Step {n+1}:``."""
    raise NotImplementedError("Person 3 (Akul)")


# === JUDGE SECTION (owner: Person 2 (Jagriti)) ===========================================
# Draft texts from SPEC.md §5.3; tune in Phase 1 (on the train split only).

SOUND_INSTRUCTION = (
    "Is every step in SOLUTION SO FAR mathematically correct and logically valid? "
    "Ignore whether the solution is finished."
)
NEXT_INSTRUCTION = "Which candidate next step is most likely to lead to a correct final answer?"
FINAL_INSTRUCTION = "Is the final answer of this solution correct?"

# Three phrasings for the pilot's prompt-sensitivity check; the first must equal the main one.
SOUND_VARIANTS: list[str] = [SOUND_INSTRUCTION]


def judge_state(problem: str, steps: list[str]) -> str:
    """``PROBLEM:\\n...\\n\\nSOLUTION SO FAR:\\nStep 1: ...`` (SPEC.md §5.3)."""
    raise NotImplementedError("Person 2 (Jagriti)")
