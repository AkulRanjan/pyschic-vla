"""All prompt templates and the step format (spec §5.2, team file §A4.1).

File owner: Person 3 (Akul). Person 2 (Jagriti) edits only the JUDGE SECTION at the bottom.
Every change to prompt *text* must bump ``PROMPT_VERSION``: it goes into cache
keys and logs.

Step convention (team file §B4, item 1): a step is stored as plain text,
stripped, WITHOUT its ``Step n:`` prefix. Numbering is added only here, by
``format_steps``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

PROMPT_VERSION = "g5"  # g5: Gemma self-judge prompts


class ChatTemplater(Protocol):
    """Anything that can render a chat into a prompt string (see ``llm/tokenize.py``)."""

    def apply_chat_template(self, messages: list[dict[str, str]]) -> str: ...


# --------------------------------------------------------------------------
# Step format
# --------------------------------------------------------------------------

# A step header at the start of a line. Tolerates the markdown variants Gemma
# tends to produce: "Step 2:", "**Step 2:**", "**Step 2**:", "## Step 2:".
# A closing "**" is consumed only if an opening "**" was seen, so a step whose
# text itself starts with bold markup keeps it.
_HEADER_RE = re.compile(
    r"^[ \t]*(?:(?P<bold>\*\*)|#{1,6}[ \t]*)?Step[ \t]+(?P<num>\d+)[ \t]*"
    r"(?:(?(bold)\*\*)[ \t]*:|:(?(bold)\*\*))[ \t]*",
    re.MULTILINE,
)


def format_steps(steps: Sequence[str]) -> str:
    """Render steps as ``"Step 1: ...\\n\\nStep 2: ...\\n\\n"``.

    Each step is stripped first, so ``split_steps(format_steps(x)) == x`` for
    any list of already-stripped steps (the storage convention).
    """
    return "".join(f"Step {i}: {s.strip()}\n\n" for i, s in enumerate(steps, start=1))


def step_header(n: int) -> str:
    """The prefix that opens step ``n`` in a generator prompt, e.g. ``"Step 3:"``."""
    return f"Step {n}:"


def split_steps(text: str, first_number: int = 1) -> list[str]:
    """Split a ``Step n:``-formatted text into steps, stripping the prefixes.

    Inverse of :func:`format_steps`. Only headers that continue the numbering
    (``first_number``, ``first_number + 1``, ...) at the start of a line count as
    boundaries, so a step that mentions "Step 5:" out of sequence, or the word
    "Step" mid-line, is left intact.

    Text before the first header (a preamble) is kept as its own step so
    nothing is lost. Text with no header at all is a single step. Known limit:
    a step whose own text contains, at a line start, exactly the next header
    in sequence cannot round-trip.
    """
    accepted: list[re.Match[str]] = []
    expected = first_number
    for m in _HEADER_RE.finditer(text):
        if int(m.group("num")) == expected:
            accepted.append(m)
            expected += 1

    if not accepted:
        stripped = text.strip()
        return [stripped] if stripped else []

    steps: list[str] = []
    preamble = text[: accepted[0].start()].strip()
    if preamble:
        steps.append(preamble)
    for i, m in enumerate(accepted):
        end = accepted[i + 1].start() if i + 1 < len(accepted) else len(text)
        steps.append(text[m.end() : end].strip())
    return steps


def truncate_at_next_step(text: str) -> str:
    """Cut a single proposed step at any step header that slipped past the stop sequence.

    The server stops on ``"\\n\\nStep"``, but variants such as ``"\\nStep 4:"`` or
    ``"\\n\\n**Step 4:**"`` get through. The text is the continuation after
    ``"Step n:"``, so any header at a later line start opens the next step.
    """
    body = text.lstrip()
    for m in _HEADER_RE.finditer(body):
        if m.start() > 0:
            return body[: m.start()].strip()
    return body.strip()


# --------------------------------------------------------------------------
# Generator prompt
# --------------------------------------------------------------------------

# Tuned on the MATH *train* split only (team file §B11.1). The worked example
# is hand-written and does not come from any benchmark.
GENERATOR_SYSTEM = (
    "You are a careful mathematician. You solve problems one short, numbered step at a time."
)

GENERATOR_INSTRUCTIONS = r"""Solve the math problem below step by step.

Format rules:
- Write numbered steps: "Step 1:", "Step 2:", and so on.
- Keep each step short: one idea or one calculation.
- Separate steps with a blank line.
- In the last step, put the final answer in \boxed{}.

Example of the format:

Problem: What is the sum of the first 5 positive odd integers?

Step 1: The first 5 positive odd integers are 1, 3, 5, 7 and 9.

Step 2: Their sum is 1 + 3 + 5 + 7 + 9 = 25.

Step 3: The final answer is \boxed{25}.

Now solve this problem.

Problem: {problem}"""


def generator_messages(problem: str, use_system_role: bool = True) -> list[dict[str, str]]:
    """Chat messages for the generator.

    Gemma 4's template supports a system turn (verified). For a template
    without one, ``use_system_role=False`` folds the system text into the user turn.
    """
    user = GENERATOR_INSTRUCTIONS.replace("{problem}", problem.strip())
    if use_system_role:
        return [
            {"role": "system", "content": GENERATOR_SYSTEM},
            {"role": "user", "content": user},
        ]
    return [{"role": "user", "content": f"{GENERATOR_SYSTEM}\n\n{user}"}]


def generator_prompt(
    problem: str,
    steps: Sequence[str],
    tokenizer: ChatTemplater,
    use_system_role: bool = True,
) -> str:
    """Raw completions-endpoint prompt that continues an open assistant turn (spec §5.2).

    ``chat_template(system?, user) + format_steps(steps) + "Step {n+1}:"``
    """
    head = tokenizer.apply_chat_template(generator_messages(problem, use_system_role))
    return head + format_steps(steps) + step_header(len(steps) + 1)


# Chat-API generator (llm/chat_generator.py): for hosted APIs such as OpenRouter that can't
# continue a partial assistant turn, the steps so far go into the user turn with an
# explicit request for the next step (or the rest of the solution).
NEXT_STEP_REQUEST = (
    'Write ONLY the next step, starting with "{header}". Do not repeat earlier steps.'
)
CONTINUE_REQUEST = (
    'Continue the solution from "{header}" to the end. Do not repeat earlier steps. '
    "In the last step, put the final answer in \\boxed{{}}."
)


def generator_chat_messages(
    problem: str, steps: Sequence[str], *, next_step_only: bool, use_system_role: bool = True
) -> list[dict[str, str]]:
    """Chat messages asking for the next step (``next_step_only``) or the rest of the solution.

    With no steps and the whole solution wanted, this is just ``generator_messages``.
    """
    messages = generator_messages(problem, use_system_role)
    if not steps and not next_step_only:
        return messages
    header = step_header(len(steps) + 1)
    so_far = format_steps(steps).rstrip() or "(no steps yet)"
    request = (NEXT_STEP_REQUEST if next_step_only else CONTINUE_REQUEST).format(header=header)
    last = messages[-1]
    messages[-1] = {
        "role": last["role"],
        "content": f"{last['content']}\n\nSolution so far:\n\n{so_far}\n\n{request}",
    }
    return messages


# === JUDGE SECTION (owner: Person 2 (Jagriti)) ===========================================
# Draft texts from SPEC.md §5.3; tune in Phase 1 (on the train split only).

SOUND_INSTRUCTION = (
    "Is every step in SOLUTION SO FAR mathematically correct and logically valid? "
    "Ignore whether the solution is finished."
)
NEXT_INSTRUCTION = "Which candidate next step is most likely to lead to a correct final answer?"
FINAL_INSTRUCTION = "Is the final answer of this solution correct?"

# per_candidate_noul ablation (SPEC.md §8.4): one noul question per candidate
# instead of a single choice question.
CANDIDATE_INSTRUCTION_TEMPLATE = (
    "Does the following step correctly continue the solution so far?\n\nSTEP: {candidate}"
)

# Three phrasings for the pilot's prompt-sensitivity check (spec §7); the
# first must equal the main instruction used everywhere else.
SOUND_VARIANTS: list[str] = [
    SOUND_INSTRUCTION,
    (
        "Check the solution so far step by step. Are all steps valid and free of "
        "mathematical or logical errors, regardless of whether it reaches a final "
        "answer yet?"
    ),
    # every variant must ask so that "yes" means SOUND (the value is P(yes))
    (
        "Would a careful grader accept all of the reasoning and arithmetic shown so far "
        "as free of errors, even though the solution may not be finished?"
    ),
]


def judge_state(problem: str, steps: list[str]) -> str:
    """``PROBLEM:\\n...\\n\\nSOLUTION SO FAR:\\nStep 1: ...`` (SPEC.md §5.3)."""
    return f"PROBLEM:\n{problem.strip()}\n\nSOLUTION SO FAR:\n{format_steps(steps)}".rstrip()


def judge_state_final(problem: str, steps: list[str]) -> str:
    """Like ``judge_state``, but headed ``SOLUTION:`` for a terminal state
    (used by ``final_correct``, which asks about the finished solution)."""
    return f"PROBLEM:\n{problem.strip()}\n\nSOLUTION:\n{format_steps(steps)}".rstrip()


# Gemma self-judge (baseline B4, judge/self_judge.py): the same questions as Jev, posed to the
# generator model as chat prompts; the answer is read from the first token's logprobs.
SELF_JUDGE_SYSTEM = "You are a careful, strict grader of math solutions."
SELF_JUDGE_YES_NO = "Answer with exactly one word: Yes or No."
SELF_JUDGE_LETTER = "Answer with exactly one letter: the letter of your choice."
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def self_judge_yes_no_messages(
    problem: str, steps: list[str], question: str, *, final: bool = False
) -> list[dict[str, str]]:
    """A yes/no question about the state (``final``: about the finished solution)."""
    state = judge_state_final(problem, steps) if final else judge_state(problem, steps)
    return [
        {"role": "system", "content": SELF_JUDGE_SYSTEM},
        {"role": "user", "content": f"{state}\n\n{question}\n{SELF_JUDGE_YES_NO}"},
    ]


def self_judge_choice_messages(
    problem: str, steps: list[str], candidates: list[str]
) -> list[dict[str, str]]:
    """A multiple-choice question over candidate next steps, lettered A, B, C, ..."""
    if len(candidates) > len(LETTERS):
        raise ValueError(f"at most {len(LETTERS)} candidates, got {len(candidates)}")
    options = "\n".join(f"{LETTERS[i]}. {c.strip()}" for i, c in enumerate(candidates))
    state = judge_state(problem, steps)
    return [
        {"role": "system", "content": SELF_JUDGE_SYSTEM},
        {
            "role": "user",
            "content": f"{state}\n\nCANDIDATE NEXT STEPS:\n{options}\n\n"
            f"{NEXT_INSTRUCTION}\n{SELF_JUDGE_LETTER}",
        },
    ]
