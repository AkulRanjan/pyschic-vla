"""Prompts (spec B4 item 9). File owner: Person 3 (Akul) on the real team
repo. This local copy holds only what the judge needs to develop against
(format_steps + the JUDGE section), since that section is the one Person 2
is allowed to edit. Non-judge content is intentionally absent here.
"""

PROMPT_VERSION = "p2-local-0.1.0"


def format_steps(steps: list[str]) -> str:
    if not steps:
        return ""
    return "\n".join(f"Step {i + 1}: {s}" for i, s in enumerate(steps))


# ---- JUDGE section (owned by Person 2) ----

SOUND_INSTRUCTION = (
    "Is every step in SOLUTION SO FAR mathematically correct and logically "
    "valid? Ignore whether the solution is finished."
)

SOUND_VARIANTS = [
    SOUND_INSTRUCTION,
    (
        "Check the solution so far step by step. Are all steps valid and free "
        "of mathematical or logical errors, regardless of whether it reaches "
        "a final answer yet?"
    ),
    (
        "Would a careful grader find any error in the reasoning or arithmetic "
        "shown so far, even if the solution isn't finished?"
    ),
]

NEXT_STEP_INSTRUCTION = (
    "Which candidate next step is most likely to lead to a correct final answer?"
)

CANDIDATE_SOUND_INSTRUCTION = (
    "Does the following step correctly continue the solution so far? STEP: {candidate}"
)

FINAL_CORRECT_INSTRUCTION = "Is the final answer of this solution correct?"


def judge_state(problem: str, steps: list[str]) -> str:
    body = format_steps(steps)
    return f"PROBLEM:\n{problem}\n\nSOLUTION SO FAR:\n{body}"


def judge_state_final(problem: str, steps: list[str]) -> str:
    body = format_steps(steps)
    return f"PROBLEM:\n{problem}\n\nSOLUTION:\n{body}"
