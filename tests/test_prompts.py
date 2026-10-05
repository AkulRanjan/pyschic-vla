"""Step format and generator prompt tests (team file §A4.1)."""

from __future__ import annotations

import pytest

from thoughtzero.llm import prompts
from thoughtzero.llm.prompts import (
    format_steps,
    generator_messages,
    generator_prompt,
    split_steps,
    truncate_at_next_step,
)
from thoughtzero.llm.tokenize import ApproxTokenizer


def test_format_steps() -> None:
    assert format_steps(["a", "b"]) == "Step 1: a\n\nStep 2: b\n\n"
    assert format_steps([]) == ""


@pytest.mark.parametrize(
    "steps",
    [
        [],
        ["Compute 2 + 2 = 4."],
        ["First line.\nSecond line of the same step.", "Next."],
        ["Paragraph one.\n\nParagraph two, same step.", "Done: \\boxed{4}."],
        [r"Let $f(x) = \frac{x^2}{\sqrt{x+1}}$.", r"So $\boxed{\frac{1}{2}}$."],
        ["Step size is 2, so we Step forward.", "The next Step: continue."],
        ["See Step 5: it is referenced out of order.", "ok"],
        ["", "after an empty step", ""],
        ["Step", "Step 1"],
        ["**bold** text", "## not a header"],
    ],
)
def test_split_inverts_format(steps: list[str]) -> None:
    assert split_steps(format_steps(steps)) == steps


def test_split_tolerates_markdown_headers() -> None:
    text = "**Step 1:** a\n\n**Step 2**: b\n\n## Step 3: c"
    assert split_steps(text) == ["a", "b", "c"]


def test_split_single_newline_and_first_number() -> None:
    assert split_steps("Step 3: x\nStep 4: y", first_number=3) == ["x", "y"]


def test_split_ignores_out_of_sequence_headers() -> None:
    text = "Step 1: a\nStep 3: still a\n\nStep 2: b"
    assert split_steps(text) == ["a\nStep 3: still a", "b"]


def test_split_without_headers_and_with_preamble() -> None:
    assert split_steps("just some text") == ["just some text"]
    assert split_steps("   ") == []
    assert split_steps("Intro.\n\nStep 1: a") == ["Intro.", "a"]


def test_truncate_at_next_step() -> None:
    assert truncate_at_next_step(" a short step") == "a short step"
    assert truncate_at_next_step(" a\nStep 4: leaked") == "a"
    assert truncate_at_next_step(" a\n\n**Step 4:** leaked") == "a"
    assert truncate_at_next_step(" mentions Step 4: inline") == "mentions Step 4: inline"


def test_generator_messages_system_role() -> None:
    folded = generator_messages("What is 1+1?", use_system_role=False)
    assert [m["role"] for m in folded] == ["user"]
    assert prompts.GENERATOR_SYSTEM in folded[0]["content"]
    assert "What is 1+1?" in folded[0]["content"]

    separate = generator_messages("What is 1+1?")  # Gemma 4 default
    assert [m["role"] for m in separate] == ["system", "user"]


def test_generator_prompt_continues_open_assistant_turn() -> None:
    tok = ApproxTokenizer()
    p = generator_prompt("Find x.", ["first", "second"], tok)
    assert p.startswith("<bos><|turn>system\n")
    assert p.endswith("<|turn>model\nStep 1: first\n\nStep 2: second\n\nStep 3:")
    assert "Find x." in p
    assert generator_prompt("Find x.", [], tok).endswith("<|turn>model\nStep 1:")


def test_instructions_contain_problem_with_braces() -> None:
    # The problem is inserted with str.replace, so LaTeX braces are safe.
    msgs = generator_messages(r"Evaluate $\frac{1}{2} + \{x\}$.")
    assert r"\frac{1}{2} + \{x\}" in msgs[-1]["content"]


def test_prompt_version_is_set() -> None:
    assert isinstance(prompts.PROMPT_VERSION, str) and prompts.PROMPT_VERSION
