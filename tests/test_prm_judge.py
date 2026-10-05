"""PRMJudge with a fake scorer (no torch, no model download)."""

import pytest

from thoughtzero.config import JudgeCfg
from thoughtzero.judge.factory import make_judge
from thoughtzero.judge.prm import STEP_SEPARATOR, PRMJudge, prm_messages, torch_available


def test_messages_follow_the_model_card() -> None:
    msgs = prm_messages(" Q? ", ["a ", "b"])
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[1]["content"] == "Q?"
    assert msgs[2]["content"] == f"a{STEP_SEPARATOR}b{STEP_SEPARATOR}"


async def test_value_is_the_weakest_step_and_priors_are_uniform() -> None:
    seen: list[list[str]] = []

    def scorer(problem: str, steps: list[str]) -> list[float]:
        seen.append(steps)
        return [0.9, 0.2, 0.8][: len(steps)]

    judge = PRMJudge(JudgeCfg(kind="prm", root_value=0.5), scorer)
    priors, value = await judge.prior_and_value("p", ["s1", "s2", "s3"], ["a", "b"])
    assert priors == [0.5, 0.5] and value == 0.2
    assert await judge.final_correct("p", ["s1"]) == 0.9
    assert await judge.step_sound("p", []) == 0.5 and len(seen) == 2  # root: no model call


async def test_score_count_mismatch_is_an_error() -> None:
    judge = PRMJudge(JudgeCfg(kind="prm"), lambda p, s: [0.5])
    with pytest.raises(ValueError, match="scores"):
        await judge.step_sound("p", ["s1", "s2"])


def test_factory_skips_prm_without_torch() -> None:
    if torch_available():
        assert isinstance(make_judge(JudgeCfg(kind="prm")), PRMJudge)
    else:
        with pytest.raises(NotImplementedError, match=r"\[prm\]"):
            make_judge(JudgeCfg(kind="prm"))
