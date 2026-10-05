"""GemmaSelfJudge with a fake logprob source (no model, no network)."""

from __future__ import annotations

import math
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI

from thoughtzero.config import GeneratorCfg, JudgeCfg
from thoughtzero.eval.methods import MethodUnavailable, build_judge, judge_needs_generator
from thoughtzero.judge.factory import make_judge
from thoughtzero.judge.hybrid import HybridJudge
from thoughtzero.judge.self_judge import (
    GemmaSelfJudge,
    NoLogprobsError,
    letter_probabilities,
    yes_probability,
)
from thoughtzero.llm.chat_generator import ChatGenerator


class FakeSource:
    """Answers by question type: the last user message decides which logprobs come back."""

    def __init__(self, yes_no: dict[str, float], letters: dict[str, float]) -> None:
        self.yes_no, self.letters = yes_no, letters
        self.calls: list[list[dict[str, str]]] = []

    async def first_token_logprobs(
        self, messages: list[dict[str, str]], top_k: int = 20
    ) -> dict[str, float]:
        self.calls.append(messages)
        return self.letters if "CANDIDATE NEXT STEPS" in messages[-1]["content"] else self.yes_no


def ln(p: float) -> float:
    return math.log(p)


def test_yes_probability_sums_token_variants() -> None:
    lps = {"Yes": ln(0.5), " yes": ln(0.1), "▁No": ln(0.2), "Maybe": ln(0.2)}
    assert yes_probability(lps) == pytest.approx(0.6 / 0.8)
    assert yes_probability({"Sure": 0.0}) is None


def test_letter_probabilities_with_variants_and_gaps() -> None:
    lps = {"A": ln(0.5), " B.": ln(0.2), "b": ln(0.1), "Z": ln(0.1)}
    probs = letter_probabilities(lps, 3)
    assert probs[0] == pytest.approx(0.5) and probs[1] == pytest.approx(0.3)
    assert probs[2] is None  # "C" not returned: renormalize fills an epsilon


async def test_prior_and_value() -> None:
    src = FakeSource({"Yes": ln(0.9), "No": ln(0.1)}, {"A": ln(0.2), "B": ln(0.6)})
    judge = GemmaSelfJudge(JudgeCfg(kind="self"), src)
    priors, value = await judge.prior_and_value("p", ["s1"], ["x", "y", "z"])
    assert value == pytest.approx(0.9)
    assert sum(priors) == pytest.approx(1.0) and priors[1] > priors[0] > priors[2]
    choice = next(m for m in src.calls if "CANDIDATE" in m[-1]["content"])[-1]["content"]
    assert "A. x\nB. y\nC. z" in choice and "SOLUTION SO FAR:\nStep 1: s1" in choice


async def test_root_single_candidate_and_final() -> None:
    src = FakeSource({"Yes": ln(0.3), "No": ln(0.7)}, {})
    judge = GemmaSelfJudge(JudgeCfg(kind="self", root_value=0.5), src)
    priors, value = await judge.prior_and_value("p", [], ["only"])
    assert priors == [1.0] and value == 0.5 and src.calls == []  # nothing to ask
    assert await judge.final_correct("p", ["s"]) == pytest.approx(0.3)
    assert "SOLUTION:\nStep 1: s" in src.calls[-1][-1]["content"]


async def test_no_logprobs_is_an_error_and_no_yes_no_is_neutral() -> None:
    with pytest.raises(NoLogprobsError):
        await GemmaSelfJudge(JudgeCfg(kind="self"), FakeSource({}, {})).step_sound("p", ["s"])
    judge = GemmaSelfJudge(JudgeCfg(kind="self"), FakeSource({"Hmm": 0.0}, {}))
    assert await judge.step_sound("p", ["s"]) == 0.5


async def test_sound_variant_selects_the_question() -> None:
    from thoughtzero.llm.prompts import SOUND_VARIANTS

    src = FakeSource({"Yes": 0.0}, {})
    await GemmaSelfJudge(JudgeCfg(kind="self", sound_variant=2), src).step_sound("p", ["s"])
    assert SOUND_VARIANTS[2] in src.calls[-1][-1]["content"]


def test_factory_and_build_judge_wire_the_generator() -> None:
    src = FakeSource({}, {})
    assert isinstance(make_judge(JudgeCfg(kind="self"), src), GemmaSelfJudge)
    hybrid = make_judge(JudgeCfg(kind="hybrid", prior_from="self", value_from="uniform"), src)
    assert isinstance(hybrid, HybridJudge) and isinstance(hybrid.prior_from, GemmaSelfJudge)
    with pytest.raises(MethodUnavailable):  # no generator: reported unavailable, skipped
        build_judge(JudgeCfg(kind="self"))
    assert isinstance(build_judge(JudgeCfg(kind="self"), generator=src), GemmaSelfJudge)  # type: ignore[arg-type]
    assert judge_needs_generator(JudgeCfg(kind="hybrid", prior_from="jev", value_from="self"))
    assert not judge_needs_generator(JudgeCfg(kind="jev"))


async def test_chat_generator_reads_first_token_logprobs(respx_mock: Any) -> None:
    body = {
        "id": "c",
        "object": "chat.completion",
        "created": 0,
        "model": "gemma",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Yes"},
                "finish_reason": "length",
                "logprobs": {
                    "content": [
                        {
                            "token": "Yes",
                            "logprob": -0.1,
                            "bytes": None,
                            "top_logprobs": [
                                {"token": "Yes", "logprob": -0.1, "bytes": None},
                                {"token": "No", "logprob": -2.4, "bytes": None},
                            ],
                        }
                    ]
                },
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 1, "total_tokens": 51},
    }
    route = respx_mock.post("http://chat.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=body)
    )
    client = AsyncOpenAI(
        base_url="http://chat.test/v1", api_key="k", max_retries=0, http_client=httpx.AsyncClient()
    )

    class Tok:
        def apply_chat_template(self, messages: list[dict[str, str]]) -> str:
            return ""

        def count_tokens(self, text: str) -> int:
            return len(text.split())

    gen = ChatGenerator(GeneratorCfg(api="chat", base_url="http://chat.test/v1"), Tok(), client)
    lps = await gen.first_token_logprobs([{"role": "user", "content": "q"}], top_k=5)
    assert lps == {"Yes": -0.1, "No": -2.4}
    import json

    sent = json.loads(route.calls.last.request.content)
    assert sent["logprobs"] is True and sent["top_logprobs"] == 5 and sent["max_tokens"] == 1
