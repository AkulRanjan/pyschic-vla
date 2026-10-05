"""Hosted-Gemma cost: ledger USD, the spend cap, search stopping on it, and the estimate."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI

from thoughtzero.accounting import ledger_scope
from thoughtzero.config import Config, GeneratorCfg, SearchCfg
from thoughtzero.eval.runner import GEMMA_SOLUTION_CALL, GEMMA_STEP_CALL, estimate_gemma_usd
from thoughtzero.judge import budget
from thoughtzero.judge.budget import BudgetExceeded
from thoughtzero.llm.chat_generator import ChatGenerator
from thoughtzero.mocks import TOY, MockGenerator, MockJudge, toy_extract_answer
from thoughtzero.search.mcts import search

BASE = "http://chat.test/v1"


class Tok:
    def apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        return ""

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def reply() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "c",
            "object": "chat.completion",
            "created": 0,
            "model": "gemma",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Step 2: a"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1_000, "completion_tokens": 100, "total_tokens": 1_100},
        },
    )


def priced_gen() -> ChatGenerator:
    cfg = GeneratorCfg(
        api="chat", base_url=BASE, model="g", usd_per_mtok_in=0.1, usd_per_mtok_out=0.3
    )
    client = AsyncOpenAI(base_url=BASE, api_key="k", max_retries=0, http_client=httpx.AsyncClient())
    return ChatGenerator(cfg, Tok(), client=client)


@pytest.fixture(autouse=True)
def no_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(budget, "gemma_cap", None)


async def test_ledger_records_gemma_usd(respx_mock: Any) -> None:
    respx_mock.post(f"{BASE}/chat/completions").mock(return_value=reply())
    with ledger_scope() as led:
        await priced_gen().propose("p", ["s"], 2)
    assert led.gemma_usd == pytest.approx(2 * (1_000 * 0.1 + 100 * 0.3) / 1e6)


async def test_spend_cap_raises_budget_exceeded(respx_mock: Any) -> None:
    respx_mock.post(f"{BASE}/chat/completions").mock(return_value=reply())
    budget.configure_gemma(max_usd=0.0002)  # one call costs $0.00013
    gen = priced_gen()
    await gen.propose("p", ["s"], 1)
    with pytest.raises(BudgetExceeded, match="Gemma budget"):
        await gen.propose("p", ["s"], 1)


class BrokeJudge(MockJudge):
    async def prior_and_value(self, *a: Any, **k: Any) -> Any:
        raise BudgetExceeded("over budget")


async def test_search_stops_on_budget_instead_of_retrying() -> None:
    judge = BrokeJudge()
    with pytest.raises(BudgetExceeded):
        await search(
            TOY.question,
            MockGenerator(),
            judge,
            SearchCfg(n_simulations=8, k=3),
            extract=toy_extract_answer,
            normalize=str.strip,
        )


def test_estimate_scales_with_simulations_and_samples() -> None:
    class M:
        def __init__(self, name: str, **params: int) -> None:
            self.name, self.params = name, params

    free = Config()
    assert estimate_gemma_usd(free, 100, [M("tz", n_simulations=16)]) == 0.0  # no prices
    cfg = Config(generator=GeneratorCfg(usd_per_mtok_in=0.1, usd_per_mtok_out=0.3))
    step = (GEMMA_STEP_CALL[0] * 0.1 + GEMMA_STEP_CALL[1] * 0.3) / 1e6
    sol = (GEMMA_SOLUTION_CALL[0] * 0.1 + GEMMA_SOLUTION_CALL[1] * 0.3) / 1e6
    tz = estimate_gemma_usd(cfg, 10, [M("tz", n_simulations=16)])
    assert tz == pytest.approx(10 * (16 * cfg.search.k * step + sol))
    assert estimate_gemma_usd(cfg, 10, [M("sc", n=8)]) == pytest.approx(10 * 8 * sol)
