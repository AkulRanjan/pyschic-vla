"""ChatGenerator against a respx-mocked chat endpoint (no real network)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI

from thoughtzero import accounting
from thoughtzero.accounting import Ledger
from thoughtzero.config import Config, GeneratorCfg
from thoughtzero.llm.chat_generator import ChatGenerator, api_key_for, steps_from_reply
from thoughtzero.llm.factory import make_generator
from thoughtzero.llm.prompts import generator_chat_messages

BASE = "http://chat.test/v1"
URL = f"{BASE}/chat/completions"


class WordTokenizer:
    def apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        return messages[-1]["content"]

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def reply(text: str, usage: tuple[int, int] | None = (100, 10)) -> httpx.Response:
    body: dict[str, Any] = {
        "id": "c",
        "object": "chat.completion",
        "created": 0,
        "model": "gemma",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
    }
    if usage:
        body["usage"] = {
            "prompt_tokens": usage[0],
            "completion_tokens": usage[1],
            "total_tokens": sum(usage),
        }
    return httpx.Response(200, json=body)


@pytest.fixture
def ledger() -> Iterator[Ledger]:
    led = Ledger()
    token = accounting.current_ledger.set(led)
    yield led
    accounting.current_ledger.reset(token)


def make_gen(**kwargs: Any) -> ChatGenerator:
    cfg = GeneratorCfg(api="chat", base_url=BASE, model="gemma")
    client = AsyncOpenAI(
        base_url=BASE, api_key="test", max_retries=0, http_client=httpx.AsyncClient()
    )
    kwargs.setdefault("max_attempts", 3)
    return ChatGenerator(cfg, WordTokenizer(), client=client, **kwargs)


def sent(call: Any) -> dict[str, Any]:
    return json.loads(call.request.content)


@pytest.mark.parametrize(
    ("text", "first", "expected"),
    [
        ("Step 2: Divide by 2.", 2, ["Divide by 2."]),
        ("Divide by 2.", 2, ["Divide by 2."]),  # no header at all
        ("Step 2: a\n\nStep 3: b", 2, ["a", "b"]),
        ("Step 1: x\n\nStep 2: y\n\nStep 3: z", 3, ["z"]),  # restarted at Step 1
        ("Step 1: Find the factors.", 1, ["Find the factors."]),
        ("", 2, []),
    ],
)
def test_steps_from_reply(text: str, first: int, expected: list[str]) -> None:
    assert steps_from_reply(text, first) == expected


def test_chat_messages_carry_the_steps_and_the_request() -> None:
    msgs = generator_chat_messages("What is 1+1?", ["One plus one."], next_step_only=True)
    assert msgs[0]["role"] == "system"
    assert "Step 1: One plus one." in msgs[-1]["content"]
    assert msgs[-1]["content"].endswith('starting with "Step 2:". Do not repeat earlier steps.')
    plain = generator_chat_messages("What is 1+1?", [], next_step_only=False)
    assert "Solution so far" not in plain[-1]["content"]  # a fresh solve is the plain prompt


async def test_propose_makes_k_requests_and_strips_headers(respx_mock: Any, ledger: Ledger) -> None:
    route = respx_mock.post(URL).mock(
        side_effect=[reply("Step 2: a"), reply("Step 2: b"), reply(""), reply("Step 2: c")]
    )
    outs = await make_gen().propose("p", ["s"], 4)
    # parallel requests: replies arrive in any order; the empty one is dropped
    assert sorted(o.text for o in outs) == ["a", "b", "c"]
    assert route.call_count == 4
    bodies = [sent(c) for c in route.calls]
    assert all("n" not in b for b in bodies)  # one choice per request
    assert len({b["seed"] for b in bodies}) == 4  # different seed per sample
    assert bodies[0]["stop"] == ["\n\nStep", "\n\n\n"] and bodies[0]["temperature"] == 0.9
    assert ledger.gemma_calls == 4 and ledger.gemma_completion_tokens == 40  # API usage


async def test_complete_returns_new_steps_cut_at_the_answer(respx_mock: Any) -> None:
    respx_mock.post(URL).mock(
        return_value=reply("Step 2: w = 6.\n\nStep 3: area 54, \\boxed{54}.\n\nStep 4: extra")
    )
    steps = await make_gen().complete("p", ["s"], temperature=0.0)
    assert steps == ["w = 6.", "area 54, \\boxed{54}."]


async def test_sample_solutions_and_depth_cap(respx_mock: Any) -> None:
    route = respx_mock.post(URL).mock(return_value=reply("Step 1: go"))
    sols = await make_gen().sample_solutions("p", 3, temperature=0.7)
    assert sols == ["Step 1: go"] * 3 and route.call_count == 3
    capped = await make_gen(max_depth=1).sample_completions("p", ["s"], 2, 0.7)
    assert capped == [[], []] and route.call_count == 3  # no request at max depth


async def test_usage_falls_back_to_the_tokenizer(respx_mock: Any, ledger: Ledger) -> None:
    respx_mock.post(URL).mock(return_value=reply("Step 2: one two three", usage=None))
    await make_gen().propose("p", ["s"], 1)
    assert ledger.gemma_completion_tokens == 5  # "Step 2: one two three" = 5 words


def test_openrouter_key_is_used_for_an_openrouter_endpoint(monkeypatch: Any) -> None:
    monkeypatch.delenv("GEMMA_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    assert api_key_for("https://openrouter.ai/api/v1") == "or-key"
    assert api_key_for("http://localhost:8000/v1") == "EMPTY"
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    assert api_key_for("https://generativelanguage.googleapis.com/v1beta/openai/") == "g-key"
    monkeypatch.setenv("GEMMA_API_KEY", "own")
    assert api_key_for("https://openrouter.ai/api/v1") == "own"


def test_factory_picks_the_class_from_generator_api() -> None:
    cfg = Config(generator=GeneratorCfg(api="chat", base_url=BASE, model="gemma"))
    assert isinstance(make_generator(cfg, tokenizer=WordTokenizer()), ChatGenerator)


async def test_openrouter_requests_carry_provider_routing(respx_mock: Any) -> None:
    or_base = "https://openrouter.ai/api/v1"
    route = respx_mock.post(f"{or_base}/chat/completions").mock(return_value=reply("Step 2: a"))
    cfg = GeneratorCfg(api="chat", base_url=or_base, model="google/gemma-4-26b-a4b-it")
    client = AsyncOpenAI(
        base_url=or_base, api_key="k", max_retries=0, http_client=httpx.AsyncClient()
    )
    await ChatGenerator(cfg, WordTokenizer(), client=client).propose("p", ["s"], 1)
    assert sent(route.calls.last)["provider"] == {
        "quantizations": ["bf16"],
        "require_parameters": True,
    }


async def test_other_endpoints_get_no_provider_field(respx_mock: Any) -> None:
    route = respx_mock.post(URL).mock(return_value=reply("Step 2: a"))
    await make_gen().propose("p", ["s"], 1)
    assert "provider" not in sent(route.calls.last)
