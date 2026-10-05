"""OpenAICompatibleGenerator tests against a respx-mocked server (no real network)."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from openai import AsyncOpenAI

from thoughtzero import accounting
from thoughtzero.accounting import Ledger
from thoughtzero.config import GeneratorCfg, load_config
from thoughtzero.llm.gemma import OpenAICompatibleGenerator, _derive_seed

BASE = "http://gemma.test/v1"
URL = f"{BASE}/completions"


class WordTokenizer:
    """Deterministic fake: one token per whitespace-separated word."""

    def apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        content = messages[-1]["content"]
        return f"<bos><start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n"

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def completion(*texts: str, logprobs: Any = None) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "cmpl-1",
            "object": "text_completion",
            "created": 0,
            "model": "gemma",
            "choices": [
                {"index": i, "text": t, "finish_reason": "stop", "logprobs": logprobs}
                for i, t in enumerate(texts)
            ],
            "usage": {"prompt_tokens": 999, "completion_tokens": 999, "total_tokens": 1998},
        },
    )


@pytest.fixture
def ledger() -> Iterator[Ledger]:
    led = Ledger()
    token = accounting.current_ledger.set(led)
    yield led
    accounting.current_ledger.reset(token)


def make_gen(**kwargs: Any) -> OpenAICompatibleGenerator:
    cfg = GeneratorCfg(base_url=BASE, model="gemma")
    # Recent openai SDKs default to their own httpx fork, which respx cannot
    # intercept; a plain httpx client is accepted and mockable.
    client = AsyncOpenAI(
        base_url=BASE, api_key="test", max_retries=0, http_client=httpx.AsyncClient()
    )
    kwargs.setdefault("max_attempts", 3)
    return OpenAICompatibleGenerator(cfg, WordTokenizer(), client=client, **kwargs)


@pytest.fixture
def gen() -> OpenAICompatibleGenerator:
    return make_gen()


def body(call: Any) -> dict[str, Any]:
    return json.loads(call.request.content)


@respx.mock
async def test_propose_request_body(gen: OpenAICompatibleGenerator, ledger: Ledger) -> None:
    route = respx.post(URL).mock(return_value=completion(" a", " b", " c", " d"))
    await gen.propose("Find x.", ["one", "two"], k=4)

    b = body(route.calls.last)
    assert b["model"] == "gemma"
    assert b["n"] == 4
    assert b["stop"] == ["\n\nStep", "\n\n\n"]
    assert b["temperature"] == 0.9
    assert b["top_p"] == 0.95
    assert b["max_tokens"] == 256
    assert isinstance(b["seed"], int)
    assert b["prompt"].endswith("Step 1: one\n\nStep 2: two\n\nStep 3:")


@respx.mock
async def test_propose_cleans_outputs_and_counts_tokens(
    gen: OpenAICompatibleGenerator, ledger: Ledger
) -> None:
    respx.post(URL).mock(
        return_value=completion(" x equals two", "   ", " y is 3\nStep 4: leaked text", "")
    )
    outs = await gen.propose("Find x.", ["s1", "s2"], k=4)

    assert [o.text for o in outs] == ["x equals two", "y is 3"]
    # Completion tokens are counted per choice with the tokenizer (not the usage block),
    # including empty and truncated text the model still generated.
    assert [o.completion_tokens for o in outs] == [3, 7]
    prompt = gen._prompt("Find x.", ["s1", "s2"])
    assert all(o.prompt_tokens == len(prompt.split()) for o in outs)
    assert ledger.gemma_calls == 1
    assert ledger.gemma_completion_tokens == 3 + 0 + 7 + 0
    assert ledger.gemma_prompt_tokens == len(prompt.split())  # once per request, not per choice


@respx.mock
async def test_propose_all_empty_retries_once_with_new_seed(
    gen: OpenAICompatibleGenerator, ledger: Ledger
) -> None:
    route = respx.post(URL).mock(side_effect=[completion("", " "), completion("", "")])
    outs = await gen.propose("p", [], k=2)

    assert outs == []
    assert route.call_count == 2
    assert body(route.calls[0])["seed"] != body(route.calls[1])["seed"]
    assert ledger.gemma_calls == 2


@respx.mock
async def test_propose_retry_recovers(gen: OpenAICompatibleGenerator, ledger: Ledger) -> None:
    respx.post(URL).mock(side_effect=[completion(""), completion(" ok")])
    outs = await gen.propose("p", [], k=1)
    assert [o.text for o in outs] == ["ok"]


@respx.mock
async def test_complete_returns_only_new_steps(
    gen: OpenAICompatibleGenerator, ledger: Ledger
) -> None:
    route = respx.post(URL).mock(
        return_value=completion(
            " third\n\nStep 4: fourth, so \\boxed{7}\n\nStep 5: rambling after the answer"
        )
    )
    new = await gen.complete("p", ["first", "second"])

    assert new == ["third", "fourth, so \\boxed{7}"]
    b = body(route.calls.last)
    assert b["n"] == 1
    assert b["temperature"] == 0.0
    assert "top_p" not in b  # greedy
    assert "stop" not in b  # no step stop: run to the end
    assert b["max_tokens"] == 2048


@respx.mock
async def test_complete_respects_max_depth(ledger: Ledger) -> None:
    g = make_gen(max_depth=3)
    route = respx.post(URL).mock(return_value=completion(" b\n\nStep 3: c\n\nStep 4: d"))

    assert await g.complete("p", ["a"]) == ["b", "c"]
    assert await g.complete("p", ["a", "b", "c"]) == []
    assert route.call_count == 1  # already at max depth: no request


@respx.mock
async def test_sample_completions_batches_n(gen: OpenAICompatibleGenerator, ledger: Ledger) -> None:
    route = respx.post(URL).mock(
        return_value=completion(" \\boxed{1}", " x\n\nStep 3: \\boxed{2}", "")
    )
    out = await gen.sample_completions("p", ["s1"], n=3, temperature=0.7)

    assert out == [["\\boxed{1}"], ["x", "\\boxed{2}"], []]
    b = body(route.calls.last)
    assert b["n"] == 3 and b["temperature"] == 0.7 and b["top_p"] == 0.95


@respx.mock
async def test_sample_solutions_are_full_texts(
    gen: OpenAICompatibleGenerator, ledger: Ledger
) -> None:
    route = respx.post(URL).mock(return_value=completion(" a\n\nStep 2: \\boxed{3}", " b"))
    sols = await gen.sample_solutions("p", n=2, temperature=0.7)

    assert sols == ["Step 1: a\n\nStep 2: \\boxed{3}", "Step 1: b"]
    assert body(route.calls.last)["prompt"].endswith("Step 1:")


@respx.mock
async def test_retries_5xx_but_not_4xx(gen: OpenAICompatibleGenerator, ledger: Ledger) -> None:
    route = respx.post(URL).mock(
        side_effect=[httpx.Response(503, json={"error": "busy"}), completion(" ok")]
    )
    outs = await gen.propose("p", [], k=1)
    assert [o.text for o in outs] == ["ok"]
    assert route.call_count == 2

    route.mock(side_effect=None, return_value=httpx.Response(400, json={"error": "bad"}))
    with pytest.raises(Exception, match="400"):
        await gen.propose("p", [], k=1)
    assert route.call_count == 3  # the 400 was not retried


@respx.mock
async def test_raw_completion_logprobs(gen: OpenAICompatibleGenerator, ledger: Ledger) -> None:
    lp = {
        "tokens": ["Yes"],
        "token_logprobs": [-0.1],
        "top_logprobs": [{"Yes": -0.1, "No": -2.4}],
        "text_offset": [0],
    }
    route = respx.post(URL).mock(return_value=completion("Yes", logprobs=lp))
    r = await gen.raw_completion("Is it right? Answer:", max_tokens=1, logprobs=5)

    assert r["text"] == "Yes"
    assert r["top_logprobs"] == [{"Yes": -0.1, "No": -2.4}]
    assert r["completion_tokens"] == 1
    b = body(route.calls.last)
    assert b["prompt"] == "Is it right? Answer:"  # no template applied
    assert b["logprobs"] == 5 and b["max_tokens"] == 1
    assert ledger.gemma_calls == 1


@respx.mock
async def test_raw_completion_without_logprobs(
    gen: OpenAICompatibleGenerator, ledger: Ledger
) -> None:
    route = respx.post(URL).mock(return_value=completion("No"))
    r = await gen.raw_completion("Q:")
    assert r["text"] == "No" and r["top_logprobs"] == []
    assert "logprobs" not in body(route.calls.last)  # default None: not requested


@respx.mock
async def test_thinking_markup_warns(
    gen: OpenAICompatibleGenerator, ledger: Ledger, caplog: pytest.LogCaptureFixture
) -> None:
    respx.post(URL).mock(return_value=completion("<think>hmm</think> step"))
    with caplog.at_level(logging.WARNING):
        await gen.propose("p", [], k=1)
    assert "Thinking-mode" in caplog.text


def test_seed_depends_on_prompt_not_call_order() -> None:
    assert _derive_seed(0, "abc", 0) == _derive_seed(0, "abc", 0)
    assert _derive_seed(0, "abc", 0) != _derive_seed(0, "abd", 0)
    assert _derive_seed(0, "abc", 0) != _derive_seed(1, "abc", 0)
    assert 0 <= _derive_seed(0, "abc", 0) < 2**31


def test_from_config_uses_seed_and_search_max_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMMA_BASE_URL", "http://x:1/v1")
    monkeypatch.setenv("GEMMA_MODEL", "gemma4:e4b")
    cfg = load_config(
        Path(__file__).parents[1] / "configs" / "default.yaml",
        ["seed=7", "search.max_depth=12", "generator.max_step_tokens=96"],
    )
    g = OpenAICompatibleGenerator.from_config(cfg, WordTokenizer())
    assert (g.cfg.base_url, g.cfg.model, g.cfg.max_step_tokens) == (
        "http://x:1/v1",
        "gemma4:e4b",
        96,
    )
    assert (g.seed, g.max_depth) == (7, 12)
    # explicit kwargs win over the config
    assert OpenAICompatibleGenerator.from_config(cfg, WordTokenizer(), seed=None).seed is None
