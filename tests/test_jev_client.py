"""Offline tests for the generic HTTP retry/semaphore machinery and the
local_stub request/response shape translation (spec A4.6) — no network,
no model load (respx mocks HTTP; the local_stub backend is monkeypatched).
"""

import asyncio

import httpx
import pytest

from thoughtzero.config import JudgeCfg
from thoughtzero.judge import budget as budget_module
from thoughtzero.judge.client import (
    PROVIDERS,
    JevClient,
    JevConfigError,
    JevHTTPError,
    request_with_retry,
)
from thoughtzero.judge.jev import JevJudge


@pytest.mark.respx(base_url="https://example.test")
async def test_429_then_200_succeeds_after_retry(respx_mock):
    route = respx_mock.get("/x").mock(
        side_effect=[httpx.Response(429), httpx.Response(200, json={"ok": True})]
    )
    async with httpx.AsyncClient(base_url="https://example.test") as client:
        resp = await request_with_retry(client, "GET", "/x")
    assert resp.json() == {"ok": True}
    assert route.call_count == 2


@pytest.mark.respx(base_url="https://example.test")
async def test_500_times_5_raises(respx_mock):
    route = respx_mock.get("/x").mock(return_value=httpx.Response(500))
    async with httpx.AsyncClient(base_url="https://example.test") as client:
        with pytest.raises(JevHTTPError):
            await request_with_retry(client, "GET", "/x")
    assert route.call_count == 5


@pytest.mark.respx(base_url="https://example.test")
async def test_400_is_not_retried(respx_mock):
    route = respx_mock.get("/x").mock(return_value=httpx.Response(400, text="bad field name"))
    async with httpx.AsyncClient(base_url="https://example.test") as client:
        with pytest.raises(JevHTTPError, match="400"):
            await request_with_retry(client, "GET", "/x")
    assert route.call_count == 1


@pytest.mark.parametrize("transport", sorted(PROVIDERS))
def test_missing_api_key_fails_at_construction_with_the_env_var_name(transport, monkeypatch):
    monkeypatch.delenv(PROVIDERS[transport].key_env, raising=False)
    with pytest.raises(JevConfigError, match=PROVIDERS[transport].key_env):
        JevClient(JudgeCfg(transport=transport))


async def test_local_stub_maps_question_ids_to_answers(monkeypatch):
    from thoughtzero.judge import client as client_module

    async def fake_decide(self, state, questions):
        assert state == "a state"
        assert questions[1]["options"] == {"c0": "x"}  # the stand-in's name for criteria
        return [{"noul": 0.7}, {"choice": "c0", "probabilities": {"c0": 1.0}}]

    monkeypatch.setattr(client_module._LocalStubBackend, "decide", fake_decide)

    client = JevClient(JudgeCfg(transport="local_stub"))
    reply = await client.ask(
        "a state",
        {
            "sound": {"type": "noul", "instructions": "is it sound?"},
            "next": {"type": "choice", "instructions": "pick", "criteria": {"c0": "x"}},
        },
    )
    assert reply.input_tokens is None
    assert reply.answers == {
        "sound": {"noul": 0.7},
        "next": {"choice": "c0", "probabilities": {"c0": 1.0}},
    }


async def test_semaphore_caps_concurrency(monkeypatch):
    from thoughtzero.judge import client as client_module

    in_flight = 0
    max_in_flight = 0

    async def fake_decide(self, state, questions):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return [{"noul": 0.5}]

    monkeypatch.setattr(client_module._LocalStubBackend, "decide", fake_decide)

    client = JevClient(JudgeCfg(transport="local_stub", max_concurrency=2))
    questions = {"sound": {"type": "noul", "instructions": "x"}}
    await asyncio.gather(*(client.ask("s", questions) for _ in range(8)))
    assert max_in_flight <= 2


async def test_budget_reservation_released_when_client_call_fails(monkeypatch):
    """A failed call must not leak its reservation, or repeated failures
    eventually trip a false BudgetExceeded for calls that never happened."""
    from thoughtzero.judge import client as client_module

    async def failing_decide(self, state, questions):
        raise RuntimeError("simulated backend failure")

    monkeypatch.setattr(client_module._LocalStubBackend, "decide", failing_decide)

    budget_module.configure(max_usd=1.0, usd_per_mtok=0.042)
    try:
        judge = JevJudge(JudgeCfg(transport="local_stub"))
        with pytest.raises(RuntimeError, match="simulated backend failure"):
            await judge.step_sound("a problem never seen before", ["a step never seen before"])
        assert budget_module.current_guard._reserved_usd == 0.0
        assert budget_module.current_guard.spent_usd == 0.0
    finally:
        budget_module.current_guard = None


# --------------------------------------------------------------------------- real HTTP routes
# Response shapes follow the official API reference (docs.typesafe.ai/api.md).

URL = PROVIDERS["direct"].url
REAL_RESPONSE = {
    "model": "jev-1.13.0",
    "answers": {
        "sound": {"type": "noul", "noul": 0.8},
        "next": {
            "type": "choice",
            "choice": "c1",
            "probabilities": {"c0": 0.2, "c1": 0.7, "c2": 0.1},
            "confidence": 0.6,
        },
    },
    "usage": {"input_tokens": 321, "output_tokens": 30},
}


@pytest.fixture
def direct_judge(monkeypatch, tmp_path):
    from thoughtzero.judge.cache import DiskCache

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key-123")
    judge = JevJudge(JudgeCfg(transport="direct"))
    judge.cache = DiskCache(str(tmp_path / "cache"))
    return judge


async def test_http_request_matches_the_documented_protocol(direct_judge, respx_mock):
    import json

    route = respx_mock.post(URL).mock(return_value=httpx.Response(200, json=REAL_RESPONSE))
    priors, value = await direct_judge.prior_and_value("2+2?", ["2+2 = 4"], ["a", "b", "c"])

    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer test-key-123"
    body = json.loads(request.content)
    assert body["model"] == "jev-1.13.0"  # pinned, not the floating jev-latest
    assert body["state"].startswith("PROBLEM:\n2+2?")
    assert body["questions"]["sound"]["type"] == "noul"
    nxt = body["questions"]["next"]
    assert nxt["type"] == "choice" and nxt["criteria"] == {"c0": "a", "c1": "b", "c2": "c"}
    assert priors == pytest.approx([0.2, 0.7, 0.1]) and value == 0.8


async def test_reported_usage_is_what_gets_billed(direct_judge, respx_mock):
    from thoughtzero.accounting import ledger_scope

    respx_mock.post(URL).mock(return_value=httpx.Response(200, json=REAL_RESPONSE))
    with ledger_scope() as ledger:
        await direct_judge.prior_and_value("p", ["s"], ["a", "b", "c"])
        await direct_judge.prior_and_value("p", ["s"], ["a", "b", "c"])  # cache hit
    assert ledger.jev_calls == 1 and ledger.jev_cache_hits == 1
    assert ledger.jev_input_tokens == 321
    assert ledger.jev_usd == pytest.approx(321 * 0.042 / 1e6)


async def test_overloaded_529_is_retried_and_401_is_not(direct_judge, respx_mock):
    route = respx_mock.post(URL).mock(
        side_effect=[httpx.Response(529), httpx.Response(200, json=REAL_RESPONSE)]
    )
    await direct_judge.prior_and_value("p", ["s1"], ["a", "b", "c"])
    assert route.call_count == 2

    route = respx_mock.post(URL).mock(return_value=httpx.Response(401, json={"error": "bad key"}))
    before = route.call_count
    with pytest.raises(JevHTTPError, match="401"):
        await direct_judge.step_sound("p", ["a different step"])
    assert route.call_count - before == 1


async def test_response_without_answers_is_an_error(direct_judge, respx_mock):
    respx_mock.post(URL).mock(return_value=httpx.Response(200, json={"error": "?"}))
    with pytest.raises(JevHTTPError, match="answers"):
        await direct_judge.step_sound("p", ["s"])


async def test_single_candidate_needs_no_choice_question(direct_judge, respx_mock):
    route = respx_mock.post(URL).mock(return_value=httpx.Response(200, json=REAL_RESPONSE))
    priors, value = await direct_judge.prior_and_value("p", [], ["only one"])  # root
    assert priors == [1.0] and value == direct_judge.cfg.root_value
    assert route.call_count == 0  # nothing to ask: no request, no cost
    priors, _ = await direct_judge.prior_and_value("p", ["s"], ["only one"])
    assert priors == [1.0] and "next" not in route.calls.last.request.content.decode()


@pytest.mark.parametrize("transport", sorted(PROVIDERS))
def test_every_route_resolves_url_model_and_overrides(transport, monkeypatch):
    from thoughtzero.judge.client import resolve_route

    monkeypatch.setenv(PROVIDERS[transport].key_env, "k")
    client = JevClient(JudgeCfg(transport=transport))
    assert (client.url, client.model) == (PROVIDERS[transport].url, PROVIDERS[transport].model)
    custom = JudgeCfg(transport=transport, jev_model="m", jev_url="https://x.test/v1/systemone")
    assert resolve_route(custom) == ("https://x.test/v1/systemone", "m")


async def test_different_models_never_share_cached_answers(monkeypatch, tmp_path, respx_mock):
    from thoughtzero.judge.cache import DiskCache

    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    monkeypatch.setenv("BOCHA_API_KEY", "k")
    cache = DiskCache(str(tmp_path / "cache"))
    real, other = JevJudge(JudgeCfg(transport="direct")), JevJudge(JudgeCfg(transport="bocha"))
    real.cache = other.cache = cache
    a = respx_mock.post(URL).mock(return_value=httpx.Response(200, json=REAL_RESPONSE))
    b = respx_mock.post(PROVIDERS["bocha"].url).mock(
        return_value=httpx.Response(200, json=REAL_RESPONSE)
    )
    await real.step_sound("p", ["s"])
    await other.step_sound("p", ["s"])
    assert a.call_count == 1 and b.call_count == 1
