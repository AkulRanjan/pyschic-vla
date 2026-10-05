"""Offline tests for the generic HTTP retry/semaphore machinery and the
local_stub request/response shape translation (spec A4.6) — no network,
no model load (respx mocks HTTP; the local_stub backend is monkeypatched).
"""

import asyncio

import httpx
import pytest

from thoughtzero.config import JudgeCfg
from thoughtzero.judge import budget as budget_module
from thoughtzero.judge.client import JevClient, JevHTTPError, request_with_retry
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


async def test_direct_transport_raises_not_implemented_with_pointer_to_docs():
    client = JevClient(JudgeCfg(transport="direct"))
    with pytest.raises(NotImplementedError, match="verified_apis"):
        await client.ask("state", {"q": {"type": "noul", "instructions": "x"}})


async def test_local_stub_maps_question_ids_to_answers(monkeypatch):
    from thoughtzero.judge import client as client_module

    async def fake_decide(self, state, questions):
        assert state == "a state"
        return [{"noul": 0.7}, {"choice": "c0", "probabilities": {"c0": 1.0}}]

    monkeypatch.setattr(client_module._LocalStubBackend, "decide", fake_decide)

    client = JevClient(JudgeCfg(transport="local_stub"))
    answers = await client.ask(
        "a state",
        {
            "sound": {"type": "noul", "instructions": "is it sound?"},
            "next": {"type": "choice", "instructions": "pick", "options": {"c0": "x"}},
        },
    )
    assert answers == {
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
