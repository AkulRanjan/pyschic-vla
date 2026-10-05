"""Offline cache + budget tests (spec A4.6) — no network, no model load."""

import asyncio

import pytest

from thoughtzero.judge.budget import BudgetExceeded, BudgetGuard, usd
from thoughtzero.judge.cache import CacheEntry, DiskCache, cache_key


def test_cache_key_changes_with_prompt_version():
    k1 = cache_key("state", [{"id": "q"}], "model-a", "v1")
    k2 = cache_key("state", [{"id": "q"}], "model-a", "v2")
    assert k1 != k2


def test_cache_key_stable_across_transport_choice():
    # transport is deliberately not part of the key inputs at all
    k1 = cache_key("state", [{"id": "q"}], "model-a", "v1")
    k2 = cache_key("state", [{"id": "q"}], "model-a", "v1")
    assert k1 == k2


def test_disk_cache_roundtrip(tmp_path):
    cache = DiskCache(directory=str(tmp_path / "cache"))
    key = cache_key("state", [{"id": "q"}], "model-a", "v1")
    assert cache.get(key) is None

    entry = CacheEntry(
        raw_response=[{"noul": 0.5}], parsed=None, input_tokens=10, latency_ms=5.0, created_at=0.0
    )
    cache.set(key, entry)
    got = cache.get(key)
    assert got is not None
    assert got.raw_response == [{"noul": 0.5}]


def test_cache_export_import_roundtrip(tmp_path):
    cache1 = DiskCache(directory=str(tmp_path / "cache1"))
    key = cache_key("state", [{"id": "q"}], "model-a", "v1")
    cache1.set(
        key,
        CacheEntry(
            raw_response=[{"noul": 0.1}],
            parsed=None,
            input_tokens=1,
            latency_ms=1.0,
            created_at=0.0,
        ),
    )
    export_path = tmp_path / "export.jsonl"
    n = cache1.export(str(export_path))
    assert n == 1

    cache2 = DiskCache(directory=str(tmp_path / "cache2"))
    n_imported = cache2.import_(str(export_path))
    assert n_imported == 1
    assert cache2.get(key).raw_response == [{"noul": 0.1}]

    # import again: existing key is skipped
    n_imported_again = cache2.import_(str(export_path))
    assert n_imported_again == 0


def test_usd_price():
    assert usd(1_000_000) == pytest.approx(0.042)


def test_budget_raises_at_cap():
    async def run():
        guard = BudgetGuard(max_usd=0.0001, spend_log="/dev/null")
        # ~2381 tokens @ $0.042/1M = $0.0001
        estimate = await guard.check(2000)
        await guard.record(estimate, 2000)
        with pytest.raises(BudgetExceeded):
            await guard.check(100_000)

    asyncio.run(run())


def test_budget_concurrent_calls_dont_overshoot(tmp_path):
    async def run():
        guard = BudgetGuard(max_usd=usd(16 * 1000) + 1e-9, spend_log=str(tmp_path / "spend.jsonl"))

        async def one_call():
            estimate = await guard.check(1000)
            await asyncio.sleep(0.001)
            await guard.record(estimate, 1000)

        # 16 concurrent calls at exactly the budget boundary; a 17th must fail
        await asyncio.gather(*(one_call() for _ in range(16)))
        with pytest.raises(BudgetExceeded):
            await guard.check(1000)

    asyncio.run(run())


def test_budget_reservation_prevents_concurrent_overshoot(tmp_path):
    async def run():
        guard = BudgetGuard(max_usd=usd(1000), spend_log=str(tmp_path / "spend.jsonl"))
        results = []

        async def try_call():
            try:
                estimate = await guard.check(1000)
            except BudgetExceeded:
                results.append(False)
                return
            await asyncio.sleep(0.001)
            await guard.record(estimate, 1000)
            results.append(True)

        await asyncio.gather(*(try_call() for _ in range(5)))
        # only 1 of 5 concurrent $1000-token calls should fit a $1000-token budget
        assert sum(results) == 1

    asyncio.run(run())
