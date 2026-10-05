"""Offline cache + budget tests (spec A4.6) — no network, no model load."""

import pytest

from thoughtzero.judge import budget as budget_module
from thoughtzero.judge.budget import BudgetExceeded, BudgetGuard
from thoughtzero.judge.cache import DiskCache, cache_key


def test_cache_key_changes_with_prompt_version():
    k1 = cache_key("state", {"q": {"type": "noul"}}, "model-a", "v1")
    k2 = cache_key("state", {"q": {"type": "noul"}}, "model-a", "v2")
    assert k1 != k2


def test_cache_key_stable_for_same_inputs():
    # transport is deliberately not part of the key inputs at all
    k1 = cache_key("state", {"q": {"type": "noul"}}, "model-a", "v1")
    k2 = cache_key("state", {"q": {"type": "noul"}}, "model-a", "v1")
    assert k1 == k2


def test_disk_cache_roundtrip(tmp_path):
    cache = DiskCache(directory=str(tmp_path / "cache"))
    key = cache_key("state", {"q": {"type": "noul"}}, "model-a", "v1")
    assert cache.get(key) is None

    cache.set(key, {"answers": {"sound": {"noul": 0.5}}, "input_tokens": 10})
    got = cache.get(key)
    assert got is not None
    assert got["answers"] == {"sound": {"noul": 0.5}}


def test_cache_export_import_roundtrip(tmp_path):
    cache1 = DiskCache(directory=str(tmp_path / "cache1"))
    key = cache_key("state", {"q": {"type": "noul"}}, "model-a", "v1")
    cache1.set(key, {"answers": {"sound": {"noul": 0.1}}, "input_tokens": 1})

    export_path = tmp_path / "export.jsonl"
    n = cache1.export(str(export_path))
    assert n == 1

    cache2 = DiskCache(directory=str(tmp_path / "cache2"))
    n_imported = cache2.import_(str(export_path))
    assert n_imported == 1
    assert cache2.get(key)["answers"] == {"sound": {"noul": 0.1}}

    # import again: existing key is skipped
    assert cache2.import_(str(export_path)) == 0


def test_budget_raises_at_cap():
    guard = BudgetGuard(max_usd=0.0001, usd_per_mtok=0.042)
    # ~2381 tokens @ $0.042/1M = $0.0001
    guard.check(2000)
    guard.record(2000, 2000)
    with pytest.raises(BudgetExceeded):
        guard.check(100_000)


def test_budget_reserve_before_settle_prevents_overshoot():
    """Two calls for 1000 tokens each against a budget sized for exactly
    one: the first check reserves, so the second's check (before its own
    record runs) correctly sees that reservation and fails."""
    guard_usd = 1000 * 0.042 / 1e6
    guard = BudgetGuard(max_usd=guard_usd, usd_per_mtok=0.042)
    guard.check(1000)  # reserves the whole budget
    with pytest.raises(BudgetExceeded):
        guard.check(1000)  # a second concurrent call must not also pass
    guard.record(1000, 1000)
    assert guard.spent_usd == pytest.approx(guard_usd)


def test_configure_sets_module_level_guard():
    budget_module.configure(max_usd=1.0, usd_per_mtok=0.042)
    assert budget_module.current_guard is not None
    assert budget_module.current_guard.max_usd == 1.0
    budget_module.current_guard = None


def test_estimate_run_cost():
    assert budget_module.estimate_run_cost(1_000_000, 1.0, usd_per_mtok=0.042) == pytest.approx(
        0.042
    )


@pytest.mark.parametrize("mock", [False, True])
def test_run_scripts_arm_the_budget_cap_for_real_runs(mock, monkeypatch):
    import argparse

    from thoughtzero.eval.cli import add_run_args, setup

    monkeypatch.setattr(budget_module, "current_guard", None)
    parser = argparse.ArgumentParser()
    add_run_args(parser, "configs/default.yaml")
    setup(parser.parse_args(["--set", "budget.max_usd=2.5", *(["--mock"] if mock else [])]))
    if mock:
        assert budget_module.current_guard is None
    else:
        assert budget_module.current_guard is not None
        assert budget_module.current_guard.max_usd == 2.5
