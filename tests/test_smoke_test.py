"""scripts/smoke_test.py: the real-run logic, driven by the mocks (no server, no key)."""

import importlib.util
import sys
from pathlib import Path

import pytest

from thoughtzero.config import Config, SearchCfg
from thoughtzero.mocks import MockGenerator, MockJudge

_path = Path(__file__).resolve().parents[1] / "scripts" / "smoke_test.py"
_spec = importlib.util.spec_from_file_location("smoke_test", _path)
assert _spec is not None and _spec.loader is not None
smoke = importlib.util.module_from_spec(_spec)
sys.modules["smoke_test"] = smoke  # @dataclass looks its module up here
_spec.loader.exec_module(smoke)


async def test_real_run_reports_each_problem_and_totals() -> None:
    cfg = Config(search=SearchCfg(n_simulations=4, k=3))
    summary = await smoke.run_real(cfg, MockGenerator(), MockJudge())
    assert [r["id"] for r in summary.rows] == [p[0] for p in smoke.SMOKE_PROBLEMS]
    assert summary.jev_calls == sum(r["jev_calls"] for r in summary.rows) > 0
    assert summary.jev_calls <= 3 * 4  # at most one Jev call per simulation
    assert summary.solved == 0  # the toy mocks can't solve real problems


@pytest.mark.parametrize(
    ("solved", "calls", "usd", "passed"),
    [(1, 20, 0.01, True), (0, 5, 0.0, False), (3, 21, 0.0, False), (3, 5, 0.011, False)],
)
def test_pass_criteria(solved: int, calls: int, usd: float, passed: bool) -> None:
    assert smoke.SmokeSummary(solved, calls, usd, []).passed is passed


def test_unreachable_server_is_reported() -> None:
    assert smoke._check_gemma_server("http://127.0.0.1:9/v1") is not None
