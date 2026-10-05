"""Offline tests for uniform/constant/hybrid judges, the factory, and
truncation (spec A4.6) — no network, no model load.
"""

import asyncio

import pytest

from thoughtzero.judge.factory import make_judge
from thoughtzero.judge.hybrid import HybridJudge
from thoughtzero.judge.jev import _truncate_state
from thoughtzero.judge.uniform import ConstantValueJudge, UniformJudge


def run(coro):
    return asyncio.run(coro)


def test_uniform_judge_priors_and_value():
    priors, value = run(UniformJudge().prior_and_value("p", ["s1"], ["a", "b", "c"]))
    assert priors == [pytest.approx(1 / 3)] * 3
    assert value == 0.5


def test_constant_value_judge():
    priors, value = run(ConstantValueJudge(0.9).prior_and_value("p", [], ["a", "b"]))
    assert value == 0.9
    assert priors == [0.5, 0.5]


class _FakePriorJudge:
    """A fake judge used only to test HybridJudge wiring, not Jev itself."""

    def __init__(self, priors: list[float]) -> None:
        self._priors = priors
        self.prior_only_calls = 0

    async def prior_only(self, problem, steps, candidates):
        self.prior_only_calls += 1
        return self._priors

    async def prior_and_value(self, problem, steps, candidates):
        raise AssertionError("should have used prior_only, not prior_and_value")

    async def final_correct(self, problem, steps):
        return 1.0

    async def step_sound(self, problem, steps):
        return 1.0


def test_hybrid_judge_uses_prior_only_when_available():
    prior_judge = _FakePriorJudge([0.2, 0.8])
    hybrid = HybridJudge(prior_from=prior_judge, value_from=ConstantValueJudge(0.4))
    priors, value = run(hybrid.prior_and_value("p", ["s"], ["a", "b"]))
    assert priors == [0.2, 0.8]
    assert value == 0.4
    assert prior_judge.prior_only_calls == 1


def test_hybrid_judge_falls_back_to_prior_and_value_without_prior_only():
    hybrid = HybridJudge(prior_from=UniformJudge(), value_from=ConstantValueJudge(0.7))
    priors, value = run(hybrid.prior_and_value("p", ["s"], ["a", "b", "c"]))
    assert priors == [pytest.approx(1 / 3)] * 3
    assert value == 0.7


def test_factory_builds_uniform_and_constant():
    assert isinstance(make_judge({"kind": "uniform"}), UniformJudge)
    j = make_judge({"kind": "constant", "value": 0.3})
    assert isinstance(j, ConstantValueJudge)
    assert j.value == 0.3


def test_factory_builds_hybrid_recursively():
    j = make_judge(
        {
            "kind": "hybrid",
            "prior_from": {"kind": "uniform"},
            "value_from": {"kind": "constant", "value": 0.1},
        }
    )
    assert isinstance(j, HybridJudge)
    assert isinstance(j.prior_from, UniformJudge)
    assert isinstance(j.value_from, ConstantValueJudge)


def test_factory_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown judge kind"):
        make_judge({"kind": "nonsense"})


def test_truncation_keeps_problem_and_last_steps_and_marks_truncated():
    problem = "p" * 100
    steps = ["x" * 20000 for _ in range(10)]  # way over MAX_STATE_CHARS
    truncated_steps, was_truncated = _truncate_state(problem, steps)
    assert was_truncated
    assert truncated_steps[0] == steps[0]
    assert any("omitted" in s for s in truncated_steps)
    assert truncated_steps[-1] == steps[-1]


def test_no_truncation_for_short_state():
    problem = "What is 2 + 2?"
    steps = ["2 + 2 = 4"]
    truncated_steps, was_truncated = _truncate_state(problem, steps)
    assert not was_truncated
    assert truncated_steps == steps
