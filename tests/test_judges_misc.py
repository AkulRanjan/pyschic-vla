"""Offline tests for uniform/constant/hybrid judges, the factory, and
truncation/shuffle (spec A4.6) — no network, no model load.
"""

import pytest

from thoughtzero.config import JudgeCfg
from thoughtzero.judge.factory import make_constant_value_judge, make_judge
from thoughtzero.judge.hybrid import HybridJudge
from thoughtzero.judge.jev import _shuffle_permutation, _truncate_steps
from thoughtzero.judge.uniform import ConstantValueJudge, UniformJudge


async def test_uniform_judge_priors_and_value():
    priors, value = await UniformJudge().prior_and_value("p", ["s1"], ["a", "b", "c"])
    assert priors == [pytest.approx(1 / 3)] * 3
    assert value == 0.5


async def test_constant_value_judge():
    priors, value = await ConstantValueJudge(0.9).prior_and_value("p", [], ["a", "b"])
    assert value == 0.9
    assert priors == [0.5, 0.5]


class _FakePriorJudge:
    """A fake judge used only to test HybridJudge's wiring, not Jev itself."""

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


async def test_hybrid_judge_uses_prior_only_when_available():
    prior_judge = _FakePriorJudge([0.2, 0.8])
    hybrid = HybridJudge(prior_from=prior_judge, value_from=ConstantValueJudge(0.4))
    priors, value = await hybrid.prior_and_value("p", ["s"], ["a", "b"])
    assert priors == [0.2, 0.8]
    assert value == 0.4
    assert prior_judge.prior_only_calls == 1


async def test_hybrid_judge_falls_back_to_prior_and_value_without_prior_only():
    hybrid = HybridJudge(prior_from=UniformJudge(), value_from=ConstantValueJudge(0.7))
    priors, value = await hybrid.prior_and_value("p", ["s"], ["a", "b", "c"])
    assert priors == [pytest.approx(1 / 3)] * 3
    assert value == 0.7


def test_factory_builds_uniform():
    assert isinstance(make_judge(JudgeCfg(kind="uniform")), UniformJudge)


def test_make_constant_value_judge_helper():
    j = make_constant_value_judge(0.3)
    assert isinstance(j, ConstantValueJudge)
    assert j.value == 0.3


def test_factory_builds_hybrid_from_named_sub_kinds():
    cfg = JudgeCfg(kind="hybrid", prior_from="uniform", value_from="uniform")
    j = make_judge(cfg)
    assert isinstance(j, HybridJudge)
    assert isinstance(j.prior_from, UniformJudge)
    assert isinstance(j.value_from, UniformJudge)


def test_factory_hybrid_without_prior_from_raises():
    cfg = JudgeCfg(kind="hybrid")
    with pytest.raises(ValueError, match="prior_from"):
        make_judge(cfg)


def test_factory_builds_jev():
    from thoughtzero.judge.jev import JevJudge

    j = make_judge(JudgeCfg(kind="jev", transport="local_stub"))
    assert isinstance(j, JevJudge)


def test_truncation_keeps_problem_and_last_steps_and_marks_truncated():
    problem = "p" * 100
    steps = ["x" * 500 for _ in range(10)]  # over the token budget, but each step still fits alone
    truncated_steps, was_truncated = _truncate_steps(problem, steps, max_state_tokens=1000)
    assert was_truncated
    assert truncated_steps[0] == steps[0]
    assert any("omitted" in s for s in truncated_steps)
    assert truncated_steps[-1] == steps[-1]


def test_no_truncation_for_short_state():
    problem = "What is 2 + 2?"
    steps = ["2 + 2 = 4"]
    truncated_steps, was_truncated = _truncate_steps(problem, steps, max_state_tokens=28_000)
    assert not was_truncated
    assert truncated_steps == steps


def test_shuffle_permutation_is_deterministic_and_a_valid_permutation():
    p1 = _shuffle_permutation("some state text", 5)
    p2 = _shuffle_permutation("some state text", 5)
    assert p1 == p2
    assert sorted(p1) == [0, 1, 2, 3, 4]


def test_shuffle_permutation_varies_with_seed():
    p1 = _shuffle_permutation("state A", 6)
    p2 = _shuffle_permutation("state B", 6)
    assert p1 != p2
