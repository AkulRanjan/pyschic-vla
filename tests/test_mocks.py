import pytest

from thoughtzero.accounting import ledger_scope
from thoughtzero.mocks import (
    TARGET,
    TOY,
    MockFailure,
    MockGenerator,
    MockJudge,
    greedy_by_prior,
    running_total,
    toy_extract_answer,
)
from thoughtzero.types import Generator, Judge


def test_mocks_satisfy_protocols() -> None:
    assert isinstance(MockGenerator(), Generator)
    assert isinstance(MockJudge(), Judge)


async def test_propose_is_deterministic() -> None:
    a = await MockGenerator(seed=0).propose(TOY.question, ["add 5 -> 5"], 3)
    b = await MockGenerator(seed=0).propose(TOY.question, ["add 5 -> 5"], 3)
    assert [o.text for o in a] == [o.text for o in b]
    assert sorted(o.text for o in a) == ["add 1 -> 6", "add 2 -> 7", "add 5 -> 10"]


async def test_propose_k_larger_than_increments_has_duplicates() -> None:
    outs = await MockGenerator().propose(TOY.question, [], 6)
    assert len(outs) == 6
    assert len({o.text for o in outs}) == 3


async def test_judge_prior_favours_plus_five_and_sums_to_one() -> None:
    cands = ["add 1 -> 1", "add 5 -> 5", "add 2 -> 2"]
    priors, value = await MockJudge().prior_and_value(TOY.question, [], cands)
    assert abs(sum(priors) - 1) < 1e-9
    assert max(range(3), key=lambda i: priors[i]) == 1
    assert value == 0.9


async def test_judge_values() -> None:
    judge = MockJudge()
    assert await judge.final_correct(TOY.question, ["add 5 -> 13. \\boxed{13}"]) == 0.95
    assert await judge.final_correct(TOY.question, ["add 5 -> 15. \\boxed{15}"]) == 0.05
    assert await judge.step_sound(TOY.question, ["add 5 -> 15"]) == 0.1


async def test_greedy_by_prior_is_wrong_on_toy() -> None:
    for seed in range(5):
        answer = await greedy_by_prior(TOY.question, MockGenerator(seed), MockJudge(seed))
        assert answer == "15" != TOY.answer


async def test_complete_and_samples_reach_a_boxed_answer() -> None:
    gen = MockGenerator()
    greedy = await gen.complete(TOY.question, ["add 2 -> 2"])
    assert greedy == ["add 5 -> 7", "add 5 -> 12", "add 5 -> 17. \\boxed{17}"]
    samples = await gen.sample_completions(TOY.question, [], n=8, temperature=0.7)
    assert len(samples) == 8
    assert all(toy_extract_answer(s[-1]) is not None for s in samples)
    assert any(running_total(s) == TARGET for s in samples)
    solutions = await gen.sample_solutions(TOY.question, n=2, temperature=0.7)
    assert all(s.startswith("Step 1: ") for s in solutions)


async def test_mocks_record_into_ledger() -> None:
    with ledger_scope() as ledger:
        await greedy_by_prior(TOY.question, MockGenerator(), MockJudge())
    assert ledger.gemma_calls == 3 and ledger.jev_calls == 3
    assert ledger.gemma_completion_tokens > 0 and ledger.jev_usd > 0


async def test_fail_rate_injects_failures_and_counts_calls() -> None:
    judge = MockJudge(fail_rate=1.0)
    with pytest.raises(MockFailure):
        await judge.step_sound(TOY.question, [])
    assert judge.calls["step_sound"] == 1
