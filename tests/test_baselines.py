"""Baseline tests with a scripted mock generator and judge (no network)."""

from __future__ import annotations

import pytest

from thoughtzero.baselines import best_of_n, self_consistency
from thoughtzero.baselines.best_of_n import BestOfN
from thoughtzero.baselines.cot import ChainOfThought
from thoughtzero.baselines.self_consistency import SCSamples, SelfConsistency, majority_vote
from thoughtzero.eval.runner import Method
from thoughtzero.types import GenOut, Problem

PROBLEM = Problem(id="t/1", question="What is 2+2?", answer="4", source="test")


class ScriptedGenerator:
    """Returns pre-set solutions in order; records each sample_solutions call."""

    def __init__(self, solutions: list[str]) -> None:
        self.solutions = list(solutions)
        self.calls: list[tuple[int, float]] = []

    async def sample_solutions(self, problem: str, n: int, temperature: float) -> list[str]:
        self.calls.append((n, temperature))
        out, self.solutions = self.solutions[:n], self.solutions[n:]
        return out

    async def propose(self, problem: str, steps: list[str], k: int) -> list[GenOut]:
        raise NotImplementedError

    async def complete(self, problem: str, steps: list[str], temperature: float = 0.0) -> list[str]:
        raise NotImplementedError

    async def sample_completions(
        self, problem: str, steps: list[str], n: int, temperature: float
    ) -> list[list[str]]:
        raise NotImplementedError


class ScoreJudge:
    """final_correct = a fixed score looked up by the boxed answer in the last step."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.seen: list[list[str]] = []

    async def final_correct(self, problem: str, steps: list[str]) -> float:
        self.seen.append(steps)
        for ans, s in self.scores.items():
            if f"\\boxed{{{ans}}}" in steps[-1]:
                return s
        return 0.0

    async def prior_and_value(
        self, problem: str, steps: list[str], candidates: list[str]
    ) -> tuple[list[float], float]:
        raise NotImplementedError

    async def step_sound(self, problem: str, steps: list[str]) -> float:
        raise NotImplementedError


def sol(ans: str) -> str:
    return f"Step 1: Think.\n\nStep 2: So \\boxed{{{ans}}}"


def words(text: str) -> int:
    return len(text.split())


def test_classes_satisfy_method_protocol() -> None:
    gen = ScriptedGenerator([])
    methods: list[Method] = [
        ChainOfThought(gen),
        SelfConsistency(gen, n=4, count_tokens=words),
        BestOfN(ScoreJudge({}), n=4, generator=gen, count_tokens=words),
    ]
    assert [m.name for m in methods] == ["cot", "self_consistency", "best_of_n"]


# -------------------------------------------------------------------- voting


def test_majority_vote_basic() -> None:
    assert majority_vote(["4", "5", "4"]) == "4"


def test_majority_vote_groups_by_normalized_form() -> None:
    # "\frac{1}{2}" and "\dfrac12" normalize alike and outvote "3".
    assert majority_vote(["3", r"\frac{1}{2}", r"\dfrac12"]) == r"\frac{1}{2}"


def test_majority_vote_tie_goes_to_earliest_first_occurrence() -> None:
    assert majority_vote(["5", "4", "4", "5"]) == "5"
    assert majority_vote(["4", "5", "5", "4"]) == "4"


def test_majority_vote_ignores_none_and_empty() -> None:
    assert majority_vote([None, None, "7"]) == "7"
    assert majority_vote([None, ""]) is None
    assert majority_vote([]) is None


# ----------------------------------------------------------------- B1 / C


async def test_cot_is_greedy_single_sample() -> None:
    gen = ScriptedGenerator([sol("4")])
    res = await ChainOfThought(gen, where="vllm-local/fp16", name="ceiling_31b").solve(PROBLEM)
    assert gen.calls == [(1, 0.0)]
    assert res.answer == "4"
    assert res.raw == {"solution": sol("4"), "where": "vllm-local/fp16"}


async def test_cot_no_answer() -> None:
    res = await ChainOfThought(ScriptedGenerator(["Step 1: I give up."])).solve(PROBLEM)
    assert res.answer is None


# ---------------------------------------------------------------------- B2


async def test_self_consistency_batches_and_counts_tokens() -> None:
    sols = [sol(a) for a in ["4", "5", "4", "5", "5"]]
    gen = ScriptedGenerator(sols)
    s = await self_consistency.sample_self_consistency(
        gen, PROBLEM, n_max=5, count_tokens=words, batch_size=2
    )
    assert [n for n, _ in gen.calls] == [2, 2, 1]
    assert all(t == 0.7 for _, t in gen.calls)
    assert s.solutions == sols
    assert s.answers == ["4", "5", "4", "5", "5"]
    # "Step 1:" belongs to the prompt and is not counted.
    assert s.completion_tokens == [words(x.removeprefix("Step 1:")) for x in sols]


async def test_self_consistency_method_votes_at_n_but_stores_n_max() -> None:
    sols = [sol(a) for a in ["4", "5", "4", "5", "5"]]
    m = SelfConsistency(ScriptedGenerator(sols), n=3, n_max=5, count_tokens=words)
    res = await m.solve(PROBLEM)

    assert res.answer == "4"  # vote over the first 3
    assert res.raw["n"] == 3
    assert len(res.raw["solutions"]) == 5  # all samples kept for the post-hoc curve
    stored = SCSamples.from_raw(res.raw)
    assert stored.vote(5) == "5"
    assert res.raw["completion_tokens_at_n"] == stored.tokens(3)


def test_self_consistency_rejects_n_max_below_n() -> None:
    with pytest.raises(ValueError):
        SelfConsistency(ScriptedGenerator([]), n=4, n_max=2, count_tokens=words)


def test_prefix_subsampling_curve() -> None:
    s = SCSamples(
        solutions=["a", "b", "c", "d", "e"],
        answers=["4", "5", "5", None, "4"],
        completion_tokens=[10, 20, 30, 40, 50],
    )
    assert s.vote(1) == "4"
    assert s.vote(2) == "4"  # tie 1-1: earliest first occurrence
    assert s.vote(3) == "5"
    assert s.vote(4) == "5"  # None ignored
    assert s.vote(5) == "4"  # tie 2-2: "4" appeared first
    assert self_consistency.curve(s, [1, 3, 5]) == {1: ("4", 10), 3: ("5", 60), 5: ("4", 150)}
    with pytest.raises(ValueError):
        s.vote(6)


def test_sc_samples_roundtrip_raw() -> None:
    s = SCSamples(solutions=["a"], answers=[None], completion_tokens=[3], temperature=0.5)
    assert SCSamples.from_raw(s.to_raw()) == s


# ---------------------------------------------------------------------- B3


async def test_best_of_n_picks_highest_judge_score_from_stored_samples() -> None:
    samples = SCSamples(
        solutions=[sol("5"), sol("4"), sol("5")],
        answers=["5", "4", "5"],
        completion_tokens=[1, 1, 1],
    )
    judge = ScoreJudge({"4": 0.9, "5": 0.2})
    res = await BestOfN(judge, n=3, stored={PROBLEM.id: samples}).solve(PROBLEM)

    assert res.answer == "4"  # the minority answer wins on judge score
    assert res.raw["picked"] == 1
    assert res.raw["reused_samples"] is True
    assert len(judge.seen) == 3
    assert judge.seen[0] == ["Think.", "So \\boxed{5}"]  # judged as split steps


async def test_best_of_n_samples_fresh_when_not_stored() -> None:
    gen = ScriptedGenerator([sol("5"), sol("4")])
    res = await BestOfN(ScoreJudge({"4": 0.9}), n=2, generator=gen, count_tokens=words).solve(
        PROBLEM
    )
    assert res.answer == "4"
    assert res.raw["reused_samples"] is False
    assert gen.calls == [(2, 0.7)]


def test_best_of_n_needs_samples_or_generator() -> None:
    with pytest.raises(ValueError):
        BestOfN(ScoreJudge({}), n=2)


def test_pick_best_ties_and_prefix() -> None:
    scores = [0.5, 0.9, 0.9, 0.95]
    assert best_of_n.pick_best(scores, 3) == 1  # tie: earliest
    assert best_of_n.pick_best(scores, 4) == 3
    assert best_of_n.pick_best(scores, 1) == 0
    with pytest.raises(ValueError):
        best_of_n.pick_best(scores, 0)


def test_best_of_n_curve() -> None:
    samples = SCSamples(
        solutions=["a", "b", "c"], answers=["1", None, "3"], completion_tokens=[5, 5, 5]
    )
    assert best_of_n.curve(samples, [0.1, 0.8, 0.5], [1, 2, 3]) == {
        1: ("1", 5),
        2: (None, 10),  # the best-scored sample may have no answer
        3: (None, 15),
    }


def test_estimate_usd() -> None:
    samples = SCSamples(solutions=["x" * 400] * 2, answers=["1", "1"], completion_tokens=[1, 1])
    p = Problem(id="t/2", question="q" * 100, answer="1", source="t")
    # (100*2 + 800) chars / 4 = 250 tokens
    assert best_of_n.estimate_usd([p], [samples]) == pytest.approx(250 * 0.042 / 1e6)
