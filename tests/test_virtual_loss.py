"""Async MCTS with virtual loss (SPEC.md §6.2; team/Prakhar.md §A5.2)."""

import asyncio
import gc
import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from thoughtzero.config import SearchCfg
from thoughtzero.mocks import TOY, MockFailure, MockGenerator, MockJudge, toy_extract_answer
from thoughtzero.search.mcts import SearchResult, _Ctx, _simulate, search
from thoughtzero.search.node import Node
from thoughtzero.types import GenOut

REFERENCE = Path(__file__).parent / "fixtures" / "search" / "sequential_reference.json"


async def run(
    n: int = 32,
    k: int = 3,
    seed: int = 0,
    gen: MockGenerator | None = None,
    judge: MockJudge | None = None,
    max_failures: int | None = None,
    **cfg_kwargs: Any,
) -> SearchResult:
    cfg = SearchCfg(n_simulations=n, k=k, **cfg_kwargs)
    return await search(
        TOY.question,
        gen or MockGenerator(seed),
        judge or MockJudge(seed),
        cfg,
        seed=seed,
        extract=toy_extract_answer,
        normalize=str.strip,
        max_failures=max_failures,
    )


def walk(node: Node) -> list[Node]:
    out = [node]
    for child in node.children:
        out.extend(walk(child))
    return out


def assert_clean(root: Node) -> None:
    """No virtual loss leaked and no evaluation left marked in flight."""
    for node in walk(root):
        assert node.virtual_loss == 0
        assert node.expanding is None


class UniformJudge(MockJudge):
    """Equal priors, so only virtual loss can separate concurrent simulations."""

    async def prior_and_value(
        self, problem: str, steps: list[str], candidates: list[str]
    ) -> tuple[list[float], float]:
        _, value = await super().prior_and_value(problem, steps, candidates)
        return [1 / len(candidates)] * len(candidates), value


class StateCountingGenerator(MockGenerator):
    """Counts ``propose`` calls per state, to check that no state is expanded twice."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.states: Counter[tuple[str, ...]] = Counter()

    async def propose(self, problem: str, steps: list[str], k: int) -> list[GenOut]:
        self.states[tuple(steps)] += 1
        return await super().propose(problem, steps, k)


class PeakGenerator(MockGenerator):
    """Records the largest number of ``propose`` calls in flight at once."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.in_flight = self.peak = 0

    async def propose(self, problem: str, steps: list[str], k: int) -> list[GenOut]:
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            return await super().propose(problem, steps, k)
        finally:
            self.in_flight -= 1


async def concurrent_below_root(vl: int) -> tuple[Node, StateCountingGenerator, _Ctx]:
    """Expand the root, then start 3 simulations at once (k=3 -> 3 root children)."""
    gen = StateCountingGenerator(latency_s=0.02)
    cfg = SearchCfg(n_simulations=4, k=3, virtual_loss=vl)
    ctx = _Ctx(TOY.question, gen, UniformJudge(), cfg, toy_extract_answer, random.Random(0))
    root = Node(steps=[])
    await _simulate(root, ctx)
    assert len(root.children) == 3
    await asyncio.gather(*(_simulate(root, ctx) for _ in range(3)))
    return root, gen, ctx


def case_id(case: dict[str, Any]) -> str:
    return f"seed{case['case']['seed']}"


@pytest.mark.parametrize("case", json.loads(REFERENCE.read_text("utf-8")), ids=case_id)
@pytest.mark.parametrize("latency", [0.0, 0.001])
async def test_one_parallel_sim_reproduces_sequential_search(
    case: dict[str, Any], latency: float
) -> None:
    # The fixture was recorded from the sequential Phase-3 search before async landed.
    cfg = dict(case["case"])
    seed = cfg.pop("seed")
    result = await search(
        TOY.question,
        MockGenerator(seed, latency_s=latency),
        MockJudge(seed, latency_s=latency),
        SearchCfg(parallel_sims=1, **cfg),
        seed=seed,
        extract=toy_extract_answer,
        normalize=str.strip,
    )
    assert result.tree_dump == case["tree_dump"]
    assert result.answer == case["answer"]


async def test_virtual_loss_spreads_concurrent_simulations() -> None:
    root, gen, ctx = await concurrent_below_root(vl=1)
    assert [c.N for c in root.children] == [1, 1, 1]  # each sim took a different child
    assert gen.calls["propose"] == 4 and ctx.shared_evals == 0
    assert_clean(root)


async def test_racing_simulations_share_one_expansion() -> None:
    # without virtual loss all three pick the same child (tie -> lowest index) and race
    root, gen, ctx = await concurrent_below_root(vl=0)
    shared = root.children[0]
    assert [c.N for c in root.children] == [3, 0, 0]
    assert gen.states[tuple(shared.steps)] == 1  # expanded once, not three times
    assert max(gen.states.values()) == 1  # no state anywhere was expanded twice
    assert ctx.shared_evals >= 2
    # the two simulations that waited then went on below the shared child
    assert sum(c.N for c in shared.children) == 2
    assert_clean(root)


async def test_concurrent_terminal_is_judged_once() -> None:
    judge = MockJudge(latency_s=0.02)
    cfg = SearchCfg(n_simulations=3, k=3, virtual_loss=0)
    ctx = _Ctx(TOY.question, MockGenerator(), judge, cfg, toy_extract_answer, random.Random(0))
    root = Node(steps=["add 5 -> 5", "add 5 -> 10", "add 2 -> 12", "add 1 -> 13. \\boxed{13}"])
    root.terminal = True
    await asyncio.gather(*(_simulate(root, ctx) for _ in range(3)))
    assert judge.calls["final_correct"] == 1
    assert root.N == 3 and root.value == 0.95
    assert_clean(root)


async def test_parallel_search_runs_concurrently_and_stays_consistent() -> None:
    gen = PeakGenerator(latency_s=0.005)
    result = await run(n=48, gen=gen, judge=MockJudge(latency_s=0.005), parallel_sims=4)
    root = result.root
    assert 1 < gen.peak <= 4
    assert root.N == 48 == result.stats["simulations"]
    assert result.stats["failed_simulations"] == 0
    for node in walk(root):
        if node.children:  # one expansion visit; every later visit passes to a child
            assert 1 + sum(c.N for c in node.children) == node.N
        if node.N:
            assert 0.0 <= node.Q <= 1.0
    assert_clean(root)


@pytest.mark.parametrize("seed", range(5))
async def test_parallel_search_still_beats_greedy(seed: int) -> None:
    latency = {"latency_s": 0.001}
    result = await run(
        n=64,
        seed=seed,
        gen=MockGenerator(seed, **latency),
        judge=MockJudge(seed, **latency),
        parallel_sims=8,
    )
    assert result.answer == TOY.answer


async def test_failures_are_survived_counted_and_retried(
    caplog: pytest.LogCaptureFixture,
) -> None:
    gen = MockGenerator(latency_s=0.002, fail_rate=0.3)
    judge = MockJudge(latency_s=0.002, fail_rate=0.3)
    with caplog.at_level(logging.WARNING):
        result = await run(n=48, gen=gen, judge=judge, parallel_sims=4)
        del gen, judge
        gc.collect()  # surfaces "Future exception was never retrieved" if we leaked one
    assert result.root.N == 48  # failed simulations were replaced
    assert result.stats["failed_simulations"] > 0
    assert_clean(result.root)
    assert not any("never retrieved" in r.getMessage() for r in caplog.records)


async def test_every_call_failing_reraises_after_the_failure_budget() -> None:
    gen = MockGenerator(fail_rate=1.0)
    with pytest.raises(MockFailure):
        await run(n=8, gen=gen, parallel_sims=4, max_failures=5)
    assert gen.calls["propose"] == 5


async def test_failure_budget_stops_search_early() -> None:
    class FailsAfterRoot(MockGenerator):
        async def propose(self, problem: str, steps: list[str], k: int) -> list[GenOut]:
            if steps:
                raise MockFailure("only the root expands")
            return await super().propose(problem, steps, k)

    result = await run(n=16, gen=FailsAfterRoot(), parallel_sims=2, max_failures=3)
    assert result.root.N == 1 and result.stats["failed_simulations"] == 3
    assert_clean(result.root)


async def test_cancellation_releases_virtual_loss() -> None:
    gen = MockGenerator(latency_s=10.0)
    cfg = SearchCfg(n_simulations=1, k=3)
    ctx = _Ctx(TOY.question, gen, MockJudge(), cfg, toy_extract_answer, random.Random(0))
    root = Node(steps=[])
    task = asyncio.ensure_future(_simulate(root, ctx))
    await asyncio.sleep(0.01)
    assert root.virtual_loss == 1 and root.expanding is not None
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert_clean(root)
