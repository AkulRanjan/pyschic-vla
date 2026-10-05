"""Sequential MCTS on the toy task (SPEC.md §10 Phase 2 definition of done).

The toy judge's prior always favours "+5", so greedy goes 5 -> 10 -> 15 (wrong). The value
signal must steer the search to an exact 13.
"""

import json

import pytest

from thoughtzero.accounting import ledger_scope
from thoughtzero.config import SearchCfg
from thoughtzero.mocks import TOY, MockGenerator, MockJudge, greedy_by_prior, toy_extract_answer
from thoughtzero.search.mcts import SearchResult, is_final_step, make_child, search
from thoughtzero.search.node import Node
from thoughtzero.types import GenOut


async def run(
    n: int = 64,
    k: int = 3,
    seed: int = 0,
    gen: MockGenerator | None = None,
    judge: MockJudge | None = None,
    **cfg_kwargs: object,
) -> SearchResult:
    cfg = SearchCfg(n_simulations=n, k=k, **cfg_kwargs)  # type: ignore[arg-type]
    return await search(
        TOY.question,
        gen or MockGenerator(seed),
        judge or MockJudge(seed),
        cfg,
        seed=seed,
        extract=toy_extract_answer,
        normalize=str.strip,
    )


def walk(node: Node) -> list[Node]:
    out = [node]
    for child in node.children:
        out.extend(walk(child))
    return out


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("mode", ["most_visited", "value_vote"])
async def test_search_beats_greedy(seed: int, mode: str) -> None:
    greedy = await greedy_by_prior(TOY.question, MockGenerator(seed), MockJudge(seed))
    assert greedy == "15" != TOY.answer
    result = await run(seed=seed, extract_mode=mode)
    assert result.answer == TOY.answer
    assert result.stats["modes_agree"] is True
    assert result.answer_steps[-1].endswith("\\boxed{13}")


async def test_visit_count_invariants() -> None:
    result = await run(n=64)
    root = result.root
    assert root.N == 64 == result.stats["simulations"]
    assert sum(c.N for c in root.children) == root.N - 1  # first visit only expands the root
    for node in walk(root):
        if node.children:  # expanded: 1 expansion visit + every visit passed to a child
            assert 1 + sum(c.N for c in node.children) == node.N
        if node.N:
            assert 0.0 <= node.Q <= 1.0
        assert node.virtual_loss == 0


async def test_judge_is_called_once_per_expansion_and_terminal() -> None:
    gen, judge = MockGenerator(), MockJudge()
    result = await run(gen=gen, judge=judge)
    expanded = [n for n in walk(result.root) if n.children]
    evaluated_terminals = [n for n in walk(result.root) if n.terminal and n.value is not None]
    assert gen.calls["propose"] == judge.calls["prior_and_value"] == len(expanded)
    assert judge.calls["final_correct"] == len(evaluated_terminals)
    assert result.stats["expansions"] == len(expanded)
    assert result.stats["terminal_evals"] == len(evaluated_terminals)
    # terminals are revisited (their N exceeds 1) without calling the judge again
    assert sum(n.N for n in evaluated_terminals) > len(evaluated_terminals)


async def test_search_is_deterministic() -> None:
    a, b = await run(seed=3), await run(seed=3)
    assert json.dumps(a.tree_dump) == json.dumps(b.tree_dump)
    assert a.stats == b.stats


async def test_ledger_records_search_diagnostics() -> None:
    with ledger_scope() as ledger:
        result = await run()
    assert ledger.expansions == result.stats["expansions"]
    assert ledger.terminal_leaves == result.stats["terminal_evals"]
    assert ledger.max_depth == result.stats["max_depth"]
    assert ledger.gemma_calls == result.stats["expansions"]
    assert ledger.jev_calls == result.stats["expansions"] + result.stats["terminal_evals"]


async def test_one_simulation_completes_greedily() -> None:
    gen = MockGenerator()
    result = await run(n=1, gen=gen)
    assert result.root.N == 1 and len(result.root.children) == 3
    assert result.stats["most_visited_reached_terminal"] is False
    assert gen.calls["complete"] == 1  # unvisited children -> greedy completion
    assert result.answer == "15"  # highest-prior child, then +5 +5


async def test_value_vote_mode_never_calls_complete() -> None:
    gen = MockGenerator()
    await run(n=4, gen=gen, extract_mode="value_vote")
    assert gen.calls["complete"] == 0


async def test_duplicate_candidates_are_merged() -> None:
    result = await run(n=16, k=6)  # 6 samples over only 3 distinct steps
    for node in walk(result.root):
        assert len(node.children) <= 3
        assert len({c.steps[-1] for c in node.children}) == len(node.children)


async def test_max_depth_makes_nodes_terminal() -> None:
    result = await run(n=32, max_depth=2)
    assert result.stats["max_depth"] <= 2
    capped = [n for n in walk(result.root) if n.depth == 2]
    assert capped and all(n.terminal for n in capped)
    assert any(n.final_answer is None for n in capped)  # capped without an answer


async def test_root_dirichlet_noise_changes_priors_deterministically() -> None:
    plain = await run(n=2)
    noisy = await run(n=2, root_dirichlet_alpha=0.3)
    noisy_again = await run(n=2, root_dirichlet_alpha=0.3)
    priors = [c.prior for c in noisy.root.children]
    assert priors != [c.prior for c in plain.root.children]
    assert priors == [c.prior for c in noisy_again.root.children]
    assert sum(priors) == pytest.approx(1.0)


class EmptyGenerator(MockGenerator):
    async def propose(self, problem: str, steps: list[str], k: int) -> list[GenOut]:
        return [GenOut("  ", 1, 1) for _ in range(k)]


async def test_all_empty_candidates_make_a_dead_end() -> None:
    judge = MockJudge()
    result = await run(n=3, gen=EmptyGenerator(), judge=judge)
    assert result.root.terminal and not result.root.children
    assert result.stats["dead_ends"] == 1
    assert judge.calls["final_correct"] == 1  # evaluated once, revisited from cache
    assert result.answer is None


class ShortPriorJudge(MockJudge):
    async def prior_and_value(
        self, problem: str, steps: list[str], candidates: list[str]
    ) -> tuple[list[float], float]:
        return [1.0], 0.5


async def test_wrong_number_of_priors_is_an_error() -> None:
    with pytest.raises(ValueError, match="priors"):
        await run(n=1, judge=ShortPriorJudge())


@pytest.mark.parametrize(
    ("text", "final"),
    [
        ("so x = \\boxed{4}", True),
        ("Final answer: 4", True),
        ("the FINAL ANSWER is 4", True),
        ("add 5 -> 10", False),
    ],
)
def test_is_final_step(text: str, final: bool) -> None:
    assert is_final_step(text) is final


def test_make_child() -> None:
    parent = Node(steps=["a"])
    child = make_child(parent, "add 1 -> 13. \\boxed{13}", 0.4, 20, toy_extract_answer)
    assert child.steps == ["a", "add 1 -> 13. \\boxed{13}"] and parent.steps == ["a"]
    assert child.terminal and child.final_answer == "13" and child.prior == 0.4
    capped = make_child(parent, "keep going", 0.4, 2, toy_extract_answer)
    assert capped.terminal and capped.final_answer is None


class ZeroPriorJudge(MockJudge):
    """Puts all prior mass on the misleading "+5" step, exact zeros elsewhere (like real Jev)."""

    async def prior_and_value(
        self, problem: str, steps: list[str], candidates: list[str]
    ) -> tuple[list[float], float]:
        _, value = await super().prior_and_value(problem, steps, candidates)
        return [1.0 if "add 5" in c else 0.0 for c in candidates], value


async def test_prior_floor_lets_search_explore_zero_prior_children() -> None:
    plain = await run(n=48, judge=ZeroPriorJudge(), extract_mode="value_vote")
    floored = await run(n=48, judge=ZeroPriorJudge(), extract_mode="value_vote", prior_floor=0.3)
    root_plain = {c.steps[-1]: c.N for c in plain.root.children}
    root_floor = {c.steps[-1]: c.N for c in floored.root.children}
    assert all(n == 0 for s, n in root_plain.items() if "add 5" not in s)  # never explored
    assert all(n > 0 for n in root_floor.values())  # the floor reaches every child
