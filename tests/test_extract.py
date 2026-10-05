import json

import pytest

from thoughtzero.mocks import MockGenerator, toy_extract_answer
from thoughtzero.search.extract import (
    answers_agree,
    most_visited,
    most_visited_path,
    terminal_nodes,
    value_vote,
)
from thoughtzero.search.mcts import dump_tree
from thoughtzero.search.node import Node

PROBLEM = "toy"


def strip(answer: str) -> str:
    return answer.strip()


def leaf(steps: list[str], N: int, value: float | None, answer: str | None) -> Node:
    return Node(
        steps=steps, N=N, W=N * (value or 0.0), value=value, terminal=True, final_answer=answer
    )


def tree_with_disagreement() -> Node:
    """most_visited -> "15" (N=5), value_vote -> "13" (3*.9 + 1*.9 = 3.6 > 5*.5 = 2.5)."""
    root = Node(steps=[], N=10, W=6.0)
    a = Node(steps=["a"], N=4, W=3.6, prior=0.5)
    a.children = [leaf(["a", "a1"], 3, 0.9, "13"), leaf(["a", "a2"], 1, 0.9, " 13 ")]
    b = leaf(["b"], 5, 0.5, "15")
    unvisited = Node(steps=["c"], terminal=True, final_answer="99")  # never evaluated
    no_answer = leaf(["d"], 2, 0.9, None)  # depth cap without an answer
    root.children = [a, b, unvisited, no_answer]
    return root


async def test_most_visited_follows_max_n_to_terminal() -> None:
    answer, steps, diag = await most_visited(tree_with_disagreement(), MockGenerator(), PROBLEM)
    assert answer == "15"
    assert steps == ["b"]
    assert diag == {"path_visits": [10, 5], "reached_terminal": True, "completed_steps": 0}


async def test_most_visited_completes_non_terminal_path_greedily() -> None:
    root = Node(steps=[], N=3)
    root.children = [Node(steps=["add 5 -> 5"], N=2), Node(steps=["add 1 -> 1"], N=1)]
    answer, steps, diag = await most_visited(
        root, MockGenerator(), PROBLEM, extract=toy_extract_answer
    )
    assert steps == ["add 5 -> 5", "add 5 -> 10", "add 5 -> 15. \\boxed{15}"]
    assert answer == "15"
    assert diag["reached_terminal"] is False and diag["completed_steps"] == 2


async def test_most_visited_from_unexpanded_root() -> None:
    gen = MockGenerator()
    answer, steps, _ = await most_visited(Node(steps=[]), gen, PROBLEM, extract=toy_extract_answer)
    assert answer == "15" and len(steps) == 3
    assert gen.calls["complete"] == 1


def test_most_visited_tie_breaks_on_q_then_prior() -> None:
    root = Node(steps=[], N=5)
    low_q = Node(steps=["x"], N=2, W=0.2, prior=0.9)
    high_q = Node(steps=["y"], N=2, W=1.8, prior=0.1)
    root.children = [low_q, high_q]
    assert most_visited_path(root)[-1] is high_q
    high_q.W = 0.2
    assert most_visited_path(root)[-1] is low_q  # equal N and Q -> higher prior


def test_value_vote_sums_n_times_value_per_normalized_answer() -> None:
    answer, steps, diag = value_vote(tree_with_disagreement(), normalize=strip)
    assert answer == "13"
    assert steps == ["a", "a1"]  # highest N x value leaf of the winning group
    assert diag["group_scores"] == pytest.approx({"13": 3.6, "15": 2.5})
    assert diag["n_groups"] == 2  # unvisited and answer-less terminals ignored


def test_value_vote_ties() -> None:
    root = Node(steps=[], N=6)
    root.children = [leaf(["p"], 2, 0.5, "7"), leaf(["q"], 1, 1.0, "3")]
    answer, _, _ = value_vote(root, normalize=strip)
    assert answer == "7"  # equal score 1.0 -> larger total N
    root.children = [leaf(["p"], 1, 0.5, "7"), leaf(["q"], 1, 0.5, "3")]
    answer, _, _ = value_vote(root, normalize=strip)
    assert answer == "3"  # equal score and N -> smaller normalized answer


def test_value_vote_without_answers() -> None:
    assert value_vote(Node(steps=[]), normalize=strip) == (
        None,
        [],
        {"group_scores": {}, "n_groups": 0},
    )


def test_terminal_nodes_preorder() -> None:
    names = [n.steps[-1] for n in terminal_nodes(tree_with_disagreement())]
    assert names == ["a1", "a2", "b", "c", "d"]


def test_answers_agree() -> None:
    assert answers_agree("13", " 13 ", normalize=strip)
    assert not answers_agree("13", "15", normalize=strip)
    assert not answers_agree(None, None, normalize=strip)


def test_dump_tree_is_compact_json() -> None:
    root = tree_with_disagreement()
    root.children[0].children[0].steps[-1] = "z" * 400
    dump = dump_tree(root, max_step_chars=10)
    json.dumps(dump)
    assert dump["id"] == 0 and dump["step"] == "" and dump["N"] == 10 and dump["Q"] == 0.6
    a = dump["children"][0]
    assert [a["id"]] + [c["id"] for c in a["children"]] == [1, 2, 3]
    assert a["children"][0]["step"] == "z" * 10 + "..."
    assert dump["children"][2]["value"] is None and dump["children"][3]["final_answer"] is None
