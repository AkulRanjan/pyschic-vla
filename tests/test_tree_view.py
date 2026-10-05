"""Tree rendering for scripts/view_tree.py."""

import json

from thoughtzero.config import SearchCfg
from thoughtzero.mocks import TOY, MockGenerator, MockJudge, toy_extract_answer
from thoughtzero.search.mcts import search
from thoughtzero.search.tree_view import format_tree


def leaf(N: int, prior: float, step: str, **extra: object) -> dict:
    return {"N": N, "Q": 0.5, "prior": prior, "value": 0.9, "step": step, "children": [], **extra}


ROOT = {
    "N": 9,
    "Q": 0.6,
    "prior": 1.0,
    "value": 0.9,
    "step": "",
    "children": [
        leaf(1, 0.7, "rarely visited"),
        leaf(5, 0.1, "most visited"),
        leaf(3, 0.2, "done \\boxed{4}", terminal=True, final_answer="4"),
    ],
}


def test_children_sorted_by_visits_and_terminals_marked() -> None:
    lines = format_tree(ROOT).splitlines()
    assert lines[0].endswith("(root)") and "P=" not in lines[0]
    assert "most visited" in lines[1] and lines[1].startswith("├─ ")
    assert "[T] 4" in lines[2]
    assert "rarely visited" in lines[3] and lines[3].startswith("└─ ")


def test_min_visits_and_max_depth_hide_nodes() -> None:
    out = format_tree(ROOT, min_visits=2)
    assert "rarely visited" not in out and "... 1 hidden" in out
    assert format_tree(ROOT, max_depth=0).splitlines()[1] == "└─ ... 3 hidden"


def test_long_and_multiline_steps_are_shortened() -> None:
    tree = {**ROOT, "children": [leaf(1, 1.0, "word\nword " * 50)]}
    line = format_tree(tree, step_chars=20).splitlines()[1]
    assert "\n" not in line and line.endswith("...")


async def test_renders_a_real_tree_dump() -> None:
    result = await search(
        TOY.question,
        MockGenerator(),
        MockJudge(),
        SearchCfg(n_simulations=16, k=3),
        extract=toy_extract_answer,
        normalize=str.strip,
    )
    dump = json.loads(json.dumps(result.tree_dump))  # as stored in per_problem.jsonl
    out = format_tree(dump)
    assert len(out.splitlines()) == result.stats["n_nodes"]


def test_html_page_marks_the_most_visited_path() -> None:
    from thoughtzero.search.tree_html import most_visited_ids, tree_to_html

    tree = {
        **ROOT,
        "id": 0,
        "children": [
            {**leaf(1, 0.7, "rarely <visited>"), "id": 1},
            {**leaf(5, 0.1, "most visited"), "id": 2},
        ],
    }
    assert most_visited_ids(tree) == {0, 2}
    page = tree_to_html(tree, problem="2+2?", answer="4", gold="4")
    assert page.startswith("<!doctype html>") and "✓ correct" in page
    assert "rarely &lt;visited&gt;" in page  # escaped
    assert page.index("most visited") < page.index("rarely")  # ordered by visits
