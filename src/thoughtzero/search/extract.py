"""Answer extraction: most_visited / value_vote (SPEC.md §4). Owner: Person 1 (Prakhar).

``extract`` / ``normalize`` default to Person 3's ``data.grading`` functions; tests inject toy
versions so the search does not depend on grading being finished.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from thoughtzero.search.node import Node
from thoughtzero.types import Generator

ExtractFn = Callable[[str], str | None]
NormalizeFn = Callable[[str], str]


def default_extract(text: str) -> str | None:
    from thoughtzero.data.grading import extract_answer

    return extract_answer(text)


def default_normalize(answer: str) -> str:
    from thoughtzero.data.grading import normalize_answer

    return normalize_answer(answer)


def _most_visited_child(node: Node) -> Node:
    # ties: higher Q, then higher prior, then lower index
    best = max(
        range(len(node.children)),
        key=lambda i: (node.children[i].N, node.children[i].Q, node.children[i].prior, -i),
    )
    return node.children[best]


def most_visited_path(root: Node) -> list[Node]:
    """Root -> ... following the most-visited child until a terminal or unexpanded node."""
    path, node = [root], root
    while node.children and not node.terminal:
        node = _most_visited_child(node)
        path.append(node)
    return path


async def most_visited(
    root: Node,
    generator: Generator,
    problem: str,
    extract: ExtractFn = default_extract,
) -> tuple[str | None, list[str], dict[str, Any]]:
    """Follow max-N children; if the path ends non-terminal, finish it greedily (temperature 0)."""
    path = most_visited_path(root)
    end = path[-1]
    diag: dict[str, Any] = {
        "path_visits": [n.N for n in path],
        "reached_terminal": end.terminal,
        "completed_steps": 0,
    }
    if end.terminal:
        return end.final_answer, list(end.steps), diag
    new_steps = await generator.complete(problem, end.steps, temperature=0.0)
    diag["completed_steps"] = len(new_steps)
    answer = extract("\n\n".join(new_steps)) if new_steps else None
    return answer, end.steps + new_steps, diag


def terminal_nodes(root: Node) -> list[Node]:
    """All terminal nodes in the tree (iterative DFS, preorder)."""
    found, stack = [], [root]
    while stack:
        node = stack.pop()
        if node.terminal:
            found.append(node)
        stack.extend(reversed(node.children))
    return found


def value_vote(
    root: Node, normalize: NormalizeFn = default_normalize
) -> tuple[str | None, list[str], dict[str, Any]]:
    """Group evaluated terminal leaves by normalized answer; the largest sum of N x value wins.

    Ties: larger total N, then the lexicographically smaller normalized answer. The returned
    answer and steps come from the group's highest N x value leaf.
    """
    groups: dict[str, list[Node]] = {}
    for leaf in terminal_nodes(root):
        if leaf.final_answer is None or leaf.value is None or leaf.N == 0:
            continue
        groups.setdefault(normalize(leaf.final_answer), []).append(leaf)

    scores = {key: sum(n.N * (n.value or 0.0) for n in nodes) for key, nodes in groups.items()}
    diag: dict[str, Any] = {"group_scores": scores, "n_groups": len(groups)}
    if not groups:
        return None, [], diag

    winner = min(groups, key=lambda key: (-scores[key], -sum(n.N for n in groups[key]), key))
    best = max(groups[winner], key=lambda n: n.N * (n.value or 0.0))
    return best.final_answer, list(best.steps), diag


def answers_agree(a: str | None, b: str | None, normalize: NormalizeFn = default_normalize) -> bool:
    """Whether two extracted answers match after normalization (None never matches)."""
    return a is not None and b is not None and normalize(a) == normalize(b)
