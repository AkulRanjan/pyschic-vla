import math
import random

import pytest

from thoughtzero.search.node import Node
from thoughtzero.search.puct import dirichlet_noise, puct_score, select_child

C_PUCT, FPU = 1.5, 0.0


def make_root(priors: list[float], n: int = 1) -> Node:
    root = Node(steps=[], N=n)
    root.children = [Node(steps=[f"s{i}"], prior=p) for i, p in enumerate(priors)]
    return root


def scores(root: Node, fpu: float = FPU) -> list[float]:
    return [puct_score(root, c, C_PUCT, fpu) for c in root.children]


def test_node_q_and_virtual_loss() -> None:
    node = Node(steps=[], N=2, W=1.0)
    assert node.Q == 0.5
    assert node.q_eff(0.3) == 0.5
    node.virtual_loss = 2  # in-flight visits count as value 0
    assert node.n_eff == 4
    assert node.q_eff(0.3) == 0.25
    assert node.Q == 0.5  # real statistics untouched
    assert Node(steps=[]).q_eff(0.3) == 0.3  # FPU for never-visited


def test_worked_example_first_selection() -> None:
    # PRAKHAR_GUIDE.md P2 worked example: priors .5/.3/.2, root visited once
    root = make_root([0.5, 0.3, 0.2])
    assert scores(root) == pytest.approx([0.75, 0.45, 0.30])
    assert select_child(root, C_PUCT, FPU) is root.children[0]


def test_worked_example_low_value_moves_search_away() -> None:
    root = make_root([0.5, 0.3, 0.2], n=2)
    a, b, _ = root.children
    a.N, a.W = 1, 0.1  # judge said V(A) = 0.1
    assert scores(root)[0] == pytest.approx(0.1 + 1.5 * 0.5 * math.sqrt(2) / 2)  # 0.630
    assert scores(root)[1] == pytest.approx(1.5 * 0.3 * math.sqrt(2))  # 0.636
    assert select_child(root, C_PUCT, FPU) is b


def test_worked_example_high_value_keeps_search_on_a() -> None:
    root = make_root([0.5, 0.3, 0.2], n=2)
    a = root.children[0]
    a.N, a.W = 1, 0.9
    assert scores(root)[0] == pytest.approx(1.4303, abs=1e-4)
    assert select_child(root, C_PUCT, FPU) is a


def test_q_is_never_negated() -> None:
    root = make_root([0.5, 0.5], n=4)
    good, bad = root.children
    good.N, good.W = 2, 1.8
    bad.N, bad.W = 2, 0.2
    assert select_child(root, C_PUCT, FPU) is good


def test_unvisited_parent_still_uses_priors() -> None:
    # sqrt(max(1, N)): with N(parent) = 0 the prior must still decide
    root = make_root([0.2, 0.5, 0.3], n=0)
    assert select_child(root, C_PUCT, FPU) is root.children[1]


def test_fpu_value_controls_unvisited_children() -> None:
    root = make_root([0.9, 0.1], n=10)
    visited, fresh = root.children
    visited.N, visited.W = 5, 2.5  # Q = 0.5 -> 0.5 + 0.711 = 1.21; fresh = fpu + 0.474
    assert select_child(root, C_PUCT, fpu_value=0.0) is visited
    assert select_child(root, C_PUCT, fpu_value=1.0) is fresh


def test_virtual_loss_diverts_concurrent_selection() -> None:
    root = make_root([0.5, 0.3, 0.2])
    a = root.children[0]
    assert select_child(root, C_PUCT, FPU) is a
    root.virtual_loss, a.virtual_loss = 1, 1  # one simulation already in flight through A
    assert select_child(root, C_PUCT, FPU) is root.children[1]


def test_ties_break_on_lowest_index() -> None:
    root = make_root([0.25, 0.25, 0.25, 0.25])
    assert select_child(root, C_PUCT, FPU) is root.children[0]


def test_select_child_requires_children() -> None:
    with pytest.raises(ValueError):
        select_child(Node(steps=[]), C_PUCT, FPU)


def test_dirichlet_noise() -> None:
    priors = [0.7, 0.2, 0.1]
    noisy = dirichlet_noise(priors, alpha=0.3, rng=random.Random(0))
    assert sum(noisy) == pytest.approx(1.0)
    assert noisy != priors
    assert noisy == dirichlet_noise(priors, alpha=0.3, rng=random.Random(0))
    assert dirichlet_noise(priors, 0.3, random.Random(0), epsilon=0.0) == pytest.approx(priors)
    assert dirichlet_noise([], 0.3, random.Random(0)) == []


def test_floor_priors_keeps_every_candidate_explorable() -> None:
    from thoughtzero.search.puct import floor_priors

    floored = floor_priors([1.0, 0.0, 0.0], 0.06)
    assert floored == pytest.approx([0.96, 0.02, 0.02]) and sum(floored) == pytest.approx(1.0)
    assert floor_priors([0.7, 0.3], 0.0) == [0.7, 0.3]  # 0 = spec behaviour, unchanged
