"""PUCT selection and root noise, pure functions (SPEC.md §4, §6.2). Owner: Person 1 (Prakhar).

    score(a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

Single-agent search: Q is never negated (no opponent, unlike two-player AlphaZero).
"""

from __future__ import annotations

import math
import random

from thoughtzero.search.node import Node


def puct_score(parent: Node, child: Node, c_puct: float, fpu_value: float) -> float:
    # max(1, .): with a parent count of 0 the exploration term would vanish and ignore priors
    explore = c_puct * child.prior * math.sqrt(max(1, parent.n_eff)) / (1 + child.n_eff)
    return child.q_eff(fpu_value) + explore


def select_child(parent: Node, c_puct: float, fpu_value: float) -> Node:
    """Highest PUCT score; ties go to the higher prior, then the lower index."""
    if not parent.children:
        raise ValueError("select_child called on a node without children")
    best_i = max(
        range(len(parent.children)),
        key=lambda i: (
            puct_score(parent, parent.children[i], c_puct, fpu_value),
            parent.children[i].prior,
            -i,
        ),
    )
    return parent.children[best_i]


def dirichlet_noise(
    priors: list[float], alpha: float, rng: random.Random, epsilon: float = 0.25
) -> list[float]:
    """AlphaZero root noise: ``(1 - eps) * P + eps * Dir(alpha)``. Still sums to 1."""
    if not priors:
        return []
    draws = [rng.gammavariate(alpha, 1.0) for _ in priors]
    total = sum(draws) or 1.0
    return [(1 - epsilon) * p + epsilon * d / total for p, d in zip(priors, draws, strict=True)]


def floor_priors(priors: list[float], floor: float) -> list[float]:
    """``(1 - floor) * P + floor / k``: every candidate keeps at least ``floor / k``.

    A judge can give a candidate exactly 0 (real Jev rounds to 2 decimals); PUCT would then
    never explore it, even when the judge is wrong (PLAN D9). Still sums to 1.
    """
    if not priors or floor <= 0:
        return list(priors)
    k = len(priors)
    return [(1 - floor) * p + floor / k for p in priors]
