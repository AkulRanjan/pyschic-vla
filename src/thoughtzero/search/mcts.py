"""MCTS over reasoning steps (SPEC.md §4, §6). Owner: Person 1 (Prakhar).

Phase P3: sequential (one simulation at a time). Phase P5 adds concurrent simulations with
virtual loss; only ``_simulate`` and the driver loop in ``search`` change for that.

One simulation = select (PUCT) -> expand a leaf OR evaluate a terminal -> backup.
- Expansion: ``generator.propose`` k steps -> dedupe -> ONE ``judge.prior_and_value`` call on
  the unique candidates -> children. The leaf's value is the judge's V(s).
- Terminal: ``judge.final_correct`` once; the value is cached and re-backed-up on revisits.
- Backup: ``N += 1``, ``W += v`` for every node on the path, root included. Single-agent:
  values are never negated.

``search`` never receives the ground-truth answer: only the question string.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any

from thoughtzero.accounting import record_search
from thoughtzero.config import SearchCfg
from thoughtzero.search.dedupe import dedupe
from thoughtzero.search.extract import (
    ExtractFn,
    NormalizeFn,
    answers_agree,
    default_extract,
    default_normalize,
    most_visited,
    most_visited_path,
    value_vote,
)
from thoughtzero.search.node import Node
from thoughtzero.search.puct import dirichlet_noise, select_child
from thoughtzero.types import Generator, Judge


@dataclass
class SearchResult:
    answer: str | None
    answer_steps: list[str]
    root: Node
    tree_dump: dict[str, Any]
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Ctx:
    problem: str
    generator: Generator
    judge: Judge
    cfg: SearchCfg
    extract: ExtractFn
    rng: random.Random
    expansions: int = 0
    terminal_evals: int = 0
    dead_ends: int = 0


def is_final_step(text: str) -> bool:
    """A step that states the final answer (SPEC.md §4 terminal rule)."""
    return "\\boxed{" in text or "final answer" in text.lower()


def make_child(parent: Node, text: str, prior: float, max_depth: int, extract: ExtractFn) -> Node:
    """Child for step ``text``; terminal if it states an answer or hits ``max_depth``."""
    steps = [*parent.steps, text]
    final = is_final_step(text)
    return Node(
        steps=steps,
        prior=prior,
        terminal=final or len(steps) >= max_depth,
        final_answer=extract(text) if final else None,
    )


def backup(path: list[Node], value: float) -> None:
    for node in path:
        node.N += 1
        node.W += value


def _problem_rng(problem: str, seed: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}|{problem}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


async def _evaluate_terminal(node: Node, ctx: _Ctx) -> float:
    if node.value is None:
        node.value = await ctx.judge.final_correct(ctx.problem, node.steps)
        ctx.terminal_evals += 1
        record_search(terminal_leaves=1)
    return node.value


async def _expand(node: Node, ctx: _Ctx) -> float:
    outs = await ctx.generator.propose(ctx.problem, node.steps, ctx.cfg.k)
    unique, _ = dedupe(outs, ctx.cfg.dedupe_jaccard)
    if not unique:
        # the generator produced only empty steps: treat this state as a dead end
        ctx.dead_ends += 1
        node.terminal = True
        return await _evaluate_terminal(node, ctx)

    texts = [c.text for c in unique]
    priors, value = await ctx.judge.prior_and_value(ctx.problem, node.steps, texts)
    if len(priors) != len(texts):
        raise ValueError(f"judge returned {len(priors)} priors for {len(texts)} candidates")
    if not node.steps and ctx.cfg.root_dirichlet_alpha:
        priors = dirichlet_noise(priors, ctx.cfg.root_dirichlet_alpha, ctx.rng)

    node.value = value
    node.children = [
        make_child(node, t, p, ctx.cfg.max_depth, ctx.extract)
        for t, p in zip(texts, priors, strict=True)
    ]
    ctx.expansions += 1
    record_search(expansions=1, depth=node.depth + 1)
    return value


async def _simulate(root: Node, ctx: _Ctx) -> None:
    path, node = [root], root
    while node.children and not node.terminal:
        node = select_child(node, ctx.cfg.c_puct, ctx.cfg.fpu_value)
        path.append(node)

    if node.terminal:
        value = await _evaluate_terminal(node, ctx)
    else:
        value = await _expand(node, ctx)
    backup(path, value)


def _count(root: Node) -> tuple[int, int, int]:
    """(number of nodes, number of terminal nodes, maximum depth)."""
    nodes = terminals = max_depth = 0
    stack = [root]
    while stack:
        node = stack.pop()
        nodes += 1
        terminals += node.terminal
        max_depth = max(max_depth, node.depth)
        stack.extend(node.children)
    return nodes, terminals, max_depth


async def search(
    problem: str,
    generator: Generator,
    judge: Judge,
    cfg: SearchCfg,
    *,
    seed: int = 0,
    extract: ExtractFn | None = None,
    normalize: NormalizeFn | None = None,
) -> SearchResult:
    """Run ``cfg.n_simulations`` simulations and extract an answer (``cfg.extract_mode``).

    ``extract`` / ``normalize`` default to ``data.grading``; tests inject toy versions.
    """
    extract = extract or default_extract
    normalize = normalize or default_normalize
    ctx = _Ctx(problem, generator, judge, cfg, extract, _problem_rng(problem, seed))
    root = Node(steps=[])

    for _ in range(cfg.n_simulations):
        await _simulate(root, ctx)

    # value_vote is free; most_visited may call the generator to finish a non-terminal path,
    # so only the configured mode is allowed to spend tokens.
    vv_answer, vv_steps, vv_diag = value_vote(root, normalize)
    mv_end = most_visited_path(root)[-1]
    if cfg.extract_mode == "most_visited":
        answer, steps, mv_diag = await most_visited(root, generator, problem, extract)
        mv_answer = answer
    else:
        answer, steps = vv_answer, vv_steps
        mv_answer = mv_end.final_answer if mv_end.terminal else None
        mv_diag = {"reached_terminal": mv_end.terminal}

    n_nodes, n_terminals, max_depth = _count(root)
    stats: dict[str, Any] = {
        "simulations": root.N,
        "expansions": ctx.expansions,
        "terminal_evals": ctx.terminal_evals,
        "dead_ends": ctx.dead_ends,
        "n_nodes": n_nodes,
        "n_terminal_nodes": n_terminals,
        "max_depth": max_depth,
        "extract_mode": cfg.extract_mode,
        "answer_most_visited": mv_answer,
        "answer_value_vote": vv_answer,
        "modes_agree": answers_agree(mv_answer, vv_answer, normalize),
        "most_visited_reached_terminal": mv_diag["reached_terminal"],
        "value_vote_groups": vv_diag["group_scores"],
    }
    return SearchResult(answer, steps, root, dump_tree(root), stats)


STEP_PREVIEW_CHARS = 300


def dump_tree(root: Node, max_step_chars: int = STEP_PREVIEW_CHARS) -> dict[str, Any]:
    """Compact JSON-serialisable tree for per_problem.jsonl (SPEC.md §8.5).

    Each node stores only its own (last) step, truncated; ids are preorder DFS indices.
    """
    counter = 0

    def visit(node: Node) -> dict[str, Any]:
        nonlocal counter
        node_id, counter = counter, counter + 1
        step = node.steps[-1] if node.steps else ""
        if len(step) > max_step_chars:
            step = step[:max_step_chars] + "..."
        return {
            "id": node_id,
            "step": step,
            "N": node.N,
            "Q": round(node.Q, 4),
            "prior": round(node.prior, 4),
            "value": None if node.value is None else round(node.value, 4),
            "terminal": node.terminal,
            "final_answer": node.final_answer,
            "children": [visit(c) for c in node.children],
        }

    return visit(root)
