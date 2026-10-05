"""Live demonstration: watch ThoughtZero search for one problem, then compare with greedy.

    python scripts/demo.py --mock --html demo.html          # offline toy task, no keys needed
    python scripts/demo.py --html demo.html                 # real Gemma + judge (from .env)
    python scripts/demo.py --problem "What is 3^4 - 2^5?" --gold 49 --n-sims 24

As the search runs, every expansion prints the generator's candidate next steps with the
judge's prior for each (a bar) and the judge's value for the state; every new terminal prints
its answer and the judge's verdict. At the end: the chosen path, the answer, the search tree,
a greedy baseline on the same problem and, with --html, an interactive page of the whole tree
(open it in a browser; the green path is what the search picked).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any

from thoughtzero.config import Config, add_config_args, config_from_args
from thoughtzero.search.mcts import search
from thoughtzero.search.tree_html import tree_to_html
from thoughtzero.search.tree_view import format_tree

DEFAULT_PROBLEM = "A rectangle has perimeter 30 and length 9. What is its area?"
DEFAULT_GOLD = "54"
BAR = 24


def short(text: str, width: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 3] + "..."


class LiveJudge:
    """Wraps a judge and narrates every call (otherwise identical)."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.expansions = 0

    def __getattr__(self, name: str) -> Any:  # step_sound etc. go straight through
        return getattr(self.inner, name)

    async def prior_and_value(
        self, problem: str, steps: list[str], candidates: list[str]
    ) -> tuple[list[float], float]:
        priors, value = await self.inner.prior_and_value(problem, steps, candidates)
        self.expansions += 1
        where = f"after step {len(steps)}: {short(steps[-1], 60)}" if steps else "at the start"
        print(
            f"\n▶ expansion {self.expansions} ({where})   judge value of this state = {value:.2f}"
        )
        for p, c in sorted(zip(priors, candidates, strict=True), reverse=True):
            bar = "█" * round(p * BAR)
            print(f"   prior {p:4.2f} {bar:<{BAR}} {short(c, 70)}")
        return priors, value

    async def final_correct(self, problem: str, steps: list[str]) -> float:
        p = await self.inner.final_correct(problem, steps)
        last = short(steps[-1], 60)
        print(f"   ■ finished solution ({len(steps)} steps): {last}  →  judge: {p:.2f}")
        return p


async def run(cfg: Config, args: argparse.Namespace) -> int:
    from thoughtzero.mocks import TOY, MockGenerator, MockJudge, greedy_by_prior, toy_extract_answer

    if args.mock:
        problem, gold = TOY.question, TOY.answer
        generator: Any = MockGenerator(cfg.seed)
        judge: Any = MockJudge(cfg.seed)
        extract: Any = toy_extract_answer
        normalize: Any = str.strip
        cfg = cfg.model_copy(update={"search": cfg.search.model_copy(update={"k": 3})})
        route = "mock generator + mock judge (toy task: reach exactly 13 by adding 1, 2 or 5)"
    else:
        from thoughtzero.data.grading import extract_answer, normalize_answer
        from thoughtzero.judge.client import resolve_route
        from thoughtzero.judge.factory import make_judge
        from thoughtzero.llm.factory import make_generator

        problem, gold = args.problem, args.gold
        generator = make_generator(cfg)
        judge = make_judge(cfg.judge, generator)
        extract, normalize = extract_answer, normalize_answer
        model = resolve_route(cfg.judge)[1] if cfg.judge.kind == "jev" else cfg.judge.kind
        route = f"generator {cfg.generator.model} | judge {model}"

    search_cfg = cfg.search.model_copy(
        update={"n_simulations": args.n_sims, "parallel_sims": args.parallel}
    )
    print(f"ThoughtZero demo — {route}")
    print(f"Problem: {problem}")
    print(f"Search: {args.n_sims} simulations, k={search_cfg.k} candidates per expansion")

    live = LiveJudge(judge)
    t0 = time.time()
    result = await search(
        problem, generator, live, search_cfg, seed=cfg.seed, extract=extract, normalize=normalize
    )
    seconds = time.time() - t0

    print("\n" + "=" * 78)
    print("Chosen solution (most-visited path):")
    for i, step in enumerate(result.answer_steps, 1):
        print(f"  Step {i}: {step}")
    verdict = ""
    if gold is not None:
        ok = result.answer is not None and normalize(result.answer) == normalize(gold)
        verdict = "  ✓ correct" if ok else f"  ✗ (expected {gold})"
    print(f"\nThoughtZero answer: {result.answer}{verdict}")
    s = result.stats
    print(
        f"({s['expansions']} expansions, {s['n_nodes']} nodes, max depth {s['max_depth']}, "
        f"{seconds:.1f} s; most-visited and value-vote answers agree: {s['modes_agree']})"
    )

    print("\nSearch tree (children ordered by visits; nodes visited at least twice):")
    print(format_tree(result.tree_dump, min_visits=2, step_chars=70))

    print("\nBaseline on the same problem:")
    if args.mock:
        greedy = await greedy_by_prior(problem, MockGenerator(cfg.seed), MockJudge(cfg.seed))
        print(f"  greedy (always take the judge's top step): {greedy}  (expected {gold})")
    else:
        steps = await generator.complete(problem, [], temperature=0.0)
        greedy = extract(steps[-1]) if steps else None
        print(f"  plain chain of thought (one greedy solution, no search): {greedy}")

    if args.html:
        notes = [
            route,
            f"{args.n_sims} simulations · {s['expansions']} expansions · {seconds:.1f} s",
        ]
        page = tree_to_html(
            result.tree_dump, problem=problem, answer=result.answer, gold=gold, notes=notes
        )
        await asyncio.to_thread(Path(args.html).write_text, page, encoding="utf-8")
        print(f"\nInteractive tree written to {args.html} — open it in a browser.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_config_args(parser, default="configs/default.yaml")
    parser.add_argument("--mock", action="store_true", help="offline toy task, no keys")
    parser.add_argument("--problem", default=DEFAULT_PROBLEM)
    parser.add_argument("--gold", default=DEFAULT_GOLD, help="expected answer, for the ✓/✗")
    parser.add_argument("--n-sims", type=int, default=32, help="32+ finds the toy answer")
    parser.add_argument("--parallel", type=int, default=1, help="1 keeps the narration in order")
    parser.add_argument("--html", default=None, help="write an interactive tree page here")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    import logging

    logging.basicConfig(level=logging.WARNING)
    for noisy in ("httpx", "httpx2", "openai", "math_verify"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    return asyncio.run(run(config_from_args(args), args))


if __name__ == "__main__":
    raise SystemExit(main())
