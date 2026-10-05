"""End-to-end smoke test on 3 problems (SPEC.md §10, Phase 0). Owner: Prakhar.

    python scripts/smoke_test.py --mock                  # offline: mock generator + judge, toy task
    python scripts/smoke_test.py --n-sims 4 --k 3        # real Gemma server + real Jev
    python scripts/smoke_test.py --tokenizer google/gemma-4-E4B-it   # when GEMMA_MODEL is a
                                                         # quantized checkpoint or an Ollama tag

``--mock`` runs greedy-by-prior (expected WRONG on the toy task) and then MCTS with the
config's search settings (k forced to 3, the toy's branching), and checks MCTS gets it right.

The real run searches 3 easy, hand-written problems (no benchmark problems, so nothing
leaks into the evaluation) with the configured Gemma server (``GEMMA_BASE_URL``) and Jev
route (``JEV_TRANSPORT`` + its key in ``.env``). It passes (SPEC.md Phase 0) when at least
1 of 3 is solved with at most 20 Jev calls and $0.01 of Jev spend in total.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass
from typing import Any

import httpx

from thoughtzero.accounting import ledger_scope
from thoughtzero.config import Config, add_config_args, config_from_args, config_hash
from thoughtzero.logging_setup import setup_logging
from thoughtzero.mocks import TOY, MockGenerator, MockJudge, greedy_by_prior, toy_extract_answer
from thoughtzero.search.mcts import search
from thoughtzero.types import Generator, Judge

# Hand-written, deliberately easy (not from MATH-500, AIME or the MATH train split).
SMOKE_PROBLEMS = [
    ("smoke/1", "What is 15% of 80?", "12"),
    ("smoke/2", "Solve for x: 3x + 7 = 22.", "5"),
    ("smoke/3", "What is the sum of the interior angles of a pentagon, in degrees?", "540"),
]
MAX_JEV_CALLS = 20
MAX_JEV_USD = 0.01


async def run_mock(cfg: Config) -> None:
    greedy = await greedy_by_prior(TOY.question, MockGenerator(cfg.seed), MockJudge(cfg.seed))
    print(f"[mock] greedy-by-prior: {greedy} (gold {TOY.answer}; expected wrong)")

    search_cfg = cfg.search.model_copy(update={"k": 3})
    with ledger_scope() as ledger:
        result = await search(
            TOY.question,
            MockGenerator(cfg.seed),
            MockJudge(cfg.seed),
            search_cfg,
            seed=cfg.seed,
            extract=toy_extract_answer,
            normalize=str.strip,
        )
    ok = result.answer == TOY.answer
    verdict = "OK" if ok else "WRONG"
    print(f"[mock] MCTS ({search_cfg.n_simulations} sims): {result.answer} -> {verdict}")
    print(f"[mock] path: {result.answer_steps}")
    print(f"[mock] stats: { {k: v for k, v in result.stats.items() if k != 'value_vote_groups'} }")
    print(f"[mock] ledger: {ledger.to_dict()}")
    if not ok:
        raise SystemExit(1)


@dataclass
class SmokeSummary:
    solved: int
    jev_calls: int
    jev_usd: float
    rows: list[dict[str, Any]]

    @property
    def passed(self) -> bool:
        return self.solved >= 1 and self.jev_calls <= MAX_JEV_CALLS and self.jev_usd <= MAX_JEV_USD


async def run_real(cfg: Config, generator: Generator, judge: Judge) -> SmokeSummary:
    """Search each smoke problem; grade with ``data.grading`` (never inside the search)."""
    from thoughtzero.data.grading import is_equivalent

    rows: list[dict[str, Any]] = []
    for pid, question, gold in SMOKE_PROBLEMS:
        t0 = time.monotonic()
        with ledger_scope() as ledger:
            result = await search(question, generator, judge, cfg.search, seed=cfg.seed)
        correct = is_equivalent(result.answer, gold)
        row = {
            "id": pid,
            "answer": result.answer,
            "gold": gold,
            "correct": correct,
            "jev_calls": ledger.jev_calls,
            "jev_cache_hits": ledger.jev_cache_hits,
            "jev_usd": ledger.jev_usd,
            "gemma_completion_tokens": ledger.gemma_completion_tokens,
            "expansions": result.stats["expansions"],
            "failed_simulations": result.stats["failed_simulations"],
            "seconds": round(time.monotonic() - t0, 1),
        }
        rows.append(row)
        print(
            f"{pid}: answer={result.answer!r} gold={gold} {'OK' if correct else 'WRONG'} | "
            f"Jev calls={ledger.jev_calls} (cached {ledger.jev_cache_hits}) "
            f"${ledger.jev_usd:.5f} | Gemma tokens={ledger.gemma_completion_tokens} | "
            f"{row['seconds']} s"
        )
        print(f"    path: {result.answer_steps}")
    return SmokeSummary(
        solved=sum(r["correct"] for r in rows),
        jev_calls=sum(r["jev_calls"] for r in rows),
        jev_usd=sum(r["jev_usd"] for r in rows),
        rows=rows,
    )


def _check_gemma_server(base_url: str) -> str | None:
    """None if the OpenAI-compatible server answers ``/models``, else what went wrong."""
    try:
        resp = httpx.get(base_url.rstrip("/") + "/models", timeout=10)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        return f"{type(e).__name__}: {e}"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_config_args(parser, default="configs/default.yaml")
    parser.add_argument("--mock", action="store_true", help="offline mocks, toy task")
    parser.add_argument("--n-sims", type=int, default=4, help="simulations per problem (real)")
    parser.add_argument("--k", type=int, default=3, help="candidate steps per expansion (real)")
    parser.add_argument(
        "--tokenizer", default=None, help="HF tokenizer id if GEMMA_MODEL isn't one"
    )
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    setup_logging()
    cfg = config_from_args(args)
    print(f"config hash: {config_hash(cfg)}")
    if args.mock:
        asyncio.run(run_mock(cfg))
        return

    cfg = cfg.model_copy(
        update={"search": cfg.search.model_copy(update={"n_simulations": args.n_sims, "k": args.k})}
    )
    problem = _check_gemma_server(cfg.generator.base_url)
    if problem:
        raise SystemExit(
            f"Gemma server not reachable at {cfg.generator.base_url} ({problem}). "
            "Start one (docs/gpu_setup.md) and set GEMMA_BASE_URL / GEMMA_MODEL in .env."
        )

    from thoughtzero.judge import budget
    from thoughtzero.judge.factory import make_judge
    from thoughtzero.llm.gemma import OpenAICompatibleGenerator

    budget.configure(5 * MAX_JEV_USD, cfg.judge.usd_per_mtok)  # hard stop well above the target
    try:
        judge = make_judge(cfg.judge)
    except (NotImplementedError, ValueError) as e:
        raise SystemExit(f"Jev judge unavailable: {e}") from e
    generator = OpenAICompatibleGenerator.from_config(cfg, tokenizer_name=args.tokenizer)
    print(
        f"Gemma {cfg.generator.model} @ {cfg.generator.base_url} | Jev route "
        f"{cfg.judge.transport} | n_sims={args.n_sims} k={args.k}"
    )

    summary = asyncio.run(run_real(cfg, generator, judge))
    print(
        f"\nsolved {summary.solved}/3 | Jev calls {summary.jev_calls} (limit {MAX_JEV_CALLS}) | "
        f"Jev ${summary.jev_usd:.5f} (limit ${MAX_JEV_USD})"
    )
    print("PASS" if summary.passed else "FAIL")
    if not summary.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
