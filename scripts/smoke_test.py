"""End-to-end smoke test on 3 problems (SPEC.md §10, Phase 0). Owner: Person 1 (Prakhar).

    python scripts/smoke_test.py --mock      # offline: mock generator + judge, toy task
    python scripts/smoke_test.py             # real Gemma + Jev (available from CP1)

``--mock`` runs greedy-by-prior (expected WRONG on the toy task) and then MCTS with the
config's search settings (k forced to 3, the toy's branching), and checks MCTS gets it right.
"""

from __future__ import annotations

import argparse
import asyncio

from thoughtzero.accounting import ledger_scope
from thoughtzero.config import Config, add_config_args, config_from_args, config_hash
from thoughtzero.logging_setup import setup_logging
from thoughtzero.mocks import TOY, MockGenerator, MockJudge, greedy_by_prior, toy_extract_answer
from thoughtzero.search.mcts import search


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser, default="configs/default.yaml")
    parser.add_argument("--mock", action="store_true", help="offline mocks, toy task")
    args = parser.parse_args()
    setup_logging()
    cfg = config_from_args(args)
    print(f"config hash: {config_hash(cfg)}")
    if args.mock:
        asyncio.run(run_mock(cfg))
    else:
        raise SystemExit("real smoke test lands at CP1 (needs generator, judge, search)")


if __name__ == "__main__":
    main()
