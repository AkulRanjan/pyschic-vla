"""Shared CLI plumbing for Person 4's scripts (run_experiment, run_pilot). Owner: Person 4."""

from __future__ import annotations

import argparse
import re
import sys
from typing import Any

from thoughtzero.config import Config, add_config_args, apply_variant, config_from_args
from thoughtzero.logging_setup import setup_logging
from thoughtzero.types import Problem

SUBSETS = ("full", "dev", "ablation")


def add_run_args(parser: argparse.ArgumentParser, default_config: str) -> None:
    add_config_args(parser, default=default_config)
    parser.add_argument("--variant", default=None, help="named override set from cfg.variants")
    parser.add_argument("--shard", default=None, help="i/n: run only shard i of n (0-based)")
    parser.add_argument("--yes", action="store_true", help="confirm a run estimated above $1")
    parser.add_argument(
        "--mock", action="store_true", help="offline: mock generator/judge on the toy task"
    )
    parser.add_argument("--log-level", default="INFO")


def setup(args: argparse.Namespace) -> Config:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    setup_logging(args.log_level)
    cfg = config_from_args(args)
    if args.variant:
        cfg = apply_variant(cfg, args.variant)
    if args.mock:  # the toy task has exactly three increments to branch on
        cfg = cfg.model_copy(update={"search": cfg.search.model_copy(update={"k": 3})})
    else:  # hard cap on Jev spend for this run (SPEC.md §5.4); JevJudge raises BudgetExceeded
        from thoughtzero.judge import budget

        budget.configure(cfg.budget.max_usd, cfg.judge.usd_per_mtok)
    return cfg


def mock_problems(n: int, source: str = "toy") -> list[Problem]:
    """n distinct instances of the mocks' reach-13 toy task (answer "13")."""
    from thoughtzero.mocks import TOY

    return [
        Problem(
            id=f"{source}/{i:04d}",
            question=f"{TOY.question} (instance {i})",
            answer=TOY.answer,
            source=source,
            level=1 + i % 5,
        )
        for i in range(n)
    ]


_AIME = re.compile(r"aime(\d{4})$")


def load_problems(
    cfg: Config, subset: str, *, mock: bool = False, n_mock: int = 12
) -> list[Problem]:
    """Problems for a run, always via Person 3's deterministic loaders (one subset function)."""
    if mock:
        return mock_problems(cfg.eval.n_problems or n_mock)
    from thoughtzero.data import datasets

    n, seed = cfg.eval.n_problems, cfg.eval.subset_seed
    if subset == "dev":
        return datasets.dev_subset(n or 50, seed)
    if subset == "ablation":
        return datasets.ablation_subset(n or 200, seed)
    if cfg.eval.dataset == "math500":
        probs = datasets.load_math500()
    elif m := _AIME.match(cfg.eval.dataset):
        probs = datasets.load_aime(int(m.group(1)))
    else:
        raise ValueError(f"unknown eval.dataset {cfg.eval.dataset!r} (math500 | aimeYYYY)")
    return probs[:n] if n else probs


def fmt_summary(rows: list[dict[str, Any]]) -> str:
    head = (
        f"{'method':<28} {'N':>5} {'err':>4} {'acc':>6} {'95% CI':>15} {'tok/prob':>9} {'Jev $':>8}"
    )
    out = [head]
    for r in rows:
        out.append(
            f"{r['method_id']:<28} {r['n_problems']:>5} {r['n_errors']:>4} "
            f"{100 * float(r['accuracy']):>5.1f}% "
            f"[{100 * float(r['ci_lo']):>5.1f},{100 * float(r['ci_hi']):>5.1f}] "
            f"{float(r['mean_completion_tokens']):>9.0f} {float(r['jev_usd_total']):>8.4f}"
        )
    return "\n".join(out)
