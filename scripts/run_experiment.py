"""Run methods over a dataset; resumable, shardable (SPEC.md §8). Owner: Person 4 (Harjas).

    python scripts/run_experiment.py --config configs/exp_main.yaml \\
        --set "eval.methods=[tz]" --set search.n_simulations=16 --run-id tz16_math500 \\
        [--subset full|dev|ablation] [--shard 0/3] [--yes]

    # self-consistency with N=8 samples, best-of-4:
    python scripts/run_experiment.py --set "eval.methods=['sc:8','bon:4']" --run-id sc8_bon4

    # ablation variant (configs/ablations.yaml):
    python scripts/run_experiment.py --config configs/ablations.yaml --subset ablation \\
        --variant prior_only --run-id abl_prior_only

    # merge shards (or per-method runs) into one folder:
    python scripts/run_experiment.py --merge results/tz16_s0 results/tz16_s1 --run-id tz16_math500

    # offline plumbing check (mock generator + judge on the toy task):
    python scripts/run_experiment.py --mock --run-id mock_check
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from thoughtzero.eval.cli import SUBSETS, add_run_args, fmt_summary, load_problems, setup
from thoughtzero.eval.methods import MethodUnavailable, build_generator, build_methods
from thoughtzero.eval.runner import (
    CostConfirmationRequired,
    check_cost,
    estimate_jev_usd,
    merge_runs,
    parse_shard,
    prepare_run_dir,
    read_summary,
    run_methods,
    shard_problems,
)
from thoughtzero.eval.toolkit import mock_toolkit, real_toolkit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_run_args(parser, default_config="configs/exp_main.yaml")
    parser.add_argument("--run-id", default=None, help="results/<run-id>/ (default: eval.run_id)")
    parser.add_argument("--subset", choices=SUBSETS, default="full")
    parser.add_argument("--timeout", type=float, default=None, help="per-problem seconds")
    parser.add_argument("--retry-errors", action="store_true", help="re-run errored problems")
    parser.add_argument("--merge", nargs="+", metavar="RUN_DIR", help="merge into --run-id")
    parser.add_argument("--allow-mixed-config", action="store_true")
    args = parser.parse_args(argv)
    cfg = setup(args)
    run_id = args.run_id or cfg.eval.run_id
    if not run_id:
        parser.error("--run-id is required (or set eval.run_id)")
    run_dir = Path(cfg.eval.out_dir) / run_id

    if args.merge:
        rows = merge_runs(args.merge, run_dir, allow_mixed_config=args.allow_mixed_config)
        print(f"merged {len(args.merge)} runs -> {run_dir}\n{fmt_summary(rows)}")
        return 0

    shard = parse_shard(args.shard)
    problems = load_problems(cfg, args.subset, mock=args.mock)
    tools = mock_toolkit() if args.mock else real_toolkit()
    generator = build_generator(cfg, mock=args.mock)
    try:
        methods = build_methods(
            cfg.eval.methods, cfg, generator, mock=args.mock, tools=tools if args.mock else None
        )
    except MethodUnavailable as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 4
    n_mine = len(shard_problems(problems, shard))
    print(
        f"{n_mine} problems (shard {args.shard or 'all'}); methods {cfg.eval.methods} -> {run_dir}"
    )
    try:
        check_cost(estimate_jev_usd(cfg, n_mine, methods), args.yes, cfg.budget.confirm_above_usd)
    except CostConfirmationRequired as e:
        print(f"REFUSING: {e}", file=sys.stderr)
        return 2

    ctx = prepare_run_dir(run_dir, cfg)
    summaries = asyncio.run(
        run_methods(
            methods,
            problems,
            cfg,
            ctx,
            shard=shard,
            retry_errors=args.retry_errors,
            timeout_s=args.timeout,
            grader=tools.grade,
        )
    )
    for s in summaries:
        stop = f", STOPPED: {s.stopped_reason}" if s.stopped else ""
        print(f"{s.method_id}: ran {s.n_run} (skipped {s.n_skipped}), {s.n_errors} errors{stop}")
    print(fmt_summary(read_summary(run_dir)))
    return 3 if any(s.stopped for s in summaries) else 0


if __name__ == "__main__":
    sys.exit(main())
