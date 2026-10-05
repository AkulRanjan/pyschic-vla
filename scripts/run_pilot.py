"""Pilot: --stage traces|label|score|sensitivity|report|all|merge (SPEC.md §7).

Owner: Person 4 (Harjas). Every stage is resumable and can be re-run alone.

    python scripts/run_pilot.py --stage traces                         # GPU
    python scripts/run_pilot.py --stage label --shard 0/3              # GPU, heavy; 1 shard/account
    python scripts/run_pilot.py --stage merge --from dl/acct0 dl/acct1 dl/acct2
    python scripts/run_pilot.py --stage score --judges jev [--yes]     # Jev; self/prm need GPU
    python scripts/run_pilot.py --stage sensitivity                    # Jev x SOUND_VARIANTS
    python scripts/run_pilot.py --stage report                         # -> results/pilot/report.md

    python scripts/run_pilot.py --stage all --mock                     # offline check
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from thoughtzero.config import Config, JudgeCfg, dump_resolved
from thoughtzero.eval.cli import add_run_args, mock_problems, setup
from thoughtzero.eval.methods import (
    MethodUnavailable,
    build_generator,
    build_judge,
    judge_needs_generator,
)
from thoughtzero.eval.runner import CostConfirmationRequired, check_cost, git_commit, parse_shard
from thoughtzero.eval.toolkit import Toolkit, mock_toolkit, real_toolkit
from thoughtzero.llm.prompts import SOUND_VARIANTS
from thoughtzero.pilot import mc_label, report, score, traces
from thoughtzero.pilot.common import load_rows
from thoughtzero.types import Problem

STAGES = ["traces", "label", "score", "sensitivity", "report"]
PROMPT_OVERHEAD_TOKENS = 400  # rough: instructions and formatting around problem + prefix


def load_pilot_problems(cfg: Config, mock: bool) -> list[Problem]:
    if mock:
        return mock_problems(min(cfg.pilot.n_problems, 12), source="toypilot")
    from thoughtzero.data.datasets import pilot_subset

    return pilot_subset(cfg.pilot.n_problems, cfg.seed, split=cfg.pilot.source_split)


def estimate_jev_usd(
    cfg: Config, prefixes: list[dict[str, Any]], tr: dict[str, Any], pb: dict[str, Problem]
) -> float:
    """~4 chars/token over problem + prefix, plus overhead; ignores cache hits."""
    tok = 0
    for px in prefixes:
        steps = traces.prefix_steps(px, tr)
        chars = len(pb[px["problem_id"]].question) + sum(len(s) for s in steps)
        tok += chars // 4 + PROMPT_OVERHEAD_TOKENS
    return tok * cfg.judge.usd_per_mtok / 1e6


def merge_dirs(srcs: list[str], dst: Path) -> None:
    """Combine stage outputs downloaded from several Kaggle accounts (de-duplicated)."""
    files: dict[str, list[Path]] = {}
    for s in srcs:
        for f in Path(s).glob("*.jsonl"):
            files.setdefault(f.name, []).append(f)
    for name, paths in files.items():
        key = "problem_id" if name == traces.TRACES_FILE else "prefix_id"
        tmp = dst / f".{name}.merge"
        with tmp.open("w", encoding="utf-8") as out:
            for p in [*paths, *([dst / name] if (dst / name).exists() else [])]:
                for line in p.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        out.write(line + "\n")
        rows = load_rows(tmp, key)
        with (dst / name).open("w", encoding="utf-8") as out:
            for r in sorted(rows, key=lambda r: r[key]):
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.unlink()
        print(f"merged {name}: {len(rows)} rows from {len(paths)} source(s)")


async def stage_score(
    cfg: Config,
    args: argparse.Namespace,
    out: Path,
    tr: dict[str, Any],
    pb: dict[str, Problem],
    prefixes: list[dict[str, Any]],
    sensitivity: bool,
) -> int:
    shard = parse_shard(args.shard)
    judges = [j.strip() for j in args.judges.split(",")]
    variants: list[int | None] = [None]
    if sensitivity:
        if "sound_variant" not in JudgeCfg.model_fields:
            print(
                "sensitivity skipped: JudgeCfg has no `sound_variant` field yet "
                "(request from Person 2 / Jagriti)",
                file=sys.stderr,
            )
            return 0
        ids = set(score.sensitivity_subset(list(tr), cfg.pilot.sensitivity_subset, cfg.seed))
        prefixes = [p for p in prefixes if p["problem_id"] in ids]
        judges, variants = [report.DECISION_JUDGE], list(range(len(SOUND_VARIANTS)))
    if report.DECISION_JUDGE in judges and not args.mock:
        est = len(variants) * estimate_jev_usd(cfg, prefixes, tr, pb)
        try:
            check_cost(est, args.yes, cfg.budget.confirm_above_usd)
        except CostConfirmationRequired as e:
            print(f"REFUSING: {e}", file=sys.stderr)
            return 2
    for name in judges:
        for v in variants:
            update: dict[str, Any] = {"kind": name}
            if v is not None:
                update["sound_variant"] = v
            jcfg = cfg.judge.model_copy(update=update)
            try:
                gen = (
                    build_generator(cfg, mock=args.mock)
                    if judge_needs_generator(jcfg) and not args.mock
                    else None
                )
                judge = build_judge(jcfg, seed=cfg.seed, mock=args.mock, generator=gen)
            except (MethodUnavailable, ValueError) as e:  # e.g. PRM not available: skip it
                print(f"skipping judge {name}: {e}", file=sys.stderr)
                continue
            tag = None if v is None else f"v{v}"
            st = await score.score_prefixes(
                prefixes,
                tr,
                pb,
                judge,
                name,
                out,
                variant=tag,
                concurrency=args.concurrency,
                shard=shard,
            )
            print(f"score {name}{'/' + tag if tag else ''}: {st}")
            if st.stopped_reason:
                return 3
    return 0


def do_report(cfg: Config, args: argparse.Namespace, out: Path) -> int:
    tl = traces.load_traces(out)
    ids = score.sensitivity_subset(
        [t["problem_id"] for t in tl], cfg.pilot.sensitivity_subset, cfg.seed
    )
    variants = [
        f"v{i}"
        for i in range(len(SOUND_VARIANTS))
        if score.scores_path(out, report.DECISION_JUDGE, f"v{i}").exists()
    ]
    _, decision = report.build_report(
        out,
        judges=[j.strip() for j in args.judges.split(",")],
        variants=variants,
        sensitivity_ids=ids,
        max_prefixes=cfg.pilot.max_prefixes,
        seed=cfg.seed,
    )
    print(f"wrote {out / report.REPORT_FILE}\n\n{decision.markdown()}")
    return 0


def prepare_out(cfg: Config, args: argparse.Namespace) -> Path:
    """Results folder with the resolved config and git commit (SPEC.md §13.7)."""
    out = Path(args.out or Path(cfg.eval.out_dir) / "pilot")
    out.mkdir(parents=True, exist_ok=True)
    dump_resolved(cfg, out / "config.yaml")
    (out / "git_commit.txt").write_text(git_commit() + "\n")
    return out


async def run(cfg: Config, args: argparse.Namespace, out: Path) -> int:
    stages = STAGES if args.stage == "all" else [args.stage]
    tk: Toolkit = mock_toolkit() if args.mock else real_toolkit()
    shard = parse_shard(args.shard)
    problems = load_pilot_problems(cfg, args.mock)
    pb = {p.id: p for p in problems}

    if "traces" in stages:
        st = await traces.generate_traces(
            problems,
            build_generator(cfg, mock=args.mock),
            out,
            temperature=cfg.pilot.temperature,
            concurrency=args.concurrency,
            shard=shard,
            tools=tk,
        )
        print(f"traces: {st}")
        tl = traces.load_traces(out)
        sanity = traces.sanity_markdown(tl, seed=cfg.seed)
        (out / "trace_sanity.md").write_text(sanity, encoding="utf-8")
        print(f"trace stats: {traces.trace_stats(tl)} (details: {out / 'trace_sanity.md'})")

    tl = traces.load_traces(out)
    tr = {t["problem_id"]: t for t in tl}
    prefixes = traces.make_prefixes(tl, cfg.pilot.max_prefixes)
    if stages != ["traces"]:
        print(f"{len(tl)} traces -> {len(prefixes)} prefixes")

    if "label" in stages:
        st = await mc_label.label_prefixes(
            prefixes,
            tr,
            pb,
            build_generator(cfg, mock=args.mock),
            out,
            n_samples=cfg.pilot.m_completions,
            temperature=cfg.pilot.temperature,
            concurrency=args.concurrency,
            shard=shard,
            tools=tk,
        )
        print(f"labels: {st}")
    for stage, sens in (("score", False), ("sensitivity", True)):
        if stage in stages:
            rc = await stage_score(cfg, args, out, tr, pb, prefixes, sensitivity=sens)
            if rc:
                return rc
    if "report" in stages:
        return do_report(cfg, args, out)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_run_args(parser, default_config="configs/pilot.yaml")
    parser.add_argument("--stage", required=True, choices=[*STAGES, "all", "merge"])
    parser.add_argument("--judges", default="jev,self,prm", help="comma list for score/report")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--out", default=None, help="default: <eval.out_dir>/pilot")
    parser.add_argument("--from", dest="from_dirs", nargs="+", default=[], help="merge sources")
    args = parser.parse_args(argv)
    cfg = setup(args)
    out = prepare_out(cfg, args)
    if args.stage == "merge":
        merge_dirs(args.from_dirs, out)
        return 0
    if args.stage == "report":
        return do_report(cfg, args, out)
    return asyncio.run(run(cfg, args, out))


if __name__ == "__main__":
    sys.exit(main())
