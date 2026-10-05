"""Merge runs and produce the main figures and tables (spec §8.3, §9, §11).

    python scripts/make_plots.py --runs results/tz_math500 results/sc_math500 results/c31b \
        --out results/figures [--baseline sc] [--aime-cutoff-year 2024] [--allow-mismatch]

Writes into --out: accuracy_vs_compute.png, accuracy_by_level.png, main_table.md,
by_level.md, aime_split.md, diagnostics.md, audit.md, tz_tokens_per_nsim.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from thoughtzero.eval import analysis, plots


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--baseline", default="sc", help="method for matched-compute paired tests (B2)")
    ap.add_argument("--aime-cutoff-year", type=int, default=None)
    ap.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="plot even if methods cover different problem sets (comparisons still refuse)",
    )
    ap.add_argument(
        "--sample-ns",
        default=None,
        help="comma list, e.g. 1,2,4,8,16,32,64: derive B2/B3 points from stored samples",
    )
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = analysis.load_runs(args.runs)
    if not rows:
        print("no rows found", file=sys.stderr)
        return 1
    if args.sample_ns:
        from thoughtzero.data.grading import is_equivalent

        ns = [int(x) for x in args.sample_ns.split(",")]
        rows = analysis.expand_sample_curves(rows, ns, is_equivalent)
    by_source: dict[str, list[dict]] = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)

    audit = analysis.sanity_audit(rows, seed=args.seed)
    audit_md = [
        "# Sanity audit",
        "",
        f"Problem-ID sets: {audit['ids_detail']}",
        "",
        analysis.md_table(
            [{"method": m, "rows": n, "errors": audit["errors"][m]} for m, n in audit["n"].items()],
            ["method", "rows", "errors"],
        ),
        "",
        "## 20 random rows: check the grader by hand",
        "",
        analysis.md_table(
            audit["manual_sample"], ["problem_id", "method_id", "answer", "gold", "correct"]
        ),
    ]
    (out / "audit.md").write_text("\n".join(audit_md) + "\n", encoding="utf-8")

    tables = ["# Main results", ""]
    for src, rs in sorted(by_source.items()):
        try:
            analysis.assert_same_problem_ids(analysis.rows_by_method(rs))
        except analysis.ProblemSetMismatch as e:
            print(f"[{src}] {e}", file=sys.stderr)
            if not args.allow_mismatch:
                return 2
        pts = analysis.points(rs, args.n_boot, args.seed)
        plots.plot_accuracy_vs_compute(
            pts, out / f"accuracy_vs_compute_{src}.png", f"Accuracy vs compute: {src}"
        )
        try:
            table = analysis.main_table(rs, args.baseline, args.n_boot, args.seed)
        except analysis.ProblemSetMismatch as e:
            print(f"[{src}] paired tests skipped: {e}", file=sys.stderr)
            table = [dict(p) for p in pts]
        tables += [f"## {src}", "", analysis.main_table_markdown(table), ""]
    (out / "main_table.md").write_text("\n".join(tables), encoding="utf-8")

    lvl = analysis.by_level(rows, args.n_boot, args.seed)
    if lvl:
        top = analysis.top_compute_ids(rows)  # one line per method: its highest compute level
        plots.plot_accuracy_by_level(
            [r for r in lvl if r["method_id"] in top], out / "accuracy_by_level.png"
        )
        (out / "by_level.md").write_text(
            analysis.md_table(lvl, ["method_id", "level", "n", "accuracy", "ci_lo", "ci_hi"])
            + "\n",
            encoding="utf-8",
        )
    if args.aime_cutoff_year is not None:
        split = analysis.aime_split(rows, args.aime_cutoff_year, args.n_boot, args.seed)
        (out / "aime_split.md").write_text(
            analysis.md_table(split, ["method_id", "group", "n", "accuracy", "ci_lo", "ci_hi"])
            + "\n",
            encoding="utf-8",
        )

    diag = analysis.diagnostics_table(rows)
    if diag:
        (out / "diagnostics.md").write_text(
            analysis.md_table(diag, list(diag[0].keys())) + "\n", encoding="utf-8"
        )
    tok = analysis.tz_tokens_per_nsim(rows)
    (out / "tz_tokens_per_nsim.json").write_text(json.dumps(tok, indent=2) + "\n")
    print(f"wrote figures and tables to {out}")
    if tok:
        print(
            f"TZ mean completion tokens per n_simulations (send to Person 3 for B2 matching): {tok}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
