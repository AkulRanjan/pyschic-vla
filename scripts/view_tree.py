"""Pretty-print ThoughtZero search trees from a run's per_problem.jsonl. Owner: Person 1 (Prakhar).

    python scripts/view_tree.py results/tz16_dev                  # list problems with trees
    python scripts/view_tree.py results/tz16_dev --problem math500/0042
    python scripts/view_tree.py results/tz16_dev/per_problem.jsonl --problem math500/0042 \
        --method tz_self --min-visits 2 --max-depth 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from thoughtzero.search.tree_view import format_tree


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        path = path / "per_problem.jsonl"
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if isinstance(row.get("raw"), dict) and row["raw"].get("tree"):
                    rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("path", type=Path, help="run folder or per_problem.jsonl")
    parser.add_argument("--problem", help="problem id; omit to list the problems")
    parser.add_argument("--method", help="method_id substring, e.g. tz_self or n_simulations=16")
    parser.add_argument("--min-visits", type=int, default=0, help="hide children with fewer")
    parser.add_argument("--max-depth", type=int, default=None, help="levels to show")
    parser.add_argument("--width", type=int, default=80, help="characters of each step shown")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # box-drawing characters on Windows consoles

    rows = load_rows(args.path)
    if args.method:
        rows = [r for r in rows if args.method in r["method_id"]]
    if not rows:
        print("no rows with a search tree found", file=sys.stderr)
        return 1

    if args.problem is None:
        for r in rows:
            mark = "ok " if r.get("correct") else "BAD"
            stats = r["raw"].get("stats", {})
            print(
                f"{mark} {r['problem_id']:<24} {r['method_id']:<28} answer={r.get('answer')!s:<10} "
                f"gold={r.get('gold')!s:<10} nodes={stats.get('n_nodes', '?')}"
            )
        return 0

    matches = [r for r in rows if r["problem_id"] == args.problem]
    if not matches:
        print(
            f"problem {args.problem!r} not found (run without --problem to list)", file=sys.stderr
        )
        return 1
    for r in matches:
        stats = r["raw"].get("stats", {})
        print(
            f"# {r['problem_id']}  {r['method_id']}  answer={r.get('answer')}  "
            f"gold={r.get('gold')}  correct={r.get('correct')}"
        )
        print(
            f"# most_visited={stats.get('answer_most_visited')}  "
            f"value_vote={stats.get('answer_value_vote')}  "
            f"expansions={stats.get('expansions')}  max_depth={stats.get('max_depth')}"
        )
        print(
            format_tree(
                r["raw"]["tree"],
                max_depth=args.max_depth,
                min_visits=args.min_visits,
                step_chars=args.width,
            )
        )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
