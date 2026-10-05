"""Cross-run analysis for the main results (spec §8.3, §9, §11).

Operates on per-problem rows (runner JSONL). Every comparison between two
methods first asserts they cover the identical problem-ID set.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from thoughtzero.eval.metrics import (
    accuracy,
    bootstrap_ci,
    paired_bootstrap_diff,
    search_diagnostics,
)
from thoughtzero.eval.runner import ROWS_FILE, dedupe_rows, read_jsonl

Row = dict[str, Any]

# Display order for tables; unknown methods sort after these, alphabetically.
METHOD_ORDER = [
    "tz",
    "tz_self",
    "tz_prm",
    "tz_hybrid",
    "tz_prior_only",
    "tz_value_only",
    "sc",
    "bon",
    "cot",
    "c31b",
]
# Methods whose compute is not Gemma tokens: never compute-matched against B2.
UNMATCHED_METHODS = {"c31b"}


def _order(p: Row) -> tuple[int, str, float]:
    m = p["method"]
    return (
        METHOD_ORDER.index(m) if m in METHOD_ORDER else len(METHOD_ORDER),
        m,
        p["mean_completion_tokens"],
    )


def mid(r: Row) -> str:
    return r.get("method_id") or r["method"]


def load_runs(run_dirs: Sequence[str | Path]) -> list[Row]:
    rows: list[Row] = []
    for d in run_dirs:
        rows.extend(read_jsonl(Path(d) / ROWS_FILE))
    return dedupe_rows(rows)


def rows_by_method(rows: Sequence[Row]) -> dict[str, list[Row]]:
    out: dict[str, list[Row]] = {}
    for r in rows:
        out.setdefault(mid(r), []).append(r)
    return out


class ProblemSetMismatch(ValueError):
    pass


def assert_same_problem_ids(groups: Mapping[str, Sequence[Row]]) -> set[str]:
    """Raise if the methods were not run on exactly the same problems."""
    sets = {m: {r["problem_id"] for r in rs} for m, rs in groups.items()}
    if not sets:
        return set()
    ref_m, ref = next(iter(sets.items()))
    bad = {m: (len(s - ref), len(ref - s)) for m, s in sets.items() if s != ref}
    if bad:
        detail = ", ".join(f"{m}: +{a}/-{b} vs {ref_m}" for m, (a, b) in bad.items())
        raise ProblemSetMismatch(f"problem-ID sets differ: {detail}")
    return ref


# --------------------------------------------------------------------------- points & tables


def points(rows: Sequence[Row], n_boot: int = 1000, seed: int = 0) -> list[Row]:
    """One point per method_id: accuracy [CI], mean tokens, wall time, Jev USD."""
    out = []
    for m, rs in sorted(rows_by_method(rows).items()):
        corr = [bool(r["correct"]) for r in rs]
        lo, hi = bootstrap_ci(corr, n=n_boot, seed=seed)
        out.append(
            {
                "method_id": m,
                "method": rs[0]["method"],
                "params": rs[0].get("params") or {},
                "n": len(rs),
                "n_errors": sum(1 for r in rs if r.get("error")),
                "accuracy": accuracy(corr),
                "ci_lo": lo,
                "ci_hi": hi,
                "mean_completion_tokens": float(
                    np.mean([r.get("gemma_completion_tokens", 0) for r in rs])
                ),
                "mean_wall_time_s": float(np.mean([r.get("wall_time_s", 0.0) for r in rs])),
                "jev_usd": float(np.sum([r.get("jev_usd", 0.0) for r in rs])),
            }
        )
    return sorted(out, key=_order)


def tz_tokens_per_nsim(rows: Sequence[Row], method: str = "tz") -> dict[int, float]:
    """Mean completion tokens per n_simulations, for B2 compute matching (Person 3)."""
    out: dict[int, list[float]] = {}
    for r in rows:
        if r["method"] == method and "n_simulations" in (r.get("params") or {}):
            out.setdefault(int(r["params"]["n_simulations"]), []).append(
                r.get("gemma_completion_tokens", 0)
            )
    return {k: float(np.mean(v)) for k, v in sorted(out.items())}


def nearest_by_compute(target_tokens: float, candidates: Sequence[Row]) -> Row | None:
    """Candidate point with the closest mean completion tokens (log scale)."""
    cs = [c for c in candidates if c["mean_completion_tokens"] > 0]
    if not cs or target_tokens <= 0:
        return None
    return min(
        cs, key=lambda c: abs(math.log(c["mean_completion_tokens"]) - math.log(target_tokens))
    )


def paired_vs(
    rows: Sequence[Row], a: str, b: str, n_boot: int = 1000, seed: int = 0
) -> dict[str, Any]:
    by = rows_by_method(rows)
    assert_same_problem_ids({a: by[a], b: by[b]})
    ca = {r["problem_id"]: bool(r["correct"]) for r in by[a]}
    cb = {r["problem_id"]: bool(r["correct"]) for r in by[b]}
    ids = sorted(ca)
    d = paired_bootstrap_diff([ca[i] for i in ids], [cb[i] for i in ids], n=n_boot, seed=seed)
    return {"a": a, "b": b, "diff": d.diff, "lo": d.lo, "hi": d.hi, "p": d.p_value, "n": d.n}


def main_table(
    rows: Sequence[Row], baseline: str = "sc", n_boot: int = 1000, seed: int = 0
) -> list[Row]:
    """One row per method/compute level, plus the paired difference vs B2 at matched compute."""
    pts = points(rows, n_boot, seed)
    base = [p for p in pts if p["method"] == baseline]
    out = []
    for p in pts:
        row = dict(p)
        if p["method"] != baseline and p["method"] not in UNMATCHED_METHODS and base:
            match = nearest_by_compute(p["mean_completion_tokens"], base)
            if match is not None:
                pv = paired_vs(rows, p["method_id"], match["method_id"], n_boot, seed)
                row.update(
                    vs=match["method_id"],
                    diff=pv["diff"],
                    diff_lo=pv["lo"],
                    diff_hi=pv["hi"],
                    p=pv["p"],
                    compute_ratio=p["mean_completion_tokens"] / match["mean_completion_tokens"],
                )
        out.append(row)
    return out


def accuracy_by(
    rows: Sequence[Row], key: Callable[[Row], Any], n_boot: int = 1000, seed: int = 0
) -> list[Row]:
    """Accuracy [CI] per (method_id, key(row)); rows where key is None are dropped."""
    groups: dict[tuple[str, Any], list[Row]] = {}
    for r in rows:
        k = key(r)
        if k is not None:
            groups.setdefault((mid(r), k), []).append(r)
    out = []
    for (m, k), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        corr = [bool(r["correct"]) for r in rs]
        lo, hi = bootstrap_ci(corr, n=n_boot, seed=seed)
        out.append(
            {
                "method_id": m,
                "method": rs[0]["method"],
                "label": m,
                "group": k,
                "n": len(rs),
                "accuracy": accuracy(corr),
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
    return out


def by_level(rows: Sequence[Row], n_boot: int = 1000, seed: int = 0) -> list[Row]:
    out = accuracy_by(
        [r for r in rows if r.get("source", "").startswith("math")],
        lambda r: r.get("level"),
        n_boot,
        seed,
    )
    for r in out:
        r["level"] = r["group"]
    return out


def top_compute_ids(rows: Sequence[Row]) -> set[str]:
    """For each method, the method_id with the highest mean completion tokens."""
    best: dict[str, tuple[float, str]] = {}
    for p in points(rows, n_boot=1):
        t = (p["mean_completion_tokens"], p["method_id"])
        if p["method"] not in best or t > best[p["method"]]:
            best[p["method"]] = t
    return {mid_ for _, mid_ in best.values()}


_YEAR = re.compile(r"(19|20)\d{2}")


def aime_split(
    rows: Sequence[Row], cutoff_year: int, n_boot: int = 1000, seed: int = 0
) -> list[Row]:
    """AIME accuracy split by problem year vs. the generator's training cutoff year.

    Year is parsed from `source` or `problem_id` (e.g. "aime2025/I/3"). Problems
    from years > cutoff_year are "post-cutoff" (uncontaminated).
    """

    def key(r: Row) -> str | None:
        if "aime" not in r.get("source", ""):
            return None
        m = _YEAR.search(r.get("source", "")) or _YEAR.search(r["problem_id"])
        if not m:
            return None
        return "post-cutoff" if int(m.group(0)) > cutoff_year else "pre-cutoff"

    return accuracy_by(rows, key, n_boot, seed)


def diagnostics_table(rows: Sequence[Row], prefix: str = "tz") -> list[Row]:
    """Spec §9 search diagnostics per TZ method_id."""
    return [
        {"method_id": m, **search_diagnostics(rs)}
        for m, rs in sorted(rows_by_method(rows).items())
        if rs[0]["method"].startswith(prefix)
    ]


def sanity_audit(rows: Sequence[Row], n_manual: int = 20, seed: int = 0) -> dict[str, Any]:
    """Error counts per method, problem-ID set agreement, and a sample to grade by hand."""
    by = rows_by_method(rows)
    errors = {m: sum(1 for r in rs if r.get("error")) for m, rs in by.items()}
    try:
        assert_same_problem_ids(by)
        ids_ok, ids_msg = True, "identical"
    except ProblemSetMismatch as e:
        ids_ok, ids_msg = False, str(e)
    rng = np.random.default_rng(seed)
    ok_rows = [r for r in rows if not r.get("error")]
    k = min(n_manual, len(ok_rows))
    pick: list[int] = rng.choice(len(ok_rows), size=k, replace=False).tolist() if ok_rows else []
    sample = [
        {
            k: ok_rows[int(i)].get(k)
            for k in ("problem_id", "method_id", "answer", "gold", "correct")
        }
        for i in pick
    ]
    return {
        "errors": errors,
        "n": {m: len(rs) for m, rs in by.items()},
        "ids_identical": ids_ok,
        "ids_detail": ids_msg,
        "manual_sample": sample,
    }


# --------------------------------------------------------------------------- markdown


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return "n/a" if math.isnan(v) else f"{v:.3f}" if abs(v) < 10 else f"{v:,.1f}"
    if isinstance(v, dict):
        return json.dumps(v, sort_keys=True)
    return "" if v is None else str(v)


def md_table(rows: Sequence[Row], columns: Sequence[str]) -> str:
    head = "| " + " | ".join(columns) + " |\n|" + "---|" * len(columns)
    body = ["| " + " | ".join(_fmt(r.get(c)) for c in columns) + " |" for r in rows]
    return "\n".join([head, *body])


def main_table_markdown(table: Sequence[Row]) -> str:
    out = []
    for r in table:
        acc = f"{100 * r['accuracy']:.1f} [{100 * r['ci_lo']:.1f}, {100 * r['ci_hi']:.1f}]"
        diff = ""
        if "diff" in r:
            diff = (
                f"{100 * r['diff']:+.1f} [{100 * r['diff_lo']:+.1f}, {100 * r['diff_hi']:+.1f}] "
                f"vs {r['vs']} (p={r['p']:.3f}; tokens ×{r['compute_ratio']:.2f})"
            )
        out.append(
            {
                "method": r["method_id"],
                "N": r["n"],
                "errors": r["n_errors"],
                "accuracy % [95% CI]": acc,
                "mean completion tokens": f"{r['mean_completion_tokens']:,.0f}",
                "wall time / problem (s)": f"{r['mean_wall_time_s']:.1f}",
                "Jev USD": f"{r['jev_usd']:.4f}",
                "Δ vs B2 @ matched compute (pp)": diff,
            }
        )
    return md_table(out, list(out[0].keys()) if out else ["method"])
