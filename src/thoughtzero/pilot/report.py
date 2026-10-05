"""Pilot step 6 (spec §7.5-7.6): metrics, figures and the GO / PARTIAL / NO-GO decision.

The thresholds below are fixed in code BEFORE any numbers are seen and are
printed at the top of the report. Do not change them after looking at results
(team pitfall A9). If the CI straddles a threshold the report says so and the
team decides at CP2.

All CIs resample by PROBLEM (cluster bootstrap): prefixes of one trace are
correlated, so resampling prefixes would make the CIs too narrow.
"""

from __future__ import annotations

import datetime as _dt
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from thoughtzero.eval import plots
from thoughtzero.eval.metrics import (
    DEPTH_BUCKETS,
    auroc,
    brier,
    by_depth,
    ece,
    grouped_bootstrap_ci,
    reliability_data,
)
from thoughtzero.eval.runner import git_commit
from thoughtzero.pilot.mc_label import load_labels
from thoughtzero.pilot.score import load_scores, scores_path
from thoughtzero.pilot.traces import load_traces, make_prefixes, trace_stats

GO_THRESHOLD = 0.75
PARTIAL_THRESHOLD = 0.65
DECISION_JUDGE = "jev"
REPORT_FILE = "report.md"


# --------------------------------------------------------------------------- decision


@dataclass(frozen=True)
class Decision:
    verdict: str  # "GO" | "PARTIAL" | "NO-GO" | "UNAVAILABLE"
    auroc: float
    ci: tuple[float, float]
    straddles: tuple[float, ...]  # thresholds that fall inside the CI
    reason: str = "no Jev scores joined with labels yet."  # why UNAVAILABLE

    def markdown(self) -> str:
        if self.verdict == "UNAVAILABLE":
            return f"**DECISION: UNAVAILABLE**: {self.reason}"
        lo, hi = self.ci
        s = (
            f"**DECISION: {self.verdict}**: Jev AUROC = {self.auroc:.3f} "
            f"(95% CI {lo:.3f}–{hi:.3f}, resampled by problem)"
        )
        if self.straddles:
            th = " and ".join(f"{t:.2f}" for t in self.straddles)
            s += (
                f"\n\n**The CI straddles the {th} threshold.** The point estimate gives "
                f"{self.verdict}, but the data cannot rule out the neighbouring outcome; "
                "the team decides at CP2."
            )
        return s


def decide(
    auroc_value: float, ci: tuple[float, float] = (math.nan, math.nan), reason: str | None = None
) -> Decision:
    if auroc_value is None or math.isnan(auroc_value):
        if reason:
            return Decision("UNAVAILABLE", math.nan, ci, (), reason)
        return Decision("UNAVAILABLE", math.nan, ci, ())
    if auroc_value >= GO_THRESHOLD:
        v = "GO"
    elif auroc_value >= PARTIAL_THRESHOLD:
        v = "PARTIAL"
    else:
        v = "NO-GO"
    lo, hi = ci
    strad = tuple(
        t for t in (PARTIAL_THRESHOLD, GO_THRESHOLD) if not math.isnan(lo) and lo < t <= hi
    )
    return Decision(v, float(auroc_value), ci, strad)


# --------------------------------------------------------------------------- metrics


def join_scores(
    labels: Sequence[dict[str, Any]], scores: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    by = {s["prefix_id"]: s for s in scores}
    out = []
    for lab in labels:
        s = by.get(lab["prefix_id"])
        if s is not None:
            out.append(
                {
                    **lab,
                    "score": s["score"],
                    "latency_s": s.get("latency_s"),
                    "jev_usd": s.get("jev_usd", 0.0),
                    "jev_calls": s.get("jev_calls", 0),
                    "jev_cache_hits": s.get("jev_cache_hits", 0),
                }
            )
    return out


def _arr(rows: Sequence[Mapping[str, Any]], k: str) -> np.ndarray:
    return np.array([float(r[k]) for r in rows], dtype=float)


def _ci(
    fn: Callable[..., float],
    rows: Sequence[dict[str, Any]],
    keys: Sequence[str],
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    return grouped_bootstrap_ci(
        fn, [_arr(rows, k) for k in keys], [r["problem_id"] for r in rows], n=n_boot, seed=seed
    )


def _auroc_q(s: np.ndarray, y: np.ndarray) -> float:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return auroc(s, y)


def judge_metrics(
    rows: Sequence[dict[str, Any]], n_boot: int = 1000, seed: int = 0
) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    s, hard, soft = _arr(rows, "score"), _arr(rows, "hard_label"), _arr(rows, "soft_label")
    m: dict[str, Any] = {
        "n": len(rows),
        "n_problems": len({r["problem_id"] for r in rows}),
        "auroc": _auroc_q(s, hard),
        "auroc_ci": _ci(_auroc_q, rows, ["score", "hard_label"], n_boot, seed),
        "brier": brier(s, soft),
        "brier_ci": _ci(brier, rows, ["score", "soft_label"], n_boot, seed),
        "ece": ece(s, soft),
        "ece_ci": _ci(ece, rows, ["score", "soft_label"], n_boot, seed),
        "reliability": reliability_data(s, soft),
        "depth": [],
    }
    auc_d = by_depth(
        rows, lambda rs: _auroc_q(_arr(rs, "score"), _arr(rs, "hard_label")), DEPTH_BUCKETS
    )
    ece_d = by_depth(rows, lambda rs: ece(_arr(rs, "score"), _arr(rs, "soft_label")), DEPTH_BUCKETS)
    pos_d = by_depth(rows, lambda rs: float(_arr(rs, "hard_label").sum()), DEPTH_BUCKETS)
    for a, e, p in zip(auc_d, ece_d, pos_d, strict=True):
        m["depth"].append(
            {
                "bucket": a["bucket"],
                "count": a["count"],
                "positives": 0 if math.isnan(p["value"]) else int(p["value"]),
                "auroc": a["value"],
                "ece": e["value"],
            }
        )
    return m


def label_stats(labels: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not labels:
        return {"n": 0}
    soft = _arr(labels, "soft_label")
    hist, _edges = np.histogram(soft, bins=[0, 1e-9, 0.25, 0.5, 0.75, 1 - 1e-9, 1.0 + 1e-9])
    names = ["0", "(0, .25)", "[.25, .5)", "[.5, .75)", "[.75, 1)", "1"]
    return {
        "n": len(labels),
        "frac_hard_pos": float(_arr(labels, "hard_label").mean()),
        "mean_soft": float(soft.mean()),
        "frac_terminal": float(np.mean([r.get("label_source") == "terminal" for r in labels])),
        "soft_hist": dict(zip(names, (int(h) for h in hist), strict=True)),
    }


# --------------------------------------------------------------------------- rendering


def _f(x: float, nd: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _fci(x: float, ci: tuple[float, float]) -> str:
    return f"{_f(x)} [{_f(ci[0])}, {_f(ci[1])}]"


def render_report(ctx: dict[str, Any]) -> str:
    """Pure: dict of computed results -> markdown. Thresholds come first, numbers after."""
    L: list[str] = []
    L += [
        "# ThoughtZero pilot report (spec §7)",
        "",
        f"Generated {ctx.get('generated', '')} · git `{ctx.get('git_commit', 'unknown')}`",
        "",
        "## Decision rule (fixed in code before the numbers were computed)",
        "",
        f"- **GO** if Jev AUROC ≥ {GO_THRESHOLD:.2f}",
        f"- **PARTIAL** if {PARTIAL_THRESHOLD:.2f} ≤ AUROC < {GO_THRESHOLD:.2f}",
        f"- **NO-GO** if AUROC < {PARTIAL_THRESHOLD:.2f}",
        (
            "- AUROC is computed against the hard MC label (any of the samples correct). "
            "CIs come from a 1,000-resample bootstrap **by problem**."
        ),
        "",
    ]
    L += ["## Decision", "", ctx["decision"].markdown(), ""]

    cov = ctx.get("coverage", {})
    if cov.get("incomplete"):
        L += [
            (
                f"> **INCOMPLETE DATA:** {cov['n_labelled']} of {cov['n_prefixes']} prefixes "
                f"labelled ({cov['frac']:.0%}); Jev scored {cov.get('n_jev', 0)}. "
                "This report covers "
                "only what is finished. Thresholds are unchanged."
            ),
            "",
        ]

    ts = ctx.get("trace_stats", {})
    L += [
        "## Data",
        "",
        (
            f"- Traces: {ts.get('n', 0)} · trace accuracy {_f(ts.get('accuracy', math.nan), 2)} · "
            f"no parsed answer {_f(ts.get('frac_no_answer', math.nan), 2)} · "
            f"mean steps {_f(ts.get('mean_steps', math.nan), 1)}"
        ),
        (
            f"- Prefixes: {cov.get('n_prefixes', 0)} (≤ 8 per trace, spread evenly over depth) · "
            f"labelled {cov.get('n_labelled', 0)}"
        ),
        "",
    ]

    ls = ctx.get("label_stats", {})
    if ls.get("n"):
        L += [
            "## Label statistics",
            "",
            f"- Fraction of prefixes with `hard_label = 1`: **{ls['frac_hard_pos']:.1%}**",
            (
                f"- Mean soft label: {ls['mean_soft']:.3f} · labelled without sampling (terminal): "
                f"{ls['frac_terminal']:.1%}"
            ),
            "",
            "| soft label | " + " | ".join(ls["soft_hist"]) + " |",
            "|---|" + "---|" * len(ls["soft_hist"]),
            "| prefixes | " + " | ".join(str(v) for v in ls["soft_hist"].values()) + " |",
            "",
        ]
        if not 0.1 <= ls["frac_hard_pos"] <= 0.9:
            L += [
                (
                    f"> **Warning:** {ls['frac_hard_pos']:.0%} of prefixes are positive. "
                    "With so few "
                    "examples of one class, AUROC is noisy; read its CI, not the point estimate."
                ),
                "",
            ]
        if "soft_hist_png" in ctx:
            L += [f"![soft labels]({ctx['soft_hist_png']})", ""]

    L += [
        "## Judges",
        "",
        "| judge | prefixes | problems | AUROC vs hard [95% CI] | Brier vs soft [CI] "
        "| ECE vs soft [CI] |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in ctx["judges"].items():
        if not m.get("n"):
            L.append(f"| {name} | 0 | 0 | not scored | | |")
            continue
        L.append(
            f"| {name} | {m['n']} | {m['n_problems']} | {_fci(m['auroc'], m['auroc_ci'])} | "
            f"{_fci(m['brier'], m['brier_ci'])} | {_fci(m['ece'], m['ece_ci'])} |"
        )
    L.append("")
    for name, m in ctx["judges"].items():
        if m.get("reliability_png"):
            L += [f"![reliability {name}]({m['reliability_png']})"]
    L.append("")

    L += [
        "## By depth",
        "",
        "AUROC and ECE per prefix-depth bucket. `n/a` = empty or single-class bucket.",
        "",
    ]
    for name, m in ctx["judges"].items():
        if not m.get("n"):
            continue
        L += [
            f"**{name}**",
            "",
            "| depth | prefixes | positives | AUROC | ECE |",
            "|---|---|---|---|---|",
        ]
        L += [
            f"| {d['bucket']} | {d['count']} | {d['positives']} "
            f"| {_f(d['auroc'])} | {_f(d['ece'])} |"
            for d in m["depth"]
        ]
        L.append("")

    sens = ctx.get("sensitivity", {})
    L += ["## Prompt sensitivity (Jev, `SOUND_VARIANTS`)", ""]
    if sens.get("variants"):
        L += [
            f"{sens['n_problems']}-problem subset.",
            "",
            "| variant | prefixes | AUROC [95% CI] | ECE |",
            "|---|---|---|---|",
        ]
        L += [
            f"| {v} | {r['n']} | {_fci(r['auroc'], r['auroc_ci'])} | {_f(r['ece'])} |"
            for v, r in sens["variants"].items()
        ]
        aus = [r["auroc"] for r in sens["variants"].values() if not math.isnan(r["auroc"])]
        if aus:
            L += ["", f"AUROC spread across phrasings: {max(aus) - min(aus):.3f}"]
    else:
        L.append("Not run.")
    L.append("")

    cost = ctx.get("cost", {})
    L += ["## Jev cost and latency", ""]
    if cost:
        L += [
            (
                f"- Total: **${cost['usd']:.4f}** over {cost['calls']} paid calls "
                f"({cost['cache_hits']} cache hits); ${cost['usd_per_call'] * 1e3:.4f} per 1k calls"
            ),
            f"- Latency (paid calls): p50 {_f(cost['p50'], 2)} s · p95 {_f(cost['p95'], 2)} s",
            "",
        ]
        if cost.get("latency_png"):
            L += [f"![latency]({cost['latency_png']})", ""]
    else:
        L += ["No Jev scores yet.", ""]
    return "\n".join(L)


# --------------------------------------------------------------------------- assembly


def _quantile(xs: Sequence[float], q: float) -> float:
    return float(np.percentile(xs, q)) if xs else math.nan


def build_report(
    out_dir: str | Path,
    *,
    judges: Sequence[str] = ("jev", "self", "prm"),
    variants: Sequence[str] = (),
    sensitivity_ids: Sequence[str] = (),
    max_prefixes: int = 8,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[str, Decision]:
    """Load every stage's output from `out_dir`; write report.md + PNGs; return (md, decision)."""
    d = Path(out_dir)
    fig_dir = d / "figures"
    traces = load_traces(d)
    prefixes = make_prefixes(traces, max_prefixes)
    labels = load_labels(d)
    ctx: dict[str, Any] = {
        "generated": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "git_commit": git_commit(),
        "trace_stats": trace_stats(traces),
        "label_stats": label_stats(labels),
        "judges": {},
    }
    if labels:
        plots.plot_histogram(
            [r["soft_label"] for r in labels],
            fig_dir / "soft_labels.png",
            "Soft-label distribution",
            "soft label (fraction of MC completions correct)",
            bins=[i / 8 - 1 / 16 for i in range(10)],
        )
        ctx["soft_hist_png"] = "figures/soft_labels.png"

    jev_rows: list[dict[str, Any]] = []
    for name in judges:
        rows = join_scores(labels, load_scores(d, name)) if scores_path(d, name).exists() else []
        m = judge_metrics(rows, n_boot, seed)
        if rows:
            png = plots.plot_reliability(
                m["reliability"],
                fig_dir / f"reliability_{name}.png",
                f"Reliability: {name}",
                plots.JUDGE_COLORS.get(name, plots.PALETTE[0]),
            )
            m["reliability_png"] = f"figures/{png.name}"
        ctx["judges"][name] = m
        if name == DECISION_JUDGE:
            jev_rows = rows

    jm = ctx["judges"].get(DECISION_JUDGE, {})
    reason = None
    classes = {bool(r["hard_label"]) for r in jev_rows}
    if len(classes) == 1:  # AUROC needs both correct and incorrect prefixes
        which = "correct" if classes == {True} else "incorrect"
        reason = (
            f"all {len(jev_rows)} scored prefixes are labelled {which}, so AUROC is undefined. "
            "The pilot needs prefixes of both kinds (more or harder problems)."
        )
    ctx["decision"] = decide(
        jm.get("auroc", math.nan), jm.get("auroc_ci", (math.nan, math.nan)), reason
    )
    ctx["coverage"] = {
        "n_prefixes": len(prefixes),
        "n_labelled": len(labels),
        "n_jev": len(jev_rows),
        "frac": len(labels) / len(prefixes) if prefixes else 0.0,
        "incomplete": bool(prefixes)
        and (len(labels) < len(prefixes) or len(jev_rows) < len(labels)),
    }

    if variants:
        ids = set(sensitivity_ids)
        sub = [r for r in labels if not ids or r["problem_id"] in ids]
        res = {}
        for v in variants:
            rows = (
                join_scores(sub, load_scores(d, DECISION_JUDGE, v))
                if scores_path(d, DECISION_JUDGE, v).exists()
                else []
            )
            mm = judge_metrics(rows, n_boot, seed)
            res[v] = {
                "n": mm.get("n", 0),
                "auroc": mm.get("auroc", math.nan),
                "auroc_ci": mm.get("auroc_ci", (math.nan, math.nan)),
                "ece": mm.get("ece", math.nan),
            }
        ctx["sensitivity"] = {
            "n_problems": len(ids) or len({r["problem_id"] for r in sub}),
            "variants": res,
        }

    if jev_rows:
        paid = [r for r in jev_rows if r.get("jev_calls", 0) > 0]
        lat = [float(r["latency_s"]) for r in (paid or jev_rows) if r.get("latency_s") is not None]
        calls = sum(int(r.get("jev_calls", 0)) for r in jev_rows)
        usd = sum(float(r.get("jev_usd", 0.0)) for r in jev_rows)
        ctx["cost"] = {
            "usd": usd,
            "calls": calls,
            "cache_hits": sum(int(r.get("jev_cache_hits", 0)) for r in jev_rows),
            "usd_per_call": usd / calls if calls else 0.0,
            "p50": _quantile(lat, 50),
            "p95": _quantile(lat, 95),
        }
        if lat:
            plots.plot_histogram(
                lat,
                fig_dir / "jev_latency.png",
                "Jev latency per call",
                "seconds",
                mark_quantiles=True,
            )
            ctx["cost"]["latency_png"] = "figures/jev_latency.png"

    md = render_report(ctx)
    (d / REPORT_FILE).write_text(md, encoding="utf-8")
    return md, ctx["decision"]
