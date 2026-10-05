"""Static report figures (PNG via matplotlib, headless).

Colors follow the method (entity), never its rank: a method keeps its color
in every figure. The categorical order is the validated reference palette
(CVD-safe on adjacent pairs). Identity is never color-alone: every figure
has a legend and/or direct labels.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#8a8984"

# Fixed categorical order (reference palette, light mode).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
METHOD_COLORS: dict[str, str] = {
    "tz": PALETTE[0],
    "sc": PALETTE[1],
    "bon": PALETTE[2],
    "cot": PALETTE[3],
    "tz_self": PALETTE[4],
    "tz_prm": PALETTE[5],
    "tz_hybrid": PALETTE[6],
    "tz_prior_only": PALETTE[7],
    # beyond 8 entities: fold to neutral rather than generating hues
}
JUDGE_COLORS: dict[str, str] = {"jev": PALETTE[0], "self": PALETTE[1], "prm": PALETTE[2]}
METHOD_LABELS = {
    "tz": "ThoughtZero (Jev)",
    "sc": "B2 self-consistency",
    "bon": "B3 best-of-N",
    "cot": "B1 CoT",
    "c31b": "C large model",
    "tz_self": "TZ self-judge",
    "tz_prm": "TZ PRM",
    "tz_hybrid": "TZ hybrid",
    "tz_prior_only": "TZ prior-only",
    "tz_value_only": "TZ value-only",
}
REFERENCE_METHODS = {"c31b"}  # drawn as horizontal reference lines
LEGEND_ORDER = [
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


def _legend_key(m: str) -> tuple[int, str]:
    return (LEGEND_ORDER.index(m) if m in LEGEND_ORDER else len(LEGEND_ORDER), m)


def color_for(method: str) -> str:
    return METHOD_COLORS.get(method, NEUTRAL)


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.xaxis.label.set_color(TEXT_2)
    ax.yaxis.label.set_color(TEXT_2)
    ax.title.set_color(TEXT)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fig(w: float = 7.0, h: float = 4.5) -> tuple[Any, Any]:
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    return fig, ax


def _save(fig: Any, out: str | Path) -> Path:
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    return p


def _param_label(method: str, params: Mapping[str, Any]) -> str | None:
    if "n_simulations" in params:
        return f"n={params['n_simulations']}"
    if "n" in params:
        return f"N={params['n']}"
    return None


def plot_accuracy_vs_compute(
    points: Sequence[Mapping[str, Any]], out: str | Path, title: str = "Accuracy vs compute"
) -> Path:
    """points: {method, params, accuracy, ci_lo, ci_hi, mean_completion_tokens}; acc in [0,1]."""
    fig, ax = _fig(8.6, 4.6)
    by: dict[str, list[Mapping[str, Any]]] = {}
    for p in points:
        by.setdefault(p["method"], []).append(p)
    for m in sorted(by, key=_legend_key):
        pts = sorted(by[m], key=lambda p: p["mean_completion_tokens"])
        label = METHOD_LABELS.get(m, m)
        if m in REFERENCE_METHODS:
            p = pts[0]
            ax.axhline(
                100 * p["accuracy"],
                color=TEXT_2,
                linestyle="--",
                linewidth=1.5,
                label=(
                    f"{label}: {100 * p['accuracy']:.1f}% ({p['mean_completion_tokens']:,.0f} tok)"
                ),
            )
            continue
        x = [p["mean_completion_tokens"] for p in pts]
        y = [100 * p["accuracy"] for p in pts]
        yerr = [
            [100 * (p["accuracy"] - p["ci_lo"]) for p in pts],
            [100 * (p["ci_hi"] - p["accuracy"]) for p in pts],
        ]
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            color=color_for(m),
            linewidth=2,
            marker="o",
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            capsize=3,
            elinewidth=1,
            label=label,
        )
        for p, xi, yi in zip(pts, x, y, strict=True):
            pl = _param_label(m, p.get("params") or {})
            if pl:  # TZ labels above-left, others below-right, so paired points don't collide
                above = m.startswith("tz")
                ax.annotate(
                    pl,
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(-6, 7) if above else (6, -12),
                    ha="right" if above else "left",
                    fontsize=8,
                    color=TEXT_2,
                )
    ax.set_xscale("log")
    ax.set_xlabel("Mean Gemma completion tokens per problem (log)")
    ax.set_ylabel("Accuracy (%) with 95% bootstrap CI")
    ax.set_title(title, loc="left", fontsize=11)
    ax.legend(
        frameon=False, fontsize=8, labelcolor=TEXT_2, loc="upper left", bbox_to_anchor=(1.01, 1.0)
    )
    return _save(fig, out)


def plot_reliability(
    bins: Sequence[Mapping[str, float]], out: str | Path, title: str, color: str = PALETTE[0]
) -> Path:
    """Reliability diagram (top) and count per bin (bottom, same x-axis, separate y)."""
    fig, (ax, axc) = plt.subplots(
        2, 1, figsize=(4.8, 5.4), dpi=150, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.patch.set_facecolor(SURFACE)
    for a in (ax, axc):
        _style(a)
    ax.plot([0, 1], [0, 1], color=NEUTRAL, linestyle="--", linewidth=1, label="perfect calibration")
    xs = [b["mean_pred"] for b in bins if b["count"] > 0]
    ys = [b["mean_label"] for b in bins if b["count"] > 0]
    ax.plot(
        xs,
        ys,
        color=color,
        linewidth=2,
        marker="o",
        markersize=6,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        label="judge",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean MC label")
    ax.set_title(title, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_2)
    w = bins[0]["bin_hi"] - bins[0]["bin_lo"] if bins else 0.1
    axc.bar(
        [b["bin_lo"] + w / 2 for b in bins], [b["count"] for b in bins], width=w * 0.9, color=color
    )
    axc.set_ylabel("Count")
    axc.set_xlabel("Judge score (bin)")
    return _save(fig, out)


def plot_histogram(
    values: Sequence[float],
    out: str | Path,
    title: str,
    xlabel: str,
    bins: int | Sequence[float] = 30,
    mark_quantiles: bool = False,
    color: str = PALETTE[0],
) -> Path:
    fig, ax = _fig(6.0, 3.6)
    vals = [v for v in values if v is not None and not math.isnan(v)]
    ax.hist(vals, bins=bins, color=color, edgecolor=SURFACE, linewidth=1)
    if mark_quantiles and vals:
        import numpy as np

        for q, ls in ((50, "-"), (95, "--")):
            v = float(np.percentile(vals, q))
            ax.axvline(v, color=TEXT, linestyle=ls, linewidth=1)
            ax.annotate(
                f"p{q} = {v:.3g}",
                (v, ax.get_ylim()[1] * 0.92),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=8,
                color=TEXT_2,
            )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title, loc="left", fontsize=11)
    return _save(fig, out)


def plot_accuracy_by_level(
    rows: Sequence[Mapping[str, Any]],
    out: str | Path,
    title: str = "Accuracy by MATH difficulty level",
) -> Path:
    """rows: {method, label, level, accuracy, ci_lo, ci_hi}; one line per label.

    Pass ONE compute level per method (colors follow the method).
    """
    fig, ax = _fig(8.6, 4.6)
    by: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        by.setdefault(r["label"], []).append(r)
    for i, lab in enumerate(sorted(by, key=lambda lb: _legend_key(by[lb][0]["method"]))):
        rs = sorted(by[lab], key=lambda r: r["level"])
        m = rs[0]["method"]
        x = [r["level"] + (i - (len(by) - 1) / 2) * 0.06 for r in rs]  # dodge overlapping CIs
        y = [100 * r["accuracy"] for r in rs]
        yerr = [
            [100 * (r["accuracy"] - r["ci_lo"]) for r in rs],
            [100 * (r["ci_hi"] - r["accuracy"]) for r in rs],
        ]
        ls = "--" if m in REFERENCE_METHODS else "-"
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            color=color_for(m) if m not in REFERENCE_METHODS else TEXT_2,
            linestyle=ls,
            linewidth=2,
            marker="o",
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            capsize=3,
            elinewidth=1,
            label=lab,
        )
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("MATH difficulty level")
    ax.set_ylabel("Accuracy (%) with 95% CI")
    ax.set_title(title, loc="left", fontsize=11)
    ax.legend(
        frameon=False, fontsize=8, labelcolor=TEXT_2, loc="upper left", bbox_to_anchor=(1.01, 1.0)
    )
    return _save(fig, out)
