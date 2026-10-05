"""Evaluation metrics (spec §7.5, §8.3, §9). Pure, deterministic functions.

Conventions:
- Probabilities and labels are array-likes of floats in [0, 1]. Soft labels are
  allowed wherever a "soft_labels" argument appears.
- Every resampling function takes a `seed` and is deterministic given it.
- Degenerate inputs (empty, single class) return `nan` with a warning instead
  of raising, because small depth buckets hit them routinely.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

ArrayLike = Sequence[float] | np.ndarray

DEPTH_BUCKETS: list[tuple[float, float]] = [(1, 2), (3, 5), (6, 9), (10, math.inf)]


# --------------------------------------------------------------------------- accuracy & CIs


def accuracy(correct: Sequence[bool] | np.ndarray) -> float:
    a = np.asarray(correct, dtype=float)
    return float(a.mean()) if a.size else math.nan


def bootstrap_ci(
    values: ArrayLike | Sequence[bool],
    stat: Callable[[np.ndarray], float] = np.mean,
    n: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI of `stat(values)` (spec §8.3: 1,000 resamples, 95%)."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n, v.size))
    stats = np.array([stat(v[row]) for row in idx], dtype=float)
    return _percentile_ci(stats, alpha)


def _percentile_ci(stats: np.ndarray, alpha: float) -> tuple[float, float]:
    stats = stats[~np.isnan(stats)]
    if stats.size == 0:
        return (math.nan, math.nan)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


@dataclass(frozen=True)
class PairedDiff:
    diff: float  # mean(a) - mean(b)
    lo: float
    hi: float
    p_value: float  # two-sided bootstrap p for H0: diff == 0
    n: int


def paired_bootstrap_diff(
    correct_a: Sequence[bool] | np.ndarray,
    correct_b: Sequence[bool] | np.ndarray,
    n: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> PairedDiff:
    """CI on accuracy(a) - accuracy(b) on the SAME problems, resampled jointly.

    Inputs must be aligned (index i = same problem in both). This is the right
    test for "TZ vs self-consistency"; overlapping marginal CIs are not.
    """
    a = np.asarray(correct_a, dtype=float)
    b = np.asarray(correct_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired inputs must align: {a.shape} vs {b.shape}")
    if a.size == 0:
        return PairedDiff(math.nan, math.nan, math.nan, math.nan, 0)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n, d.size))
    boot = d[idx].mean(axis=1)
    lo, hi = _percentile_ci(boot, alpha)
    p = min(1.0, 2 * min(float((boot <= 0).mean()), float((boot >= 0).mean())))
    return PairedDiff(float(d.mean()), lo, hi, p, int(d.size))


def grouped_bootstrap_ci(
    metric: Callable[..., float],
    arrays: Sequence[ArrayLike],
    groups: Sequence[Hashable],
    n: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Cluster bootstrap: resample whole GROUPS (e.g. problems), not rows.

    Pilot prefixes from the same trace are correlated, so CIs must resample by
    problem. `metric(*arrays_subset)` is evaluated on the concatenated rows of
    the resampled groups; resamples where the metric is nan (e.g. AUROC on a
    single-class draw) are dropped.
    """
    arrs = [np.asarray(a, dtype=float) for a in arrays]
    g = np.asarray([str(x) for x in groups])
    if g.size == 0:
        return (math.nan, math.nan)
    uniq, inv = np.unique(g, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(uniq.size)]
    rng = np.random.default_rng(seed)
    stats = np.empty(n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(n):
            pick = rng.integers(0, uniq.size, size=uniq.size)
            rows = np.concatenate([members[k] for k in pick])
            stats[i] = metric(*(a[rows] for a in arrs))
    return _percentile_ci(stats, alpha)


# ------------------------------------------------------------------ discrimination & calibration


def auroc(scores: ArrayLike, labels: Sequence[bool] | ArrayLike) -> float:
    """ROC AUC of `scores` for binary `labels`. nan (+warning) if one class only."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float) > 0.5
    if s.size == 0 or y.all() or not y.any():
        warnings.warn("auroc: only one class present; returning nan", RuntimeWarning, stacklevel=2)
        return math.nan
    return float(roc_auc_score(y, s))


def brier(probs: ArrayLike, soft_labels: ArrayLike) -> float:
    """Mean squared error (p - y)^2; y may be soft."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(soft_labels, dtype=float)
    return float(np.mean((p - y) ** 2)) if p.size else math.nan


def _bin_index(p: np.ndarray, n_bins: int) -> np.ndarray:
    # Equal-width bins [0, 1/B), ..., [(B-1)/B, 1]; p == 1 goes in the last bin.
    return np.clip(np.floor(p * n_bins).astype(int), 0, n_bins - 1)


def ece(probs: ArrayLike, soft_labels: ArrayLike, n_bins: int = 10) -> float:
    """Expected calibration error: sum_b (n_b/N) * |mean(p_b) - mean(y_b)|."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(soft_labels, dtype=float)
    if p.size == 0:
        return math.nan
    b = _bin_index(p, n_bins)
    total = 0.0
    for k in range(n_bins):
        m = b == k
        if m.any():
            total += m.sum() / p.size * abs(p[m].mean() - y[m].mean())
    return float(total)


def reliability_data(
    probs: ArrayLike, labels: ArrayLike, n_bins: int = 10
) -> list[dict[str, float]]:
    """One dict per bin (empty bins included, with nan means and count 0)."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    b = _bin_index(p, n_bins) if p.size else np.array([], dtype=int)
    out = []
    for k in range(n_bins):
        m = b == k
        out.append(
            {
                "bin_lo": k / n_bins,
                "bin_hi": (k + 1) / n_bins,
                "mean_pred": float(p[m].mean()) if m.any() else math.nan,
                "mean_label": float(y[m].mean()) if m.any() else math.nan,
                "count": int(m.sum()),
            }
        )
    return out


# --------------------------------------------------------------------------- stratification


def bucket_label(lo: float, hi: float) -> str:
    return f"{int(lo)}+" if math.isinf(hi) else f"{int(lo)}-{int(hi)}"


def by_depth(
    records: Sequence[Mapping[str, Any]],
    fn: Callable[[list[Mapping[str, Any]]], float],
    buckets: Sequence[tuple[float, float]] = DEPTH_BUCKETS,
    depth_key: str = "depth",
) -> list[dict[str, Any]]:
    """Apply `fn(records_in_bucket)` per inclusive depth bucket (spec §7.5).

    Returns `[{bucket, lo, hi, count, value}]`; value is nan for empty buckets.
    """
    out = []
    for lo, hi in buckets:
        sub = [r for r in records if lo <= r[depth_key] <= hi]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            val = fn(sub) if sub else math.nan
        out.append(
            {"bucket": bucket_label(lo, hi), "lo": lo, "hi": hi, "count": len(sub), "value": val}
        )
    return out


def group_by(
    records: Iterable[Mapping[str, Any]], key: Callable[[Mapping[str, Any]], Hashable]
) -> dict[Hashable, list[Mapping[str, Any]]]:
    out: dict[Hashable, list[Mapping[str, Any]]] = {}
    for r in records:
        out.setdefault(key(r), []).append(r)
    return out


# ------------------------------------------------------------------ search diagnostics (spec §9)
#
# Reads ``raw["stats"]`` as written by ``search.mcts.search`` (via eval.methods.TZMethod):
#   expansions, max_depth, most_visited_reached_terminal, modes_agree.
# Rows without stats (non-TZ methods, error rows) are skipped, never treated as False.


def _mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else math.nan


def search_diagnostics(records: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    stats = [(r.get("raw") or {}).get("stats") for r in records]
    st = [x for x in stats if x]
    exp = [float(x["expansions"]) for x in st if x.get("expansions") is not None]
    dep = [float(x["max_depth"]) for x in st if x.get("max_depth") is not None]
    term = [
        float(bool(x["most_visited_reached_terminal"]))
        for x in st
        if "most_visited_reached_terminal" in x
    ]
    agree = [float(bool(x["modes_agree"])) for x in st if "modes_agree" in x]
    return {
        "n": len(records),
        "n_with_stats": len(st),
        "mean_expansions": _mean(exp),
        "mean_max_depth": _mean(dep),
        "max_max_depth": max(dep) if dep else math.nan,
        "frac_most_visited_terminal": _mean(term),
        "extract_agreement": _mean(agree),
    }
