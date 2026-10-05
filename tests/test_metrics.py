import math

import numpy as np
import pytest

from thoughtzero.eval import metrics as M


def test_accuracy():
    assert M.accuracy([True, False, True, True]) == 0.75
    assert math.isnan(M.accuracy([]))


def test_auroc_perfect_inverted_random():
    y = [0, 0, 0, 1, 1, 1]
    assert M.auroc([0.1, 0.2, 0.3, 0.7, 0.8, 0.9], y) == 1.0
    assert M.auroc([0.9, 0.8, 0.7, 0.3, 0.2, 0.1], y) == 0.0
    rng = np.random.default_rng(0)
    yr = rng.integers(0, 2, 20000)
    assert abs(M.auroc(rng.random(20000), yr) - 0.5) < 0.02


def test_auroc_single_class_is_nan_with_warning():
    with pytest.warns(RuntimeWarning):
        assert math.isnan(M.auroc([0.1, 0.9], [1, 1]))
    with pytest.warns(RuntimeWarning):
        assert math.isnan(M.auroc([], []))


def test_brier_hand_computed():
    # (0.9-1)^2 + (0.2-0)^2 + (0.5-0.25)^2 = 0.01 + 0.04 + 0.0625
    assert M.brier([0.9, 0.2, 0.5], [1, 0, 0.25]) == pytest.approx(0.1125 / 3)


def test_ece_calibrated_is_near_zero():
    rng = np.random.default_rng(1)
    p = rng.random(200000)
    y = (rng.random(200000) < p).astype(float)
    assert M.ece(p, y) < 0.01
    # soft labels equal to p: exactly calibrated
    assert M.ece(p, p) < 0.01


def test_ece_hand_computed():
    # bin [0.9,1]: p=0.95, y=0 -> |0.95| * 1/2 ; bin [0.1,0.2): p=0.15,y=0.15 -> 0
    assert M.ece([0.95, 0.15], [0.0, 0.15]) == pytest.approx(0.475)
    # p == 1.0 falls in the last bin, not out of range
    assert M.ece([1.0], [1.0]) == 0.0


def test_reliability_data():
    bins = M.reliability_data([0.05, 0.05, 0.95], [0, 1, 1], n_bins=10)
    assert len(bins) == 10
    assert bins[0]["count"] == 2 and bins[0]["mean_label"] == 0.5
    assert bins[9]["count"] == 1 and bins[9]["mean_pred"] == 0.95
    assert bins[5]["count"] == 0 and math.isnan(bins[5]["mean_pred"])
    assert sum(b["count"] for b in bins) == 3


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    rng = np.random.default_rng(2)
    v = rng.random(300) < 0.6
    lo, hi = M.bootstrap_ci(v, seed=7)
    assert lo <= v.mean() <= hi
    assert (lo, hi) == M.bootstrap_ci(v, seed=7)
    assert (lo, hi) != M.bootstrap_ci(v, seed=8)
    assert hi - lo < 0.15


def test_paired_bootstrap_diff():
    a = [True] * 60 + [False] * 40
    b = [True] * 40 + [False] * 60
    d = M.paired_bootstrap_diff(a, b, seed=0)
    assert d.diff == pytest.approx(0.2)
    assert d.lo <= 0.2 <= d.hi and d.lo > 0
    assert d.p_value < 0.05
    same = M.paired_bootstrap_diff(a, a, seed=0)
    assert same.diff == 0 and same.lo == 0 and same.hi == 0 and same.p_value == 1.0
    with pytest.raises(ValueError):
        M.paired_bootstrap_diff([True], [True, False])


def test_paired_is_tighter_than_marginal_for_correlated_methods():
    rng = np.random.default_rng(3)
    base = rng.random(200) < 0.5
    a = base.copy()
    b = base.copy()
    flip = rng.choice(200, 10, replace=False)
    a[flip[:8]] = True  # a fixes 8 problems
    d = M.paired_bootstrap_diff(a, b, seed=0)
    la, ha = M.bootstrap_ci(a, seed=0)
    _lb, hb = M.bootstrap_ci(b, seed=0)
    assert la < hb  # marginal CIs overlap...
    assert d.hi - d.lo < (ha - la)  # ...but the paired CI on the difference is much tighter


def test_grouped_bootstrap_resamples_groups():
    # 2 groups, perfectly separated within each: AUROC is 1 for any resample with both classes
    s = [0.1, 0.9, 0.2, 0.8]
    y = [0, 1, 0, 1]
    g = ["a", "a", "b", "b"]
    lo, hi = M.grouped_bootstrap_ci(lambda s_, y_: M.auroc(s_, y_), [s, y], g, n=200, seed=0)
    assert lo == hi == 1.0
    assert M.grouped_bootstrap_ci(
        np.mean, [[1, 2]], ["a", "b"], n=50, seed=1
    ) == M.grouped_bootstrap_ci(np.mean, [[1, 2]], ["a", "b"], n=50, seed=1)


def test_grouped_ci_wider_than_row_ci_for_clustered_data():
    rng = np.random.default_rng(4)
    groups, vals = [], []
    for k in range(30):  # 30 problems x 8 identical-ish prefixes
        v = float(rng.random() < 0.5)
        groups += [k] * 8
        vals += [v] * 8
    w_row = np.subtract(*M.bootstrap_ci(vals, seed=0)[::-1])
    w_grp = np.subtract(*M.grouped_bootstrap_ci(np.mean, [vals], groups, seed=0)[::-1])
    assert w_grp > 2 * w_row


def test_by_depth():
    recs = [{"depth": d, "v": d} for d in [1, 2, 3, 7, 12, 15]]
    out = M.by_depth(recs, lambda rs: float(np.mean([r["v"] for r in rs])))
    assert [o["bucket"] for o in out] == ["1-2", "3-5", "6-9", "10+"]
    assert [o["count"] for o in out] == [2, 1, 1, 2]
    assert out[0]["value"] == 1.5 and out[3]["value"] == 13.5
    empty = M.by_depth([{"depth": 1}], lambda rs: 1.0)
    assert math.isnan(empty[1]["value"]) and empty[1]["count"] == 0


def test_search_diagnostics():
    def row(exp, depth, terminal, agree):
        stats = {
            "expansions": exp,
            "max_depth": depth,
            "most_visited_reached_terminal": terminal,
            "modes_agree": agree,
        }
        return {"raw": {"stats": stats}}

    recs = [row(10, 4, True, True), row(20, 6, False, False), {"raw": {}}, {"error": "x"}]
    d = M.search_diagnostics(recs)
    assert d["n"] == 4 and d["n_with_stats"] == 2  # rows without stats are skipped, not False
    assert d["mean_expansions"] == 15 and d["mean_max_depth"] == 5 and d["max_max_depth"] == 6
    assert d["frac_most_visited_terminal"] == 0.5 and d["extract_agreement"] == 0.5
