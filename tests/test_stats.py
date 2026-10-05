"""Effect sizes, bootstrap CIs and period comparisons."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from healthlab.analysis.seasonal import compare_periods, label_days, month_periods
from healthlab.analysis.stats import bootstrap_median_diff, cliffs_delta, delta_magnitude


def test_cliffs_delta_extremes_and_brute_force():
    assert cliffs_delta([5, 6, 7], [1, 2, 3]) == 1.0
    assert cliffs_delta([1, 2, 3], [5, 6, 7]) == -1.0
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 10, 40), rng.integers(0, 10, 50)    # with ties
    brute = (np.sign(a[:, None] - b[None, :])).mean()
    assert cliffs_delta(a, b) == pytest.approx(brute)


def test_cliffs_delta_equals_rank_biserial():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0.5, 1, 60), rng.normal(0, 1, 70)
    u = stats.mannwhitneyu(a, b).statistic
    rank_biserial = 2 * u / (60 * 70) - 1
    assert cliffs_delta(a, b) == pytest.approx(rank_biserial)


def test_delta_magnitude_labels():
    assert [delta_magnitude(x) for x in (0.1, -0.2, 0.4, 0.9)] == ["negligible", "small", "medium", "large"]


def test_bootstrap_ci_covers_true_difference():
    rng = np.random.default_rng(2)
    covered = 0
    for i in range(40):
        a, b = rng.normal(10, 2, 150), rng.normal(9, 2, 150)
        lo, hi = bootstrap_median_diff(a, b, reps=400, seed=i)
        covered += lo <= 1.0 <= hi
    assert covered >= 34          # nominal 95% -> ~38/40


def test_block_bootstrap_wider_for_autocorrelated_data():
    rng = np.random.default_rng(3)
    def ar1(n, phi=0.8):
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + rng.normal()
        return x
    a, b = ar1(300), ar1(300)
    lo, hi = bootstrap_median_diff(a, b, reps=800, block=1, seed=0)
    blo, bhi = bootstrap_median_diff(a, b, reps=800, block=14, seed=0)
    assert (bhi - blo) > 1.4 * (hi - lo)


def test_label_days_multiple_ranges_and_overlap():
    idx = pd.date_range("2024-01-01", "2025-12-31")
    lab = label_days(idx, {"summer": [("2024-06-20", "2024-08-25"), ("2025-06-20", "2025-08-25")],
                           "school": [("2024-09-01", "2025-06-10")]})
    assert (lab == "summer").sum() == 67 * 2
    assert lab[pd.Timestamp("2024-07-04")] == "summer" and pd.isna(lab[pd.Timestamp("2024-01-05")])
    with pytest.raises(ValueError):
        label_days(idx, {"a": [("2024-01-01", "2024-02-01")], "b": [("2024-01-15", "2024-03-01")]})


def test_compare_periods_detects_planted_shift():
    rng = np.random.default_rng(4)
    idx = pd.date_range("2024-01-01", periods=730)
    summer = idx.month.isin([6, 7, 8])
    df = pd.DataFrame({"steps": rng.normal(7000, 1500, 730) + np.where(summer, 2500, 0),
                       "noise": rng.normal(0, 1, 730)}, index=idx)
    labels = month_periods(idx, [6, 7, 8], "summer", "rest")
    res = compare_periods(df, ["steps", "noise"], labels, "summer", "rest", reps=300).set_index("metric")
    s = res.loc["steps"]
    assert s["cliffs_delta"] > 0.6 and s["q"] < 1e-10
    assert s["ci_lo"] < 2500 < s["ci_hi"] and s["ci_block_lo"] < 2500 < s["ci_block_hi"]
    assert res.loc["noise", "q"] > 0.05
    wk = compare_periods(df, ["steps"], labels, "summer", "rest", unit="week", reps=300).iloc[0]
    assert wk["n_a"] < 30 and wk["cliffs_delta"] > 0.6
