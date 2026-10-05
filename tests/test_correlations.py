"""Correlation scan behaviour on controlled synthetic series."""

import numpy as np
import pandas as pd

from healthlab.analysis.correlations import lagged_pair, scan, transform


def _frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n)
    a = rng.normal(size=n)
    b = np.r_[np.zeros(2), 0.6 * a[:-2]] + rng.normal(size=n)       # b(t) driven by a(t-2)
    c = rng.normal(size=n)                                            # independent
    return pd.DataFrame({"a": a, "b": b, "c": c}, index=idx)


def test_planted_lag_is_found_at_right_lag():
    res = scan(_frame(), ["a", "b", "c"], max_lag=3, min_n=60, method="raw")
    top = res.iloc[0]
    assert (top["x"], top["y"], top["lag"]) == ("a", "b", 2)
    assert top["q"] < 1e-6
    others = res[~((res["x"] == "a") & (res["y"] == "b") & (res["lag"] == 2))]
    assert (others["q"] > 0.05).mean() > 0.9


def test_test_count_and_min_n():
    df = _frame()
    res = scan(df, ["a", "b", "c"], max_lag=3, min_n=60, method="raw")
    assert len(res) == 3 * (1 + 2 * 3)          # 3 pairs x (lag0 + both directions x 3 lags)
    df.loc[df.index[70:], "c"] = np.nan          # c now has only 70 points (69 after diff)
    res = scan(df, ["a", "b", "c"], max_lag=3, min_n=100, method="diff")
    assert not ((res["x"] == "c") | (res["y"] == "c")).any()


def test_differencing_kills_spurious_random_walk_correlation():
    # Independent random walks look strongly correlated in levels.
    hits_raw = hits_diff = 0
    for seed in range(20):
        rng = np.random.default_rng(100 + seed)
        idx = pd.date_range("2024-01-01", periods=300)
        df = pd.DataFrame({"u": rng.normal(size=300).cumsum(), "v": rng.normal(size=300).cumsum()}, index=idx)
        r = scan(df, ["u", "v"], max_lag=0, min_n=30, method="raw", use_effective_n=False)
        d = scan(df, ["u", "v"], max_lag=0, min_n=30, method="diff", use_effective_n=False)
        hits_raw += int(r["p"].iat[0] < 0.05)
        hits_diff += int(d["p"].iat[0] < 0.05)
    assert hits_raw >= 10          # wildly anti-conservative
    assert hits_diff <= 3          # ~nominal 5%


def test_effective_n_deflates_autocorrelated_series():
    rng = np.random.default_rng(5)
    n = 400
    idx = pd.date_range("2024-01-01", periods=n)
    x = np.zeros(n); y = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.8 * x[i - 1] + rng.normal()
        y[i] = 0.8 * y[i - 1] + rng.normal()
    res = scan(pd.DataFrame({"x": x, "y": y}, index=idx), ["x", "y"], max_lag=0, min_n=30, method="raw")
    row = res.iloc[0]
    assert row["n_eff"] < 0.4 * row["n"]
    assert row["p"] > row["p_naive"]


def test_definitional_pairs_rank_last():
    res = scan(_frame(), ["a", "b", "c"], max_lag=3, min_n=60, method="raw", definitional=[["a", "b"]])
    assert not res.iloc[0]["definitional"]
    assert res[res["definitional"]].index.min() > res[~res["definitional"]].index.max()


def test_transforms_and_lagged_pair():
    df = _frame(50)
    assert transform(df, "diff")["a"].iloc[1] == df["a"].iloc[1] - df["a"].iloc[0]
    lp = lagged_pair(df, "a", "b", 2, "raw")
    assert lp["y"].iloc[0] == df["b"].iloc[2]
