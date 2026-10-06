"""Data-quality helpers: gap runs and robust outlier flags."""

import numpy as np
import pandas as pd

from healthlab.quality import gap_runs, outlier_flags


def test_gap_runs():
    idx = pd.date_range("2024-01-01", periods=12)
    s = pd.Series([1, np.nan, np.nan, np.nan, 2, np.nan, 3, np.nan, np.nan, np.nan, np.nan, 4], index=idx)
    g = gap_runs(s, min_days=3)
    assert g["days"].tolist() == [3, 4]
    assert g["start"].iloc[0] == pd.Timestamp("2024-01-02")


def test_gaps_ignore_time_before_first_sample():
    idx = pd.date_range("2024-01-01", periods=10)
    s = pd.Series([np.nan] * 5 + [1.0] * 5, index=idx)
    assert gap_runs(s, min_days=3).empty


def test_outlier_flags_robust():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(8000, 1000, 120), index=pd.date_range("2024-01-01", periods=120))
    s.iloc[60] = 60000
    f = outlier_flags(s, window=28, k=5, plausible=(0, 50000))
    assert f["outlier"].sum() == 1 and f.loc[s.index[60], "outlier"]
    assert f.loc[s.index[60], "implausible"]
