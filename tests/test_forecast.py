"""Forecast baselines, leakage safety and rolling-origin evaluation."""

import numpy as np
import pandas as pd

from healthlab.analysis.forecast import (ForecastParams, _features, evaluate, forecast_naive,
                                         forecast_seasonal_naive, future_forecast)


def weekly_series(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n)
    pattern = np.array([0, 0, 0, 0, 2000, 4000, -1000])
    return pd.Series(8000 + pattern[idx.dayofweek] + rng.normal(0, 500, n), index=idx)


def test_baselines():
    s = pd.Series([1.0, 2, 3, 4, 5, 6, 7, 8, np.nan], index=pd.date_range("2024-01-01", periods=9))
    assert (forecast_naive(s, 3) == 8).all()
    # Next day (index 9) -> same weekday one week earlier = index 2.
    assert forecast_seasonal_naive(s, 3).tolist() == [3, 4, 5]


def test_seasonal_naive_falls_back_further_back():
    s = pd.Series(np.arange(21, dtype=float), index=pd.date_range("2024-01-01", periods=21))
    s.iloc[14] = np.nan
    assert forecast_seasonal_naive(s, 1)[0] == 7.0     # 1 week back missing -> 2 weeks back


def test_features_never_look_past_origin():
    y = np.arange(100, dtype=float)
    dates = pd.date_range("2024-01-01", periods=107)
    f1 = _features(y, dates, np.array([60]), np.array([7]), (7, 8, 14))
    y2 = y.copy()
    y2[61:] = -999
    f2 = _features(y2, dates, np.array([60]), np.array([7]), (7, 8, 14))
    np.testing.assert_array_equal(f1, f2)


def test_evaluation_structure_and_seasonal_baseline_wins_on_weekly_data():
    p = ForecastParams(n_splits=4, models=("naive", "seasonal_naive", "gbm"))
    ev = evaluate(weekly_series(), p)
    assert len(ev.folds) == 4 * 3
    s = ev.summary.set_index("model")
    assert s.loc["seasonal_naive", "mae"] < s.loc["naive", "mae"]
    assert s.loc["seasonal_naive", "skill_vs_snaive"] == 0
    # Test windows are the last 4 x 7 days, contiguous, after their training data.
    origins = ev.folds["origin"].drop_duplicates().sort_values()
    assert (origins.diff().dropna() == pd.Timedelta(days=7)).all()
    assert ev.predictions["date"].min() > origins.iloc[0]


def test_sarimax_and_future_forecast():
    p = ForecastParams(n_splits=2)
    ev = evaluate(weekly_series(250), p)
    assert set(ev.summary["model"]) == {"naive", "seasonal_naive", "sarimax", "gbm"}
    s = ev.summary.set_index("model")
    assert s.loc["sarimax", "mae"] < s.loc["naive", "mae"]          # captures the weekly cycle
    fut = future_forecast(weekly_series(250), p)
    assert len(fut) == 7 and (fut["sarimax_lo"] <= fut["sarimax_hi"]).all()
