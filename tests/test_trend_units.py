"""Smoothing helpers and display-unit conversion."""

import numpy as np
import pandas as pd
import pytest

from healthlab.analysis.trend import ewma_alpha, rolling, weight_trend
from healthlab.units import canonical_unit, convert, display


def test_ewma_alpha_matches_halflife():
    a = ewma_alpha(10)
    assert (1 - a) ** 10 == pytest.approx(0.5)


def test_weight_trend_time_aware_and_daily():
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-20"])
    w = pd.Series([80.0, 80.0, 78.0], index=idx)
    tr = weight_trend(w, 10)
    assert len(tr) == 20  # interpolated to every calendar day
    # pandas' time-aware EWMA: weights decay with elapsed time, so after an 18-day gap the
    # new reading gets weight 1 against 2 old readings decayed by 0.5**1.8 and 0.5**1.9.
    w_old = 0.5 ** 1.9 + 0.5 ** 1.8
    assert tr.iloc[-1] == pytest.approx((80 * w_old + 78) / (w_old + 1))


def test_weight_trend_tracks_linear_loss_with_lag():
    idx = pd.date_range("2024-01-01", periods=120)
    w = pd.Series(90 - 0.05 * np.arange(120), index=idx)
    tr = weight_trend(w, 10)
    # Steady state: the trend slope equals the true slope.
    assert (tr.iloc[-1] - tr.iloc[-31]) / 30 == pytest.approx(-0.05, rel=0.02)


def test_rolling_tolerates_gaps():
    s = pd.Series([1.0, np.nan, 3.0, np.nan], index=pd.date_range("2024-01-01", periods=4))
    r = rolling(s, 4)
    assert r.iloc[-1] == pytest.approx(2.0)


def test_display_units():
    assert display("kg", "imperial")[0] == "lb"
    assert 70 * display("kg", "imperial")[1] == pytest.approx(154.3236, rel=1e-5)
    assert display("km", "metric") == ("km", 1.0)
    assert canonical_unit("mi") == "km" and canonical_unit("dBASPL") == "dBASPL"
    with pytest.raises(ValueError):
        convert(1.0, "kg", "km")
