"""Smoothing helpers shared by the Trends, Energy and Forecast pages."""

from __future__ import annotations

import math

import pandas as pd


def ewma_alpha(halflife_days: float) -> float:
    """Per-day smoothing factor equivalent to a half-life: alpha = 1 - 0.5**(1/h)."""
    return 1.0 - 0.5 ** (1.0 / halflife_days)


def weight_trend(weight: pd.Series, halflife_days: float = 10.0) -> pd.Series:
    """Time-aware EWMA of irregular weigh-ins, interpolated to every calendar day.

    Uses ``pandas.ewm(halflife=..., times=...)`` so gaps between weigh-ins decay
    the old estimate by elapsed time rather than by number of readings. The
    result covers first..last weigh-in; days between readings are linearly
    interpolated (no extrapolation).
    """
    w = weight.dropna()
    if w.empty:
        return pd.Series(dtype=float, name="weight_trend")
    sm = w.ewm(halflife=pd.Timedelta(days=halflife_days), times=w.index).mean()
    full = pd.date_range(w.index.min(), w.index.max(), freq="D")
    return sm.reindex(full).interpolate(method="time").rename("weight_trend")


def rolling(series: pd.Series, window: int, min_frac: float = 0.5) -> pd.Series:
    """Trailing rolling mean that tolerates gaps (needs ``min_frac`` of the window observed)."""
    if window <= 1:
        return series
    return series.rolling(window, min_periods=max(1, math.ceil(window * min_frac))).mean()
