"""Smoothing helpers shared by the app pages."""

from __future__ import annotations

import math

import pandas as pd


def rolling(series: pd.Series, window: int, min_frac: float = 0.5) -> pd.Series:
    """Trailing rolling mean that tolerates gaps (needs ``min_frac`` of the window observed)."""
    if window <= 1:
        return series
    return series.rolling(window, min_periods=max(1, math.ceil(window * min_frac))).mean()
