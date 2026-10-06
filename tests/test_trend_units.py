"""Smoothing helper and display-unit conversion."""

import numpy as np
import pandas as pd
import pytest

from healthlab.analysis.trend import rolling
from healthlab.units import canonical_unit, convert, display


def test_rolling_tolerates_gaps():
    s = pd.Series([1.0, np.nan, 3.0, np.nan], index=pd.date_range("2024-01-01", periods=4))
    r = rolling(s, 4)
    assert r.iloc[-1] == pytest.approx(2.0)


def test_rolling_requires_half_the_window():
    s = pd.Series([1.0, np.nan, np.nan, np.nan], index=pd.date_range("2024-01-01", periods=4))
    assert np.isnan(rolling(s, 4).iloc[-1])


def test_display_units():
    assert display("kg", "imperial")[0] == "lb"
    assert 70 * display("kg", "imperial")[1] == pytest.approx(154.3236, rel=1e-5)
    assert display("km", "metric") == ("km", 1.0)
    assert display("km/hr", "imperial")[0] == "mph"
    assert canonical_unit("mi") == "km" and canonical_unit("dBASPL") == "dBASPL"
    with pytest.raises(ValueError):
        convert(1.0, "kg", "km")
