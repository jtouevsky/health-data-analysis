"""Energy-balance estimators on a controlled simulated person."""

import numpy as np
import pandas as pd
import pytest

from healthlab.analysis.energy import (EnergyParams, expected_trajectory, has_energy_data,
                                       implied_daily_discrepancy, summarize, window_tdee)

TRUE_TDEE = 2500.0


def person(days=240, seed=0, log_frac=0.8, under=0.0):
    """True TDEE 2500; eats ~2200 (300 kcal deficit); Apple says 2700."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=days)
    true_intake = rng.normal(2200, 300, days)
    weight_true = 85 + np.r_[0, np.cumsum(true_intake - TRUE_TDEE)[:-1]] / 7700
    weigh = np.where(rng.uniform(size=days) < 5 / 7, weight_true + rng.normal(0, 0.4, days), np.nan)
    logged = np.where(rng.uniform(size=days) < log_frac, true_intake * (1 - under), np.nan)
    return pd.DataFrame({"intake_kcal": logged, "weight_kg": weigh,
                         "active_kcal": rng.normal(900, 80, days), "basal_kcal": np.full(days, 1800.0)},
                        index=idx)


@pytest.mark.parametrize("estimator", ["ols", "ewma"])
def test_tdee_recovered(estimator):
    d = person()
    p = EnergyParams(estimator=estimator)
    win = window_tdee(d, p)
    s = summarize(win, p)
    assert s["tdee_mean"] == pytest.approx(TRUE_TDEE, abs=120)
    assert s["gap_mean"] == pytest.approx(-200, abs=130)          # empirical - Apple(2700)
    # Most window CIs cover the truth.
    cover = ((win["tdee_lo"] <= TRUE_TDEE) & (win["tdee_hi"] >= TRUE_TDEE)).mean()
    assert cover > 0.75


def test_underreporting_biases_low_and_adjustment_fixes_it():
    d = person(under=0.2, seed=1)
    naive = summarize(window_tdee(d, EnergyParams()), EnergyParams())
    p = EnergyParams(underreport=0.2)
    adj = summarize(window_tdee(d, p), p)
    assert naive["tdee_mean"] < TRUE_TDEE - 300                    # ~ 20% of 2200 = 440 low
    assert adj["tdee_mean"] == pytest.approx(TRUE_TDEE, abs=150)


def test_windows_respect_minimum_coverage():
    d = person(log_frac=0.3, seed=2)                               # ~8 logged days per 28
    assert window_tdee(d, EnergyParams(min_logged_days=14)).empty
    assert not window_tdee(d, EnergyParams(min_logged_days=5)).empty


def test_summary_uses_non_overlapping_windows():
    p = EnergyParams()
    s = summarize(window_tdee(person(), p), p)
    gaps = np.diff(s["windows"].index.to_numpy()).astype("timedelta64[D]").astype(int)
    assert (gaps >= 28).all()


def test_trajectory_with_apple_overestimate_predicts_too_much_loss():
    d = person(seed=3)
    p = EnergyParams()
    traj = expected_trajectory(d, p, n_sims=200)
    assert (traj["lo"] <= traj["median"]).all() and (traj["median"] <= traj["hi"]).all()
    # Apple says 2700 vs a true 2500: predicted weight falls ~200 kcal/day faster.
    assert implied_daily_discrepancy(traj, p.kcal_per_kg) == pytest.approx(-200, abs=80)


def test_has_energy_data():
    assert has_energy_data(person())
    assert not has_energy_data(person().drop(columns="weight_kg"))
