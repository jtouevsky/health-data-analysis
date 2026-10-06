"""Changepoint primitives on controlled series."""

import numpy as np
import pandas as pd
import pytest

from healthlab.analysis.changepoints import (CPParams, bic_penalty, binned_series, bocpd, detect, match_events,
                                            noise_scale, pelt, segment_effects)


def daily_series(levels, days_each, sd=1.0, seed=0, start="2024-01-01"):
    rng = np.random.default_rng(seed)
    vals = np.concatenate([lvl + rng.normal(0, sd, n) for lvl, n in zip(levels, days_each)])
    return pd.Series(vals, index=pd.date_range(start, periods=len(vals)))


def test_pelt_finds_planted_mean_shifts():
    rng = np.random.default_rng(1)
    z = np.r_[rng.normal(0, 1, 60), rng.normal(3, 1, 40), rng.normal(-1, 1, 50)]
    assert pelt(z, bic_penalty(len(z)), 4) == [60, 100]


def test_bic_penalty_rarely_fires_on_pure_noise():
    fp = sum(len(pelt(np.random.default_rng(s).normal(size=100), bic_penalty(100), 4)) > 0 for s in range(30))
    assert fp <= 3


def test_noise_scale_robust_to_shifts_and_inflates_for_autocorrelation():
    rng = np.random.default_rng(2)
    shifted = np.r_[rng.normal(0, 1, 100), rng.normal(5, 1, 100)]
    sigma, phi, lr = noise_scale(shifted)
    assert sigma == pytest.approx(1.0, rel=0.2)          # the 5-sigma shift doesn't inflate it
    assert phi < 0.15
    x = np.zeros(2000)
    for t in range(1, 2000):
        x[t] = 0.6 * x[t - 1] + rng.normal()
    sigma, phi, lr = noise_scale(x)
    assert 0.45 <= phi <= 0.6               # lower confidence bound of a true 0.6
    assert sigma == pytest.approx(1 / np.sqrt(1 - 0.36), rel=0.15)     # marginal SD 1.25
    assert lr == pytest.approx(1 / (1 - 0.6), rel=0.3)                 # long-run SD 2.5


def test_autocorrelation_adjustment_reduces_false_changepoints():
    p = CPParams(resolution="day", min_segment=10, log_metrics=())
    naive_fp = adj_fp = 0
    for seed in range(10):
        rng = np.random.default_rng(10 + seed)
        x = np.zeros(300)
        for t in range(1, 300):
            x[t] = 0.7 * x[t - 1] + rng.normal()
        s = pd.Series(x, index=pd.date_range("2024-01-01", periods=300), name="m")
        sigma, _, _ = noise_scale(x)
        naive_fp += len(pelt(np.clip(x / sigma, -3, 3), bic_penalty(300), 10))
        adj_fp += len(detect(s, p).changepoints)
    assert adj_fp < naive_fp / 2


def test_weekly_binning_and_detection():
    s = daily_series([8000, 5000, 8000], [140, 112, 140], sd=1500, seed=3).rename("steps")
    b = binned_series(s, CPParams(), log=True)
    assert b.index.dayofweek.unique().tolist() == [0] and len(b) >= 55
    r = detect(s, CPParams())
    starts = pd.to_datetime(["2024-05-20", "2024-09-09"])
    assert len(r.changepoints) == 2
    assert all(abs(cp - st).days <= 7 for cp, st in zip(r.changepoints, starts))
    assert r.segments["mean"].iloc[1] == pytest.approx(5000, rel=0.1)   # back on the original scale
    assert all(v >= 0.5 for v in r.stability.values())


def test_sparse_weeks_are_dropped():
    s = daily_series([1.0], [70])
    s.iloc[7:12] = np.nan     # second week has only 2 observed days
    b = binned_series(s, CPParams(min_obs_per_week=4))
    assert pd.Timestamp("2024-01-08") not in b.index


def test_sensitivity_monotone_in_penalty():
    s = daily_series([0, 1, 0, 2, 0], [100, 60, 80, 60, 100], sd=1.5, seed=4).rename("m")
    r = detect(s, CPParams(log_metrics=()))
    n = r.sensitivity.sort_values("multiplier")["n_changepoints"].to_numpy()
    assert (np.diff(n) <= 0).all()


def test_bocpd_detects_shift():
    rng = np.random.default_rng(5)
    z = pd.Series(np.r_[rng.normal(0, 1, 40), rng.normal(3, 1, 40)], index=pd.date_range("2024-01-01", periods=80, freq="W-MON"))
    trace, cps = bocpd(z, expected_run=26, min_run=4)
    assert len(cps) >= 1 and min(abs(i - z.index[40]).days for i in cps) <= 14
    assert trace["p_new_regime"].iloc[41:46].max() > 0.5


def test_segment_effects_sign_and_ci():
    s = daily_series([8000, 5000], [120, 120], sd=1500, seed=6).rename("steps")
    r = detect(s, CPParams())
    eff = segment_effects(s, r, CPParams(bootstrap_reps=300)).iloc[0]
    assert eff["diff"] < 0 and eff["ci_hi"] < 0 and eff["cliffs_delta"] < -0.5
    assert eff["pct_change"] == pytest.approx(-0.375, abs=0.08)


def test_match_events_within_tolerance():
    events = pd.DataFrame({"date": pd.to_datetime(["2024-03-01", "2024-08-01"]), "label": ["moved", "new phone"],
                           "kind": ["life event", "device"]})
    m = match_events([pd.Timestamp("2024-03-10"), pd.Timestamp("2024-06-01")], events, 21)
    assert m["event"].tolist()[0] == "moved" and m["days_from_event"].iloc[0] == -9
    assert m["event"].isna().tolist() == [False, True]
