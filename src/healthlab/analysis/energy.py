"""Energy balance: empirical TDEE from logged intake and weight change.

Energy balance over a window of W days::

    mean intake − TDEE = (Δweight / W) × energy density

so ``TDEE = mean intake − (Δweight per day) × kcal_per_kg``. Using only
well-logged days for intake (see ``rollup.add_nutrition_coverage``), this gives
an *empirical* TDEE to compare against Apple's active + basal estimate.

Weight change per day within a window can be estimated two ways:

* ``ols`` (default): the slope of an ordinary least-squares line through the
  window's raw weigh-ins. No lag, with an analytic standard error.
* ``ewma``: the change in the EWMA weight trend across the window. This is the
  classic "smoothed trend" approach, but a causal EWMA lags a changing trend by
  ~(1−α)/α days, which biases TDEE right after a change in intake.

Known bias: **intake under-reporting**. Food logs typically miss 10–30 % of intake
(forgotten snacks, oils, portion underestimates). Under-reported intake makes the
empirical TDEE too low by the same amount. The ``underreport`` parameter scales
logged intake up by 1/(1 − u) to show the sensitivity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .trend import weight_trend

ESTIMATORS = {
    "ols": "Linear fit through the window's weigh-ins (no lag)",
    "ewma": "Change in the EWMA weight trend (causal, lags ~14 d)",
}


@dataclass(frozen=True)
class EnergyParams:
    """Parameters for the energy-balance model (defaults mirror the YAML config)."""

    kcal_per_kg: float = 7700.0
    window_days: int = 28
    min_logged_days: int = 14
    min_weigh_ins: int = 4
    halflife_days: float = 10.0
    estimator: str = "ols"
    underreport: float = 0.0
    seed: int = 7

    @classmethod
    def from_config(cls, energy: dict, trends: dict, **overrides) -> "EnergyParams":
        base = cls(
            kcal_per_kg=float(energy.get("kcal_per_kg", 7700)),
            window_days=int(energy.get("window_days", 28)),
            min_logged_days=int(energy.get("min_logged_days", 14)),
            min_weigh_ins=int(energy.get("min_weigh_ins", 4)),
            halflife_days=float(trends.get("weight_ewma_halflife_days", 10)),
            estimator=str(energy.get("estimator", "ols")),
            seed=int(energy.get("seed", 7)),
        )
        return cls(**{**base.__dict__, **overrides})


def has_energy_data(daily: pd.DataFrame, intake: str = "intake_kcal", weight: str = "weight_kg",
                    min_days: int = 28) -> bool:
    """True when there is enough logged intake and weight to fit at least one window."""
    return (intake in daily and weight in daily and daily[intake].notna().sum() >= min_days // 2
            and daily[weight].notna().sum() >= 4)


def window_tdee(daily: pd.DataFrame, p: EnergyParams, intake: str = "intake_kcal",
                weight: str = "weight_kg", step: int = 1) -> pd.DataFrame:
    """Empirical TDEE for every W-day window ending on each day (every ``step`` days).

    ``daily[intake]`` must already be masked to well-logged days.

    Returns columns: ``n_logged, n_weigh, intake_mean, intake_se, slope_kg_day,
    slope_se, tdee, tdee_lo, tdee_hi, apple_tdee, n_apple, gap`` indexed by the
    window end date. CIs are 95 %.
    """
    W = p.window_days
    scale = 1.0 / (1.0 - p.underreport)
    df = daily.asfreq("D")
    kcal = df[intake] * scale
    w = df[weight]
    apple = (df["active_kcal"] + df["basal_kcal"]) if {"active_kcal", "basal_kcal"} <= set(df) else pd.Series(np.nan, index=df.index)
    trend = weight_trend(w, p.halflife_days).reindex(df.index) if p.estimator == "ewma" else None
    resid_sd = _ewma_resid_sd(w, trend) if trend is not None else np.nan

    rows = []
    ends = df.index[W - 1::step]
    for t in ends:
        lo_t = t - pd.Timedelta(days=W - 1)
        k = kcal.loc[lo_t:t].dropna()
        wi = w.loc[lo_t:t].dropna()
        if len(k) < p.min_logged_days or len(wi) < p.min_weigh_ins:
            continue
        im, ise = k.mean(), k.std(ddof=1) / np.sqrt(len(k))
        if p.estimator == "ols":
            x = (wi.index - lo_t).days.to_numpy(dtype=float)
            if np.ptp(x) < W / 3:  # weigh-ins clustered in a few days -> slope unreliable
                continue
            fit = stats.linregress(x, wi.to_numpy())
            slope, sse = fit.slope, fit.stderr
        else:
            t0 = trend.get(lo_t - pd.Timedelta(days=1), np.nan)
            t1 = trend.get(t, np.nan)
            if np.isnan(t0) or np.isnan(t1):
                continue
            slope = (t1 - t0) / W
            # Approximate SE of an EWMA difference: each end has variance ~ s^2 * alpha/(2-alpha).
            a = 1 - 0.5 ** (1 / p.halflife_days)
            sse = resid_sd * np.sqrt(2 * a / (2 - a)) / W
        tdee = im - slope * p.kcal_per_kg
        se = np.sqrt(ise ** 2 + (sse * p.kcal_per_kg) ** 2)
        ap = apple.loc[lo_t:t].dropna()
        rows.append({
            "end": t, "n_logged": len(k), "n_weigh": len(wi), "intake_mean": im, "intake_se": ise,
            "slope_kg_day": slope, "slope_se": sse, "tdee": tdee,
            "tdee_lo": tdee - 1.96 * se, "tdee_hi": tdee + 1.96 * se,
            "apple_tdee": ap.mean() if len(ap) >= W // 2 else np.nan, "n_apple": len(ap),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.set_index("end")
    out["gap"] = out["tdee"] - out["apple_tdee"]
    return out


def _ewma_resid_sd(w: pd.Series, trend: pd.Series) -> float:
    r = (w - trend).dropna()
    return float(r.std(ddof=1)) if len(r) > 2 else np.nan


def summarize(windows: pd.DataFrame, p: EnergyParams) -> dict:
    """Summary over *non-overlapping* windows (every W-th window from the latest).

    Overlapping windows share most of their days, so their spread would overstate
    the evidence. Reports mean empirical TDEE, the Apple estimate, the gap (mean,
    SD, 95 % CI of the mean), the ratio, and a linear drift test of the gap over time.
    """
    if windows.empty:
        return {}
    ends = windows.index[::-1]
    keep, last = [], None
    for e in ends:
        if last is None or (last - e).days >= p.window_days:
            keep.append(e)
            last = e
    nw = windows.loc[sorted(keep)]
    g = nw["gap"].dropna()
    out = {
        "n_windows": len(nw), "tdee_mean": nw["tdee"].mean(), "tdee_sd": nw["tdee"].std(ddof=1),
        "apple_mean": nw["apple_tdee"].mean(), "gap_mean": g.mean() if len(g) else np.nan,
        "gap_sd": g.std(ddof=1) if len(g) > 1 else np.nan, "ratio": (nw["tdee"] / nw["apple_tdee"]).mean(),
        "n_gap": len(g), "windows": nw,
    }
    if len(g) > 1:
        half = stats.t.ppf(0.975, len(g) - 1) * out["gap_sd"] / np.sqrt(len(g))
        out["gap_ci"] = (out["gap_mean"] - half, out["gap_mean"] + half)
    if len(g) >= 4:
        x = (g.index - g.index[0]).days.to_numpy(dtype=float)
        fit = stats.linregress(x, g.to_numpy())
        out["gap_drift_per_90d"] = fit.slope * 90
        out["gap_drift_p"] = fit.pvalue
    return out


def expected_trajectory(daily: pd.DataFrame, p: EnergyParams, *, logging_error_sd: float = 0.10,
                        apple_bias_sd: float = 0.08, n_sims: int = 400, intake: str = "intake_kcal",
                        weight: str = "weight_kg") -> pd.DataFrame:
    """Weight trajectory predicted from logged intake and Apple's expenditure estimate.

    Starting from the mean of the first week's weigh-ins, weight evolves as
    ``w_t = w_0 + Σ (intake − expenditure) / kcal_per_kg``. Monte Carlo bands combine:

    * per-day logging error on logged days (relative SD ``logging_error_sd``);
    * imputation for unlogged days: the local (±14 d) mean of well-logged intake plus
      a resampled residual;
    * a run-level systematic error in Apple's expenditure (relative SD ``apple_bias_sd``).

    Returns daily ``median, lo (5 %), hi (95 %)`` of predicted weight, plus the
    observed ``weight`` and EWMA ``trend`` for comparison.
    """
    df = daily.asfreq("D")
    w = df[weight]
    first, last = w.first_valid_index(), w.last_valid_index()
    if first is None:
        return pd.DataFrame()
    df = df.loc[first:last]
    scale = 1.0 / (1.0 - p.underreport)
    logged = df[intake]
    local = logged.rolling(29, center=True, min_periods=5).mean().fillna(logged.mean())
    resid = (logged - local).dropna().to_numpy()
    if resid.size == 0:
        return pd.DataFrame()
    expend = df["active_kcal"] + df["basal_kcal"]
    expend = expend.fillna(expend.rolling(29, center=True, min_periods=5).mean()).fillna(expend.mean())

    rng = np.random.default_rng(p.seed)
    n = len(df)
    is_logged = logged.notna().to_numpy()
    base = np.where(is_logged, logged.to_numpy(), local.to_numpy())
    noise = np.where(is_logged[None, :],
                     base[None, :] * rng.normal(0, logging_error_sd, (n_sims, n)),
                     rng.choice(resid, (n_sims, n)))
    intake_sim = (base[None, :] + noise) * scale
    expend_sim = expend.to_numpy()[None, :] * (1 + rng.normal(0, apple_bias_sd, (n_sims, 1)))
    w0 = w.loc[first:first + pd.Timedelta(days=6)].mean()
    # Balance on day t changes weight from day t+1 onward.
    delta = np.cumsum(intake_sim - expend_sim, axis=1) / p.kcal_per_kg
    pred = w0 + np.concatenate([np.zeros((n_sims, 1)), delta[:, :-1]], axis=1)
    q = np.quantile(pred, [0.05, 0.5, 0.95], axis=0)
    return pd.DataFrame({"lo": q[0], "median": q[1], "hi": q[2], "weight": w.loc[first:last].to_numpy(),
                         "trend": weight_trend(w, p.halflife_days).reindex(df.index).to_numpy()},
                        index=df.index)


def implied_daily_discrepancy(traj: pd.DataFrame, kcal_per_kg: float) -> float:
    """Average kcal/day by which (logged intake − Apple expenditure) misses the observed trend."""
    t = traj.dropna(subset=["trend"])
    if len(t) < 14:
        return float("nan")
    days = (t.index[-1] - t.index[0]).days
    return float((t["median"].iloc[-1] - t["trend"].iloc[-1]) * kcal_per_kg / days)
