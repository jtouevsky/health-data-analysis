"""Data-quality diagnostics: coverage, gaps, sources, outliers, logging coverage."""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from .config import Config

MAD_SCALE = 1.4826  # makes MAD a consistent estimator of sigma under normality


def coverage_matrix(daily: pd.DataFrame, metrics: list[str], freq: str = "MS") -> pd.DataFrame:
    """Share of days with data, per metric (rows) and period (columns).

    Args:
        freq: pandas period alias; ``"MS"`` = calendar month, ``"W-MON"`` = week.
    """
    present = daily[metrics].notna().astype(float)
    return present.resample(freq).mean().T


def gap_runs(series: pd.Series, min_days: int = 3) -> pd.DataFrame:
    """Runs of consecutive missing days of length >= ``min_days``.

    Only gaps between the metric's first and last observation are reported
    (before the first sample the metric simply did not exist yet).
    """
    s = series.loc[series.first_valid_index(): series.last_valid_index()] if series.notna().any() else series.iloc[:0]
    miss = s.isna().to_numpy()
    if not miss.any():
        return pd.DataFrame(columns=["start", "end", "days"])
    # Label runs by the cumulative count of value changes.
    edges = np.flatnonzero(np.diff(np.r_[0, miss.astype(int), 0]))
    starts, ends = edges[::2], edges[1::2]  # [start, end) index ranges
    rows = [(s.index[a], s.index[b - 1], b - a) for a, b in zip(starts, ends) if b - a >= min_days]
    return pd.DataFrame(rows, columns=["start", "end", "days"])


def gap_summary(daily: pd.DataFrame, metrics: list[str], min_days: int) -> pd.DataFrame:
    """All gap runs across metrics, longest first."""
    parts = [gap_runs(daily[m], min_days).assign(metric=m) for m in metrics]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=["metric", "start", "end", "days"])
    return pd.concat(parts, ignore_index=True)[["metric", "start", "end", "days"]].sort_values("days", ascending=False)


def outlier_flags(series: pd.Series, window: int = 28, k: float = 5.0,
                  plausible: tuple[float, float] | None = None) -> pd.DataFrame:
    """Flag robust outliers and implausible values.

    A value is an outlier when it lies more than ``k`` scaled MADs from the
    centred rolling median. Robust statistics are used so the outliers being
    hunted do not inflate the threshold. Values outside the configured
    plausible range are flagged separately.
    """
    s = series.astype(float)
    med = s.rolling(window, center=True, min_periods=max(7, window // 4)).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=max(7, window // 4)).median() * MAD_SCALE
    # Avoid flagging everything when a metric is locally near-constant: the local
    # scale may not drop below half the series-wide MAD.
    floor = MAD_SCALE * (s - s.median()).abs().median() * 0.5
    mad = mad.clip(lower=floor if np.isfinite(floor) and floor > 0 else 1e-9)
    z = (s - med) / mad
    out = pd.DataFrame({"value": s, "rolling_median": med, "robust_z": z})
    out["outlier"] = z.abs() > k
    if plausible is not None:
        out["implausible"] = s.notna() & ~s.between(*plausible)
    else:
        out["implausible"] = False
    return out[s.notna()]


def outlier_table(daily: pd.DataFrame, catalog: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Flagged days across all metrics."""
    q = cfg.section("quality")
    rows = []
    for _, c in catalog.iterrows():
        m = c["metric"]
        if m not in daily:
            continue
        pl = (c["plausible_lo"], c["plausible_hi"]) if pd.notna(c.get("plausible_lo")) else None
        f = outlier_flags(daily[m], int(q.get("outlier_window_days", 28)), float(q.get("outlier_mad_k", 5)), pl)
        if c["agg"] == "count" or bool(c.get("zero_fill", False)):
            # Zero-inflated counts have a degenerate MAD; only plausibility applies.
            f["outlier"] = False
        f = f[f["outlier"] | f["implausible"]]
        if not f.empty:
            rows.append(f.assign(metric=m).reset_index())
    if not rows:
        return pd.DataFrame(columns=["date", "metric", "value", "rolling_median", "robust_z", "outlier", "implausible"])
    return pd.concat(rows, ignore_index=True)[["date", "metric", "value", "rolling_median", "robust_z", "outlier", "implausible"]]


def source_breakdown(con: sqlite3.Connection) -> pd.DataFrame:
    """Per type and source: record counts, date span, and days on which the source was used."""
    raw = pd.read_sql_query(
        """SELECT type, source_name, COUNT(*) AS n_records, MIN(local_date) AS first_date,
                  MAX(local_date) AS last_date, COUNT(DISTINCT local_date) AS days_with_data
           FROM records GROUP BY type, source_name""", con)
    try:
        used = pd.read_sql_query(
            "SELECT hk_type AS type, source_used AS source_name, COUNT(*) AS days_selected "
            "FROM daily_sources GROUP BY 1, 2", con)
        raw = raw.merge(used, on=["type", "source_name"], how="left")
        raw["days_selected"] = raw["days_selected"].fillna(0).astype(int)
    except Exception:  # rollup not run yet
        raw["days_selected"] = 0
    raw["days_overridden"] = raw["days_with_data"] - raw["days_selected"]
    return raw.sort_values(["type", "n_records"], ascending=[True, False])


def logging_summary(daily: pd.DataFrame, cfg: Config) -> pd.DataFrame | None:
    """Monthly counts of well-logged, partially logged and unlogged days (None if no nutrition)."""
    energy = cfg.section("nutrition").get("energy_metric", "intake_kcal")
    if "well_logged" not in daily or energy not in daily:
        return None
    status = np.where(daily["well_logged"], "well logged",
                      np.where(daily["n_food_entries"] > 0, "partial / implausible", "not logged"))
    return (pd.Series(status, index=daily.index).groupby(pd.Grouper(freq="MS"))
            .value_counts().unstack(fill_value=0))


def activity_timing(con: sqlite3.Connection, cfg: Config, hk_type: str = "HKQuantityTypeIdentifierStepCount",
                    min_total: float = 30000.0) -> pd.DataFrame:
    """Share of each month's steps that fall in each local hour (month x hour).

    Hours are local per the zone resolver (default zone + override periods). A
    sudden shift of the daily rhythm is the signature of travel that is *not*
    covered by an override period (it shows up ~10 h earlier for Bulgaria). Months
    with fewer than ``min_total`` steps are dropped as too sparse to profile.
    """
    df = pd.read_sql_query("SELECT start_ts, value FROM records WHERE type = ?", con, params=(hk_type,))
    if df.empty:
        return pd.DataFrame()
    t = cfg.zone_resolver().localize(pd.to_datetime(df["start_ts"], utc=True))
    g = df.assign(month=t.dt.to_period("M").dt.to_timestamp(), hour=t.dt.hour)
    m = g.pivot_table(index="month", columns="hour", values="value", aggfunc="sum", fill_value=0.0)
    m = m[m.sum(axis=1) >= min_total]
    return m.div(m.sum(axis=1), axis=0)
