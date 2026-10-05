"""Seasonal / period comparisons (e.g. summers in Bulgaria vs the school year).

Periods are labels mapped to one or more inclusive date ranges in config. For
each metric the two labelled groups of days are compared with:

* Mann–Whitney U (two-sided): does one group tend to have larger values?
* Cliff's delta: effect size, P(A > B) − P(A < B);
* bootstrap 95 % CIs for the difference in medians, both iid and moving-block
  (the block version respects day-to-day autocorrelation);
* Benjamini–Hochberg across metrics.

Daily values are autocorrelated, so the MWU p-value on days is optimistic. The
``unit="week"`` option compares weekly means instead, a coarser but more
honest unit of analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .stats import bh_fdr, bootstrap_median_diff, cliffs_delta, delta_magnitude

DateRanges = list[tuple[str, str]]


def label_days(index: pd.DatetimeIndex, periods: dict[str, DateRanges]) -> pd.Series:
    """Label each day with the period it falls in (NaN if none).

    Raises:
        ValueError: if a day falls in two differently labelled periods.
    """
    lab = pd.Series(pd.NA, index=index, dtype="object")
    for name, ranges in periods.items():
        for start, end in ranges:
            m = (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
            clash = m & lab.notna().to_numpy() & (lab != name).to_numpy()
            if clash.any():
                raise ValueError(f"period {name!r} overlaps another period on {index[clash][0].date()}")
            lab[m] = name
    return lab


def month_periods(index: pd.DatetimeIndex, months_a: list[int], name_a: str = "selected months",
                  name_b: str = "other months") -> pd.Series:
    """Label days by calendar month: ``months_a`` vs every other month."""
    return pd.Series(np.where(index.month.isin(months_a), name_a, name_b), index=index, dtype="object")


def _group_values(s: pd.Series, labels: pd.Series, name: str, unit: str, min_days_per_week: int) -> np.ndarray:
    v = s[labels == name].dropna()
    if unit == "week":
        g = v.resample("W-MON", label="left", closed="left")
        wk = g.mean()[g.count() >= min_days_per_week]
        return wk.to_numpy()
    return v.to_numpy()


def compare_periods(daily: pd.DataFrame, metrics: list[str], labels: pd.Series, a: str, b: str, *,
                    unit: str = "day", reps: int = 2000, block: int = 7, seed: int = 42,
                    min_n: int = 10, min_days_per_week: int = 4) -> pd.DataFrame:
    """Compare period ``a`` vs period ``b`` for each metric.

    Returns one row per metric: group sizes and medians, median difference (a − b)
    with iid and block-bootstrap CIs, Mann–Whitney U and p, Cliff's delta and its
    magnitude, and BH q across the metrics compared.
    """
    rows = []
    for m in metrics:
        xa = _group_values(daily[m], labels, a, unit, min_days_per_week)
        xb = _group_values(daily[m], labels, b, unit, min_days_per_week)
        if xa.size < min_n or xb.size < min_n:
            continue
        u, p = stats.mannwhitneyu(xa, xb, alternative="two-sided")
        d = cliffs_delta(xa, xb)
        lo, hi = bootstrap_median_diff(xa, xb, reps=reps, block=1, seed=seed)
        blo, bhi = bootstrap_median_diff(xa, xb, reps=reps, block=block if unit == "day" else 1, seed=seed)
        rows.append({
            "metric": m, "n_a": xa.size, "n_b": xb.size,
            "median_a": float(np.median(xa)), "median_b": float(np.median(xb)),
            "diff_median": float(np.median(xa) - np.median(xb)),
            "ci_lo": lo, "ci_hi": hi, "ci_block_lo": blo, "ci_block_hi": bhi,
            "U": float(u), "p": float(p), "cliffs_delta": d, "magnitude": delta_magnitude(d),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q"] = bh_fdr(out["p"].to_numpy())
    out["abs_d"] = out["cliffs_delta"].abs()
    return out.sort_values("abs_d", ascending=False).drop(columns="abs_d").reset_index(drop=True)


CONFOUNDS = """
**Likely confounds. Read before interpreting any difference:**
- **Place and built environment.** A walkable European city and a car-centric LA neighbourhood change steps,
  distance and gait speed regardless of anything else.
- **Schedule.** Summer means no classes, different sleep and wake times, and holidays. "Season" is mostly "routine".
- **Time-zone artifact.** Local dates are always computed in Los Angeles time (DECISIONS.md T1). While in Bulgaria,
  each "day" runs about 10:00 to 10:00 Sofia time. Totals over a stay are right, but daily values are blurred.
- **Phone carry habits.** Carrying the phone more or less often changes step counts and the iPhone basal estimate,
  with no change in actual activity.
- **Weather and daylight.** Heat, rain and day length affect activity directly.
- **Device and software changes.** A new iPhone or iOS update can shift gait metrics. Check the source breakdown on
  the Data Quality page.
- **Regression to the mean and year effects.** With only a few summers, one unusual year can dominate.

With one person and a handful of periods, these comparisons *describe* differences. They don't attribute them.
"""
