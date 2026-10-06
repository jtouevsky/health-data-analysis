"""Lagged Spearman correlation scan across all daily metrics, with FDR control.

Raw daily health series are autocorrelated and trend together (seasons, travel,
life phases), which makes naive correlation p-values far too small. The scan
therefore:

1. transforms each series (first difference by default, or residual from a
   rolling median, or raw for comparison);
2. computes Spearman's rho for every metric pair at lags 0..max_lag
   (``x`` on day t vs ``y`` on day t+lag);
3. optionally deflates n to an *effective* n using the lag-1 autocorrelation that
   remains after transformation;
4. applies Benjamini–Hochberg across *all* tests in the scan.

The output is a ranked list of hypotheses, not causal claims.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from .stats import bh_fdr, corr_pvalue, effective_n, lag1_autocorr

TRANSFORMS = {
    "diff": "First difference (today − yesterday)",
    "detrend": "Deviation from a centred rolling median",
    "raw": "Raw daily values (inflated significance, for comparison)",
}


def transform(df: pd.DataFrame, method: str, window: int = 28) -> pd.DataFrame:
    """Apply a stationarising transform column-wise on a daily-indexed frame.

    ``diff`` requires both days to be observed (gaps propagate as NaN). ``detrend``
    subtracts a centred rolling median, which is robust to the outliers it would
    otherwise chase.
    """
    df = df.asfreq("D") if df.index.freq is None else df
    if method == "diff":
        return df.diff()
    if method == "detrend":
        med = df.rolling(window, center=True, min_periods=max(5, window // 3)).median()
        return df - med
    if method == "raw":
        return df
    raise ValueError(f"unknown transform {method!r}")


def _pair_key(a: str, b: str) -> frozenset:
    return frozenset((a, b))


def scan(daily: pd.DataFrame, metrics: list[str], *, max_lag: int = 3, min_n: int = 60,
         method: str = "diff", detrend_window: int = 28, use_effective_n: bool = True,
         definitional: list[list[str]] | None = None) -> pd.DataFrame:
    """Run the lagged correlation scan.

    Args:
        daily: daily table (DatetimeIndex), as returned by ``load_daily``.
        metrics: columns to include.
        max_lag: maximum lag in days; lag k pairs x(t) with y(t+k).
        min_n: pairs with fewer overlapping observations are not tested.
        method: ``diff`` | ``detrend`` | ``raw``.
        use_effective_n: compute the p used for FDR from the effective n.
        definitional: metric pairs that are mechanically linked; flagged.

    Returns:
        One row per test with ``x, y, lag, n, rho, p_naive, r1_x, r1_y, n_eff, p, q,
        definitional, diff_artifact``; sorted by q then |rho|, with mechanically
        linked pairs and differencing artifacts last. ``p`` is the p-value entering FDR.
    """
    t = transform(daily[metrics].astype(float), method, detrend_window)
    arr = {m: t[m].to_numpy() for m in metrics}
    r1 = {m: lag1_autocorr(arr[m]) for m in metrics}
    defs = {_pair_key(*p) for p in (definitional or [])}

    tests: list[tuple[str, str, int]] = []
    for a, b in combinations(metrics, 2):
        tests.append((a, b, 0))
        for k in range(1, max_lag + 1):
            tests.append((a, b, k))
            tests.append((b, a, k))

    rows = []
    for x, y, k in tests:
        xv = arr[x][: len(arr[x]) - k] if k else arr[x]
        yv = arr[y][k:]
        ok = ~(np.isnan(xv) | np.isnan(yv))
        n = int(ok.sum())
        if n < min_n:
            continue
        xs, ys = xv[ok], yv[ok]
        if np.ptp(xs) == 0 or np.ptp(ys) == 0:
            continue
        rho, p_naive = stats.spearmanr(xs, ys)
        ne = effective_n(n, r1[x], r1[y]) if use_effective_n else float(n)
        p = corr_pvalue(rho, ne) if use_effective_n else float(p_naive)
        rows.append((x, y, k, n, float(rho), float(p_naive), r1[x], r1[y], ne, p,
                     _pair_key(x, y) in defs))

    cols = ["x", "y", "lag", "n", "rho", "p_naive", "r1_x", "r1_y", "n_eff", "p", "definitional"]
    res = pd.DataFrame(rows, columns=cols)
    if res.empty:
        res["q"] = []
        return res
    res["q"] = bh_fdr(res["p"].to_numpy())
    res["diff_artifact"] = _diff_artifacts(res) if method == "diff" else False
    res["abs_rho"] = res["rho"].abs()
    res = res.sort_values(["definitional", "diff_artifact", "q", "abs_rho"], ascending=[True, True, True, False])
    return res.drop(columns="abs_rho").reset_index(drop=True)


def _diff_artifacts(res: pd.DataFrame, min_rho0: float = 0.1) -> pd.Series:
    """Flag lag-1 results that differencing manufactures from a same-day correlation.

    If x and y correlate at rho0 on the same day, then
    corr(x_t - x_{t-1}, y_{t+1} - y_t) is about -rho0/2, because the differenced
    y contains -y_t. A lag-1 result with the opposite sign to a non-trivial lag-0
    correlation is therefore expected under the null of no lagged effect.
    """
    lag0 = {frozenset((r.x, r.y)): r.rho for r in res[res["lag"] == 0].itertuples()}
    rho0 = pd.Series([lag0.get(frozenset((x, y)), 0.0) for x, y in zip(res["x"], res["y"])], index=res.index)
    return (res["lag"] == 1) & (rho0.abs() >= min_rho0) & (np.sign(res["rho"]) == -np.sign(rho0))


def best_lag_per_pair(res: pd.DataFrame) -> pd.DataFrame:
    """Collapse the scan to the strongest (lowest-q) lag per ordered direction."""
    if res.empty:
        return res
    return (res.sort_values(["q", "n"], ascending=[True, False])
            .drop_duplicates(subset=["x", "y"]).reset_index(drop=True))


def lagged_pair(daily: pd.DataFrame, x: str, y: str, lag: int, method: str,
                detrend_window: int = 28) -> pd.DataFrame:
    """Aligned transformed values for the drill-down plot: x(t) and y(t+lag)."""
    t = transform(daily[[x, y]].astype(float), method, detrend_window)
    out = pd.DataFrame({"x": t[x], "y": t[y].shift(-lag)}, index=t.index)
    return out.dropna()
