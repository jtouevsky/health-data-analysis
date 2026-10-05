"""Small, dependency-light statistical primitives used across analyses."""

from __future__ import annotations

import numpy as np
from scipy import stats


def bh_fdr(pvalues: np.ndarray | list[float]) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values (q-values).

    ``q_(i) = min_{j >= i} p_(j) * m / j`` over the sorted p-values, capped at 1.
    NaNs are passed through and excluded from the family size ``m``.

    A test is a discovery at FDR level alpha iff ``q <= alpha``.
    """
    p = np.asarray(pvalues, dtype=float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    if m == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    q[ok] = out
    return q


def lag1_autocorr(x: np.ndarray) -> float:
    """Lag-1 autocorrelation of a (possibly gappy) series; NaN pairs are skipped."""
    x = np.asarray(x, dtype=float)
    a, b = x[:-1], x[1:]
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 5:
        return 0.0
    a, b = a[ok], b[ok]
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def effective_n(n: int, r1x: float, r1y: float) -> float:
    """Effective sample size for a correlation between two AR(1)-like series.

    Bartlett's first-order approximation, ``n (1 - r1x r1y) / (1 + r1x r1y)``,
    clipped to ``[3, n]``. Positive shared autocorrelation shrinks n; it never
    grows beyond the nominal n (conservative).
    """
    prod = r1x * r1y
    ne = n * (1 - prod) / (1 + prod)
    return float(np.clip(ne, 3, n))


def corr_pvalue(r: float, n: float) -> float:
    """Two-sided p-value for a correlation via the t approximation with ``n - 2`` df."""
    if n <= 2 or not np.isfinite(r):
        return float("nan")
    r = float(np.clip(r, -0.999999, 0.999999))
    t = r * np.sqrt((n - 2) / (1 - r * r))
    return float(2 * stats.t.sf(abs(t), n - 2))
