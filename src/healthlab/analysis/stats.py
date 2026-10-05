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


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta: P(A > B) - P(A < B), in [-1, 1].

    Computed from the Mann–Whitney U statistic (ties count half), so it equals
    the rank-biserial correlation for the same comparison. Rough magnitude
    guide (Romano et al. 2006): |d| < 0.147 negligible, < 0.33 small,
    < 0.474 medium, otherwise large.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic  # U for sample a
    return float(2.0 * u / (a.size * b.size) - 1.0)


def delta_magnitude(d: float) -> str:
    """Verbal label for |Cliff's delta| (Romano et al. 2006 thresholds)."""
    if not np.isfinite(d):
        return "n/a"
    d = abs(d)
    return "negligible" if d < 0.147 else "small" if d < 0.33 else "medium" if d < 0.474 else "large"


def _block_resample(x: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap resample of a time-ordered array (same length as x)."""
    n = x.size
    if block <= 1 or n <= block:
        return x[rng.integers(0, n, n)]
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
    return x[idx]


def bootstrap_median_diff(a: np.ndarray, b: np.ndarray, reps: int = 2000, block: int = 1,
                          seed: int | None = 0, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap CI for median(a) - median(b).

    Args:
        a, b: samples in time order (order only matters when ``block > 1``).
        block: block length for a moving-block bootstrap; 1 = ordinary iid bootstrap.
            Blocks preserve short-range autocorrelation, so the CI is honest when
            daily values are not independent.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2 or b.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    diffs = np.empty(reps)
    for i in range(reps):
        diffs[i] = np.median(_block_resample(a, block, rng)) - np.median(_block_resample(b, block, rng))
    lo, hi = np.quantile(diffs, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)
