"""Benjamini–Hochberg correction and correlation-test primitives."""

import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

from healthlab.analysis.stats import bh_fdr, corr_pvalue, effective_n, lag1_autocorr


def test_bh_textbook_example():
    # Benjamini & Hochberg (1995)-style worked example.
    p = [0.01, 0.04, 0.03, 0.005]
    q = bh_fdr(p)
    # sorted: 0.005,0.01,0.03,0.04 -> *4/rank: 0.02,0.02,0.04,0.04
    assert q.tolist() == pytest.approx([0.02, 0.04, 0.04, 0.02])


@pytest.mark.parametrize("seed", range(5))
def test_bh_matches_statsmodels(seed):
    rng = np.random.default_rng(seed)
    p = np.concatenate([rng.uniform(size=200), rng.uniform(0, 0.001, size=20)])
    ours = bh_fdr(p)
    ref = multipletests(p, method="fdr_bh")[1]
    np.testing.assert_allclose(ours, ref, rtol=1e-12)
    # Same discoveries at alpha = 0.05.
    assert ((ours <= 0.05) == multipletests(p, alpha=0.05, method="fdr_bh")[0]).all()


def test_bh_properties():
    rng = np.random.default_rng(1)
    p = rng.uniform(size=50)
    q = bh_fdr(p)
    assert (q >= p).all() and (q <= 1).all()
    order = np.argsort(p)
    assert (np.diff(q[order]) >= -1e-15).all()  # monotone in p


def test_bh_nan_passthrough_and_family_size():
    q = bh_fdr([0.01, np.nan, 0.02])
    assert np.isnan(q[1])
    assert q[0] == pytest.approx(0.02) and q[2] == pytest.approx(0.02)  # m = 2, not 3
    assert np.isnan(bh_fdr([np.nan])).all()


def test_effective_n():
    assert effective_n(100, 0.0, 0.5) == 100
    assert effective_n(100, 0.8, 0.8) == pytest.approx(100 * 0.36 / 1.64)
    assert effective_n(100, -0.5, 0.5) == 100        # never larger than n
    assert effective_n(10, 0.99, 0.99) == 3          # floor


def test_lag1_autocorr_ar1():
    rng = np.random.default_rng(0)
    x = np.zeros(5000)
    for i in range(1, 5000):
        x[i] = 0.7 * x[i - 1] + rng.normal()
    assert lag1_autocorr(x) == pytest.approx(0.7, abs=0.03)
    x[::7] = np.nan
    assert lag1_autocorr(x) == pytest.approx(0.7, abs=0.04)


def test_corr_pvalue_matches_pearson_t():
    from scipy import stats
    rng = np.random.default_rng(3)
    a = rng.normal(size=80)
    b = 0.3 * a + rng.normal(size=80)
    r, p = stats.pearsonr(a, b)
    assert corr_pvalue(r, 80) == pytest.approx(p, rel=1e-6)
