"""Short-horizon forecasting with honest rolling-origin evaluation.

Models (all forecast ``horizon`` days ahead from an origin day):

* ``naive``: last observed value, repeated;
* ``seasonal_naive``: the value from the same weekday one week earlier;
* ``sarimax``: SARIMAX(p,d,q)(P,D,Q,7), with NaNs handled by the Kalman filter;
* ``gbm``: HistGradientBoostingRegressor, *direct* multi-horizon: one model with
  the horizon as a feature, and only information available at the origin
  (lags relative to the target day of at least ``horizon`` days, recent means, weekday).

Evaluation uses ``sklearn.model_selection.TimeSeriesSplit`` with a fixed
``test_size = horizon``: an expanding-window rolling origin. Training history is
capped at ``max_train_days``. MAE is reported per model and fold, alongside
skill relative to seasonal-naive, whether or not the models win.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

MODELS = ("naive", "seasonal_naive", "sarimax", "gbm")
LABELS = {"naive": "Naive (last value)", "seasonal_naive": "Seasonal naive (same weekday last week)",
          "sarimax": "SARIMAX", "gbm": "Gradient boosting (lag features)"}


@dataclass(frozen=True)
class ForecastParams:
    horizon: int = 7
    n_splits: int = 12
    max_train_days: int = 730
    sarimax_order: tuple[int, int, int] = (1, 0, 1)
    sarimax_seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 7)
    tree_lags: tuple[int, ...] = (7, 8, 14, 21, 28)
    seed: int = 0
    models: tuple[str, ...] = MODELS

    @classmethod
    def from_config(cls, f: dict, **overrides) -> "ForecastParams":
        base = cls(horizon=int(f.get("horizon_days", 7)), n_splits=int(f.get("n_splits", 12)),
                   max_train_days=int(f.get("max_train_days", 730)),
                   sarimax_order=tuple(f.get("sarimax_order", (1, 0, 1))),
                   sarimax_seasonal_order=tuple(f.get("sarimax_seasonal_order", (1, 0, 1, 7))),
                   tree_lags=tuple(f.get("tree_lags", (7, 8, 14, 21, 28))), seed=int(f.get("seed", 0)))
        return cls(**{**base.__dict__, **overrides})


# ---------------------------------------------------------------------------
# Individual forecasters: f(history: pd.Series [daily, ends at origin], h) -> np.ndarray(h)
# ---------------------------------------------------------------------------

def forecast_naive(hist: pd.Series, h: int) -> np.ndarray:
    last = hist.dropna()
    return np.full(h, last.iloc[-1] if len(last) else np.nan)


def forecast_seasonal_naive(hist: pd.Series, h: int, period: int = 7) -> np.ndarray:
    """Same weekday in the most recent week that has it observed (up to 4 weeks back)."""
    vals = hist.to_numpy()
    n = len(vals)
    out = np.full(h, np.nan)
    for i in range(h):
        for k in range(1, 5):
            j = n + i - k * period
            if 0 <= j < n and not np.isnan(vals[j]):
                out[i] = vals[j]
                break
    if np.isnan(out).any():
        out = np.where(np.isnan(out), forecast_naive(hist, h), out)
    return out


def forecast_sarimax(hist: pd.Series, h: int, p: ForecastParams, return_ci: bool = False):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y = hist.astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(y.to_numpy(), order=p.sarimax_order, seasonal_order=p.sarimax_seasonal_order,
                        trend="c", enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False, maxiter=100)
        fc = res.get_forecast(h)
        mean = np.asarray(fc.predicted_mean)
        if return_ci:
            ci = np.asarray(fc.conf_int(alpha=0.2))  # 80 % interval
            return mean, ci[:, 0], ci[:, 1]
    return mean


def _features(y: np.ndarray, dates: pd.DatetimeIndex, origins: np.ndarray, hs: np.ndarray,
              lags: tuple[int, ...]) -> np.ndarray:
    """Feature rows for (origin index, horizon) pairs; only uses y[<= origin]."""
    rows = []
    for o, h in zip(origins, hs):
        t = o + h
        f = [h, dates[t].dayofweek]
        f.append(y[o])
        w7 = y[max(0, o - 6): o + 1]
        w28 = y[max(0, o - 27): o + 1]
        f.append(np.nanmean(w7) if np.isfinite(w7).any() else np.nan)
        f.append(np.nanmean(w28) if np.isfinite(w28).any() else np.nan)
        for L in lags:
            j = t - L
            f.append(y[j] if 0 <= j <= o else np.nan)  # never peek past the origin
        rows.append(f)
    return np.asarray(rows, dtype=float)


def forecast_gbm(hist: pd.Series, h: int, p: ForecastParams) -> np.ndarray:
    y = hist.to_numpy(dtype=float)
    n = len(y)
    dates = pd.DatetimeIndex(hist.index).append(pd.date_range(hist.index[-1] + pd.Timedelta(days=1), periods=h))
    # Training pairs: every origin with a known target inside the history.
    o_idx, h_idx = [], []
    for hh in range(1, h + 1):
        o = np.arange(28, n - hh)
        o_idx.append(o)
        h_idx.append(np.full(o.size, hh))
    o_tr, h_tr = np.concatenate(o_idx), np.concatenate(h_idx)
    target = y[o_tr + h_tr]
    keep = ~np.isnan(target)
    X = _features(y, dates, o_tr[keep], h_tr[keep], p.tree_lags)
    model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=20, l2_regularization=1.0, random_state=p.seed)
    model.fit(X, target[keep])
    Xf = _features(y, dates, np.full(h, n - 1), np.arange(1, h + 1), p.tree_lags)
    return model.predict(Xf)


def run_model(name: str, hist: pd.Series, p: ForecastParams) -> np.ndarray:
    if name == "naive":
        return forecast_naive(hist, p.horizon)
    if name == "seasonal_naive":
        return forecast_seasonal_naive(hist, p.horizon)
    if name == "sarimax":
        return forecast_sarimax(hist, p.horizon, p)
    if name == "gbm":
        return forecast_gbm(hist, p.horizon, p)
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class Evaluation:
    folds: pd.DataFrame            # one row per (fold, model): origin, mae, n
    summary: pd.DataFrame          # one row per model: mean MAE, skill vs seasonal naive, folds won
    predictions: pd.DataFrame      # long: date, model, y_true, y_pred, fold
    params: ForecastParams = field(default_factory=ForecastParams)


def evaluate(series: pd.Series, p: ForecastParams) -> Evaluation:
    """Rolling-origin evaluation of all models on a daily series."""
    s = series.asfreq("D").astype(float)
    s = s.loc[s.first_valid_index(): s.last_valid_index()]
    tss = TimeSeriesSplit(n_splits=p.n_splits, test_size=p.horizon)
    fold_rows, preds = [], []
    for fold, (tr, te) in enumerate(tss.split(s)):
        hist = s.iloc[tr][-p.max_train_days:]
        truth = s.iloc[te]
        if hist.notna().sum() < 60 or truth.notna().sum() == 0:
            continue
        for m in p.models:
            yhat = run_model(m, hist, p)
            err = np.abs(truth.to_numpy() - yhat)
            fold_rows.append({"fold": fold, "origin": hist.index[-1], "model": m,
                              "mae": float(np.nanmean(err)), "n": int(np.isfinite(err).sum())})
            preds.append(pd.DataFrame({"date": truth.index, "model": m, "y_true": truth.to_numpy(),
                                       "y_pred": yhat, "fold": fold}))
    folds = pd.DataFrame(fold_rows)
    if folds.empty:
        return Evaluation(folds, pd.DataFrame(), pd.DataFrame(), p)
    piv = folds.pivot(index="fold", columns="model", values="mae")
    ref = piv["seasonal_naive"]
    summ = pd.DataFrame({
        "model": piv.columns,
        "mae": [piv[m].mean() for m in piv.columns],
        "mae_sd": [piv[m].std(ddof=1) for m in piv.columns],
        "skill_vs_snaive": [1 - piv[m].mean() / ref.mean() for m in piv.columns],
        "folds_beating_snaive": [int((piv[m] < ref).sum()) for m in piv.columns],
        "folds": len(piv),
    }).sort_values("mae").reset_index(drop=True)
    return Evaluation(folds, summ, pd.concat(preds, ignore_index=True), p)


def future_forecast(series: pd.Series, p: ForecastParams) -> pd.DataFrame:
    """Forecast the next ``horizon`` days with every model (SARIMAX with an 80 % interval)."""
    s = series.asfreq("D").astype(float)
    s = s.loc[s.first_valid_index(): s.last_valid_index()][-p.max_train_days:]
    idx = pd.date_range(s.index[-1] + pd.Timedelta(days=1), periods=p.horizon)
    out = pd.DataFrame(index=idx)
    for m in p.models:
        if m == "sarimax":
            mean, lo, hi = forecast_sarimax(s, p.horizon, p, return_ci=True)
            out[m], out["sarimax_lo"], out["sarimax_hi"] = mean, lo, hi
        else:
            out[m] = run_model(m, s, p)
    return out
