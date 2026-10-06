"""Forecasts: SARIMAX and gradient boosting vs naive baselines, evaluated honestly."""

import plotly.graph_objects as go
import streamlit as st

from healthlab.analysis.forecast import LABELS, ForecastParams, evaluate, future_forecast
from healthlab.analysis.trend import weight_trend
from healthlab.app.common import BOTTOM_LEGEND, SERIES, MUTED, analysable_metrics, cfg, has_metric, label, require_data, style, to_display

st.title("Forecasts")
daily, catalog = require_data()
c = cfg()
fc = c.section("forecast")
COLORS = {"naive": MUTED, "seasonal_naive": SERIES[3], "sarimax": SERIES[0], "gbm": SERIES[2]}

targets = {}
if has_metric(daily, "weight_kg", 30):
    hl = float(c.section("trends").get("weight_ewma_halflife_days", 10))
    targets["weight_trend"] = ("Weight trend (EWMA)", weight_trend(daily["weight_kg"], hl), "weight_kg")
for m in analysable_metrics(daily, catalog):
    if daily[m].notna().sum() >= 120:
        targets[m] = (label(m, catalog), daily[m], m)
if not targets:
    st.info("No series has enough history (≥ 120 observed days) to forecast.")
    st.stop()

default = "weight_trend" if "weight_trend" in targets else ("steps" if "steps" in targets else next(iter(targets)))
c1, c2 = st.columns([3, 1])
key = c1.selectbox("Series", list(targets), index=list(targets).index(default), format_func=lambda k: targets[k][0])
n_splits = c2.number_input("Evaluation folds", 4, 30, int(fc["n_splits"]))
if "weight_trend" not in targets:
    st.caption("Weight forecasting appears here when body-weight data is present (optional module).")

name, series, unit_metric = targets[key]
p = ForecastParams.from_config(fc, n_splits=int(n_splits))

st.markdown(f"""
**Setup.** Forecast horizon {p.horizon} days. **Rolling origin** via `TimeSeriesSplit(n_splits={p.n_splits},
test_size={p.horizon})`: each fold trains on everything before its origin (capped at the last {p.max_train_days}
days) and forecasts the next {p.horizon} days. Any model has to beat two baselines to be worth anything:
**naive** (repeat the last value) and **seasonal naive** (same weekday last week).
""")


@st.cache_data(show_spinner="Running rolling-origin evaluation (SARIMAX refits per fold)…")
def run_eval(s, p):
    return evaluate(s, p)


@st.cache_data(show_spinner=False)
def run_future(s, p):
    return future_forecast(s, p)


ev = run_eval(series, p)
if ev.summary.empty:
    st.info("Not enough history for the requested folds.")
    st.stop()

conv = lambda v: to_display(v, unit_metric, catalog)[0]
unit = to_display(1.0, unit_metric, catalog)[1]
summ = ev.summary.copy()
summ["model"] = summ["model"].map(LABELS)
summ["mae"], summ["mae_sd"] = conv(summ["mae"]), conv(summ["mae_sd"])
st.subheader("Out-of-sample accuracy")
st.dataframe(summ, hide_index=True, width="stretch", column_config={
    "mae": st.column_config.NumberColumn(f"MAE ({unit})", format="%.3f"),
    "mae_sd": st.column_config.NumberColumn("SD across folds", format="%.3f"),
    "skill_vs_snaive": st.column_config.NumberColumn("Skill vs seasonal naive", format="percent",
                                                     help="1 − MAE / MAE(seasonal naive). Positive = better."),
    "folds_beating_snaive": st.column_config.NumberColumn("Folds beating seasonal naive"),
})

# Honest verdict, generated from the numbers.
s = ev.summary.set_index("model")
best_base = s.loc[["naive", "seasonal_naive"], "mae"].idxmin()
lines = []
for m in ("sarimax", "gbm"):
    if m not in s.index:
        continue
    rel = 1 - s.loc[m, "mae"] / s.loc[best_base, "mae"]
    piv = ev.folds.pivot(index="fold", columns="model", values="mae")
    wins = int((piv[m] < piv[best_base]).sum())
    verdict = ("beats" if rel > 0.02 else "does **not** meaningfully beat" if rel > -0.02 else "is **worse** than")
    lines.append(f"- **{LABELS[m]}** {verdict} the best baseline ({LABELS[best_base].lower()}): "
                 f"{rel:+.1%} MAE, better in {wins}/{len(piv)} folds.")
st.markdown("\n".join(lines))
st.caption(f"With only {len(ev.folds['fold'].unique())} folds of {p.horizon} days, differences of a few percent "
           "are within fold-to-fold noise; look at the fold count as well as the average.")

# -- MAE per fold ----------------------------------------------------------------------
fig = go.Figure()
piv = ev.folds.pivot(index="origin", columns="model", values="mae")
for m in piv.columns:
    fig.add_scatter(x=piv.index, y=conv(piv[m]), name=LABELS[m], mode="lines+markers",
                    line=dict(color=COLORS[m], width=2), marker=dict(size=8))
fig.update_yaxes(title=f"MAE ({unit})")
st.plotly_chart(style(fig, title="MAE per fold (x = forecast origin)"), width="stretch")

# -- Backtest + next week -------------------------------------------------------------------
st.subheader("Recent backtests and next week")
models = st.multiselect("Models to show", list(LABELS), default=["seasonal_naive", "sarimax", "gbm"],
                        format_func=LABELS.get)
fut = run_future(series, p)
recent = series.dropna().iloc[-90:]
fig = go.Figure()
fig.add_scatter(x=recent.index, y=conv(recent), mode="lines+markers", name="Actual",
                line=dict(color=MUTED, width=1), marker=dict(size=4))
for m in models:
    bt = ev.predictions[ev.predictions["model"] == m]
    bt = bt[bt["date"] >= recent.index.min()]
    for i, (_, g) in enumerate(bt.groupby("fold")):
        fig.add_scatter(x=g["date"], y=conv(g["y_pred"]), mode="lines", line=dict(color=COLORS[m], width=2),
                        name=f"{LABELS[m]} (backtest)", legendgroup=m, showlegend=i == 0, opacity=0.6)
    if m == "sarimax":
        fig.add_scatter(x=fut.index, y=conv(fut["sarimax_hi"]), line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=fut.index, y=conv(fut["sarimax_lo"]), fill="tonexty", line=dict(width=0),
                        fillcolor="rgba(42,120,214,0.15)", name="SARIMAX 80% interval", hoverinfo="skip")
    fig.add_scatter(x=fut.index, y=conv(fut[m]), mode="lines+markers", name=f"{LABELS[m]} (next {p.horizon} d)",
                    line=dict(color=COLORS[m], width=3, dash="dot"), marker=dict(size=8), legendgroup=m)
fig.update_yaxes(title=f"{name} ({unit})" if unit and unit != "count" else name)
st.plotly_chart(style(fig, title=f"{name}: last 90 days, backtest forecasts, and next {p.horizon} days", height=500,
                      legend=BOTTOM_LEGEND),
                width="stretch")

with st.expander("Model details"):
    st.markdown(f"""
- **SARIMAX{tuple(p.sarimax_order)}×{tuple(p.sarimax_seasonal_order)}** with a constant. Missing days are handled
  natively by the Kalman filter. A deliberately small, fixed order (no per-fold auto-ARIMA search) keeps evaluation
  honest and fast.
- **Gradient boosting** (`HistGradientBoostingRegressor`, 200 shallow trees, strong regularisation). *Direct*
  multi-horizon: one model with horizon as a feature. Features are the value at the origin, 7- and 28-day means
  up to the origin, the target weekday, and lags {list(p.tree_lags)} days *relative to the target*, set to
  missing whenever they'd fall after the origin. So there's no look-ahead.
- **Why so modest?** A few hundred to a few thousand noisy daily points don't support complex models. The
  baselines are surprisingly hard to beat for behavioural data.
""")
