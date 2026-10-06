"""Trends: any metric over time, smoothing, weekly/monthly views, day-of-week patterns."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from healthlab.analysis.trend import rolling
from healthlab.app.common import SERIES, MUTED, analysable_metrics, label, meta, require_data, style, to_display

st.title("Trends")
daily, catalog = require_data()
metrics = analysable_metrics(daily, catalog)
if not metrics:
    st.warning("No metric has enough observed days yet.")
    st.stop()

c1, c2, c3 = st.columns([2, 1, 2])
metric = c1.selectbox("Metric", metrics, index=metrics.index("steps") if "steps" in metrics else 0,
                      format_func=lambda m: label(m, catalog))
window = c2.selectbox("Rolling window", [1, 7, 14, 28, 56], index=3, format_func=lambda w: "none" if w == 1 else f"{w} days")
s_full = daily[metric]
lo, hi = s_full.first_valid_index().date(), s_full.last_valid_index().date()
default_lo = max(lo, (pd.Timestamp(hi) - pd.Timedelta(days=730)).date())
rng = c3.slider("Date range", min_value=lo, max_value=hi, value=(default_lo, hi), format="MMM YYYY")

s, unit = to_display(s_full, metric, catalog)
# Smooth on the full series so the window is "warm" at the left edge of the view.
smooth = rolling(s, window).loc[str(rng[0]):str(rng[1])]
s = s.loc[str(rng[0]):str(rng[1])]
agg = meta(catalog, metric).get("agg")

fig = go.Figure()
fig.add_scatter(x=s.index, y=s, mode="markers", name="Daily", marker=dict(size=3, color=MUTED, opacity=0.6))
if window > 1:
    fig.add_scatter(x=smooth.index, y=smooth, name=f"{window}-day mean", line=dict(color=SERIES[0], width=2))
st.plotly_chart(style(fig, title=label(metric, catalog)), width="stretch")
st.caption(f"Daily values are the {agg} of the day's samples. Rolling means need at least half the window observed; "
           "missing days are skipped, not treated as zero.")

# -- Weekly / monthly ----------------------------------------------------------------
st.subheader("Weekly and monthly view")
period = st.radio("Resolution", ["Weekly", "Monthly"], horizontal=True)
freq = "W-MON" if period == "Weekly" else "MS"
grp = s.resample(freq, label="left", closed="left")
res = pd.DataFrame({"mean": grp.mean(), "q25": grp.quantile(0.25), "q75": grp.quantile(0.75), "n": grp.count()})
res = res[res["n"] >= (3 if period == "Weekly" else 10)]
fig = go.Figure()
fig.add_scatter(x=res.index, y=res["q75"], line=dict(width=0), showlegend=False, hoverinfo="skip")
fig.add_scatter(x=res.index, y=res["q25"], fill="tonexty", line=dict(width=0), name="Interquartile range",
                fillcolor="rgba(42,120,214,0.15)", hoverinfo="skip")
fig.add_scatter(x=res.index, y=res["mean"], mode="lines+markers", name=f"{period} mean",
                line=dict(color=SERIES[0], width=2), marker=dict(size=8),
                customdata=res["n"], hovertemplate="%{y:.1f} (n=%{customdata} days)")
st.plotly_chart(style(fig, title=f"{period} mean with interquartile range"), width="stretch")
st.caption(f"Periods with fewer than {3 if period == 'Weekly' else 10} observed days are hidden.")

# -- Day of week ----------------------------------------------------------------------
st.subheader("Day-of-week pattern")
dow = pd.DataFrame({"v": s, "dow": s.index.day_name()}).dropna()
order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
fig = go.Figure()
fig.add_box(x=dow["dow"], y=dow["v"], marker_color=SERIES[0], boxmean=True, name=label(metric, catalog),
            showlegend=False, marker=dict(size=3))
fig.update_xaxes(categoryorder="array", categoryarray=order)
st.plotly_chart(style(fig, title="Distribution by weekday (box = IQR, dashed = mean)", hovermode="closest"),
                width="stretch")
if dow["dow"].nunique() == 7:
    from scipy.stats import kruskal
    h, p = kruskal(*[g["v"].to_numpy() for _, g in dow.groupby("dow")])
    st.caption(f"Kruskal–Wallis across weekdays: H = {h:.1f}, p = {p:.2g}, n = {len(dow)} days. "
               "Daily values are autocorrelated, so treat p as optimistic.")
