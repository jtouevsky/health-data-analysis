"""Correlation Scan: lagged Spearman across all metric pairs with FDR control."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from healthlab.analysis.correlations import TRANSFORMS, lagged_pair, scan
from healthlab.analysis.trend import rolling
from healthlab.app.common import (NOT_CAUSAL, SERIES, MUTED, analysable_metrics, cfg, label, require_data,
                                  style)

st.title("Correlation Scan")
st.warning(NOT_CAUSAL, icon="⚠️")

daily, catalog = require_data()
cc = cfg().section("correlations")
metrics = analysable_metrics(daily, catalog)

with st.expander("How this scan works, and why the series are transformed first", expanded=False):
    st.markdown(f"""
For every pair of daily metrics, the scan computes **Spearman's ρ** between *x* on day *t* and *y* on day
*t + lag* for lags 0–{cc['max_lag']}. Same-day pairs are tested once. Lagged pairs are tested in both directions.

**Why not use the raw daily values?** Health series are *autocorrelated* (today looks like yesterday) and share
slow trends (seasons, travel, a new job). Two series that both drift upward over a year look strongly
correlated even when day-to-day they're unrelated. A standard p-value assumes independent days, so on raw
series it is far too small. The tests show that two independent random walks come out "significant" most of the time.

- **Detrend (default)**: subtract a centred {cc['detrend_window_days']}-day rolling median. This keeps
  day-to-day deviations from your recent normal and removes seasons and trends.
- **First difference**: today minus yesterday. It removes trends completely, but creates a known artifact.
  If x and y correlate on the same day with ρ, differencing *creates* a lag-1 correlation of about −ρ/2.
  Those rows are flagged `diff_artifact`.
- **Raw**: shown only for comparison. Expect inflated significance.

**Effective n.** Even after transforming, some autocorrelation remains. With effective-n on, the p-value
uses n_eff = n·(1 − r₁ₓr₁ᵧ)/(1 + r₁ₓr₁ᵧ) (Bartlett), where r₁ is each transformed series' lag-1 autocorrelation.

**Multiple testing.** Hundreds of tests run at once, so about 5% of null tests would pass p < 0.05 by
chance. **Benjamini–Hochberg** q-values control the *false discovery rate* across the whole scan.
q < {cc['fdr_alpha']} means that among everything you call a discovery at this threshold, you expect at most
{cc['fdr_alpha']:.0%} to be false.

**Mechanical pairs** (steps↔distance, gait biomechanics, and the basal-energy artifact) are flagged and ranked last.

**Persistence caveat.** If x and y move together on the same day and both persist for several days, x(t) will
also correlate with y(t+1) without any lead–lag mechanism. Compare lagged ρ with the same-day ρ in the drill-down.
""")

c1, c2, c3, c4 = st.columns(4)
methods = list(TRANSFORMS)
method = c1.selectbox("Transform", methods, index=methods.index(cc.get("default_transform", "detrend")),
                      format_func=lambda m: TRANSFORMS[m])
min_n = c2.number_input("Minimum overlapping days (n)", 10, 2000, int(cc["min_n"]), step=10)
alpha = c3.number_input("FDR level (q)", 0.001, 0.25, float(cc["fdr_alpha"]), step=0.01, format="%.3f")
use_eff = c4.toggle("Effective-n adjustment", value=bool(cc.get("effective_n", True)))
c5, c6 = st.columns(2)
hide_mech = c5.toggle("Hide mechanical pairs and differencing artifacts", value=True)
only_sig = c6.toggle(f"Only q < {alpha:g}", value=False)


@st.cache_data(show_spinner="Running scan…")
def run(df: pd.DataFrame, ms: tuple, method: str, min_n: int, use_eff: bool) -> pd.DataFrame:
    return scan(df, list(ms), max_lag=int(cc["max_lag"]), min_n=min_n, method=method,
                detrend_window=int(cc["detrend_window_days"]), use_effective_n=use_eff,
                definitional=cc.get("definitional_pairs"))


res = run(daily, tuple(metrics), method, int(min_n), use_eff)
if res.empty:
    st.info("No metric pair has enough overlapping days. Lower the minimum n or import more data.")
    st.stop()

n_tests = len(res)
n_sig = int((res["q"] < alpha).sum())
k1, k2, k3 = st.columns(3)
k1.metric("Tests run", f"{n_tests:,}")
k2.metric(f"Discoveries (q < {alpha:g})", f"{n_sig:,}")
k3.metric("Expected false discoveries among them", f"≤ {n_sig * alpha:.1f}")

view = res.copy()
if hide_mech:
    view = view[~view["definitional"] & ~view["diff_artifact"]]
if only_sig:
    view = view[view["q"] < alpha]
view = view.assign(x_label=view["x"].map(lambda m: label(m, catalog, False)),
                   y_label=view["y"].map(lambda m: label(m, catalog, False)))
table = view[["x_label", "y_label", "lag", "rho", "n", "n_eff", "p_naive", "p", "q", "definitional", "diff_artifact"]]
table = table.rename(columns={"x_label": "x (day t)", "y_label": "y (day t+lag)", "rho": "Spearman ρ",
                              "n_eff": "effective n", "p_naive": "p (naive)", "p": "p (used)"})
st.caption("Sorted by q, then |ρ|. **ρ is the effect size**: with thousands of days, tiny effects (|ρ| < 0.1) "
           "can be 'significant' and still practically negligible. Click a row to drill down.")
event = st.dataframe(
    table, hide_index=True, width="stretch", height=380, on_select="rerun", selection_mode="single-row",
    column_config={
        "Spearman ρ": st.column_config.NumberColumn(format="%.3f"),
        "effective n": st.column_config.NumberColumn(format="%.0f"),
        "p (naive)": st.column_config.NumberColumn(format="%.2e"),
        "p (used)": st.column_config.NumberColumn(format="%.2e"),
        "q": st.column_config.NumberColumn(format="%.2e"),
    })

# -- Overview heatmap of same-day correlations -----------------------------------------
with st.expander("Same-day correlation matrix (lag 0)"):
    l0 = res[res["lag"] == 0]
    mat = pd.DataFrame(np.nan, index=metrics, columns=metrics)
    for r in l0.itertuples():
        mat.loc[r.x, r.y] = mat.loc[r.y, r.x] = r.rho
    labels = [label(m, catalog, False) for m in metrics]
    fig = go.Figure(go.Heatmap(z=mat.to_numpy(), x=labels, y=labels, zmin=-1, zmax=1,
                               colorscale=[[0, "#e34948"], [0.5, "rgba(128,128,128,0.12)"], [1, "#2a78d6"]],
                               hovertemplate="%{y} × %{x}: ρ = %{z:.2f}<extra></extra>"))
    st.plotly_chart(style(fig, height=520, hovermode="closest"), width="stretch")

# -- Drill-down --------------------------------------------------------------------------
st.subheader("Drill-down")
rows = event.selection.rows if event and event.selection else []
if rows:
    sel = view.iloc[rows[0]]
    x, y, lag = sel["x"], sel["y"], int(sel["lag"])
else:
    d1, d2, d3 = st.columns([2, 2, 1])
    x = d1.selectbox("x", metrics, format_func=lambda m: label(m, catalog))
    y = d2.selectbox("y", [m for m in metrics if m != x], format_func=lambda m: label(m, catalog))
    lag = d3.selectbox("lag (days)", list(range(int(cc["max_lag"]) + 1)))

pair = res[(res["x"] == x) & (res["y"] == y) & (res["lag"] == lag)]
if pair.empty and lag == 0:
    pair = res[(res["x"] == y) & (res["y"] == x) & (res["lag"] == 0)]
if not pair.empty:
    p = pair.iloc[0]
    st.markdown(f"**{label(x, catalog, False)}** on day *t* vs **{label(y, catalog, False)}** on day *t + {lag}*: "
                f"ρ = **{p['rho']:.3f}**, n = {int(p['n'])}, effective n ≈ {p['n_eff']:.0f}, q = {p['q']:.2g}"
                + (" · *mechanical pair*" if p["definitional"] else "")
                + (" · *likely differencing artifact*" if p["diff_artifact"] else ""))
    same_day = res[(res["lag"] == 0) & (((res["x"] == x) & (res["y"] == y)) | ((res["x"] == y) & (res["y"] == x)))]
    if lag and not same_day.empty:
        st.caption(f"Same-day ρ for this pair is {same_day['rho'].iat[0]:.3f}; "
                   "a lagged ρ much smaller than that may just be persistence.")
else:
    st.caption("This pair/lag was not tested (insufficient overlap).")

lp = lagged_pair(daily, x, y, lag, method, int(cc["detrend_window_days"]))
a, b = st.columns(2)
with a:
    fig = go.Figure(go.Scatter(x=lp["x"], y=lp["y"], mode="markers",
                               marker=dict(size=6, color=SERIES[0], opacity=0.45),
                               hovertemplate="x=%{x:.2f}, y=%{y:.2f}<extra>%{text}</extra>",
                               text=lp.index.strftime("%Y-%m-%d")))
    if len(lp) > 10:
        # Rank-based visual guide: binned medians (consistent with Spearman, robust to outliers).
        bins = pd.qcut(lp["x"].rank(method="first"), q=min(10, len(lp) // 10), labels=False)
        g = lp.groupby(bins).median()
        fig.add_scatter(x=g["x"], y=g["y"], mode="lines+markers", name="Decile medians",
                        line=dict(color=SERIES[1], width=2), marker=dict(size=8))
    fig.update_layout(xaxis_title=f"{label(x, catalog, False)} (t, {method})",
                      yaxis_title=f"{label(y, catalog, False)} (t+{lag}, {method})", showlegend=False)
    st.plotly_chart(style(fig, title="Transformed values", hovermode="closest"), width="stretch")
with b:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=(label(x, catalog), label(y, catalog)))
    for i, (m, col) in enumerate([(x, SERIES[0]), (y, SERIES[1])], start=1):
        fig.add_scatter(x=daily.index, y=daily[m], mode="markers", marker=dict(size=2, color=MUTED),
                        showlegend=False, row=i, col=1)
        fig.add_scatter(x=daily.index, y=rolling(daily[m], 14), line=dict(color=col, width=2),
                        name=f"{label(m, catalog, False)} (14-d mean)", row=i, col=1)
    fig.update_layout(showlegend=False)  # subplot titles name each series
    st.plotly_chart(style(fig, title="Raw series with 14-day mean (separate axes, shared dates)", height=420,
                          margin=dict(l=10, r=10, t=70, b=10)),
                    width="stretch")
