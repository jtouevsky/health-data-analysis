"""Changepoints: regime shifts in activity and gait, with sensitivity, BOCPD and annotations."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from healthlab.analysis.changepoints import (CPParams, all_events, bocpd, detect_all, match_events,
                                             segment_effects)
from healthlab.app.common import (BOTTOM_LEGEND, MUTED, SERIES, cfg, connection, label, require_data, style,
                                  to_display)

st.title("Changepoints")
daily, catalog = require_data()
c = cfg()
cc = c.section("changepoints")
metrics = [m for m in cc.get("metrics", []) if m in daily and daily[m].notna().sum() >= 60]
if not metrics:
    st.info("None of the configured changepoint metrics has enough data (≥ 60 observed days).")
    st.stop()

st.markdown("""
Where did a metric's *level* change, and did something happen then? Each metric is averaged per week
(weeks with ≥ 4 observed days; steps and distance on a log scale). The weekly series is standardised by a
robust long-run noise estimate and segmented with **PELT** under a **BIC penalty** (2·log n per changepoint).
Changepoints are then matched to your **life events** (config), **device changes** (new phone hardware or a
major iOS update, which can change how Apple measures gait) and
**travel periods** (time-zone overrides).
""")

c1, c2, c3, c4 = st.columns(4)
sel = c1.multiselect("Metrics", metrics, default=metrics, format_func=lambda m: label(m, catalog, False))
mult = c2.select_slider("Penalty × BIC", options=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0],
                        value=float(cc.get("penalty_multiplier", 1.0)),
                        help="Lower finds more (smaller) shifts; higher keeps only the strongest. 1 = BIC.")
res_choice = c3.radio("Resolution", ["week", "day"], horizontal=True,
                      help="Daily resolution is noisier (weekday cycle, autocorrelation) and slower.")
min_seg = c4.number_input("Min. segment (bins)", 2, 26, int(cc.get("min_segment", 4)))
p = CPParams.from_config(cc, penalty_multiplier=float(mult), resolution=res_choice, min_segment=int(min_seg))
if not sel:
    st.stop()


@st.cache_data(show_spinner="Segmenting…")
def run(df, ms, p):
    return detect_all(df, list(ms), p)


@st.cache_data(show_spinner=False)
def events_cached(db_mtime: float):
    con = connection()
    try:
        return all_events(con, c)
    finally:
        con.close()


results = run(daily, tuple(sel), p)
events = events_cached(c.db_path.stat().st_mtime if c.db_path.exists() else 0.0)
tol = int(cc.get("event_tolerance_days", 21))
KIND_STYLE = {"life event": ("#9a9993", "dash"), "device": ("#e34948", "dot"), "travel": ("#1baf7a", "dot")}

# -- Overview across metrics -----------------------------------------------------------
st.subheader("All metrics at a glance")
fig = go.Figure()
for i, m in enumerate(sel):
    r = results.get(m)
    if r is None:
        continue
    stab = [r.stability.get(cp, np.nan) for cp in r.changepoints]
    fig.add_scatter(x=r.changepoints, y=[label(m, catalog, False)] * len(r.changepoints), mode="markers",
                    marker=dict(size=[8 + 10 * (s if np.isfinite(s) else 0) for s in stab], color=SERIES[0],
                                line=dict(width=2, color="white")),
                    customdata=np.array(stab), showlegend=False,
                    hovertemplate="%{y}<br>%{x|%d %b %Y}<br>stability %{customdata:.0%}<extra></extra>")
for o in c.timezone_overrides:
    fig.add_vrect(x0=pd.Timestamp(o.start), x1=pd.Timestamp(o.end), fillcolor="rgba(27,175,122,0.10)", line_width=0)
for e in events.itertuples():
    col, dash = KIND_STYLE.get(e.kind, (MUTED, "dot"))
    fig.add_vline(x=e.date, line_color=col, line_dash=dash, line_width=1)
fig.update_yaxes(categoryorder="array", categoryarray=[label(m, catalog, False) for m in sel][::-1])
st.plotly_chart(style(fig, height=120 + 40 * len(sel), hovermode="closest",
                      title="Detected changepoints (marker size = stability across penalties)"), width="stretch")
st.caption("Vertical lines: grey dashed = life event, red dotted = device change, green dotted / shaded = travel "
           "period. Shifts that line up across all gait metrics at a device change are likely measurement "
           "artifacts, not changes in you.")

# -- Event matching table ---------------------------------------------------------------
rows = []
for m in sel:
    r = results.get(m)
    if r is None:
        continue
    mt = match_events(r.changepoints, events, tol).assign(metric=label(m, catalog, False))
    mt["stability"] = [r.stability.get(cp, np.nan) for cp in mt["changepoint"]]
    rows.append(mt)
if rows:
    tab = pd.concat(rows, ignore_index=True)[["metric", "changepoint", "stability", "event", "kind", "days_from_event"]]
    tab["changepoint"] = tab["changepoint"].dt.date
    n_unexpl = int(tab["event"].isna().sum())
    st.markdown(f"**{len(tab)} changepoints, {len(tab) - n_unexpl} within ±{tol} days of a known event, "
                f"{n_unexpl} unexplained.** Add explanations as `changepoints.life_events` in "
                "`config/healthlab.yaml` (date + label).")
    st.dataframe(tab, hide_index=True, width="stretch",
                 column_config={"stability": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent")})

# -- Drill-down -------------------------------------------------------------------------
st.subheader("Drill-down")
m = st.selectbox("Metric", [x for x in sel if x in results], format_func=lambda x: label(x, catalog))
r = results[m]
binned, unit = to_display(r.binned, m, catalog)
fig = go.Figure()
fig.add_scatter(x=binned.index, y=binned, mode="markers", name=f"{'Weekly' if p.resolution == 'week' else 'Daily'} mean",
                marker=dict(size=5, color=MUTED))
seg_x, seg_y = [], []
for s in r.segments.itertuples():
    end = s.end + (pd.Timedelta(days=6) if p.resolution == "week" else pd.Timedelta(0))
    seg_x += [s.start, end, None]
    seg_y += [to_display(s.mean, m, catalog)[0]] * 2 + [None]
fig.add_scatter(x=seg_x, y=seg_y, mode="lines", name="Segment mean", line=dict(color=SERIES[1], width=3))
for cp in r.changepoints:
    fig.add_vline(x=cp, line_color=SERIES[0], line_width=2)
for e in events.itertuples():
    col, dash = KIND_STYLE.get(e.kind, (MUTED, "dot"))
    fig.add_vline(x=e.date, line_color=col, line_dash=dash, line_width=1)
    fig.add_annotation(x=e.date, y=1.0, yref="paper", text=e.label, showarrow=False, textangle=-90,
                       xanchor="right", yanchor="top", font=dict(size=10, color=col))
fig.update_yaxes(title=unit if unit and unit != "count" else None)
st.plotly_chart(style(fig, height=460, title=f"{label(m, catalog)}: segments (blue lines = changepoints)",
                      legend=BOTTOM_LEGEND), width="stretch")
st.caption(f"Noise scale: σ = {r.sigma_iid:.3g} (MAD of differences), lag-1 autocorrelation φ = {r.phi:.2f}; "
           f"BIC penalty × {mult:g} = {r.penalty:.1f} on the standardised series (n = {len(r.z)} bins).")

eff = segment_effects(daily[m], r, p)
if not eff.empty:
    st.markdown("**Before vs after each changepoint** (daily values of the adjacent segments)")
    show = eff.copy()
    for col in ["median_before", "median_after", "diff", "ci_lo", "ci_hi"]:
        show[col] = to_display(show[col], m, catalog)[0]
    show["changepoint"] = show["changepoint"].dt.date
    st.dataframe(show, hide_index=True, width="stretch", column_config={
        "pct_change": st.column_config.NumberColumn("% change", format="percent"),
        "cliffs_delta": st.column_config.NumberColumn("Cliff's δ (after vs before)", format="%.2f"),
        "ci_lo": st.column_config.NumberColumn("95% CI lo (7-d blocks)", format="%.3g"),
        "ci_hi": st.column_config.NumberColumn("95% CI hi", format="%.3g"),
        "stability": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
    })
    st.caption("Effect sizes compare the segments on either side, as found by the same data, so they are "
               "descriptive: by construction the detector picks boundaries where the difference is large. "
               "Use them to judge *how big* a shift is, not *whether* it is real; stability across penalties "
               "and a matching event are better evidence of that.")

# -- Sensitivity --------------------------------------------------------------------------
a, b = st.columns(2)
with a:
    sens = r.sensitivity
    fig = go.Figure(go.Scatter(x=sens["multiplier"], y=sens["n_changepoints"], mode="lines+markers",
                               line=dict(color=SERIES[0], width=2), marker=dict(size=9)))
    fig.add_vline(x=mult, line_dash="dash", line_color=MUTED)
    fig.update_xaxes(type="log", title="penalty × BIC")
    fig.update_yaxes(title="changepoints found", rangemode="tozero")
    st.plotly_chart(style(fig, height=320, title="Penalty sensitivity", hovermode="closest"), width="stretch")
with b:
    pts = [(row.multiplier, cp) for row in sens.itertuples() for cp in row.changepoints]
    fig = go.Figure(go.Scatter(x=[cp for _, cp in pts], y=[mlt for mlt, _ in pts], mode="markers",
                               marker=dict(size=9, color=SERIES[0], symbol="line-ns-open", line=dict(width=2))))
    fig.update_yaxes(type="log", title="penalty × BIC")
    st.plotly_chart(style(fig, height=320, title="Which changepoints survive which penalty", hovermode="closest"),
                    width="stretch")
st.caption("A changepoint that persists from 0.25× to 4× BIC is robust; one that appears only at low penalties "
           "is fragile. The *stability* column is the share of penalty multipliers that place a changepoint "
           f"within ±{p.stability_tolerance} bins.")

# -- BOCPD comparison ---------------------------------------------------------------------
with st.expander("Comparison: Bayesian online changepoint detection (Adams & MacKay 2007)", expanded=False):
    trace, bc = bocpd(r.z, p.bocpd_expected_run, p.bocpd_min_run)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Standardised series", f"P(regime began within last {p.bocpd_min_run} bins)"))
    fig.add_scatter(x=r.z.index, y=r.z, mode="markers", marker=dict(size=4, color=MUTED), row=1, col=1, showlegend=False)
    fig.add_scatter(x=trace.index, y=trace["p_new_regime"], line=dict(color=SERIES[2], width=2), row=2, col=1, showlegend=False)
    for cp in r.changepoints:
        fig.add_vline(x=cp, line_color=SERIES[0], line_width=2)
    for cp in bc:
        fig.add_vline(x=cp, line_color=SERIES[2], line_dash="dash", line_width=2)
    st.plotly_chart(style(fig, height=440, title="PELT (solid blue) vs BOCPD (dashed green)", hovermode="closest",
                          margin=dict(l=10, r=10, t=70, b=10)), width="stretch")
    agree = sum(any(abs((x - y).days) <= 7 * p.stability_tolerance for y in bc) for x in r.changepoints)
    st.markdown(f"BOCPD (constant hazard, expected regime length {p.bocpd_expected_run:g} bins, Normal–Gamma "
                f"model) reports {len(bc)} changepoints; **{agree} of PELT's {len(r.changepoints)}** have a BOCPD "
                f"counterpart within ±{p.stability_tolerance} bins. BOCPD processes the series online (each "
                "decision uses only the past), so it typically flags shifts a few bins later.")
