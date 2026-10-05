"""Data Quality: coverage, gaps, sources, outliers and logging coverage."""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from healthlab.app.common import (SEQ_BLUE, SERIES, MUTED, cfg, connection, data, label, require_data,
                                  style, to_display)
from healthlab.quality import (activity_timing, coverage_matrix, gap_summary, logging_summary,
                               outlier_table, source_breakdown)

st.title("Data Quality")
require_data()
# Quality checks look at *all* data, including partially logged nutrition days.
daily, catalog = data(mask_nutrition=False, respect_start=False)
c = cfg()
metrics = [m for m in catalog["metric"] if m in daily]

# -- Coverage heatmap ------------------------------------------------------------
st.subheader("Coverage: share of days with data, by month")
cov = coverage_matrix(daily, metrics, "MS")
cov.index = [label(m, catalog, with_unit=False) for m in cov.index]
fig = px.imshow(cov, aspect="auto", color_continuous_scale=SEQ_BLUE, zmin=0, zmax=1,
                labels=dict(x="Month", y="", color="Coverage"))
fig.update_traces(hovertemplate="%{y}<br>%{x|%b %Y}: %{z:.0%}<extra></extra>")
st.plotly_chart(style(fig, height=60 + 28 * len(cov), hovermode="closest"), width="stretch")
st.caption("A metric's coverage before its first sample is 0 by definition; that's not a gap.")

# -- Nutrition logging -------------------------------------------------------------
logs = logging_summary(daily, c)
st.subheader("Food-logging coverage")
if logs is None:
    st.info("No dietary energy records in this export, so the nutrition module is inactive.")
else:
    n = c.section("nutrition")
    st.markdown(f"A day is **well logged** when it has ≥ {n['min_entries']} entries and "
                f"{n['min_kcal']}–{n['max_kcal']} kcal. Only well-logged days enter any nutrition analysis; "
                "unlogged days are missing, not zero.")
    order = [k for k in ["well logged", "partial / implausible", "not logged"] if k in logs]
    fig = go.Figure()
    for k, col in zip(order, [SERIES[2], SERIES[3], MUTED]):
        fig.add_bar(x=logs.index, y=logs[k], name=k, marker_color=col)
    fig.update_layout(barmode="stack", bargap=0.15)
    st.plotly_chart(style(fig, title="Days per month by logging status"), width="stretch")

# -- Gaps -----------------------------------------------------------------------------
st.subheader("Logging gaps")
min_days = int(c.section("quality").get("gap_min_days", 3))
gaps = gap_summary(daily, metrics, min_days)
st.caption(f"Runs of ≥ {min_days} consecutive missing days between a metric's first and last sample.")
st.dataframe(gaps.assign(start=gaps["start"].dt.date, end=gaps["end"].dt.date), hide_index=True,
             width="stretch", height=260)

# -- Sources ----------------------------------------------------------------------------
st.subheader("Sources per type")
con = connection()
src = source_breakdown(con)
st.caption("For each type and day, only the highest-priority source with data is used (see DECISIONS.md R3). "
           "`days_overridden` counts days where this source had data but a higher-priority source won.")
st.dataframe(src, hide_index=True, width="stretch")

# -- Outliers ----------------------------------------------------------------------------
st.subheader("Outlier flags")
q = c.section("quality")
out = outlier_table(daily, catalog, c)
st.caption(f"Flagged when more than {q['outlier_mad_k']} robust SDs (scaled MAD) from the centred "
           f"{q['outlier_window_days']}-day median, or outside the configured plausible range. "
           "Flags are informational; nothing is removed automatically.")
counts = out.groupby("metric")[["outlier", "implausible"]].sum().astype(int)
st.dataframe(counts, width="stretch")
choices = [m for m in metrics if m in set(out["metric"])]
if choices:
    m = st.selectbox("Inspect metric", choices, format_func=lambda x: label(x, catalog))
    s, unit = to_display(daily[m], m, catalog)
    flagged = out[out["metric"] == m]
    fv, _ = to_display(flagged["value"], m, catalog)
    fig = go.Figure()
    fig.add_scatter(x=s.index, y=s, mode="markers", name="Daily value", marker=dict(size=3, color=MUTED))
    fig.add_scatter(x=flagged["date"], y=fv, mode="markers", name="Flagged",
                    marker=dict(size=9, color=SERIES[7], line=dict(width=2, color="white")))
    st.plotly_chart(style(fig, title=label(m, catalog)), width="stretch")
    st.dataframe(flagged, hide_index=True, width="stretch", height=220)

# -- Activity timing ------------------------------------------------------------------------
st.subheader("When in the day activity happens")
timing = activity_timing(con, c)
if not timing.empty:
    fig = px.imshow(timing.T, aspect="auto", color_continuous_scale=SEQ_BLUE, origin="lower",
                    zmin=0, zmax=float(np.nanquantile(timing.to_numpy(), 0.99)),
                    labels=dict(x="Month", y=f"Hour ({c.timezone})", color="Share of steps"))
    fig.update_traces(hovertemplate="%{x|%b %Y}, %{y}:00 → %{z:.1%}<extra></extra>")
    st.plotly_chart(style(fig, height=380, hovermode="closest"), width="stretch")
    st.caption(f"All local dates use {c.timezone} (DECISIONS.md T1). Months spent abroad appear as a shifted "
               "band. In those months, day boundaries split your real days, so treat their day-to-day "
               "values with caution.")
con.close()
