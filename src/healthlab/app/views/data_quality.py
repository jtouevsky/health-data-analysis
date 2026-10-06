"""Data Quality: coverage, gaps, sources, outliers and logging coverage."""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from healthlab.app.common import (SEQ_BLUE, SERIES, MUTED, cfg, connection, data, label, require_data,
                                  style, to_display)
from healthlab.quality import (activity_timing, coverage_matrix, gap_summary, hardware_timeline,
                               outlier_table, source_breakdown, source_monthly,
                               source_summary)

st.title("Data Quality")
require_data()
# Quality checks look at *all* data, ignoring analysis.start_date.
daily, catalog = data(respect_start=False)
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

# -- Gaps -----------------------------------------------------------------------------
st.subheader("Logging gaps")
min_days = int(c.section("quality").get("gap_min_days", 3))
gaps = gap_summary(daily, metrics, min_days)
st.caption(f"Runs of ≥ {min_days} consecutive missing days between a metric's first and last sample.")
st.dataframe(gaps.assign(start=gaps["start"].dt.date, end=gaps["end"].dt.date), hide_index=True,
             width="stretch", height=260)

# -- Sources & devices ---------------------------------------------------------------------
st.subheader("Sources & devices")
con = connection()
summ = source_summary(con, c)
st.caption("Every app or device that wrote data, with when it was active and which hardware it reported. "
           "Use this to check which sources are yours. To ignore one (e.g. a family member's phone), add it "
           "to `source_exclude` in `config/healthlab.yaml` and run `healthlab rollup`.")
st.dataframe(summ, hide_index=True, width="stretch", column_config={
    "span_days": st.column_config.NumberColumn("span (days)"),
    "active_days": st.column_config.NumberColumn("days with data"),
    "hardware": st.column_config.TextColumn("hardware (in order seen)"),
})
monthly = source_monthly(con)
order = summ["source_name"].tolist()
status = dict(zip(summ["source_name"], summ["status"]))
fig = go.Figure()
for i, srcname in enumerate(order):
    row = summ.iloc[i]
    excluded = status[srcname] == "excluded"
    col = MUTED if excluded else SERIES[i % len(SERIES)]
    fig.add_scatter(x=[row["first_date"], row["last_date"]], y=[srcname, srcname], mode="lines",
                    line=dict(color=col, width=2), showlegend=False, hoverinfo="skip")
    m = monthly[monthly["source_name"] == srcname]
    fig.add_scatter(x=m["month"], y=[srcname] * len(m), mode="markers", showlegend=False,
                    marker=dict(size=np.clip(3 + 2.2 * np.log10(m["n"].to_numpy() + 1), 4, 14), color=col,
                                symbol="x" if excluded else "circle"),
                    customdata=m["n"], hovertemplate="%{y}<br>%{x|%b %Y}: %{customdata:,} records<extra></extra>")
fig.update_yaxes(categoryorder="array", categoryarray=order[::-1])
st.plotly_chart(style(fig, height=110 + 45 * len(order), title="Active months per source (marker size = record volume; × = excluded)",
                      hovermode="closest"), width="stretch")
hw = hardware_timeline(con)
if not hw.empty:
    st.markdown("**Hardware reported per source.** A new model on your own source usually means a phone upgrade, "
                "which can shift gait metrics. These dates appear as device events on the Changepoints page.")
    st.dataframe(hw, hide_index=True, width="stretch")
with st.expander("Per-type source breakdown and dedup"):
    src = source_breakdown(con)
    st.caption("For each type and day, only the highest-priority source with data is used (see DECISIONS.md R3). "
               "`days_overridden` counts days where this source had data but a higher-priority source won. "
               "Excluded sources show 0 selected days.")
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
