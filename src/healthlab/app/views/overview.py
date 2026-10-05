"""Overview: what's in the database and headline numbers."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from healthlab.analysis.trend import rolling
from healthlab.app.common import (SERIES, MUTED, analysable_metrics, cfg, has_metric, label, query,
                                  require_data, style, to_display)

st.title("HealthLab")
st.caption("Local-first statistical analysis of an Apple Health export.")

daily, catalog = require_data()
runs = query("SELECT * FROM import_runs WHERE status='ok' ORDER BY id DESC")
n_records = query("SELECT COUNT(*) AS n FROM records")["n"].iat[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Records", f"{n_records:,}")
c2.metric("Days covered", f"{daily.dropna(how='all').shape[0]:,}")
c3.metric("Metrics", f"{len(catalog)}")
c4.metric("History", f"{daily.index.min():%Y}–{daily.index.max():%Y}", help=f"{daily.index.min():%d %b %Y} to {daily.index.max():%d %b %Y}")
if not runs.empty:
    r = runs.iloc[0]
    st.caption(f"Last import #{r['id']} finished {r['finished_at']} UTC · export dated {r['export_date']} UTC · "
               f"{r['n_records_inserted']:,} new of {r['n_records_seen']:,} records.")

# -- Recent vs previous year --------------------------------------------------
st.subheader("Last 90 days vs the same 90 days a year earlier")
end = daily.index.max()
recent = daily.loc[end - pd.Timedelta(days=89): end]
prior = daily.loc[end - pd.Timedelta(days=365 + 89): end - pd.Timedelta(days=365)]
rows = []
for m in analysable_metrics(daily, catalog):
    a, b = recent[m].median(), prior[m].median()
    if pd.notna(a):
        da, unit = to_display(a, m, catalog)
        db = to_display(b, m, catalog)[0] if pd.notna(b) else None
        rows.append({"Metric": label(m, catalog), "Median, last 90 d": round(da, 2),
                     "Median, year earlier": None if db is None else round(db, 2),
                     "Change": None if db is None or db == 0 else f"{(da - db) / abs(db):+.0%}",
                     "Days observed (90 d)": int(recent[m].notna().sum())})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# -- Steps at a glance ----------------------------------------------------------
if has_metric(daily, "steps"):
    s = daily["steps"]
    fig = go.Figure()
    fig.add_scatter(x=s.index, y=s, mode="markers", name="Daily", marker=dict(size=3, color=MUTED, opacity=0.5))
    fig.add_scatter(x=s.index, y=rolling(s, 28), name="28-day mean", line=dict(color=SERIES[0], width=2))
    st.plotly_chart(style(fig, title="Daily steps"), width="stretch")

# -- Module status ------------------------------------------------------------
st.subheader("Modules")
energy = cfg().section("nutrition").get("energy_metric", "intake_kcal")
has_food = has_metric(daily, energy)
has_weight = has_metric(daily, "weight_kg")
st.markdown(
    f"- **Activity, gait and audio analyses**: {'available' if len(analysable_metrics(daily, catalog)) else 'no data'}\n"
    f"- **Nutrition logging**: {'available, ' + str(int(daily['well_logged'].sum())) + ' well-logged days' if has_food else 'not present in this export'}\n"
    f"- **Body weight**: {'available, ' + str(int(daily['weight_kg'].notna().sum())) + ' days with weigh-ins' if has_weight else 'not present in this export'}\n"
    f"- **Energy balance** (needs both): {'available' if has_food and has_weight else 'inactive'}\n"
    f"- **Sleep**: {'present' if has_metric(daily, 'sleep_asleep_h') else 'no history yet'} (stored, not analysed until there's history)")

with st.expander("Metric catalog"):
    st.dataframe(catalog, hide_index=True, width="stretch")
