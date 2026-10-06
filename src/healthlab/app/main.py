"""HealthLab Streamlit entry point. Run with ``healthlab app`` (or ``healthlab demo``)."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from healthlab.app.common import cfg

st.set_page_config(page_title="HealthLab", page_icon="📈", layout="wide")

HERE = Path(__file__).parent / "views"
pages = [
    st.Page(HERE / "overview.py", title="Overview", icon="🏠", default=True),
    st.Page(HERE / "data_quality.py", title="Data Quality", icon="🧪"),
    st.Page(HERE / "trends.py", title="Trends", icon="📈"),
    st.Page(HERE / "correlations.py", title="Correlation Scan", icon="🔗"),
    st.Page(HERE / "seasonal.py", title="Seasonal Comparison", icon="☀️"),
    st.Page(HERE / "changepoints.py", title="Changepoints", icon="📍"),
    st.Page(HERE / "forecasts.py", title="Forecasts", icon="🔮"),
]

with st.sidebar:
    c = cfg()
    if c.demo_mode:
        st.success("**DEMO MODE**: synthetic data", icon="🧬")
    st.radio("Units", ["metric", "imperial"], key="unit_system", horizontal=True,
             format_func=lambda s: "kg · km" if s == "metric" else "lb · mi")
    st.caption(f"Database: `{c.db_path.name}` · runs locally, no network calls.")

st.navigation(pages).run()
