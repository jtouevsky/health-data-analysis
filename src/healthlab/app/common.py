"""Shared helpers for Streamlit pages: config, cached data access, units, styling."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from healthlab.config import Config, get_config
from healthlab.rollup import load_catalog, load_daily
from healthlab.units import display

# Validated categorical order (dataviz reference palette); colour follows the
# entity, never its rank.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MUTED = "#9a9993"
SEQ_BLUE = [[0.0, "rgba(128,128,128,0.10)"], [0.25, "#b7d3f6"], [0.5, "#6da7ec"], [0.75, "#2a78d6"], [1.0, "#104281"]]
DIVERGING = [[0.0, "#e34948"], [0.5, "#f0efec"], [1.0, "#2a78d6"]]  # negative red, positive blue

NOT_CAUSAL = ("**Hypothesis generation, not proof.** These are associations in one person's "
              "observational data. Confounding (weather, travel, schedule, phone carry habits) "
              "can produce any of them; none establishes causation.")


def cfg() -> Config:
    return get_config()


def _db_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def _load(db: str, mtime: float, mask: bool, start: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    con = sqlite3.connect(db)
    try:
        return load_daily(con, mask_nutrition=mask, start_date=start), load_catalog(con)
    finally:
        con.close()


def data(mask_nutrition: bool = True, respect_start: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(daily, catalog) for the active database, cached until the DB file changes."""
    c = cfg()
    start = c.section("analysis").get("start_date") if respect_start else None
    return _load(str(c.db_path), _db_mtime(c.db_path), mask_nutrition, start)


@st.cache_data(show_spinner=False)
def _query(db: str, mtime: float, sql: str) -> pd.DataFrame:
    con = sqlite3.connect(db)
    try:
        return pd.read_sql_query(sql, con)
    finally:
        con.close()


def query(sql: str) -> pd.DataFrame:
    c = cfg()
    return _query(str(c.db_path), _db_mtime(c.db_path), sql)


def connection() -> sqlite3.Connection:
    return sqlite3.connect(cfg().db_path)


def require_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load data or stop the page with setup instructions."""
    daily, catalog = data()
    if daily.empty:
        c = cfg()
        st.warning(f"No data in `{c.db_path}`.")
        st.markdown("Import your export with `healthlab ingest data/raw/export.zip`, "
                    "or run the synthetic demo with `healthlab demo`.")
        st.stop()
    return daily, catalog


def analysable_metrics(daily: pd.DataFrame, catalog: pd.DataFrame) -> list[str]:
    """Metrics with enough observed days for analysis, in catalog order."""
    min_days = int(cfg().section("analysis").get("min_coverage_days", 30))
    return [m for m in catalog["metric"] if m in daily and daily[m].notna().sum() >= min_days]


# -- units -------------------------------------------------------------------

def unit_system() -> str:
    return st.session_state.get("unit_system", "metric")


def meta(catalog: pd.DataFrame, metric: str) -> dict:
    row = catalog.loc[catalog["metric"] == metric]
    return row.iloc[0].to_dict() if not row.empty else {"metric": metric, "label": metric, "unit": None}


def to_display(values: pd.Series | float, metric: str, catalog: pd.DataFrame):
    """Convert canonical values to the selected unit system; returns (values, unit label)."""
    unit, factor = display(meta(catalog, metric).get("unit"), unit_system())
    return values * factor, unit or ""


def label(metric: str, catalog: pd.DataFrame, with_unit: bool = True) -> str:
    m = meta(catalog, metric)
    unit, _ = display(m.get("unit"), unit_system())
    base = m.get("label") or metric
    return f"{base} ({unit})" if with_unit and unit and unit != "count" else base


# -- optional module -----------------------------------------------------------

def has_metric(daily: pd.DataFrame, name: str, min_days: int = 1) -> bool:
    return name in daily and daily[name].notna().sum() >= min_days


def optional_module_notice(what: str, needs: str) -> None:
    st.info(
        f"**{what} is an optional module and your database doesn't have the data it needs.** "
        f"It requires {needs}. Activity, gait and audio analyses on the other pages don't depend on it. "
        "To see it working, run the synthetic demo: `healthlab demo`.", icon="ℹ️")


def style(fig: go.Figure, height: int = 380, **kw) -> go.Figure:
    """Consistent, recessive chart chrome."""
    layout = dict(height=height, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_layout(**{**layout, **kw})
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig
