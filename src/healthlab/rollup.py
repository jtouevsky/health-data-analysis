"""Daily rollups: raw records -> one row per local date, one column per metric.

The ``daily`` table is fully derived and is rebuilt from ``records`` on every
run, so changing an aggregation rule or source priority only requires
``healthlab rollup``.

Pipeline per HealthKit type:
1. pick the aggregation rule (config, or a default derived from the unit);
2. convert raw values to the rule's canonical unit;
3. **source dedup**: per local day keep only the highest-priority source with data;
4. aggregate to one value per day (sum types are prorated across local midnight).

A day with no records is NaN (missing), never zero, except for metrics marked
``zero_fill`` (e.g. flights climbed), which are 0 on days the phone was evidently
carried (the ``zero_fill_anchor`` metric has data). Nutrition days additionally
get a logging-coverage flag (``n_food_entries``, ``well_logged``).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, MetricRule
from .db import connect, table_exists
from .units import canonical_unit, convert

ASLEEP_VALUES = {
    "HKCategoryValueSleepAnalysisAsleep",
    "HKCategoryValueSleepAnalysisAsleepUnspecified",
    "HKCategoryValueSleepAnalysisAsleepCore",
    "HKCategoryValueSleepAnalysisAsleepDeep",
    "HKCategoryValueSleepAnalysisAsleepREM",
}
INBED_VALUES = {"HKCategoryValueSleepAnalysisInBed"}

AGGREGATORS = ("sum", "mean", "last", "max", "min", "leq", "count", "sleep")


# ---------------------------------------------------------------------------
# Loading raw records
# ---------------------------------------------------------------------------

def load_records(con: sqlite3.Connection, cfg: Config) -> pd.DataFrame:
    """Load the raw columns the rollup needs, with local start/end datetimes."""
    skip = list(cfg.section("ingest").get("skip_types") or [])
    q = ("SELECT type, source_name, unit, value, value_text, start_ts, end_ts, utc_offset, "
         "local_date, local_end_date FROM records")
    if skip:
        q += f" WHERE type NOT IN ({','.join('?' * len(skip))})"
    df = pd.read_sql_query(q, con, params=skip)
    if df.empty:
        return df
    # Local times are recomputed from UTC with the *current* config (default zone +
    # override periods), so editing the overrides only requires `healthlab rollup`.
    resolver = cfg.zone_resolver()
    start = pd.to_datetime(df["start_ts"], utc=True)
    end = pd.to_datetime(df["end_ts"], utc=True)
    off = df["utc_offset"].fillna(0)
    df["local_start"] = resolver.localize(start, off)
    df["local_end"] = resolver.localize(end, off)
    df["local_date"] = df["local_start"].dt.strftime("%Y-%m-%d")
    df["local_end_date"] = df["local_end"].dt.strftime("%Y-%m-%d")
    df["source_name"] = df["source_name"].fillna("(unknown)")
    return drop_excluded_sources(df, cfg)


def drop_excluded_sources(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Remove records from sources excluded in config (globally or for their type)."""
    everywhere, per_type = cfg.source_exclusions
    if not everywhere and not per_type:
        return df
    pairs = df[["type", "source_name"]].drop_duplicates()
    bad = {(t, s) for t, s in pairs.itertuples(index=False) if cfg.is_excluded(s, t)}
    if not bad:
        return df
    key = pd.MultiIndex.from_frame(df[["type", "source_name"]])
    return df[~key.isin(list(bad))]


# ---------------------------------------------------------------------------
# Source priority
# ---------------------------------------------------------------------------

def source_ranking(df_type: pd.DataFrame, hk_type: str, cfg: Config) -> list[str]:
    """Ordered source list for a type: configured priority first, then by record count."""
    prio = cfg.section("source_priority")
    configured = list(prio.get(hk_type) or []) + list(prio.get("default") or [])
    counts = df_type["source_name"].value_counts()
    # Ties broken alphabetically so the ranking is deterministic.
    ordered = sorted(counts.index, key=lambda s: (-counts[s], s))
    rest = [s for s in ordered if s not in configured]
    return [s for s in configured if s in counts.index] + rest


def select_sources(df_type: pd.DataFrame, day_col: str, ranking: list[str]) -> pd.DataFrame:
    """Keep, for each day, only rows from the best-ranked source present that day."""
    rank = {s: i for i, s in enumerate(ranking)}
    r = df_type["source_name"].map(rank).fillna(len(rank)).astype(int)
    best = r.groupby(df_type[day_col]).transform("min")
    return df_type[r == best]


# ---------------------------------------------------------------------------
# Aggregators
# ---------------------------------------------------------------------------

def prorate_across_midnight(df: pd.DataFrame) -> pd.DataFrame:
    """Split rows whose local interval crosses midnight, prorating ``value`` by time.

    Returns rows with a ``day`` column (local date). Rows within one day are
    unchanged. Zero-length intervals keep their full value on the start day, as do
    the rare samples whose local end precedes their local start (a sample that
    straddles the boundary of a time-zone override period).
    """
    d0 = df["local_start"].dt.normalize()
    d1 = df["local_end"].dt.normalize()
    one = d1 <= d0
    single = df[one].assign(day=d0[one])
    multi = df[~one]
    if multi.empty:
        return single
    pieces = []
    for row in multi.itertuples(index=False):
        s, e = row.local_start, row.local_end
        total = (e - s).total_seconds()
        for day in pd.date_range(s.normalize(), e.normalize(), freq="D"):
            lo, hi = max(s, day), min(e, day + pd.Timedelta(days=1))
            frac = (hi - lo).total_seconds() / total if total > 0 else 0.0
            if frac > 0:
                pieces.append({**row._asdict(), "value": row.value * frac, "day": day})
    return pd.concat([single, pd.DataFrame(pieces)], ignore_index=True)


def _merged_hours(intervals: pd.DataFrame) -> float:
    """Total hours covered by possibly-overlapping [start, end) intervals."""
    iv = intervals.sort_values("local_start")[["local_start", "local_end"]].to_numpy()
    total = pd.Timedelta(0)
    cur_s, cur_e = iv[0]
    for s, e in iv[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return total.total_seconds() / 3600.0


def aggregate(df: pd.DataFrame, rule: MetricRule) -> pd.DataFrame:
    """Aggregate one type's (source-deduplicated) rows to a daily frame.

    Returns a DataFrame indexed by day (Timestamp) with one or more value columns.
    """
    agg = rule.agg
    if agg == "sum":
        p = prorate_across_midnight(df)
        return p.groupby("day")["value"].sum(min_count=1).to_frame(rule.name)
    day = df["local_start"].dt.normalize()
    if agg in ("mean", "max", "min"):
        return df.groupby(day)["value"].agg(agg).to_frame(rule.name)
    if agg == "last":
        s = df.sort_values("local_start")
        return s.groupby(s["local_start"].dt.normalize())["value"].last().to_frame(rule.name)
    if agg == "count":
        return df.groupby(day).size().astype(float).to_frame(rule.name)
    if agg == "leq":
        # Energy-average sound level weighted by duration (LAeq). Arithmetic means of
        # decibels understate exposure because dB is logarithmic.
        w = (df["local_end"] - df["local_start"]).dt.total_seconds().clip(lower=1.0)
        e = w * np.power(10.0, df["value"] / 10.0)
        g = pd.DataFrame({"e": e, "w": w, "day": day}).groupby("day")[["e", "w"]].sum()
        return (10.0 * np.log10(g["e"] / g["w"])).to_frame(rule.name)
    if agg == "sleep":
        # Sleep is assigned to the local date you wake up (interval end).
        wake = df["local_end"].dt.normalize()
        out = {}
        for col, values in ((f"{rule.name}_asleep_h", ASLEEP_VALUES), (f"{rule.name}_inbed_h", INBED_VALUES)):
            sub = df[df["value_text"].isin(values)]
            if not sub.empty:
                out[col] = sub.groupby(wake.loc[sub.index]).apply(_merged_hours)
        return pd.DataFrame(out)
    raise ValueError(f"unknown aggregation {agg!r} for {rule.hk_type}")


# ---------------------------------------------------------------------------
# Building the daily table
# ---------------------------------------------------------------------------

def resolve_rule(df_type: pd.DataFrame, hk_type: str, cfg: Config) -> MetricRule:
    """Rule for a type; for unconfigured types the unit is the canonical form of the modal raw unit."""
    units = df_type["unit"].dropna()
    raw_unit = units.mode().iat[0] if not units.empty else None
    is_category = df_type["value"].isna().all()
    rule = cfg.rule_for(hk_type, canonical_unit(raw_unit), is_category)
    return rule


def build_daily(records: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute the daily table from raw records.

    Returns:
        ``(daily, catalog, sources)``: the wide daily table indexed by date; one
        catalog row per metric column; and per (date, metric) the source used and
        sample count (for the data-quality page).
    """
    frames: list[pd.DataFrame] = []
    catalog: list[dict] = []
    src_rows: list[pd.DataFrame] = []
    used_names: set[str] = set()
    zero_fill: list[str] = []

    for hk_type, g in records.groupby("type", sort=True):
        rule = resolve_rule(g, hk_type, cfg)
        name = rule.name
        while name in used_names:  # two types mapping to the same short name
            name += "_"
        used_names.add(name)
        if name != rule.name:
            rule = MetricRule(**{**rule.__dict__, "name": name})

        g = g.copy()
        if rule.agg not in ("count", "sleep"):
            g = g[g["value"].notna()]
            if g.empty:
                continue
            # Convert each raw unit to the canonical unit (a type can mix units).
            vals = g["value"].to_numpy(dtype=float, copy=True)
            for u in g["unit"].dropna().unique():
                m = (g["unit"] == u).to_numpy()
                vals[m] = convert(vals[m], u, rule.unit)
            g["value"] = vals

        day_col = "local_end_date" if rule.agg == "sleep" else "local_date"
        ranking = source_ranking(g, hk_type, cfg)
        g = select_sources(g, day_col, ranking)
        daily_t = aggregate(g, rule)
        if daily_t.empty:
            continue
        frames.append(daily_t)
        if rule.zero_fill:
            zero_fill.extend(daily_t.columns)

        per_day = g.groupby(day_col).agg(source_used=("source_name", "first"), n_samples=("value", "size"))
        per_day.index = pd.to_datetime(per_day.index)
        src_rows.append(per_day.assign(metric=rule.name, hk_type=hk_type).reset_index(names="date"))

        for col in daily_t.columns:
            s = daily_t[col].dropna()
            catalog.append({
                "metric": col, "hk_type": hk_type, "agg": rule.agg, "unit": rule.unit,
                "kind": rule.kind, "label": rule.label if col == rule.name else col,
                "configured": rule.configured, "zero_fill": rule.zero_fill, "sources_ranked": " > ".join(ranking),
                "n_days": int(s.size),
                "first_date": s.index.min().date().isoformat() if s.size else None,
                "last_date": s.index.max().date().isoformat() if s.size else None,
                "plausible_lo": rule.plausible[0] if rule.plausible else None,
                "plausible_hi": rule.plausible[1] if rule.plausible else None,
            })

    if not frames:
        return pd.DataFrame(), pd.DataFrame(catalog), pd.DataFrame()

    daily = pd.concat(frames, axis=1, sort=True)
    daily.index = pd.DatetimeIndex(daily.index, name="date")
    # Continuous calendar so gaps are explicit NaN rows.
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D", name="date"))

    anchor = cfg.section("rollup").get("zero_fill_anchor")
    if anchor in daily.columns:
        for col in zero_fill:
            first = daily[col].first_valid_index()
            fill = daily[col].isna() & daily[anchor].notna() & (daily.index >= first)
            daily.loc[fill, col] = 0.0

    sources = pd.concat(src_rows, ignore_index=True) if src_rows else pd.DataFrame()
    daily = add_nutrition_coverage(daily, sources, cfg)
    return daily, pd.DataFrame(catalog), sources


def add_nutrition_coverage(daily: pd.DataFrame, sources: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add ``n_food_entries`` and ``well_logged`` if an intake metric exists.

    A day is *well logged* when it has at least ``min_entries`` energy entries and
    a total intake inside [``min_kcal``, ``max_kcal``]. Days with no entries have
    NaN intake (missing), never zero.
    """
    n = cfg.section("nutrition")
    energy = n.get("energy_metric", "intake_kcal")
    if energy not in daily.columns:
        return daily
    entries = pd.Series(0.0, index=daily.index)
    if not sources.empty:
        e = sources[sources["metric"] == energy].set_index("date")["n_samples"]
        entries = e.reindex(daily.index).fillna(0.0)
    daily = daily.copy()
    daily["n_food_entries"] = entries.astype(int)
    kcal = daily[energy]
    daily["well_logged"] = (
        (daily["n_food_entries"] >= int(n.get("min_entries", 3)))
        & kcal.between(float(n.get("min_kcal", 1200)), float(n.get("max_kcal", 5000)))
    )
    return daily


def export_cutoff(con: sqlite3.Connection, cfg: Config) -> pd.Timestamp | None:
    """Local date of the most recent export: that day (and anything later) is incomplete."""
    row = con.execute("SELECT MAX(export_date) FROM import_runs WHERE status='ok'").fetchone()
    if not row or not row[0]:
        return None
    ts = pd.Timestamp(row[0], tz="UTC").to_pydatetime()
    return pd.Timestamp(cfg.zone_resolver().local_date(ts))


def rebuild_daily(cfg: Config, db_path: str | Path | None = None,
                  con: sqlite3.Connection | None = None) -> pd.DataFrame:
    """Recompute ``daily``, ``metric_catalog`` and ``daily_sources`` from raw records.

    The export day itself is dropped: it was exported mid-day, so its totals are
    partial and would look like an outlier (and poison naive forecasts).
    """
    own = con is None
    con = con or connect(db_path or cfg.db_path)
    try:
        records = load_records(con, cfg)
        cutoff = export_cutoff(con, cfg)
        if cutoff is not None and not records.empty:
            records = records[records["local_start"] < cutoff]
        daily, catalog, sources = build_daily(records, cfg) if not records.empty else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        if cutoff is not None and not daily.empty:
            # Samples spanning midnight can prorate into the cutoff day; drop it entirely.
            daily = daily[daily.index < cutoff]
            sources = sources[sources["date"] < cutoff] if not sources.empty else sources
        out = daily.copy()
        if not out.empty:
            out.index = out.index.strftime("%Y-%m-%d")
            if "well_logged" in out:
                out["well_logged"] = out["well_logged"].astype(int)
        out.to_sql("daily", con, if_exists="replace", index=True, index_label="date")
        catalog.to_sql("metric_catalog", con, if_exists="replace", index=False)
        if not sources.empty:
            sources = sources.assign(date=sources["date"].dt.strftime("%Y-%m-%d"))
        sources.to_sql("daily_sources", con, if_exists="replace", index=False)
        con.commit()
        return daily
    finally:
        if own:
            con.close()


# ---------------------------------------------------------------------------
# Reading the derived tables (used by analyses and the app)
# ---------------------------------------------------------------------------

def nutrition_metrics(catalog: pd.DataFrame) -> list[str]:
    """Metric columns of kind ``nutrition``."""
    if catalog.empty:
        return []
    return catalog.loc[catalog["kind"] == "nutrition", "metric"].tolist()


def load_catalog(con: sqlite3.Connection) -> pd.DataFrame:
    """The ``metric_catalog`` table (empty frame if the rollup has not run)."""
    if not table_exists(con, "metric_catalog"):
        return pd.DataFrame()
    return pd.read_sql_query("SELECT * FROM metric_catalog", con)


def load_daily(con: sqlite3.Connection, mask_nutrition: bool = True,
               start_date: str | None = None) -> pd.DataFrame:
    """Load the daily table with a DatetimeIndex.

    Args:
        mask_nutrition: set nutrition metrics to NaN on days that are not
            ``well_logged``, so every downstream analysis uses well-logged days only.
        start_date: drop rows before this date (``analysis.start_date``).
    """
    if not table_exists(con, "daily"):
        return pd.DataFrame()
    df = pd.read_sql_query("SELECT * FROM daily", con, index_col="date")
    if df.empty:
        return df
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index), name="date")
    if "well_logged" in df:
        df["well_logged"] = df["well_logged"].fillna(0).astype(bool)
        if mask_nutrition:
            cols = [c for c in nutrition_metrics(load_catalog(con)) if c in df]
            df.loc[~df["well_logged"], cols] = np.nan
    if start_date:
        df = df.loc[pd.Timestamp(start_date):]
    return df
