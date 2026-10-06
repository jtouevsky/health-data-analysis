"""Aggregation rules, source priority, unit conversion and day assignment."""

import sqlite3

import numpy as np
import pandas as pd
import pytest

from healthlab.ingest import ingest_export
from healthlab.rollup import load_catalog, load_daily, rebuild_daily
from healthlab.units import convert
from xmlfactory import rec, write_export

Q = "HKQuantityTypeIdentifier"
STEPS, DIST, SPEED = Q + "StepCount", Q + "DistanceWalkingRunning", Q + "WalkingSpeed"
ACTIVE, WEIGHT, AUDIO = Q + "ActiveEnergyBurned", Q + "BodyMass", Q + "HeadphoneAudioExposure"
FLIGHTS, DS = Q + "FlightsClimbed", Q + "WalkingDoubleSupportPercentage"
SLEEP = "HKCategoryTypeIdentifierSleepAnalysis"


def build(tmp_path, cfg, body):
    db = tmp_path / "h.db"
    ingest_export(write_export(tmp_path / "e.zip", body), cfg, db_path=db)
    daily = rebuild_daily(cfg, db_path=db)
    return daily, sqlite3.connect(db)


def test_sum_mean_and_missing_days(tmp_path, cfg):
    body = (rec(STEPS, 100, "2024-05-01 08:00:00 -0700", "2024-05-01 08:10:00 -0700")
            + rec(STEPS, 50, "2024-05-01 12:00:00 -0700", "2024-05-01 12:10:00 -0700")
            + rec(STEPS, 70, "2024-05-03 12:00:00 -0700", "2024-05-03 12:10:00 -0700")
            + rec(SPEED, 4.0, "2024-05-01 09:00:00 -0700", unit="km/hr")
            + rec(SPEED, 5.0, "2024-05-01 10:00:00 -0700", unit="km/hr"))
    d, _ = build(tmp_path, cfg, body)
    assert d.loc["2024-05-01", "steps"] == 150
    assert d.loc["2024-05-01", "walking_speed_kmh"] == pytest.approx(4.5)
    # A day without records is missing, not zero.
    assert np.isnan(d.loc["2024-05-02", "steps"])
    assert d.loc["2024-05-03", "steps"] == 70


def test_sum_prorated_across_local_midnight(tmp_path, cfg):
    # 23:00 -> 01:00 local: half the steps on each day.
    d, _ = build(tmp_path, cfg, rec(STEPS, 200, "2024-05-01 23:00:00 -0700", "2024-05-02 01:00:00 -0700"))
    assert d.loc["2024-05-01", "steps"] == pytest.approx(100)
    assert d.loc["2024-05-02", "steps"] == pytest.approx(100)


def test_unit_conversion(tmp_path, cfg):
    body = (rec(DIST, 1.0, "2024-05-01 08:00:00 -0700", unit="mi")
            + rec(DS, 0.30, "2024-05-01 08:00:00 -0700", unit="%")
            + rec(WEIGHT, 154.32358, "2024-05-01 07:00:00 -0700", unit="lb"))
    d, _ = build(tmp_path, cfg, body)
    assert d.loc["2024-05-01", "distance_km"] == pytest.approx(1.609344)
    assert d.loc["2024-05-01", "double_support_pct"] == pytest.approx(30.0)
    # Unconfigured types still roll up generically (mass -> kg, mean).
    assert d.loc["2024-05-01", "body_mass"] == pytest.approx(70.0, abs=1e-4)
    assert convert(1.0, "Cal", "kcal") == 1.0


def test_source_priority_uses_one_source_per_day(tmp_path, cfg):
    # Phone has more records overall, so it outranks the app by default.
    body = (rec(ACTIVE, 300, "2024-05-01 10:00:00 -0700", unit="kcal", source="Phone")
            + rec(ACTIVE, 100, "2024-05-01 18:00:00 -0700", unit="kcal", source="Phone")
            + rec(ACTIVE, 50, "2024-05-03 18:00:00 -0700", unit="kcal", source="Phone")
            + rec(ACTIVE, 350, "2024-05-01 10:00:00 -0700", unit="kcal", source="App")
            + rec(ACTIVE, 250, "2024-05-02 10:00:00 -0700", unit="kcal", source="App"))
    d, _ = build(tmp_path, cfg, body)
    assert d.loc["2024-05-01", "active_kcal"] == 400       # not 750: no double counting
    assert d.loc["2024-05-02", "active_kcal"] == 250       # fallback when the top source is absent


def test_configured_source_priority_overrides_counts(tmp_path, cfg):
    cfg.raw["source_priority"][ACTIVE] = ["App"]
    body = (rec(ACTIVE, 300, "2024-05-01 10:00:00 -0700", unit="kcal", source="Phone")
            + rec(ACTIVE, 100, "2024-05-01 18:00:00 -0700", unit="kcal", source="Phone")
            + rec(ACTIVE, 350, "2024-05-01 10:00:00 -0700", unit="kcal", source="App"))
    d, _ = build(tmp_path, cfg, body)
    assert d.loc["2024-05-01", "active_kcal"] == 350


def test_leq_is_energy_average(tmp_path, cfg):
    body = (rec(AUDIO, 60, "2024-05-01 10:00:00 -0700", "2024-05-01 10:01:00 -0700", unit="dBASPL")
            + rec(AUDIO, 80, "2024-05-01 11:00:00 -0700", "2024-05-01 11:01:00 -0700", unit="dBASPL"))
    d, _ = build(tmp_path, cfg, body)
    expected = 10 * np.log10((10 ** 6 + 10 ** 8) / 2)
    assert d.loc["2024-05-01", "headphone_db"] == pytest.approx(expected)
    assert d.loc["2024-05-01", "headphone_db"] > 70  # arithmetic mean would understate


def test_sleep_assigned_to_wake_date_and_overlaps_merged(tmp_path, cfg):
    core = "HKCategoryValueSleepAnalysisAsleepCore"
    body = (rec(SLEEP, core, "2024-05-01 23:00:00 -0700", "2024-05-02 03:00:00 -0700", unit=None)
            + rec(SLEEP, core, "2024-05-02 02:00:00 -0700", "2024-05-02 06:00:00 -0700", unit=None)
            + rec(SLEEP, "HKCategoryValueSleepAnalysisInBed", "2024-05-01 22:30:00 -0700", "2024-05-02 06:30:00 -0700", unit=None))
    d, _ = build(tmp_path, cfg, body)
    assert d.loc["2024-05-02", "sleep_asleep_h"] == pytest.approx(7.0)   # 23:00-06:00 merged
    assert d.loc["2024-05-02", "sleep_inbed_h"] == pytest.approx(8.0)
    assert "2024-05-01" not in d.index or np.isnan(d.loc["2024-05-01", "sleep_asleep_h"])


def test_zero_fill_only_on_days_with_anchor(tmp_path, cfg):
    body = (rec(FLIGHTS, 3, "2024-05-01 10:00:00 -0700")
            + rec(STEPS, 100, "2024-05-01 10:00:00 -0700")
            + rec(STEPS, 100, "2024-05-02 10:00:00 -0700")
            + rec(STEPS, 100, "2024-05-04 10:00:00 -0700"))
    d, _ = build(tmp_path, cfg, body)
    assert d.loc["2024-05-02", "flights"] == 0            # phone carried, no flights
    assert np.isnan(d.loc["2024-05-03", "flights"])        # no phone data at all: missing


def test_unknown_type_gets_default_rule(tmp_path, cfg):
    body = (rec(Q + "WidgetDistance", 2.0, "2024-05-01 10:00:00 -0700", unit="mi")
            + rec(Q + "WidgetDistance", 1.0, "2024-05-01 11:00:00 -0700", unit="mi")
            + rec(Q + "WidgetHeartThing", 60, "2024-05-01 10:00:00 -0700", unit="count/min")
            + rec(Q + "WidgetHeartThing", 80, "2024-05-01 11:00:00 -0700", unit="count/min"))
    d, con = build(tmp_path, cfg, body)
    assert d.loc["2024-05-01", "widget_distance"] == pytest.approx(3 * 1.609344)   # cumulative -> sum, km
    assert d.loc["2024-05-01", "widget_heart_thing"] == pytest.approx(70)          # rate -> mean
    cat = load_catalog(con).set_index("metric")
    assert not cat.loc["widget_distance", "configured"]


def test_rollup_is_recomputable(tmp_path, cfg):
    d1, con = build(tmp_path, cfg, rec(STEPS, 100, "2024-05-01 10:00:00 -0700"))
    d2 = rebuild_daily(cfg, con=con)
    pd.testing.assert_frame_equal(d1, d2)
    assert load_daily(con).shape[0] == 1


def test_partial_export_day_dropped(tmp_path, cfg):
    # xmlfactory's ExportDate is 2026-10-05 16:13 -0700 -> local day 2026-10-05 is partial.
    body = (rec(STEPS, 100, "2026-10-04 12:00:00 -0700")
            + rec(STEPS, 200, "2026-10-04 23:00:00 -0700", "2026-10-05 01:00:00 -0700")
            + rec(STEPS, 300, "2026-10-05 09:00:00 -0700"))
    d, _ = build(tmp_path, cfg, body)
    assert d.index.max() == pd.Timestamp("2026-10-04")
    assert d.loc["2026-10-04", "steps"] == pytest.approx(200)  # 100 + half of the spanning sample


def test_excluded_sources_are_ignored(tmp_path, cfg):
    cfg.raw["source_exclude"] = {"all": ["Guest's iPhone"],     # straight apostrophe in config...
                                 ACTIVE: ["App"]}
    body = (rec(STEPS, 100, "2024-05-01 10:00:00 -0700", source="Phone")
            + rec(STEPS, 999, "2024-05-02 10:00:00 -0700", source="Guest’s iPhone")   # ...curly in data
            + rec(ACTIVE, 300, "2024-05-01 10:00:00 -0700", unit="kcal", source="App")
            + rec(STEPS, 50, "2024-05-03 10:00:00 -0700", source="App"))
    d, con = build(tmp_path, cfg, body)
    assert np.isnan(d.loc["2024-05-02", "steps"])           # foreign phone's day is missing, not 999
    assert "active_kcal" not in d.columns                   # App excluded for active energy only
    assert d.loc["2024-05-03", "steps"] == 50               # ...but still used for steps
    assert con.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 4   # raw data untouched


def test_source_summary_reports_ranges_and_status(tmp_path, cfg):
    from healthlab.quality import source_summary

    cfg.raw["source_exclude"] = {"all": ["Old iPod"]}
    body = (rec(STEPS, 100, "2024-05-01 10:00:00 -0700", source="Phone")
            + rec(STEPS, 100, "2024-06-01 10:00:00 -0700", source="Phone")
            + rec(STEPS, 100, "2023-01-01 10:00:00 -0800", source="Old iPod"))
    _, con = build(tmp_path, cfg, body)
    s = source_summary(con, cfg).set_index("source_name")
    assert s.loc["Phone", "first_date"] == "2024-05-01" and s.loc["Phone", "last_date"] == "2024-06-01"
    assert s.loc["Phone", "status"] == "included" and s.loc["Old iPod", "status"] == "excluded"
