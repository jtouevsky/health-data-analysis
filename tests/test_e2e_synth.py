"""End-to-end integration test: synthetic export -> ingest -> rollup -> analyses.

The synthetic generator plants known effects (see healthlab.synth). The pipeline
must run end to end and the analyses must recover them.
"""

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from healthlab.analysis.correlations import scan
from healthlab.analysis.changepoints import CPParams, all_events, bocpd, detect_all, match_events, segment_effects
from healthlab.analysis.forecast import ForecastParams, evaluate
from healthlab.analysis.seasonal import compare_periods, label_days
from healthlab.config import load_config
from healthlab.ingest import ingest_export
from healthlab.rollup import load_catalog, load_daily, rebuild_daily
from healthlab.synth import write_synthetic_export

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    cfg = load_config(demo_mode=True)
    root = tmp_path_factory.mktemp("demo")
    export = root / "export.zip"
    truth = write_synthetic_export(export, cfg)
    db = root / "demo.db"
    first = ingest_export(export, cfg, db_path=db)
    second = ingest_export(export, cfg, db_path=db)
    rebuild_daily(cfg, db_path=db)
    con = sqlite3.connect(db)
    return {"cfg": cfg, "truth": truth, "con": con, "first": first, "second": second,
            "daily": load_daily(con), "raw": load_daily(con), "catalog": load_catalog(con),
            "truth_file": json.loads((root / "truth.json").read_text())}


def test_ingest_complete_and_idempotent(demo):
    t, s1, s2 = demo["truth"], demo["first"], demo["second"]
    assert s1.records_seen == t["n_records"] and s1.parse_errors == 0
    assert s1.records_inserted == t["n_records"]
    assert s2.records_inserted == 0 and s2.workouts_inserted == 0
    assert demo["truth_file"]["n_records"] == t["n_records"]
    # Records rendered with a Bulgarian offset were parsed and kept their offset.
    n_bg = demo["con"].execute("SELECT COUNT(*) FROM records WHERE utc_offset = 180").fetchone()[0]
    assert n_bg > 0


def test_daily_table_shape_and_cutoff(demo):
    d, t = demo["raw"], demo["truth"]
    assert str(d.index.min().date()) == t["start_date"]
    assert str(d.index.max().date()) == t["end_date"]
    expected = {"steps", "distance_km", "flights", "active_kcal", "basal_kcal", "walking_speed_kmh",
                "step_length_cm", "double_support_pct", "asymmetry_pct", "headphone_db", "sleep_asleep_h"}
    assert expected <= set(d.columns)
    # Out of scope: no nutrition / weight columns or coverage flags are produced.
    assert not {"intake_kcal", "weight_kg", "well_logged", "n_food_entries"} & set(d.columns)
    # Imperial raw units come back metric.
    assert d["step_length_cm"].dropna().between(55, 90).all()
    assert d["walking_speed_kmh"].dropna().between(3, 6.5).all()


def test_source_dedup_avoids_double_counting(demo):
    t, d = demo["truth"], demo["raw"]
    # DemoFit duplicates some samples; summing both sources would inflate the mean ~5-10 %.
    assert d["active_kcal"].mean() == pytest.approx(t["active_kcal_mean"], rel=0.03)
    used = demo["con"].execute(
        "SELECT DISTINCT source_used FROM daily_sources WHERE metric='active_kcal'").fetchall()
    assert used == [("Demo iPhone",)]


def test_sleep_only_recent(demo):
    s = demo["raw"]["sleep_asleep_h"].dropna()
    assert 12 <= len(s) <= 15 and s.between(6, 9).all()


def test_planted_lagged_correlations_recovered(demo):
    cfg, d = demo["cfg"], demo["daily"]
    cc = cfg.section("correlations")
    metrics = [m for m in demo["catalog"]["metric"] if d[m].notna().sum() >= 30]
    res = scan(d, metrics, max_lag=3, min_n=60, method="detrend", definitional=cc["definitional_pairs"])
    for pl in demo["truth"]["planted_lags"]:
        sub = res[(res["x"] == pl["x"]) & (res["y"] == pl["y"])]
        best = sub.loc[sub["rho"].abs().idxmax()]
        assert int(best["lag"]) == pl["lag"], f"{pl} recovered at lag {best['lag']}"
        assert np.sign(best["rho"]) == pl["sign"] and abs(best["rho"]) > 0.25 and best["q"] < 1e-3
        # Reverse direction at the same lag is much weaker.
        rev = res[(res["x"] == pl["y"]) & (res["y"] == pl["x"]) & (res["lag"] == pl["lag"])]
        assert abs(rev["rho"].iat[0]) < abs(best["rho"]) / 2


def test_summer_effect_recovered(demo):
    cfg, d = demo["cfg"], demo["daily"]
    labels = label_days(d.index, cfg.periods)
    res = compare_periods(d, ["steps"], labels, "bulgaria_summer", "school_year", reps=300).iloc[0]
    assert res["cliffs_delta"] > 0.3 and res["ci_block_lo"] > 0 and res["q"] < 1e-6


def test_forecast_runs_on_demo(demo):
    ev = evaluate(demo["daily"]["steps"], ForecastParams(n_splits=3))
    assert set(ev.summary["model"]) == {"naive", "seasonal_naive", "sarimax", "gbm"}
    assert np.isfinite(ev.summary["mae"]).all()


def test_timezone_overrides_put_summer_samples_on_the_right_day(demo):
    """Summer days are lived in Sofia time; only the override recovers their daily totals."""
    from healthlab.rollup import build_daily, load_records

    cfg, t = demo["cfg"], demo["truth"]
    truth = pd.Series(t["daily_steps"], dtype=float)
    truth.index = pd.to_datetime(truth.index)
    summer = pd.Series(False, index=truth.index)
    for s, e in t["summer_periods"]:
        summer.loc[s:e] = True
    check = summer & ~truth.index.isin(pd.to_datetime(t["travel_days"])) & truth.notna()

    got = demo["raw"]["steps"].reindex(truth.index)
    rel_err = ((got - truth).abs() / truth)[check]
    assert rel_err.median() < 0.01                      # exact up to per-sample rounding

    # Same raw records, overrides removed: summer days are split across LA dates.
    cfg_no = load_config(demo_mode=True)
    cfg_no.raw["demo"]["timezone_overrides"] = []
    daily_no, _, _ = build_daily(load_records(demo["con"], cfg_no), cfg_no)
    rel_err_no = ((daily_no["steps"].reindex(truth.index) - truth).abs() / truth)[check]
    # Only the (low-activity) Sofia-morning hours move to the neighbouring LA date and
    # neighbouring days have similar totals, so the error is moderate but systematic.
    assert rel_err_no.median() > 0.04 and rel_err_no.median() > 5 * rel_err.median()
    assert (rel_err_no > 0.2).mean() > 0.15


def test_foreign_source_excluded_and_device_upgrade_visible(demo):
    from healthlab.quality import hardware_timeline, source_summary
    from healthlab.rollup import build_daily, load_records

    con, cfg, t = demo["con"], demo["cfg"], demo["truth"]
    used = {r[0] for r in con.execute("SELECT DISTINCT source_used FROM daily_sources")}
    assert "Partner's iPhone" not in used
    missing = [d for d, v in t["daily_steps"].items() if v is None and d not in t["travel_days"]]
    assert len(missing) > 5
    assert demo["raw"]["steps"].reindex(pd.to_datetime(missing)).isna().all()
    # Without the exclusion, the partner's steps would fill those days.
    cfg_inc = load_config(demo_mode=True)
    cfg_inc.raw["demo"]["source_exclude"] = {"all": []}
    d_inc, _, _ = build_daily(load_records(con, cfg_inc), cfg_inc)
    assert d_inc["steps"].reindex(pd.to_datetime(missing)).notna().all()

    s = source_summary(con, cfg).set_index("source_name")
    assert s.loc["Partner's iPhone", "status"] == "excluded"
    hw = hardware_timeline(con)
    phone = hw[hw["source_name"] == "Demo iPhone"]
    assert phone["hardware"].tolist() == ["iPhone12,1", "iPhone15,2"]
    upgrade = pd.Timestamp(t["start_date"]) + pd.Timedelta(days=t["params"]["phone_upgrade_day"])
    assert abs((pd.Timestamp(phone["first_date"].iloc[1]) - upgrade).days) <= 1


def test_planted_changepoints_recovered(demo):
    """Every planted regime shift is found by PELT within ±21 days; few unexplained extras."""
    cfg, d, t = demo["cfg"], demo["daily"], demo["truth"]
    cc = cfg.section("changepoints")
    p = CPParams.from_config(cc)
    res = detect_all(d, cc["metrics"], p)
    tol = pd.Timedelta(days=21)
    events = all_events(demo["con"], cfg)
    assert set(events["kind"]) == {"life event", "device", "travel"}
    for metric, planted in t["planted_changepoints"].items():
        found = res[metric].changepoints
        for date in pd.to_datetime(planted):
            assert any(abs(f - date) <= tol for f in found), f"{metric}: planted {date.date()} not found in {found}"
        # Extras must be explainable by a known event (e.g. the commute lowering walking speed
        # through the planted steps -> speed link); at most one unexplained per metric.
        unexplained = match_events(found, events, 21)["event"].isna().sum()
        assert unexplained <= 1, f"{metric}: {unexplained} unexplained changepoints"


def test_changepoint_effects_and_bocpd_agree_with_truth(demo):
    cfg, d, t = demo["cfg"], demo["daily"], demo["truth"]
    p = CPParams.from_config(cfg.section("changepoints"))
    res = detect_all(d, ["steps", "asymmetry_pct"], p)
    # Commute: steps fall by ~30 % (Cliff's delta clearly negative, CI excludes 0).
    commute = pd.Timestamp(t["planted_changepoints"]["steps"][2])
    eff = segment_effects(d["steps"], res["steps"], p)
    row = eff.loc[(eff["changepoint"] - commute).abs().idxmin()]
    assert row["pct_change"] < -0.15 and row["ci_hi"] < 0 and row["cliffs_delta"] < -0.3
    # Injury: asymmetry rises; BOCPD independently finds both injury boundaries.
    trace, bc = bocpd(res["asymmetry_pct"].z, p.bocpd_expected_run, p.bocpd_min_run)
    for date in pd.to_datetime(t["planted_changepoints"]["asymmetry_pct"]):
        assert any(abs(b - date) <= pd.Timedelta(days=21) for b in bc)
    assert trace["p_new_regime"].between(0, 1).all()
