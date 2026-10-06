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
from healthlab.analysis.energy import EnergyParams, expected_trajectory, implied_daily_discrepancy, summarize, window_tdee
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
            "daily": load_daily(con), "raw": load_daily(con, mask_nutrition=False), "catalog": load_catalog(con),
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
    expected = {"steps", "distance_km", "active_kcal", "basal_kcal", "walking_speed_kmh", "intake_kcal",
                "protein_g", "weight_kg", "sleep_asleep_h", "well_logged", "n_food_entries"}
    assert expected <= set(d.columns)
    # Weight came in as lb and must be back in kg.
    assert d["weight_kg"].between(60, 100).all() or d["weight_kg"].dropna().between(60, 100).all()


def test_source_dedup_avoids_double_counting(demo):
    t, d = demo["truth"], demo["raw"]
    P = t["params"]
    apple_active_truth = t["apple_tdee_mean"] - P["true_basal_kcal"] - P["apple_basal_offset"]
    assert d["active_kcal"].mean() == pytest.approx(apple_active_truth, rel=0.04)
    used = demo["con"].execute(
        "SELECT DISTINCT source_used FROM daily_sources WHERE metric='active_kcal'").fetchall()
    assert used == [("Demo iPhone",)]


def test_logging_coverage(demo):
    d, t = demo["raw"], demo["truth"]
    share = d["well_logged"].mean()
    assert abs(share - t["share_well_logged_planned"]) < 0.06
    assert d.loc[d["n_food_entries"] == 0, "intake_kcal"].isna().all()          # missing, not zero
    masked = demo["daily"]
    assert masked.loc[~d["well_logged"], "intake_kcal"].isna().all()


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


def test_energy_balance_recovers_truth(demo):
    cfg, d, t = demo["cfg"], demo["daily"], demo["truth"]
    base = EnergyParams.from_config(cfg.section("energy"), cfg.section("trends"))
    naive = summarize(window_tdee(d, base), base)
    adj_p = EnergyParams(**{**base.__dict__, "underreport": t["params"]["underreport"]})
    adj = summarize(window_tdee(d, adj_p), adj_p)
    assert adj["tdee_mean"] == pytest.approx(t["true_tdee_mean"], abs=150)
    assert naive["tdee_mean"] < t["true_tdee_mean"] - 200          # under-reporting bias is visible
    assert adj["apple_mean"] == pytest.approx(t["apple_tdee_mean"], rel=0.03)
    assert naive["gap_mean"] < 0                                     # Apple over-estimates
    traj = expected_trajectory(d, base, n_sims=100)
    assert implied_daily_discrepancy(traj, base.kcal_per_kg) < -150   # predicts too much loss


def test_ewma_estimator_also_reasonable(demo):
    cfg, d, t = demo["cfg"], demo["daily"], demo["truth"]
    p = EnergyParams.from_config(cfg.section("energy"), cfg.section("trends"), estimator="ewma",
                                 underreport=t["params"]["underreport"])
    assert summarize(window_tdee(d, p), p)["tdee_mean"] == pytest.approx(t["true_tdee_mean"], abs=250)


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
