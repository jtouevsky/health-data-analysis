"""Nutrition logging coverage: missing is not zero; analyses see well-logged days only."""

import sqlite3

import numpy as np
import pandas as pd

from healthlab.ingest import ingest_export
from healthlab.quality import gap_runs, outlier_flags
from healthlab.rollup import load_daily, rebuild_daily
from xmlfactory import rec, write_export

KCAL = "HKQuantityTypeIdentifierDietaryEnergyConsumed"
PROT = "HKQuantityTypeIdentifierDietaryProtein"


def meals(day: str, kcals: list[float], protein: float = 30) -> str:
    out = ""
    for i, k in enumerate(kcals):
        t = f"{day} {8 + 4 * i:02d}:00:00 -0700"
        out += rec(KCAL, k, t, unit="kcal", source="FoodApp") + rec(PROT, protein, t, unit="g", source="FoodApp")
    return out


def build(tmp_path, cfg):
    body = (meals("2024-05-01", [600, 700, 800])          # well logged: 3 entries, 2100 kcal
            + meals("2024-05-02", [450])                   # partial: 1 entry
            + meals("2024-05-03", [300, 300, 300])         # 3 entries but implausibly low (900)
            + meals("2024-05-05", [900, 900, 900, 900])    # well logged, 3600
            + meals("2024-05-06", [2000, 2000, 2000]))     # implausibly high (6000)
    db = tmp_path / "h.db"
    ingest_export(write_export(tmp_path / "e.zip", body), cfg, db_path=db)
    rebuild_daily(cfg, db_path=db)
    return sqlite3.connect(db)


def test_well_logged_flag(tmp_path, cfg):
    d = load_daily(build(tmp_path, cfg), mask_nutrition=False)
    assert d["n_food_entries"].to_dict() == {
        pd.Timestamp("2024-05-01"): 3, pd.Timestamp("2024-05-02"): 1, pd.Timestamp("2024-05-03"): 3,
        pd.Timestamp("2024-05-04"): 0, pd.Timestamp("2024-05-05"): 4, pd.Timestamp("2024-05-06"): 3}
    assert d["well_logged"].tolist() == [True, False, False, False, True, False]
    # Unlogged day is missing, not zero.
    assert np.isnan(d.loc["2024-05-04", "intake_kcal"])
    assert d.loc["2024-05-02", "intake_kcal"] == 450


def test_analyses_see_only_well_logged_days(tmp_path, cfg):
    d = load_daily(build(tmp_path, cfg))  # default: masked
    assert d["intake_kcal"].dropna().tolist() == [2100, 3600]
    assert d["protein_g"].dropna().tolist() == [90, 120]
    assert d["intake_kcal"].mean() == 2850


def test_thresholds_are_configurable(tmp_path, cfg):
    cfg.raw["nutrition"]["min_entries"] = 1
    cfg.raw["nutrition"]["min_kcal"] = 400
    d = load_daily(build(tmp_path, cfg), mask_nutrition=False)
    assert d.loc["2024-05-02", "well_logged"] and d.loc["2024-05-03", "well_logged"]


def test_gap_runs():
    idx = pd.date_range("2024-01-01", periods=12)
    s = pd.Series([1, np.nan, np.nan, np.nan, 2, np.nan, 3, np.nan, np.nan, np.nan, np.nan, 4], index=idx)
    g = gap_runs(s, min_days=3)
    assert g["days"].tolist() == [3, 4]
    assert g["start"].iloc[0] == pd.Timestamp("2024-01-02")


def test_outlier_flags_robust():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(8000, 1000, 120), index=pd.date_range("2024-01-01", periods=120))
    s.iloc[60] = 60000
    f = outlier_flags(s, window=28, k=5, plausible=(0, 50000))
    assert f["outlier"].sum() == 1 and f.loc[s.index[60], "outlier"]
    assert f.loc[s.index[60], "implausible"]
