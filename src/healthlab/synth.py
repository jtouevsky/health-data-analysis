"""Synthetic Apple Health export with planted, recoverable effects.

``healthlab synth`` writes an ``export.zip`` shaped like a real iPhone-only export
(same element/attribute layout, imperial units, Apple's fixed export-time offset)
for ~18 months, plus a ``truth.json`` with the planted ground truth. Nothing is
derived from real data.

Planted structure (all parameters in :data:`TRUTH_PARAMS`):

* **Weekly seasonality** in steps (Saturday high, Sunday low).
* **Summers in Bulgaria** (``demo.periods.bulgaria_summer``): +35 % steps, a
  holiday energy surplus, and an activity rhythm shifted to Sofia hours (visible
  on the Data Quality activity-timing heatmap).
* **Lag-1 effect**: steps on day t raise walking speed on day t+1.
* **Lag-2 effect**: active energy on day t raises energy intake on day t+2.
* **Energy balance**: weight follows true intake − true TDEE at 7,700 kcal/kg
  (a 250 kcal/day cut for the first 240 days, then maintenance).
* **Food logging**: streaky logging with partial days and gaps; logged intake
  under-reports true intake by 15 %.
* **Apple bias**: Apple's active energy overestimates the truth by 15 %, and its
  basal estimate is 50 kcal/day high.
* **Source overlap**: a second app duplicates some active-energy samples, so source
  dedup is exercised.
* **Sleep**: only the last 14 days (tracking "just enabled").
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from xml.sax.saxutils import quoteattr
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .config import Config

Q = "HKQuantityTypeIdentifier"
PHONE = "Demo iPhone"
FOOD_APP = "DemoFood"
FIT_APP = "DemoFit"
LB_PER_KG = 1 / 0.45359237
MI_PER_KM = 1 / 1.609344

TRUTH_PARAMS = {
    "kcal_per_kg": 7700.0,
    "steps_base": 7500.0,
    "weekday_factor": [0.95, 1.0, 1.0, 1.0, 1.1, 1.3, 0.8],     # Mon..Sun
    "summer_steps_multiplier": 1.35,
    "steps_log_ar": 0.5, "steps_log_sd": 0.25,
    "speed_base_kmh": 4.6, "speed_per_steps_z_lag1": 0.10, "speed_noise_sd": 0.12,
    "true_basal_kcal": 1750.0, "active_kcal_per_step": 0.045,
    "apple_active_multiplier": 1.15, "apple_basal_offset": 50.0,
    "cut_kcal": -250.0, "cut_days": 240, "summer_surplus_kcal": 200.0,
    "intake_per_active_dev_lag2": 1.5, "intake_noise_sd": 250.0,
    "underreport": 0.15,
    "logging_stay_on": 0.88, "logging_stay_off": 0.6, "partial_share": 0.12,
    "weight_start_kg": 84.0, "weight_noise_sd": 0.45, "weigh_in_prob": 5 / 7,
    "missing_phone_day_prob": 0.03,
}


@dataclass
class _Writer:
    """Accumulates XML lines; renders timestamps the way Apple does."""

    render_offset: timezone
    lines: list[str]
    n: int = 0

    def ts(self, dt_utc: datetime, offset: timezone | None = None) -> str:
        d = dt_utc.astimezone(offset or self.render_offset)
        return d.strftime("%Y-%m-%d %H:%M:%S ") + d.strftime("%z")

    def record(self, type_: str, value, start: datetime, end: datetime, *, unit: str | None, source: str = PHONE,
               meta: dict | None = None, offset: timezone | None = None) -> None:
        a = {"type": type_, "sourceName": source, "sourceVersion": "17.5"}
        if unit is not None:
            a["unit"] = unit
        a.update({"creationDate": self.ts(end, offset), "startDate": self.ts(start, offset),
                  "endDate": self.ts(end, offset), "value": value if isinstance(value, str) else f"{value:.6g}"})
        attrs = " ".join(f"{k}={quoteattr(v)}" for k, v in a.items())
        if meta:
            inner = "".join(f'  <MetadataEntry key={quoteattr(k)} value={quoteattr(str(v))}/>\n' for k, v in meta.items())
            self.lines.append(f" <Record {attrs}>\n{inner} </Record>\n")
        else:
            self.lines.append(f" <Record {attrs}/>\n")
        self.n += 1


def _latent(days: pd.DatetimeIndex, summer: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """Daily latent truth: steps, gait, energy, intake, weight, logging status."""
    P = TRUTH_PARAMS
    n = len(days)
    # Steps: weekly pattern x summer x AR(1) log-noise.
    eps = np.zeros(n)
    for i in range(1, n):
        eps[i] = P["steps_log_ar"] * eps[i - 1] + rng.normal(0, P["steps_log_sd"])
    wf = np.asarray(P["weekday_factor"])[days.dayofweek]
    steps = P["steps_base"] * wf * np.where(summer, P["summer_steps_multiplier"], 1.0) * np.exp(eps)
    steps = np.round(steps)
    z_steps = (steps - steps.mean()) / steps.std()

    speed = P["speed_base_kmh"] + rng.normal(0, P["speed_noise_sd"], n)
    speed[1:] += P["speed_per_steps_z_lag1"] * z_steps[:-1]                       # planted lag-1
    step_len = 70 + 9 * (speed - P["speed_base_kmh"]) + rng.normal(0, 1.5, n)
    dsupport = 30 - 2.5 * (speed - P["speed_base_kmh"]) + rng.normal(0, 0.6, n)
    asym = rng.gamma(1.2, 1.4, n)

    active_true = P["active_kcal_per_step"] * steps * rng.normal(1, 0.05, n)
    tdee_true = P["true_basal_kcal"] + active_true
    plan = np.where(np.arange(n) < P["cut_days"], P["cut_kcal"], 0.0) + np.where(summer, P["summer_surplus_kcal"], 0.0)
    act_dev = active_true - active_true.mean()
    intake_true = tdee_true + plan + rng.normal(0, P["intake_noise_sd"], n)
    intake_true[2:] += P["intake_per_active_dev_lag2"] * act_dev[:-2]                # planted lag-2

    weight = np.empty(n)
    weight[0] = P["weight_start_kg"]
    for i in range(1, n):
        weight[i] = weight[i - 1] + (intake_true[i - 1] - tdee_true[i - 1]) / P["kcal_per_kg"]

    # Streaky logging: Markov chain on/off, with some "on" days only partially logged.
    on = np.zeros(n, dtype=bool)
    on[0] = True
    for i in range(1, n):
        stay = P["logging_stay_on"] if on[i - 1] else P["logging_stay_off"]
        on[i] = on[i - 1] if rng.uniform() < stay else not on[i - 1]
    partial = on & (rng.uniform(size=n) < P["partial_share"])
    status = np.where(~on, "none", np.where(partial, "partial", "well"))
    phone_missing = rng.uniform(size=n) < P["missing_phone_day_prob"]

    return pd.DataFrame({
        "steps": steps, "speed": speed, "step_len": step_len, "dsupport": dsupport, "asym": asym,
        "active_true": active_true, "tdee_true": tdee_true, "intake_true": intake_true, "weight_true": weight,
        "log_status": status, "phone_missing": phone_missing, "summer": summer,
    }, index=days)


def _hour_weights(summer: bool) -> np.ndarray:
    """Share of the day's activity per LA-local hour (Sofia-hours rhythm in summer)."""
    h = np.arange(24)
    if summer:
        # Awake ~09:00-24:00 Sofia = 23:00(prev)-14:00 LA; kept inside the LA day.
        w = np.exp(-0.5 * ((h - 7) / 3.5) ** 2) * (h <= 14)
    else:
        w = np.exp(-0.5 * ((h - 14) / 4.0) ** 2) * (h >= 7)
    return w / w.sum()


def write_synthetic_export(out: str | Path, cfg: Config, days: int | None = None, seed: int | None = None) -> dict:
    """Generate the synthetic export zip and ``truth.json`` next to it. Returns the truth dict."""
    demo = cfg.section("demo")
    days = int(days or demo.get("days", 548))
    seed = int(seed if seed is not None else demo.get("seed", 2024))
    rng = np.random.default_rng(seed)
    tz = ZoneInfo(cfg.timezone)
    start = pd.Timestamp(demo.get("start_date", "2025-04-01"))
    idx = pd.date_range(start, periods=days, freq="D")

    summer = np.zeros(days, dtype=bool)
    for s, e in (demo.get("periods") or {}).get("bulgaria_summer", []):
        summer |= (idx >= pd.Timestamp(s)) & (idx <= pd.Timestamp(e))
    lat = _latent(idx, summer, rng)

    apple_offset = timezone(timedelta(hours=-7))       # Apple renders everything in the export-time offset
    sofia_render = timezone(timedelta(hours=3))
    w = _Writer(apple_offset, [])
    P = TRUTH_PARAMS

    def at(day: pd.Timestamp, hour: float) -> datetime:
        local = datetime(day.year, day.month, day.day, tzinfo=tz) + timedelta(hours=hour)
        return local.astimezone(timezone.utc)

    for i, day in enumerate(idx):
        r = lat.iloc[i]
        # A few records per summer day are rendered as if exported from Bulgaria (+0300):
        # same instant, different text. Exercises offset parsing and hash dedup.
        odd_offset = sofia_render if (r.summer and i % 9 == 0) else None
        if not r.phone_missing:
            hw = _hour_weights(bool(r.summer))
            hours = rng.choice(24, size=12, p=hw)
            shares = rng.dirichlet(np.ones(12))
            for k, (hr, sh) in enumerate(zip(hours, shares)):
                st_ = at(day, hr + rng.uniform(0, 0.6))
                en_ = st_ + timedelta(minutes=float(rng.uniform(4, 20)))
                stp = max(1, round(r.steps * sh))
                w.record(Q + "StepCount", stp, st_, en_, unit="count", offset=odd_offset if k == 0 else None)
                km = stp * r.step_len / 100000
                w.record(Q + "DistanceWalkingRunning", km * MI_PER_KM, st_, en_, unit="mi")
                w.record(Q + "ActiveEnergyBurned", r.active_true * P["apple_active_multiplier"] * sh, st_, en_, unit="Cal")
                if k % 4 == 0 and rng.uniform() < 0.3:
                    # Duplicate from a second app: must not be double counted.
                    w.record(Q + "ActiveEnergyBurned", r.active_true * sh, st_, en_, unit="Cal", source=FIT_APP)
            n_fl = rng.poisson(10 * r.steps / P["steps_base"])
            # Distinct whole seconds: identical (type, value, start, end) samples would be
            # deduplicated by design (DECISIONS.md I5), and real sensors never emit them.
            fl_secs = np.unique((rng.choice(24, n_fl, p=hw) * 3600 + rng.integers(0, 3240, n_fl)))
            for sec in fl_secs:
                t = at(day, sec / 3600)
                w.record(Q + "FlightsClimbed", 1, t, t + timedelta(seconds=20), unit="count")
            for g in range(4):
                t = at(day, rng.choice(24, p=hw) + rng.uniform(0, 0.9))
                w.record(Q + "WalkingSpeed", (r.speed + rng.normal(0, 0.15)) * MI_PER_KM, t, t + timedelta(seconds=30), unit="mi/hr")
                w.record(Q + "WalkingStepLength", (r.step_len + rng.normal(0, 2)) / 2.54, t, t + timedelta(seconds=30), unit="in")
                w.record(Q + "WalkingDoubleSupportPercentage", (r.dsupport + rng.normal(0, 0.8)) / 100, t, t + timedelta(seconds=30), unit="%")
            w.record(Q + "WalkingAsymmetryPercentage", r.asym / 100, at(day, 12), at(day, 12.01), unit="%",
                     meta={"HKMetadataKeyDevicePlacementSide": "Left"})
            if rng.uniform() < 0.6:
                for _ in range(rng.integers(3, 12)):
                    t = at(day, rng.choice(24, p=hw) + rng.uniform(0, 0.9))
                    w.record(Q + "HeadphoneAudioExposure", rng.normal(68, 6), t, t + timedelta(minutes=float(rng.uniform(5, 30))), unit="dBASPL")
        # Basal: hourly through the whole day (Apple estimate, slightly high).
        basal = (P["true_basal_kcal"] + P["apple_basal_offset"]) / 24
        for hr in range(24):
            w.record(Q + "BasalEnergyBurned", basal * rng.normal(1, 0.02), at(day, hr), at(day, hr + 1), unit="Cal")

        # Food log.
        logged_total = r.intake_true * (1 - P["underreport"])
        if r.log_status == "well":
            parts = rng.dirichlet(np.ones(int(rng.integers(3, 6))) * 3)
        elif r.log_status == "partial":
            parts = rng.dirichlet(np.ones(int(rng.integers(1, 3)))) * 0.35
        else:
            parts = []
        for j, sh in enumerate(parts):
            t = at(day, 8 + j * 3.5 + rng.uniform(0, 1))
            kcal = logged_total * sh
            prot, carb, fat = rng.dirichlet(np.array([0.25, 0.45, 0.30]) * 20)  # meal-level macro split
            w.record(Q + "DietaryEnergyConsumed", kcal, t, t, unit="kcal", source=FOOD_APP)
            w.record(Q + "DietaryProtein", prot * kcal / 4, t, t, unit="g", source=FOOD_APP)
            w.record(Q + "DietaryCarbohydrates", carb * kcal / 4, t, t, unit="g", source=FOOD_APP)
            w.record(Q + "DietaryFatTotal", fat * kcal / 9, t, t, unit="g", source=FOOD_APP)

        # Weigh-ins (morning, in lb, written by the food app).
        if rng.uniform() < P["weigh_in_prob"]:
            t = at(day, 7.5 + rng.uniform(0, 0.5))
            kg = r.weight_true + rng.normal(0, P["weight_noise_sd"])
            w.record(Q + "BodyMass", kg * LB_PER_KG, t, t, unit="lb", source=FOOD_APP)

        # Sleep only for the last two weeks.
        if i >= days - 14:
            bed = at(day - pd.Timedelta(days=1), 23.5 + rng.uniform(-0.5, 0.5))
            wake = at(day, 7 + rng.uniform(-0.5, 0.5))
            w.record("HKCategoryTypeIdentifierSleepAnalysis", "HKCategoryValueSleepAnalysisInBed", bed, wake, unit=None)
            w.record("HKCategoryTypeIdentifierSleepAnalysis", "HKCategoryValueSleepAnalysisAsleepCore",
                     bed + timedelta(minutes=15), wake - timedelta(minutes=10), unit=None)

    # Weekly walk workouts.
    workouts = []
    for day in idx[idx.dayofweek == 5]:
        st_, en_ = at(day, 10), at(day, 11)
        workouts.append(
            f' <Workout workoutActivityType="HKWorkoutActivityTypeWalking" duration="60" durationUnit="min" '
            f'sourceName="{PHONE}" sourceVersion="17.5" creationDate="{w.ts(en_)}" startDate="{w.ts(st_)}" endDate="{w.ts(en_)}">\n'
            f'  <WorkoutStatistics type="{Q}ActiveEnergyBurned" startDate="{w.ts(st_)}" endDate="{w.ts(en_)}" '
            f'sum="{rng.normal(250, 30):.1f}" unit="Cal"/>\n </Workout>\n')

    export_date = at(idx[-1] + pd.Timedelta(days=1), 9)
    header = ('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE HealthData [\n'
              '<!ELEMENT HealthData (ExportDate,Me,(Record|Correlation|Workout|ActivitySummary|ClinicalRecord)*)>\n'
              '<!ATTLIST HealthData locale CDATA #REQUIRED>\n]>\n<HealthData locale="en_US">\n'
              f' <ExportDate value="{w.ts(export_date)}"/>\n'
              ' <Me HKCharacteristicTypeIdentifierBiologicalSex="HKBiologicalSexNotSet"/>\n')
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("apple_health_export/export.xml", header + "".join(w.lines) + "".join(workouts) + "</HealthData>\n")

    apple_tdee = lat["active_true"] * P["apple_active_multiplier"] + P["true_basal_kcal"] + P["apple_basal_offset"]
    truth = {
        "seed": seed, "days": days, "start_date": str(idx[0].date()), "end_date": str(idx[-1].date()),
        "n_records": w.n, "params": TRUTH_PARAMS,
        "true_tdee_mean": float(lat["tdee_true"].mean()),
        "apple_tdee_mean": float(apple_tdee.mean()),
        "true_intake_mean": float(lat["intake_true"].mean()),
        "weight_start": float(lat["weight_true"].iloc[0]), "weight_end": float(lat["weight_true"].iloc[-1]),
        "share_well_logged_planned": float((lat["log_status"] == "well").mean()),
        "planted_lags": [
            {"x": "steps", "y": "walking_speed_kmh", "lag": 1, "sign": 1},
            {"x": "active_kcal", "y": "intake_kcal", "lag": 2, "sign": 1},
        ],
        "summer_periods": (demo.get("periods") or {}).get("bulgaria_summer", []),
    }
    (out.parent / "truth.json").write_text(json.dumps(truth, indent=2))
    return truth
