"""Synthetic Apple Health export with planted, recoverable effects.

``healthlab synth`` writes an ``export.zip`` shaped like a real iPhone-only export
(same element/attribute layout, imperial units, Apple's fixed export-time offset)
for ~18 months, plus a ``truth.json`` with the planted ground truth. Nothing is
derived from real data.

Planted structure (all parameters in :data:`TRUTH_PARAMS`):

* **Weekly seasonality** in steps (Saturday high, Sunday low).
* **Summers in Bulgaria** (``demo.timezone_overrides``): +35 % steps. Those days
  are lived in Europe/Sofia local time, so the configured time-zone override is
  needed to put each sample on the right day. Samples of LA days that fall inside
  an override window (the departure afternoon) are dropped, mimicking a travel day.
* **Lag-1 effect**: steps on day t raise walking speed on day t+1.
* **Lag-2 effect**: flights climbed on day t raise walking asymmetry on day t+2.
* **Regime shifts (changepoints)**: a car commute from ``commute_day`` cuts steps
  by 30 % for good; an injury from ``injury_start_day`` to ``injury_end_day``
  slows walking and raises asymmetry and double support; the phone upgrade on
  ``phone_upgrade_day`` shifts measured step length and double support (a device
  artifact, not a change in the person). Summer starts/ends are shifts in steps too.
* **Source overlap**: a second app duplicates some active-energy samples, so source
  dedup is exercised.
* **Sleep**: only the last 14 days (tracking "just enabled").
* **Devices**: the phone reports hardware ``iPhone12,1`` and is upgraded to
  ``iPhone15,2`` on day ``phone_upgrade_day``. A foreign "Partner's iPhone" writes
  steps on some days, including days the demo phone has no data; demo mode
  excludes it via ``demo.source_exclude``.

Nutrition and body weight are deliberately not generated (out of scope).
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from xml.sax.saxutils import quoteattr
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .config import Config
from .timeutil import ZoneOverride

Q = "HKQuantityTypeIdentifier"
PHONE = "Demo iPhone"
FIT_APP = "DemoFit"
PARTNER = "Partner's iPhone"
MI_PER_KM = 1 / 1.609344

TRUTH_PARAMS = {
    "steps_base": 7500.0,
    "weekday_factor": [0.95, 1.0, 1.0, 1.0, 1.1, 1.3, 0.8],     # Mon..Sun
    "summer_steps_multiplier": 1.35,
    "steps_log_ar": 0.3, "steps_log_sd": 0.18,
    "speed_base_kmh": 4.6, "speed_per_steps_z_lag1": 0.10, "speed_noise_sd": 0.12,
    "flights_per_base_steps": 10.0, "asym_per_flight_dev_lag2": 0.14,
    "basal_kcal": 1800.0, "active_kcal_per_step": 0.05,
    "missing_phone_day_prob": 0.03,
    # Regime shifts (day indices from the start date).
    "commute_day": 230, "commute_steps_multiplier": 0.70,
    "injury_start_day": 349, "injury_end_day": 391,
    "injury_speed_kmh": -0.35, "injury_asymmetry_pct": 4.0, "injury_double_support_pct": 1.5,
    "phone_upgrade_day": 300, "hardware_before": "iPhone12,1", "hardware_after": "iPhone15,2",
    "upgrade_step_length_cm": 3.0, "upgrade_double_support_pct": -0.8,
    "partner_extra_day_prob": 0.05,
}


def _device(hardware: str, software: str) -> str:
    """An HKDevice description string in Apple's export format."""
    return (f"<<HKDevice: 0x7d153ce6a0>, name:iPhone, manufacturer:Apple Inc., model:iPhone, "
            f"hardware:{hardware}, software:{software}, creation date:2025-01-01 12:00:00 +0000>")


@dataclass
class _Writer:
    """Accumulates XML lines; renders timestamps the way Apple does."""

    render_offset: timezone
    lines: list[str]
    n: int = 0
    blocked: "Callable[[datetime], bool] | None" = None   # drop samples starting in these instants
    device: str | None = None                              # HKDevice string for the demo phone's records

    def ts(self, dt_utc: datetime, offset: timezone | None = None) -> str:
        d = dt_utc.astimezone(offset or self.render_offset)
        return d.strftime("%Y-%m-%d %H:%M:%S ") + d.strftime("%z")

    def record(self, type_: str, value, start: datetime, end: datetime, *, unit: str | None, source: str = PHONE,
               meta: dict | None = None, offset: timezone | None = None) -> None:
        if self.blocked is not None and self.blocked(start):
            return
        a = {"type": type_, "sourceName": source, "sourceVersion": "17.5"}
        if source == PHONE and self.device:
            a["device"] = self.device
        elif source == PARTNER:
            a["device"] = _device("iPhone13,2", "17.4")
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
    """Daily latent truth: steps, flights, gait metrics, energy estimates."""
    P = TRUTH_PARAMS
    n = len(days)
    t = np.arange(n)
    commute = t >= P["commute_day"]
    injured = (t >= P["injury_start_day"]) & (t < P["injury_end_day"])
    upgraded = t >= P["phone_upgrade_day"]

    # Steps: weekly pattern x summer x commute regime x AR(1) log-noise.
    eps = np.zeros(n)
    for i in range(1, n):
        eps[i] = P["steps_log_ar"] * eps[i - 1] + rng.normal(0, P["steps_log_sd"])
    wf = np.asarray(P["weekday_factor"])[days.dayofweek]
    steps = (P["steps_base"] * wf * np.where(summer, P["summer_steps_multiplier"], 1.0)
             * np.where(commute, P["commute_steps_multiplier"], 1.0) * np.exp(eps))
    steps = np.round(steps)
    z_steps = (steps - steps.mean()) / steps.std()
    flights = rng.poisson(P["flights_per_base_steps"] * steps / P["steps_base"])

    speed = P["speed_base_kmh"] + rng.normal(0, P["speed_noise_sd"], n) + np.where(injured, P["injury_speed_kmh"], 0.0)
    speed[1:] += P["speed_per_steps_z_lag1"] * z_steps[:-1]                       # planted lag-1
    # Step length / double support follow speed; the new phone measures them differently.
    step_len = (70 + 9 * (speed - P["speed_base_kmh"]) + rng.normal(0, 1.5, n)
                + np.where(upgraded, P["upgrade_step_length_cm"], 0.0))
    dsupport = (30 - 2.5 * (speed - P["speed_base_kmh"]) + rng.normal(0, 0.6, n)
                + np.where(injured, P["injury_double_support_pct"], 0.0)
                + np.where(upgraded, P["upgrade_double_support_pct"], 0.0))
    asym = rng.gamma(1.2, 1.4, n) + np.where(injured, P["injury_asymmetry_pct"], 0.0)
    asym[2:] += P["asym_per_flight_dev_lag2"] * (flights[:-2] - flights.mean())   # planted lag-2
    asym = np.clip(asym, 0, None)

    active = P["active_kcal_per_step"] * steps * rng.normal(1, 0.05, n)
    phone_missing = rng.uniform(size=n) < P["missing_phone_day_prob"]
    return pd.DataFrame({
        "steps": steps, "flights": flights, "speed": speed, "step_len": step_len, "dsupport": dsupport,
        "asym": asym, "active": active, "phone_missing": phone_missing, "summer": summer,
    }, index=days)


def _hour_weights() -> np.ndarray:
    """Share of the day's activity per *local* hour (awake ~07:00-23:00, peak mid-afternoon)."""
    h = np.arange(24)
    w = np.exp(-0.5 * ((h - 14) / 4.0) ** 2) * (h >= 7)
    return w / w.sum()


def planted_changepoints(start: pd.Timestamp, overrides: list[ZoneOverride]) -> dict[str, list[str]]:
    """Dates (first day of the new regime) of every planted shift, per daily metric."""
    P = TRUTH_PARAMS
    d = lambda k: str((start + pd.Timedelta(days=P[k])).date())  # noqa: E731
    summer = []
    for o in overrides:
        summer += [o.start.isoformat(), (o.end + timedelta(days=1)).isoformat()]
    injury = [d("injury_start_day"), d("injury_end_day")]
    activity = sorted(summer + [d("commute_day")])
    return {
        "steps": activity,
        "distance_km": activity,
        "walking_speed_kmh": injury,
        "step_length_cm": sorted([d("phone_upgrade_day")] + injury),
        "double_support_pct": sorted([d("phone_upgrade_day")] + injury),
        "asymmetry_pct": injury,
    }


def write_synthetic_export(out: str | Path, cfg: Config, days: int | None = None, seed: int | None = None) -> dict:
    """Generate the synthetic export zip and ``truth.json`` next to it. Returns the truth dict."""
    demo = cfg.section("demo")
    days = int(days or demo.get("days", 548))
    seed = int(seed if seed is not None else demo.get("seed", 2024))
    rng = np.random.default_rng(seed)
    tz = ZoneInfo(cfg.timezone)
    start = pd.Timestamp(demo.get("start_date", "2025-04-01"))
    idx = pd.date_range(start, periods=days, freq="D")

    overrides = [ZoneOverride(date.fromisoformat(str(o["start"])), date.fromisoformat(str(o["end"])),
                              o["timezone"], o.get("label")) for o in (demo.get("timezone_overrides") or [])]
    day_zone = {d: tz for d in idx}
    summer = np.zeros(days, dtype=bool)
    travel_days: set[pd.Timestamp] = set()
    for o in overrides:
        m = (idx >= pd.Timestamp(o.start)) & (idx <= pd.Timestamp(o.end))
        summer |= m
        for d in idx[m]:
            day_zone[d] = ZoneInfo(o.timezone)
        for d in (o.start - timedelta(days=1), o.start, o.end, o.end + timedelta(days=1)):
            travel_days.add(pd.Timestamp(d))

    def in_override(u: datetime) -> bool:
        return any(o.start_utc <= u < o.end_utc for o in overrides)

    lat = _latent(idx, summer, rng)

    apple_offset = timezone(timedelta(hours=-7))       # Apple renders everything in the export-time offset
    sofia_render = timezone(timedelta(hours=3))
    w = _Writer(apple_offset, [])
    P = TRUTH_PARAMS

    def at(day: pd.Timestamp, hour: float) -> datetime:
        """UTC instant of a local hour on a day, in the zone that day is lived in."""
        zone = day_zone.get(pd.Timestamp(day.date()), tz)
        local = datetime(day.year, day.month, day.day, tzinfo=zone) + timedelta(hours=hour)
        return local.astimezone(timezone.utc)

    for i, day in enumerate(idx):
        r = lat.iloc[i]
        # LA-lived days lose samples that fall inside an override window (departure afternoon).
        w.blocked = None if r.summer else in_override
        upgraded = i >= P["phone_upgrade_day"]
        w.device = _device(P["hardware_after"] if upgraded else P["hardware_before"], "18.1" if upgraded else "17.5")
        if r.phone_missing or rng.uniform() < P["partner_extra_day_prob"]:
            # Someone else's phone: must be excluded, or it fills our missing days with their steps.
            for hr in rng.choice(np.arange(8, 22), size=8):
                st_ = at(day, hr + rng.uniform(0, 0.6))
                w.record(Q + "StepCount", max(1, round(rng.normal(1400, 300))), st_, st_ + timedelta(minutes=10),
                         unit="count", source=PARTNER)
        # A few records per summer day are rendered as if exported from Bulgaria (+0300):
        # same instant, different text. Exercises offset parsing and hash dedup.
        odd_offset = sofia_render if (r.summer and i % 9 == 0) else None
        if not r.phone_missing:
            hw = _hour_weights()
            hours = rng.choice(24, size=12, p=hw)
            shares = rng.dirichlet(np.ones(12))
            for k, (hr, sh) in enumerate(zip(hours, shares)):
                st_ = at(day, hr + rng.uniform(0, 0.6))
                en_ = st_ + timedelta(minutes=float(rng.uniform(4, 20)))
                stp = max(1, round(r.steps * sh))
                w.record(Q + "StepCount", stp, st_, en_, unit="count", offset=odd_offset if k == 0 else None)
                km = stp * r.step_len / 100000
                w.record(Q + "DistanceWalkingRunning", km * MI_PER_KM, st_, en_, unit="mi")
                w.record(Q + "ActiveEnergyBurned", r.active * sh, st_, en_, unit="Cal")
                if k % 4 == 0 and rng.uniform() < 0.3:
                    # Duplicate from a second app: must not be double counted.
                    w.record(Q + "ActiveEnergyBurned", r.active * sh * 0.9, st_, en_, unit="Cal", source=FIT_APP)
            # Distinct whole seconds: identical (type, value, start, end) samples would be
            # deduplicated by design (DECISIONS.md I5), and real sensors never emit them.
            n_fl = int(r.flights)
            fl_secs = np.unique((rng.choice(24, n_fl, p=hw) * 3600 + rng.integers(0, 3240, n_fl)))
            while len(fl_secs) < n_fl:  # top up the rare collisions so the daily count is exact
                fl_secs = np.unique(np.r_[fl_secs, rng.choice(24, 1, p=hw) * 3600 + rng.integers(0, 3240, 1)])
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
        # Basal: hourly through the whole day (Apple estimate).
        for hr in range(24):
            w.record(Q + "BasalEnergyBurned", P["basal_kcal"] / 24 * rng.normal(1, 0.02), at(day, hr), at(day, hr + 1), unit="Cal")

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

    observed = ~lat["phone_missing"]
    truth = {
        "seed": seed, "days": days, "start_date": str(idx[0].date()), "end_date": str(idx[-1].date()),
        "n_records": w.n, "params": TRUTH_PARAMS,
        "active_kcal_mean": float(lat.loc[observed, "active"].mean()),
        "planted_lags": [
            {"x": "steps", "y": "walking_speed_kmh", "lag": 1, "sign": 1},
            {"x": "flights", "y": "asymmetry_pct", "lag": 2, "sign": 1},
        ],
        "planted_changepoints": planted_changepoints(start, overrides),
        "summer_periods": [[o.start.isoformat(), o.end.isoformat()] for o in overrides],
        "travel_days": sorted(str(d.date()) for d in travel_days),
        "daily_steps": {str(d.date()): (None if lat["phone_missing"].iat[i] else float(lat["steps"].iat[i]))
                        for i, d in enumerate(idx)},
    }
    (out.parent / "truth.json").write_text(json.dumps(truth, indent=2))
    return truth
