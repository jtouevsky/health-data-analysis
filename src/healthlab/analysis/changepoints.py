"""Changepoint detection: regime shifts in activity and gait metrics.

Pipeline per metric:

1. **Bin** the daily series (weekly means by default, weeks with ≥ 4 observed days;
   ``steps``/``distance`` on a log scale so shifts are multiplicative). Weekly bins
   remove the day-of-week cycle and most day-to-day autocorrelation.
2. **Standardise** by a robust *long-run* noise scale estimated from first
   differences (insensitive to the mean shifts being detected) under an AR(1)
   noise model, so autocorrelated wandering is not mistaken for regime shifts;
   then winsorise at ±3 so a single freak week cannot create a regime.
3. **PELT** (``ruptures``, L2 cost = mean shifts) with a **BIC penalty**: each
   changepoint adds two parameters (location and new mean), so the penalty is
   2·log(n) on the standardised scale, times a user multiplier.
4. **Sensitivity**: rerun across penalty multipliers (0.25×–8×). A changepoint's
   *stability* is the share of multipliers that place one within ±2 bins.
5. **Bayesian online changepoint detection** (Adams & MacKay 2007; Normal–Gamma
   model, constant hazard) as an independent comparison.
6. **Effect sizes** before vs after each changepoint on the *daily* values of the
   adjacent segments: median difference, % change, Cliff's δ, and a 7-day
   block-bootstrap CI.
7. **Annotation**: match each changepoint to the nearest configured life event,
   device change (new hardware or major iOS version on an included source) or
   travel boundary.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import ruptures as rpt
from scipy import stats
from scipy.special import logsumexp

from .stats import bootstrap_median_diff, cliffs_delta

MAD_SCALE = 1.4826


@dataclass(frozen=True)
class CPParams:
    """Changepoint settings (defaults mirror ``changepoints`` in the YAML config)."""

    resolution: str = "week"             # week | day
    min_obs_per_week: int = 4
    min_segment: int = 4                 # bins
    penalty_multiplier: float = 1.0
    log_metrics: tuple[str, ...] = ("steps", "distance_km")
    sensitivity_multipliers: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    stability_tolerance: int = 2         # bins
    max_phi: float = 0.8
    winsor: float = 3.0                  # clip the standardised series at ±winsor
    bocpd_expected_run: float = 26.0     # bins; hazard = 1 / expected run length
    bocpd_min_run: int = 4               # bins a new MAP run must persist to count
    block_days: int = 7
    bootstrap_reps: int = 1000
    seed: int = 0
    event_tolerance_days: int = 21

    @classmethod
    def from_config(cls, c: dict, **overrides) -> "CPParams":
        keys = cls.__dataclass_fields__
        base = {k: (tuple(v) if isinstance(v, list) else v) for k, v in (c or {}).items() if k in keys}
        return cls(**{**base, **overrides})


@dataclass
class CPResult:
    """Detection output for one metric."""

    metric: str
    binned: pd.Series                  # binned values on the original scale
    z: pd.Series                       # standardised series fed to PELT
    sigma_iid: float
    phi: float
    penalty: float
    changepoints: list[pd.Timestamp]   # first bin of each new regime
    segments: pd.DataFrame             # start, end, mean (original scale), n_bins
    sensitivity: pd.DataFrame          # multiplier, penalty, n_changepoints, changepoints
    stability: dict = field(default_factory=dict)   # changepoint -> share of multipliers


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------

def binned_series(daily: pd.Series, p: CPParams, log: bool = False) -> pd.Series:
    """Weekly (or daily) values with sparse weeks removed; optional log transform."""
    s = daily.dropna().astype(float)
    if log:
        s = np.log(s.clip(lower=1.0))
    if p.resolution == "day":
        return s
    g = s.resample("W-MON", label="left", closed="left")
    out = g.mean()[g.count() >= p.min_obs_per_week]
    return out.dropna()


def noise_scale(x: np.ndarray, max_phi: float = 0.8) -> tuple[float, float, float]:
    """(σ, φ, σ_long-run) of the noise around a piecewise-constant mean, assuming AR(1) noise.

    Everything is estimated from first differences, which a few mean shifts barely
    affect (they become a handful of outlying differences):

    * for AR(1) noise, corr(Δx_t, Δx_{t−1}) = −(1 − φ)/2, so φ = 1 + 2·ρ_Δ, with ρ_Δ a
      rank (Spearman → Pearson) correlation for robustness;
    * φ is then **shrunk to its one-sided 95 % lower bound** (φ̂ − 1.645·2/√n, clipped
      to [0, max_phi]): only autocorrelation the data can demonstrate inflates the
      noise scale. On ~80 weekly bins SE(φ̂) ≈ 0.22, and using φ̂ itself made the
      effective penalty vary randomly between metrics;
    * σ (marginal SD) = MAD(Δx)·1.4826 / √(2(1 − φ));
    * σ_long-run = σ·√((1 + φ)/(1 − φ)), the SD relevant for judging whether a run of
      values differs in mean: positively autocorrelated noise wanders, and must not
      be mistaken for regime shifts.
    """
    x = np.asarray(x, dtype=float)
    d = np.diff(x)
    if d.size < 6:
        s = float(np.std(x)) or 1.0
        return s, 0.0, s
    sd_diff = MAD_SCALE * np.median(np.abs(d - np.median(d)))
    rho_s = stats.spearmanr(d[:-1], d[1:]).statistic
    rho = 2 * np.sin(np.pi * rho_s / 6) if np.isfinite(rho_s) else -0.5
    phi = float(np.clip(1 + 2 * rho - 1.645 * 2 / np.sqrt(d.size - 1), 0.0, max_phi))
    sigma = sd_diff / np.sqrt(2 * (1 - phi)) if sd_diff > 0 else (float(np.std(x)) or 1.0)
    return float(sigma), phi, float(sigma * np.sqrt((1 + phi) / (1 - phi)))


def bic_penalty(n: int) -> float:
    """BIC penalty per changepoint on a unit-variance series: 2 parameters × log(n)."""
    return 2.0 * np.log(max(n, 2))


def pelt(z: np.ndarray, penalty: float, min_size: int) -> list[int]:
    """Indices where a new segment starts (PELT, L2 cost). Excludes 0 and n."""
    if len(z) < 2 * min_size:
        return []
    bkps = rpt.Pelt(model="l2", min_size=min_size, jump=1).fit(np.asarray(z, dtype=float)).predict(pen=penalty)
    return [b for b in bkps if 0 < b < len(z)]


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect(daily: pd.Series, p: CPParams, metric: str | None = None) -> CPResult | None:
    """Detect changepoints in one daily metric; None if too little data."""
    metric = metric or str(daily.name)
    b = binned_series(daily, p, log=metric in p.log_metrics)
    if len(b) < 3 * p.min_segment:
        return None
    sigma, phi, sigma_lr = noise_scale(b.to_numpy(), p.max_phi)
    # Winsorise at ±winsor: one freak week (lognormal tails) shouldn't drag a segment mean.
    z = ((b - b.median()) / sigma_lr).clip(-p.winsor, p.winsor)
    base = bic_penalty(len(z))
    pen = base * p.penalty_multiplier
    idx = pelt(z.to_numpy(), pen, p.min_segment)
    cps = [b.index[i] for i in idx]

    sens_rows, sens_sets = [], {}
    for m in p.sensitivity_multipliers:
        ii = pelt(z.to_numpy(), base * m, p.min_segment)
        sens_sets[m] = ii
        sens_rows.append({"multiplier": m, "penalty": base * m, "n_changepoints": len(ii),
                          "changepoints": [b.index[i] for i in ii]})
    stability = {b.index[i]: float(np.mean([any(abs(i - j) <= p.stability_tolerance for j in s)
                                            for s in sens_sets.values()])) for i in idx}

    back = (lambda v: np.exp(v)) if metric in p.log_metrics else (lambda v: v)
    edges = [0] + idx + [len(b)]
    segs = pd.DataFrame([{"start": b.index[s], "end": b.index[e - 1], "n_bins": e - s,
                          "mean": float(back(b.iloc[s:e].mean()))} for s, e in zip(edges[:-1], edges[1:])])
    return CPResult(metric=metric, binned=b.apply(back), z=z, sigma_iid=sigma, phi=phi, penalty=pen,
                    changepoints=cps, segments=segs, sensitivity=pd.DataFrame(sens_rows), stability=stability)


def detect_all(daily: pd.DataFrame, metrics: list[str], p: CPParams) -> dict[str, CPResult]:
    """Run :func:`detect` for each metric present in ``daily``."""
    out = {}
    for m in metrics:
        if m in daily and daily[m].notna().sum() > 0:
            r = detect(daily[m], p, m)
            if r is not None:
                out[m] = r
    return out


# ---------------------------------------------------------------------------
# Bayesian online changepoint detection
# ---------------------------------------------------------------------------

def bocpd(z: pd.Series, expected_run: float = 26.0, min_run: int = 4, mu0: float = 0.0,
          kappa0: float = 0.5, alpha0: float = 2.0, beta0: float = 2.0) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    """Adams & MacKay (2007) BOCPD with a Normal–Gamma prior and constant hazard.

    Args:
        z: standardised series (e.g. ``CPResult.z``); the prior assumes unit-scale noise.
        expected_run: prior expected segment length in bins (hazard = 1 / expected_run).
        min_run: a new maximum-a-posteriori (MAP) run must persist this many bins to
            be reported as a changepoint.

    Returns:
        ``(trace, changepoints)``: per bin the MAP run length and the posterior
        probability that the current regime began within the last ``min_run`` bins;
        and the start dates of persistent new runs.
    """
    x = z.to_numpy(dtype=float)
    T = len(x)
    log_h, log_1mh = np.log(1.0 / expected_run), np.log(1.0 - 1.0 / expected_run)
    log_r = np.array([0.0])
    mu, kappa, alpha, beta = (np.array([v], dtype=float) for v in (mu0, kappa0, alpha0, beta0))
    map_rl = np.zeros(T, dtype=int)
    p_recent = np.zeros(T)
    for t in range(T):
        scale = np.sqrt(beta * (kappa + 1) / (alpha * kappa))
        log_pred = stats.t.logpdf(x[t], 2 * alpha, loc=mu, scale=scale)
        log_growth = log_r + log_pred + log_1mh
        log_cp = logsumexp(log_r + log_pred + log_h)
        log_r = np.r_[log_cp, log_growth]
        log_r -= logsumexp(log_r)
        # Run length r >= 1 includes x[t]; r = 0 means a change right after x[t].
        new_mu = (kappa * mu + x[t]) / (kappa + 1)
        new_beta = beta + kappa * (x[t] - mu) ** 2 / (2 * (kappa + 1))
        mu, kappa = np.r_[mu0, new_mu], np.r_[kappa0, kappa + 1]
        alpha, beta = np.r_[alpha0, alpha + 0.5], np.r_[beta0, new_beta]
        r = np.exp(log_r)
        map_rl[t] = int(np.argmax(r))
        p_recent[t] = min(1.0, float(r[: min_run + 1].sum()))

    starts = np.where(map_rl >= 1, np.arange(T) - map_rl + 1, -1)
    cps: list[int] = []
    run_start, run_len = None, 0
    for s in starts:
        if s == run_start:
            run_len += 1
        else:
            run_start, run_len = s, 1
        if run_len == min_run and s not in (None, -1, 0) and s not in cps:
            cps.append(int(s))
    trace = pd.DataFrame({"map_run_length": map_rl, "p_new_regime": p_recent}, index=z.index)
    return trace, [z.index[i] for i in sorted(cps)]


# ---------------------------------------------------------------------------
# Effects and annotation
# ---------------------------------------------------------------------------

def segment_effects(daily: pd.Series, result: CPResult, p: CPParams) -> pd.DataFrame:
    """Before/after comparison of daily values around each changepoint.

    Before = the segment ending at the changepoint, after = the segment starting
    there (segments as found by PELT, on daily data). Reports medians, the
    difference and % change, Cliff's δ (after vs before) and a block-bootstrap
    95 % CI for the median difference.
    """
    s = daily.dropna().astype(float)
    bounds = [s.index.min()] + list(result.changepoints) + [s.index.max() + pd.Timedelta(days=1)]
    rows = []
    for k, cp in enumerate(result.changepoints, start=1):
        before = s[(s.index >= bounds[k - 1]) & (s.index < cp)].to_numpy()
        after = s[(s.index >= cp) & (s.index < bounds[k + 1])].to_numpy()
        if before.size < 5 or after.size < 5:
            continue
        mb, ma = float(np.median(before)), float(np.median(after))
        lo, hi = bootstrap_median_diff(after, before, reps=p.bootstrap_reps, block=p.block_days, seed=p.seed)
        rows.append({"changepoint": cp, "n_before": before.size, "n_after": after.size,
                     "median_before": mb, "median_after": ma, "diff": ma - mb,
                     "pct_change": (ma - mb) / abs(mb) if mb else np.nan,
                     "ci_lo": lo, "ci_hi": hi, "cliffs_delta": cliffs_delta(after, before),
                     "stability": result.stability.get(cp, np.nan)})
    return pd.DataFrame(rows)


def life_events(cfg) -> pd.DataFrame:
    """Configured life events (``changepoints.life_events``, or ``demo.life_events`` in demo mode)."""
    src = (cfg.section("demo").get("life_events") if cfg.demo_mode
           else cfg.section("changepoints").get("life_events")) or []
    return pd.DataFrame([{"date": pd.Timestamp(e["date"]), "label": e["label"], "kind": "life event"} for e in src],
                        columns=["date", "label", "kind"])


def device_events(con: sqlite3.Connection, cfg, min_share: float = 0.05) -> pd.DataFrame:
    """New hardware or a new major OS version on a source the analysis relies on.

    Both can change how Apple measures steps and gait (sensor placement, motion
    algorithms), so they are candidate explanations for shifts that are measurement
    artifacts rather than changes in the person.
    """
    from ..db import table_exists
    from ..quality import hardware_timeline, software_timeline

    # Only sources the rollup actually relies on (selected on >= min_share of source-days):
    # an old secondary device's iOS update cannot explain a shift in your own phone's data.
    used = None
    if table_exists(con, "daily_sources"):
        share = pd.read_sql_query("SELECT source_used, COUNT(*) AS n FROM daily_sources GROUP BY 1", con)
        share["share"] = share["n"] / share["n"].sum()
        used = set(share.loc[share["share"] >= min_share, "source_used"])

    def relevant(src: str) -> bool:
        return not cfg.is_excluded(src) and (used is None or src in used)

    rows = []
    hw = hardware_timeline(con)
    hw = hw[hw["source_name"].map(relevant)]
    for src, g in hw.groupby("source_name", sort=False):
        g = g.sort_values("first_date")
        for prev, cur in zip(g.itertuples(), g.iloc[1:].itertuples()):
            rows.append({"date": pd.Timestamp(cur.first_date),
                         "label": f"{src}: {prev.hardware} → {cur.hardware}", "kind": "device"})
    sw = software_timeline(con)
    sw = sw[sw["source_name"].map(relevant)]
    for src, g in sw.groupby("source_name", sort=False):
        for prev, cur in zip(g.itertuples(), g.iloc[1:].itertuples()):
            rows.append({"date": pd.Timestamp(cur.first_date),
                         "label": f"{src}: iOS {prev.major} → {cur.major}", "kind": "device"})
    return pd.DataFrame(rows, columns=["date", "label", "kind"])


def travel_events(cfg) -> pd.DataFrame:
    """Start and end (first day back) of every time-zone override period."""
    rows = []
    for o in cfg.timezone_overrides:
        name = o.label or o.timezone
        rows.append({"date": pd.Timestamp(o.start), "label": f"{name} begins", "kind": "travel"})
        rows.append({"date": pd.Timestamp(o.end) + pd.Timedelta(days=1), "label": f"{name} ends", "kind": "travel"})
    return pd.DataFrame(rows, columns=["date", "label", "kind"])


def all_events(con: sqlite3.Connection, cfg) -> pd.DataFrame:
    """Life events, device changes and travel boundaries, sorted by date."""
    parts = [df for df in (life_events(cfg), device_events(con, cfg), travel_events(cfg)) if not df.empty]
    if not parts:
        return pd.DataFrame(columns=["date", "label", "kind"])
    return pd.concat(parts, ignore_index=True).sort_values("date").reset_index(drop=True)


def match_events(changepoints: list[pd.Timestamp], events: pd.DataFrame, tolerance_days: int) -> pd.DataFrame:
    """Nearest event within ±tolerance for each changepoint (NaN label if none)."""
    rows = []
    for cp in changepoints:
        best = None
        if not events.empty:
            gap = (events["date"] - cp).dt.days
            j = gap.abs().idxmin()
            if abs(gap[j]) <= tolerance_days:
                best = events.loc[j]
                rows.append({"changepoint": cp, "event": best["label"], "kind": best["kind"], "days_from_event": int(gap[j])})
                continue
        rows.append({"changepoint": cp, "event": None, "kind": None, "days_from_event": None})
    return pd.DataFrame(rows, columns=["changepoint", "event", "kind", "days_from_event"])
