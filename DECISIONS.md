# Methodology decisions

Every assumption or judgment call in the analysis design, with a one-line
rationale. Parameter values live in `config/healthlab.yaml`. This file explains
*why* each value is what it is. Entries are grouped by pipeline stage. ⚠️ marks
choices worth revisiting once more data exists.

## Data facts that shaped the design (observed in a typical iPhone-only export)

| # | Observation | Consequence |
|---|---|---|
| D1 | Several hundred thousand records across ~11 types: steps, distance, flights, walking speed/step length/double support/asymmetry, headphone audio (+ alert events), active/basal energy, and a sleep-duration goal. A handful of workouts. | The core app is built around activity, gait and audio metrics. |
| D2 | **No nutrition, body-weight or sleep samples**. | **Scope (revised 2026-10-05):** no nutrition, weight or energy-balance analysis. Ingest stays generic, so such types would still be stored and rolled up with default rules (R7), but they get no dedicated logic, pages or synthetic data. Sleep is stored but not analysed until history exists. |
| D3 | **Every timestamp carries `-0700`**, including winter and summers abroad, and no record has `HKTimeZone` metadata. | Apple renders all dates in the export-time offset, so the per-record offset is *not* the original local offset (see T1). |
| D4 | History can span many years and is often sparse early on. Older devices (a previous phone, an iPod touch) may appear as extra sources. | Source priority matters for the overlap years (R3). An `analysis.start_date` setting is available. |
| D5 | Raw units are imperial/Apple-specific: `mi`, `Cal`, `mi/hr`, `in`, and `%` stored as a 0–1 fraction. | Everything is converted to canonical metric units at rollup time (R6). |

## Ingest

| # | Decision | Rationale |
|---|---|---|
| I1 | Parse with `xml.etree.iterparse` straight from the zip. Clear each top-level child of `<HealthData>` once it closes. | Constant memory: a ~260 MB XML ingests at ~45 MB peak RSS. No `lxml` dependency needed. |
| I2 | Store every `<Record>` generically. Numeric values go to `value` and non-numeric (category) values to `value_text`. | No hardcoded type list, and new HealthKit types flow through automatically. |
| I3 | Timestamps are stored as UTC text `YYYY-MM-DD HH:MM:SS` plus the offset as written (minutes). | UTC is unambiguous and sortable. The written offset is kept for auditability even though it's unreliable (D3). |
| I4 | Content hash = SHA-1 of (type, source, unit, raw value, **UTC** start, **UTC** end). `INSERT OR IGNORE`. | Re-imports are idempotent, and an export rendered in another offset still dedups. `sourceVersion`/`creationDate` are excluded because they don't identify the measurement. |
| I5 | Two genuinely distinct samples with identical type, source, value and timestamps are treated as one. | That's practically impossible for real sensor data, and it's the price of hash-based idempotency. |
| I6 | Records deleted in Health after a previous import are **not** removed on re-import. Use `healthlab ingest --rebuild` for a clean slate. | Incremental import can't tell "deleted" from "outside the export". |
| I7 | `Records` nested inside `<Correlation>` (food/blood-pressure groupings) are ingested and deduped against their top-level copies by hash. | Some exports only contain them nested. |
| I8 | `ActivitySummary`, `ClinicalRecord` and `Me` are counted in `import_runs.other_elements` but not ingested. | They're Watch/medical-record data outside the scope of this analysis. |
| I9 | Source names are whitespace-stripped. | Some apps write source names with trailing newlines. |
| I10 | Batches of 10,000 rows, committed per batch. A failure mid-run leaves earlier batches in place, and a re-run completes the import. | Fast and safe thanks to idempotency. |

## Time

| # | Decision | Rationale |
|---|---|---|
| T1 | **Local dates use `America/Los_Angeles` (DST-aware) applied to the UTC instant by default**, plus **config-defined override periods** (`time.overrides`: inclusive date range → IANA zone, e.g. Europe/Sofia for summers in Bulgaria). | The export doesn't preserve original offsets (D3), so the zone has to come from knowledge of where you were. LA is the owner's default, and overrides fix the ~10 h misalignment that otherwise moves Sofia mornings onto the previous LA date. The synthetic test shows summer daily totals are exact with the override, and systematically wrong (median 8%, 20%+ on many days) without it. |
| T1a | An override covers **[start 00:00, end+1 00:00) local time in the override zone**, matched on the UTC instant. Overlapping overrides are a config error. | Unambiguous and independent of the offset an export was rendered with. ⚠️ Travel days are inherently ambiguous: the LA afternoon before departure falls inside the override window (it's already the next day in Sofia), and after return the LA afternoon of the end date shares that date with the last Sofia day. These 2 days per trip should be read with caution. |
| T1b | Local dates are **recomputed from UTC at every rollup** using the current config. `records.local_date` is only an ingest-time snapshot. | Editing overrides only needs `healthlab rollup`, not a re-import. |
| T1c | A **labelled override becomes a seasonal period** of that label (merged with any `seasonal.periods` ranges of the same label). | The travel dates are entered once and drive both the time zone and the "Bulgaria summer vs school year" comparison. |
| T1d | Samples whose local end precedes their local start (they straddle an override boundary) are assigned wholly to their start day. | It's rare (a few samples per trip), and prorating a negative interval is undefined. |
| T2 | `local_date_mode: offset` remains available. | Correct for exports that preserve offsets, and lets tests cover both behaviours. |
| T3 | A record's day is the local date of its **start** time. `local_end_date` is also stored. | That's the convention Apple's own daily charts use. Sleep uses the end date instead (R5). |

## Daily rollup

| # | Decision | Rationale |
|---|---|---|
| R1 | Aggregation per type comes from `metrics:` in config. Cumulative types (steps, distance, flights, energy) use **sum**. Gait samples use **mean**. | Totals are the natural daily quantity for cumulative types, and a gait sample is a level, so its daily mean is the right summary. |
| R2 | Headphone exposure uses **LAeq** (duration-weighted energy average: 10·log10 of the mean of 10^(L/10)) instead of the arithmetic mean of dB. | Decibels are logarithmic. The arithmetic mean understates exposure, and LAeq is the standard occupational/WHO metric. |
| R3 | **Source dedup**: for each (type, local day), only the highest-ranked source with data that day is used. Ranking = configured `source_priority` list, then sources by total record count (ties alphabetical). | Summing iPhone + app (or old + new phone) double-counts. Day-level selection keeps each day internally consistent. Falling back to a lower-ranked source fills days when the main source is absent (e.g. the old iPhone era). |
| R3a | **Source exclusion** via `source_exclude` in config: `all:` (every type) or per HK type. Excluded records stay in the database and are dropped at rollup time. Names match ignoring case and ’ vs '. | Other people's devices (a family member's phone, an old iPod) can be removed without re-importing. Tolerant matching avoids silent misses from Apple's curly apostrophes. |
| R3b | The Data Quality page lists every source with its first/last active date, days with data, types written, **hardware models parsed from HKDevice strings** (in the order seen), and a monthly activity timeline. | The owner can check which sources are theirs. Hardware changes on your own source mark phone upgrades, a known confounder for gait metrics. |
| R4 | Cumulative samples that cross local midnight are **prorated by time overlap** across the days. | Some step samples span up to ~10 h, and assigning them wholly to the start day biases both days. |
| R5 | Sleep intervals go to the **local date of waking** (interval end). Overlapping intervals of the same class are merged before summing. Naps also count toward their end date. | Owner's spec. Merging avoids double-counting overlapping core/deep segments or two sources. Not analysed yet (D2). |
| R6 | Raw units are converted to canonical metric units: mi→km, Cal→kcal, mi/hr→km/h, in→cm, lb→kg, and the HealthKit `%` fraction → 0–100. | Analyses need consistent units, and display conversion (kg/lb, km/mi) happens only in the UI. |
| R7 | Unlisted types get a default rule. A canonical unit in `default_rules.cumulative_units` means sum, otherwise mean. Category types are counted. | Generic ingest without silently dropping new types (e.g. dietary energy would become a daily sum, body mass a daily mean in kg). The `metric_catalog.configured` flag shows which ones were defaulted. |
| R8 | A day with **no records is NaN (missing), never 0**. | Absence of data isn't evidence of zero activity. |
| R9 | Exception: `zero_fill` metrics (flights climbed, headphone alerts) are 0 on days when **steps** were recorded, and NaN otherwise. | Apple only writes a flights sample when you climb. If the phone was demonstrably carried, "no flights" really is 0, and leaving those days NaN would bias the mean upward. |
| R10 | The daily table is a continuous calendar from the first to the last observed day. | Gaps become explicit NaN rows for coverage and differencing. |
| R11 | `daily`, `metric_catalog` and `daily_sources` are dropped and rebuilt on every rollup. | Derived data is always reproducible from raw, with no migration logic. |

## Scope change: nutrition, weight and energy balance removed (2026-10-05)

The first build included a nutrition logging-coverage flag (`well_logged`), a weight EWMA trend, an
energy-balance (empirical TDEE) module and a weight forecast as an optional module. The project's focus is activity and gait data, so these were **removed entirely** and Phase 6 was replaced by changepoint detection
(K1–K12). Their decisions (former N1–N4, P1, E1–E9) are in git history (earlier commits).

## Data quality

| # | Decision | Rationale |
|---|---|---|
| Q1 | Outliers: \|value − centred 28-day rolling median\| > 5 × scaled MAD. The local MAD is floored at 0.5 × the series-wide MAD. | Robust to the outliers being hunted and adapts to slow trends. k=5 flags only gross anomalies (partial days, sensor glitches). The floor prevents flag storms on near-constant stretches. |
| Q2 | Robust outlier flags are skipped for count / `zero_fill` metrics, which only get plausibility checks. | Zero-inflated series have a degenerate MAD. |
| Q3 | Plausibility ranges per metric in config (e.g. steps 0–60,000, basal 800–3,000 kcal). | They catch unit errors and duplicated imports, and are deliberately wide. |
| Q4 | Gaps are reported only between a metric's first and last observation, for runs ≥ 3 days. | Before the first sample the metric didn't exist. Gaps of 1–2 days are routine. |
| Q5 | Activity timing (share of steps per local hour by month, in resolver-local time) is shown on the Data Quality page. | Travel *not* covered by an override shows up as a shifted daily rhythm, and correctly configured overrides make the band disappear. It's a visual check of T1. |

## Trends & presentation

| # | Decision | Rationale |
|---|---|---|
| P1 | *(removed with the weight module, see Scope change)* | |
| P2 | Rolling means are trailing and need ≥ 50% of the window observed. Smoothing runs on the full history before the date filter. | No look-ahead, gaps don't bias the mean toward zero, and the left edge of the view isn't artificially cold. |
| P3 | The weekly/monthly view shows the mean ± interquartile range, hiding periods with < 3 (weekly) or < 10 (monthly) observed days. | Sparse periods would otherwise read as real swings. |
| P4 | The day-of-week Kruskal–Wallis p is shown with an "optimistic" caveat. | Daily values are autocorrelated, so the effective n is below the day count. |
| P5 | Charts use a validated colour-blind-safe categorical order and never a dual y-axis. Two series of different units go in stacked shared-x panels. | Dual axes invite spurious visual correlation, which this app is specifically trying to avoid. |
| P6 | Quality pages ignore `analysis.start_date`, while analysis pages respect it. | Quality diagnostics should cover the whole history, including the sparse early years. |
| P7 | Streamlit binds to 127.0.0.1 with usage statistics off (`.streamlit/config.toml` and the CLI). | The privacy requirement is no network traffic involving health data. |

## Correlation scan

| # | Decision | Rationale |
|---|---|---|
| C1 | **Spearman** ρ, not Pearson. | Daily health metrics are skewed and outlier-prone (partial days, a 30k-step hike), and rank correlation is robust to both and to monotone non-linearity. |
| C2 | **Default transform: deviation from a centred 28-day rolling median ("detrend")**. First difference and raw are toggles. | Raw series share trends and seasons, so p-values are wildly anti-conservative (independent random walks come out "significant" most of the time; see `tests/test_correlations.py`). First-differencing removes trends but induces a ≈ −ρ/2 lag-1 cross-correlation from any same-day correlation, which floods the lagged results with artifacts. Rolling-median detrending removes slow structure without that artifact. The median makes it robust to the outliers it would otherwise chase. |
| C3 | With differencing, lag-1 rows whose sign opposes a non-trivial (\|ρ₀\| ≥ 0.1) same-day correlation are flagged `diff_artifact` and ranked last. | They're expected under the null of no lagged effect (C2). |
| C4 | **Effective-n correction on by default**: n_eff = n(1 − r₁ₓr₁ᵧ)/(1 + r₁ₓr₁ᵧ) (Bartlett), with p from the t-approximation at n_eff − 2 df. It never increases n. | The transforms reduce but don't eliminate autocorrelation. This is a cheap, standard first-order correction. Higher-order corrections (Pyper–Peterman) were judged overkill for this data size. |
| C5 | **Benjamini–Hochberg across every test in a scan** (all pairs × all lags), at q = 0.05. | The scan is exploratory and FDR is the appropriate error rate (Bonferroni/FWER would be needlessly strict). The family is the whole scan, so choosing a lag after looking doesn't escape correction. ⚠️ BH assumes independence or positive dependence among tests. Lagged tests of the same pair are positively dependent, which BH tolerates (PRDS). |
| C6 | Lags 0–3 days. Lag k pairs x(t) with y(t+k). Lag 0 is tested once per unordered pair, lags ≥ 1 in both directions. | Plausible physiological and behavioural carry-over is short. Both directions let the table separate "x leads y" from "y leads x". |
| C7 | Minimum overlap **n = 60** days (configurable). | Below ~60, Spearman's SE (≈ 1/√n ≈ 0.13) is too large for the small effects expected here, and those tests would only add to the multiple-testing burden. |
| C8 | Metrics need ≥ 30 observed days to enter any analysis (`analysis.min_coverage_days`). | Very sparse metrics can't reach min_n in any pair anyway. |
| C9 | **Mechanical pairs are flagged and ranked last**: definitional (steps↔distance, active energy↔steps), biomechanical (double support↔speed/step length), and the **basal-energy artifact** (basal↔steps/distance/gait). | On iPhone-only data, basal energy correlates strongly with steps day to day. The iPhone basal estimate only accrues while the phone records, so its daily total partly measures phone-carry time rather than metabolism. |
| C10 | The table leads with effect size (ρ) and caveats that \|ρ\| < 0.1 can be "significant" yet negligible with thousands of days. | Statistical significance isn't practical significance. |
| C11 | Persistence caveat: lagged associations between metrics that already correlate on the same day may reflect multi-day states, not lead–lag. The drill-down shows the same-day ρ next to the lagged ρ. | A full Granger-style partial test was judged out of scope. The UI makes the comparison easy instead. |

## Seasonal comparison

| # | Decision | Rationale |
|---|---|---|
| S1 | Periods are config labels → lists of inclusive date ranges. Overlapping differently labelled ranges are an error. | Supports "every summer in Bulgaria" as one label, and ambiguous days would contaminate both groups. |
| S2 | When fewer than two periods are configured, the page falls back to **calendar months** (default Jun–Aug vs the rest). | The shipped config has no travel ranges, because they're personal. ⚠️ Fill in `seasonal.periods` for the intended comparison. |
| S3 | **Mann–Whitney U** (two-sided) for the location/dominance test. | Distribution-free and robust to the skew and outliers in daily metrics. It tests stochastic dominance, which matches the effect size reported. |
| S4 | **Cliff's δ** as the effect size, computed as 2U/(n₁n₂) − 1 (identical to the rank-biserial r). Magnitude labels follow Romano et al. (2006): 0.147 / 0.33 / 0.474. | It's the effect size that pairs naturally with MWU, it's ordinal and interpretable as P(A>B) − P(A<B), and it isn't driven by outliers. |
| S5 | **Bootstrap 95% CI for the difference in medians** with 2,000 reps, percentile method, fixed seed. Both an **iid** and a **7-day moving-block** CI are shown. | The median difference is in interpretable units (e.g. +2,100 steps). The iid bootstrap assumes independent days. Blocks of one week keep weekly seasonality and short-range autocorrelation inside resamples, so that CI is the honest one. The percentile method was chosen over BCa for simplicity given ≥ 60 days per group. |
| S6 | A toggle compares **weekly means** (weeks with ≥ 4 observed days) instead of days. | Weeks are much closer to independent, so MWU on weeks is a conservative sensitivity check on the daily p. |
| S7 | BH-FDR across the metrics compared in one A-vs-B comparison. | Same reasoning as C5, with the family being one comparison's metrics. |
| S8 | A confounds panel is always shown: environment, schedule, the LA-timezone artifact, carry habits, weather, device changes, and few periods. | With n = 1 person and a handful of summers, period differences are descriptive, not attributable. |

## Changepoint detection (replaces the energy-balance module)

| # | Decision | Rationale |
|---|---|---|
| K1 | Metrics: steps, distance, walking speed, step length, asymmetry and double support (configurable). **Weekly means** of daily values by default, with weeks needing ≥ 4 observed days. A daily resolution is a toggle. | Weekly bins remove the day-of-week cycle and most day-to-day autocorrelation, and life regimes last weeks to months. The changepoint is reported as the first week of the new regime, so date precision is ±1 week. |
| K2 | Steps and distance are segmented on a **log scale**. Gait metrics are segmented raw. | Activity changes are multiplicative (−30%, not −2,000 steps at every level). Gait metrics are bounded and roughly additive. |
| K3 | **PELT** (`ruptures`, L2 cost, i.e. mean shifts), minimum segment 4 bins. | It finds the exact optimum of penalised segmentation in ~linear time. The 4-week minimum prevents one odd week becoming a regime, so regimes shorter than a month aren't reported. |
| K4 | **Penalty = BIC on a standardised series: 2·log(n) per changepoint** (each adds a location and a new mean), times a user multiplier (default 1). | It's a standard, data-size-aware criterion with no tuning to the answer. The multiplier exists only for the sensitivity view. |
| K5 | **Standardisation by a robust long-run noise scale from first differences, assuming AR(1) noise**: φ̂ = 1 + 2·corr(Δxₜ, Δxₜ₋₁) (Spearman, converted to Pearson), σ = MAD(Δx)·1.4826/√(2(1−φ)), σ_LR = σ·√((1+φ)/(1−φ)). | Differences are almost unaffected by the mean shifts being detected. Positively autocorrelated noise wanders, and scaling by the long-run SD keeps that wander from being called a regime shift. A first version (residual-from-rolling-median φ) underestimated φ = 0.6 as 0.30 and produced ~10 false changepoints per pure-noise series (caught by `test_autocorrelation_adjustment_reduces_false_changepoints`). |
| K6 | **φ is shrunk to its one-sided 95% lower bound**, φ̂ − 1.645·2/√n, clipped to [0, 0.8]. | On ~80 weekly bins SE(φ̂) ≈ 0.22. Using φ̂ directly made the effective penalty vary randomly between metrics (one metric lost all its true changepoints at φ̂ = 0.38 from sampling noise alone). Only demonstrable autocorrelation inflates the scale. Long daily series keep most of the correction. |
| K7 | The standardised series is **winsorised at ±3**. | Daily steps are lognormal-ish, so an occasional week runs at twice the norm. Without clipping, one such week could drag a whole 4-week segment. |
| K8 | **Sensitivity view**: rerun at 0.25×–8× BIC. A changepoint's *stability* is the share of multipliers that place one within ±2 bins. | Shows whether a changepoint is robust or penalty-dependent, which is more informative than any single p-value. |
| K9 | **BOCPD comparison** (Adams & MacKay 2007): Normal–Gamma prior (μ₀=0, κ₀=0.5, α₀=β₀=2) on the same standardised series, constant hazard 1/26 per bin. Changepoint = a new MAP run that persists ≥ 4 bins. | An independent, online Bayesian method. Agreement with PELT is reassuring, and BOCPD typically flags a few bins later because it only uses the past. |
| K10 | **Before/after effect sizes on daily values** of the adjacent segments: median difference, % change, Cliff's δ, and a 7-day block-bootstrap 95% CI. | Interpretable magnitude of each shift. ⚠️ The boundaries were chosen by the same data, so these are descriptive (selection makes them optimistic) and not a test of whether the shift is real. Stability and matching events are the evidence for that. |
| K11 | **Annotation**: life events from config (`changepoints.life_events`); **device events** = new hardware *or a new major iOS version* on a source selected on ≥ 5% of source-days; travel = override period starts and ends. A changepoint "matches" an event within ±21 days. | iOS updates can change Apple's gait algorithms. A shift that hits every gait metric in the same week as a software update is the signature of a measurement change. The ≥ 5% filter stops a barely used secondary device's iOS update from "explaining" shifts in your own phone's data. 21 days = 3 weekly bins. |
| K12 | The synthetic data plants 20 changepoints across 6 metrics (summer starts/ends, a −30% commute, a 6-week injury, a phone upgrade that shifts measured step length and double support). The e2e test requires each one found within ±21 days, with at most 1 unexplained detection per metric. | Currently 12 are found in the exact week, 6 within 2–4 days, and 2 (one summer end in steps and distance) 20 days late because the following weeks were high by chance. 22 of 24 detections match a known event. |

## Forecasting

| # | Decision | Rationale |
|---|---|---|
| F1 | Horizon **7 days**. Targets are configurable (`forecast.targets`), defaulting to **daily steps and walking speed**. | A week is the actionable horizon for behaviour. The weight-trend forecast was removed with the weight module. |
| F2 | **Baselines always shown**: naive (last observed value) and seasonal naive (same weekday last week, falling back up to 4 weeks if missing). Skill is reported relative to seasonal naive. | A model is only interesting if it beats these. On weekly-cyclic data such as daily steps, seasonal naive is usually the stronger baseline. |
| F3 | **Rolling origin via `TimeSeriesSplit(n_splits=12, test_size=7)`**: expanding window with training capped at the last **730 days**. | Never trains on the future. The cap keeps SARIMAX fits fast and lets the models track regime changes (moves, new phone). Twelve weekly folds is a ~3-month evaluation period. ⚠️ With that few folds, differences of a few % are noise, so the UI reports fold win counts as well. |
| F4 | **SARIMAX(1,0,1)(1,0,1,7) + constant**, fixed order, no per-fold order search. NaNs go to the Kalman filter. | A small, standard weekly-seasonal spec. Searching orders per fold on a few hundred points would overfit the evaluation itself. |
| F5 | **HistGradientBoosting, direct multi-horizon**: one model with horizon as a feature. Features are the value at the origin, 7/28-day means to the origin, the target weekday, and lags 7, 8, 14, 21, 28 *relative to the target* (masked when they'd fall after the origin). Shallow trees, 200 iterations, lr 0.05, min leaf 20, L2 = 1. | Direct strategy avoids recursive error accumulation. The masking rule guarantees no look-ahead (unit-tested). Heavy regularisation suits small, noisy data, and HistGB handles missing days natively. |
| F6 | The verdict text is generated from the numbers. "Beats" requires > 2% lower MAE than the best baseline. Otherwise it reads "does not meaningfully beat" or "is worse than". | Requirement: report honestly even when models lose. |
| F7 | The **export day is dropped** from the daily table, since it was exported mid-day. | A partial final day (a fraction of a normal day's steps) looks like a crash in activity and wrecks the naive baseline. |

## Synthetic data & demo mode

| # | Decision | Rationale |
|---|---|---|
| Y1 | The synthetic export mimics a real iPhone export: the same XML layout, imperial units (`mi`, `Cal`, `mi/hr`, `in`, `lb`, `%` as a fraction), and every timestamp rendered in a fixed `-0700`. A few summer records are rendered at `+0300`. | It exercises the real parsing, unit-conversion and dedup paths, including the export-offset quirk (D3). |
| Y2 | Summer days (`demo.timezone_overrides`) are generated as **Europe/Sofia-local days**, and all other days as LA days. Samples of LA days that fall inside an override window are dropped (travel day). `truth.json` holds per-day true steps and the travel days. | The override must be needed and sufficient to recover daily totals. The e2e test checks both. |
| Y3 | Planted effects: steps → walking speed at lag 1; flights climbed → walking asymmetry at lag 2; +35% summer steps; a −30% car-commute regime from day 230; a 6-week injury (−0.35 km/h speed, +4 pp asymmetry, +1.5 pp double support); a phone upgrade on day 300 shifting measured step length (+3 cm) and double support (−0.8 pp); sleep in the last 14 days only. Daily step noise: lognormal AR(1), φ = 0.3, SD 0.18. All values are in `synth.TRUTH_PARAMS` and written to `truth.json`, including per-metric planted changepoint dates. | Each analysis has something specific to recover, at realistic signal-to-noise. The step-noise persistence was lowered from 0.5 after the first summer was not identifiable at BIC in that realisation (see K12). |
| Y4 | The end-to-end test asserts recovery with tolerances: correct best lag with \|ρ\| > 0.25 and q < 10⁻³; summer δ > 0.3 with block CI excluding 0; every planted changepoint within ±21 days; summer daily totals exact with the time-zone override; the foreign phone excluded. | It's a regression test for the methodology, not just for the code. |
| Y5a | The synthetic phone reports `iPhone12,1` and is upgraded to `iPhone15,2` on day 300. A foreign "Partner's iPhone" writes steps on the demo phone's missing days and on ~5% of other days. Demo mode excludes it. | It exercises the hardware timeline and shows the hazard exclusion fixes: the foreign phone would otherwise fill missing days with someone else's steps (asserted in the e2e test). |
| Y5 | Synthetic event times are made unique per second. | Identical (type, value, start, end) samples are deduplicated by design (I5). Real sensors don't emit them, and the generator shouldn't either. |

## Items to review when using your own data

1. **T1**: add your trips (e.g. summers abroad) as `time.overrides` (they also become the `bulgaria_summer` seasonal period and travel events on the Changepoints page). Check the activity-timing heatmap afterwards: correctly covered summers lose their shifted band.
2. **R3a**: review the Sources & devices table and exclude any source that isn't yours (a family member's phone, an old iPod) with `source_exclude`.
3. **K11**: add life events to `changepoints.life_events` so unexplained changepoints on your data can be labelled. Shifts that coincide with an iOS update or a phone upgrade are likely measurement artifacts.
4. **C9**: whether basal energy belongs in the correlation scan at all, given it's partly a phone-carry artifact on iPhone-only data.
5. **S2**: add `school_year` ranges under `seasonal.periods`.
6. **analysis.start_date**: early years are often sparse (older devices, partial coverage), and early changepoints can be coverage changes rather than behaviour. Consider starting the analysis later.
