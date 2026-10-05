# Methodology decisions

Every assumption or judgment call in the analysis design, with a one-line
rationale. Parameter values live in `config/healthlab.yaml`. This file explains
*why* each value is what it is. Entries are grouped by pipeline stage. ⚠️ marks
choices worth revisiting once more data exists.

## Data facts that shaped the design (observed in a typical iPhone-only export)

| # | Observation | Consequence |
|---|---|---|
| D1 | Several hundred thousand records across ~11 types: steps, distance, flights, walking speed/step length/double support/asymmetry, headphone audio (+ alert events), active/basal energy, and a sleep-duration goal. A handful of workouts. | The core app is built around activity, gait and audio metrics. |
| D2 | **No nutrition, body-weight or sleep samples.** | Nutrition coverage, energy balance and the weight trend/forecast are an *optional module*. They activate only when those types appear (they always do in demo mode). |
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
| T1 | **Local dates use `America/Los_Angeles` (DST-aware) applied to the UTC instant**, for all records, including summers in Bulgaria. | Owner's decision: the export doesn't preserve original offsets (D3). ⚠️ Known error: during Bulgaria stays, the LA date boundary falls at about 10:00 Sofia time, so mornings are attributed to the previous day. Daily *totals* over a stay are unaffected, but day-to-day alignment (and lagged correlations) are noisier in those weeks. |
| T2 | `local_date_mode: offset` remains available. | Correct for exports that preserve offsets, and lets tests cover both behaviours. |
| T3 | A record's day is the local date of its **start** time. `local_end_date` is also stored. | That's the convention Apple's own daily charts use. Sleep uses the end date instead (R5). |

## Daily rollup

| # | Decision | Rationale |
|---|---|---|
| R1 | Aggregation per type comes from `metrics:` in config. Cumulative types (steps, distance, flights, energy, nutrients) use **sum**. Gait samples use **mean**. Weight uses **mean** of the day's readings. | Totals are the natural daily quantity for cumulative types. For weight, the mean of same-day weigh-ins reduces measurement noise, and the EWMA handles day-to-day water noise anyway. |
| R2 | Headphone exposure uses **LAeq** (duration-weighted energy average: 10·log10 of the mean of 10^(L/10)) instead of the arithmetic mean of dB. | Decibels are logarithmic. The arithmetic mean understates exposure, and LAeq is the standard occupational/WHO metric. |
| R3 | **Source dedup**: for each (type, local day), only the highest-ranked source with data that day is used. Ranking = configured `source_priority` list, then sources by total record count (ties alphabetical). | Summing iPhone + app (or old + new phone) double-counts. Day-level selection keeps each day internally consistent. Falling back to a lower-ranked source fills days when the main source is absent (e.g. the old iPhone era). |
| R4 | Cumulative samples that cross local midnight are **prorated by time overlap** across the days. | Some step samples span up to ~10 h, and assigning them wholly to the start day biases both days. |
| R5 | Sleep intervals go to the **local date of waking** (interval end). Overlapping intervals of the same class are merged before summing. Naps also count toward their end date. | Owner's spec. Merging avoids double-counting overlapping core/deep segments or two sources. Not analysed yet (D2). |
| R6 | Raw units are converted to canonical metric units: mi→km, Cal→kcal, mi/hr→km/h, in→cm, lb→kg, and the HealthKit `%` fraction → 0–100. | Analyses need consistent units, and display conversion (kg/lb, km/mi) happens only in the UI. |
| R7 | Unlisted types get a default rule. A canonical unit in `default_rules.cumulative_units` means sum, otherwise mean. Category types are counted. Unlisted `Dietary*` types are treated as nutrition sums. | Generic ingest without silently dropping new types. The `metric_catalog.configured` flag shows which ones were defaulted. |
| R8 | A day with **no records is NaN (missing), never 0**. | Absence of data isn't evidence of zero activity or zero intake. |
| R9 | Exception: `zero_fill` metrics (flights climbed, headphone alerts) are 0 on days when **steps** were recorded, and NaN otherwise. | Apple only writes a flights sample when you climb. If the phone was demonstrably carried, "no flights" really is 0, and leaving those days NaN would bias the mean upward. Not applied to nutrition. |
| R10 | The daily table is a continuous calendar from the first to the last observed day. | Gaps become explicit NaN rows for coverage and differencing. |
| R11 | `daily`, `metric_catalog` and `daily_sources` are dropped and rebuilt on every rollup. | Derived data is always reproducible from raw, with no migration logic. |

## Nutrition logging coverage (optional module)

| # | Decision | Rationale |
|---|---|---|
| N1 | `n_food_entries` = number of dietary-energy records from the selected source that day. | Food apps write one energy record per logged item/meal, so the count is a proxy for logging effort. |
| N2 | `well_logged` = **≥ 3 entries AND 1200 ≤ kcal ≤ 5000**. | Fewer than 3 entries usually means a skipped meal. Under 1200 kcal for an adult is far more often partial logging than a real intake. Over 5000 kcal is usually a duplicate or entry error. All three thresholds are configurable. |
| N3 | **Every nutrition analysis sees only well-logged days**: `load_daily()` masks nutrition columns to NaN on other days by default. | Including partial days would bias intake down and make empirical TDEE look low. Enforced at one choke point. |
| N4 | ⚠️ Well-logged days may still be under-reported (forgotten snacks, portion underestimates). | The coverage flag catches *missing* logging, not *inaccurate* logging. See the energy-balance section. |

## Data quality

| # | Decision | Rationale |
|---|---|---|
| Q1 | Outliers: \|value − centred 28-day rolling median\| > 5 × scaled MAD. The local MAD is floored at 0.5 × the series-wide MAD. | Robust to the outliers being hunted and adapts to slow trends. k=5 flags only gross anomalies (partial days, sensor glitches). The floor prevents flag storms on near-constant stretches. |
| Q2 | Robust outlier flags are skipped for count / `zero_fill` metrics, which only get plausibility checks. | Zero-inflated series have a degenerate MAD. |
| Q3 | Plausibility ranges per metric in config (e.g. steps 0–60,000, basal 800–3,000 kcal). | They catch unit errors and duplicated imports, and are deliberately wide. |
| Q4 | Gaps are reported only between a metric's first and last observation, for runs ≥ 3 days. | Before the first sample the metric didn't exist. Gaps of 1–2 days are routine. |
| Q5 | Activity timing (share of steps per local hour by month) is shown on the Data Quality page. | It makes the LA-only timezone limitation (T1) visible: travel shows up as a shifted daily rhythm. |

## Trends & presentation

| # | Decision | Rationale |
|---|---|---|
| P1 | Weight trend = **time-aware EWMA, 10-day half-life** (α ≈ 0.067 per day), with daily values interpolated between weigh-ins. | Scale noise is ±0.5–1 kg/day (water, glycogen, gut content). A 10-day half-life damps it ~10× while tracking a real 0.5 kg/week change within ~2 weeks. It's time-aware (`times=`) so irregular weigh-in gaps decay correctly. It's the same idea as Hacker's Diet (α=0.1/day ≈ 6.6-day half-life). I chose slightly smoother because weigh-ins are sparser than daily. |
| P2 | Rolling means are trailing and need ≥ 50% of the window observed. Smoothing runs on the full history before the date filter. | No look-ahead, gaps don't bias the mean toward zero, and the left edge of the view isn't artificially cold. |
| P3 | The weekly/monthly view shows the mean ± interquartile range, hiding periods with < 3 (weekly) or < 10 (monthly) observed days. | Sparse periods would otherwise read as real swings. |
| P4 | The day-of-week Kruskal–Wallis p is shown with an "optimistic" caveat. | Daily values are autocorrelated, so the effective n is below the day count. |
| P5 | Charts use a validated colour-blind-safe categorical order and never a dual y-axis. Two series of different units go in stacked shared-x panels. | Dual axes invite spurious visual correlation, which this app is specifically trying to avoid. |
| P6 | Quality pages see *all* days (unmasked). Analysis pages see masked nutrition and `analysis.start_date`. | Quality diagnostics have to show partial logging. Analyses must not use it. |
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
| C9 | **Mechanical pairs are flagged and ranked last**: definitional (steps↔distance, macros↔calories), biomechanical (double support↔speed/step length), and the **basal-energy artifact** (basal↔steps/distance/gait). | On iPhone-only data, basal energy correlates strongly with steps day to day. The iPhone basal estimate only accrues while the phone records, so its daily total partly measures phone-carry time rather than metabolism. ⚠️ This is also why Apple's basal estimate should be used cautiously in the energy-balance module. |
| C10 | The table leads with effect size (ρ) and caveats that \|ρ\| < 0.1 can be "significant" yet negligible with thousands of days. | Statistical significance isn't practical significance. |
| C11 | Persistence caveat: lagged associations between metrics that already correlate on the same day may reflect multi-day states, not lead–lag. The drill-down shows the same-day ρ next to the lagged ρ. | A full Granger-style partial test was judged out of scope. The UI makes the comparison easy instead. |
