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
