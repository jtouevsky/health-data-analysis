"""Streaming ingest of an Apple Health export into SQLite.

The export XML can be hundreds of MB, so it is parsed incrementally with
``iterparse`` directly from the zip archive. Each ``Record``/``Workout`` element
is converted to a row and then cleared, and finished children of the root are
dropped, keeping memory flat. Rows are inserted in batches with
``INSERT OR IGNORE`` against a content hash, which makes re-importing the same
or a newer export idempotent and incremental.

Nothing about the set of record types is hardcoded: every ``Record`` is stored
generically and the rollup step decides how to aggregate it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import zipfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Iterator
from xml.etree.ElementTree import Element, iterparse

from .config import Config
from .db import connect
from .timeutil import local_date, offset_minutes, parse_apple_ts, to_utc_str

log = logging.getLogger(__name__)

RECORD_COLS = (
    "type", "source_name", "source_version", "device", "unit", "value", "value_text",
    "start_ts", "end_ts", "creation_ts", "utc_offset", "local_date", "local_end_date",
    "metadata_json", "content_hash", "import_run_id",
)
WORKOUT_COLS = (
    "activity_type", "source_name", "source_version", "device", "duration", "duration_unit",
    "total_distance", "total_distance_unit", "total_energy", "total_energy_unit",
    "start_ts", "end_ts", "creation_ts", "utc_offset", "local_date",
    "statistics_json", "metadata_json", "content_hash", "import_run_id",
)


@dataclass
class IngestStats:
    """Counters reported for one import run."""

    run_id: int = 0
    records_seen: int = 0
    records_inserted: int = 0
    workouts_seen: int = 0
    workouts_inserted: int = 0
    parse_errors: int = 0
    export_date: str | None = None
    other_elements: Counter = field(default_factory=Counter)

    @property
    def records_duplicate(self) -> int:
        return self.records_seen - self.parse_errors - self.records_inserted


# ---------------------------------------------------------------------------
# Opening the export
# ---------------------------------------------------------------------------

@contextmanager
def open_export(path: str | Path) -> Iterator[IO[bytes]]:
    """Open ``export.xml`` from a zip archive or a bare XML file, as a byte stream."""
    path = Path(path)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            candidates = [n for n in zf.namelist() if n.endswith("/export.xml") or n == "export.xml"]
            if not candidates:
                raise FileNotFoundError(f"no export.xml inside {path}")
            # Prefer the shallowest match (apple_health_export/export.xml).
            name = min(candidates, key=lambda n: n.count("/"))
            with zf.open(name) as fh:
                yield fh
    else:
        with open(path, "rb") as fh:
            yield fh


def file_sha1(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-1 of a file, streamed (used only to label import runs)."""
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Element -> row conversion
# ---------------------------------------------------------------------------

def _clean(s: str | None) -> str | None:
    """Strip whitespace (some apps write source names with trailing newlines)."""
    if s is None:
        return None
    s = s.strip()
    return s or None


def _metadata(el: Element) -> dict[str, str]:
    return {m.get("key", ""): m.get("value", "") for m in el if m.tag == "MetadataEntry"}


def content_hash(*parts: object) -> str:
    """Stable hash identifying a sample independently of how the export rendered it.

    Callers pass normalised fields (UTC timestamps, stripped strings), so the same
    sample re-exported with a different display offset hashes identically.
    """
    joined = "\x1f".join("" if p is None else str(p) for p in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def _float_or_none(s: str | None) -> float | None:
    if s is None:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def record_to_row(el: Element, cfg: Config, run_id: int) -> tuple:
    """Convert a ``<Record>`` element to a ``records`` row tuple (see RECORD_COLS)."""
    mode, tz = cfg.local_date_mode, cfg.timezone
    start = parse_apple_ts(el.get("startDate") or el.get("creationDate"))
    end = parse_apple_ts(el.get("endDate") or el.get("startDate"))
    created_raw = el.get("creationDate")
    created = parse_apple_ts(created_raw) if created_raw else None

    rtype = el.get("type")
    source = _clean(el.get("sourceName"))
    unit = _clean(el.get("unit"))
    raw_value = el.get("value")
    value = _float_or_none(raw_value)
    value_text = raw_value if value is None else None
    start_utc, end_utc = to_utc_str(start), to_utc_str(end)
    meta = _metadata(el)

    h = content_hash(rtype, source, unit, raw_value, start_utc, end_utc)
    return (
        rtype, source, _clean(el.get("sourceVersion")), el.get("device"), unit, value, value_text,
        start_utc, end_utc, to_utc_str(created) if created else None, offset_minutes(start),
        local_date(start, mode, tz).isoformat(), local_date(end, mode, tz).isoformat(),
        json.dumps(meta, ensure_ascii=False) if meta else None, h, run_id,
    )


def workout_to_row(el: Element, cfg: Config, run_id: int) -> tuple:
    """Convert a ``<Workout>`` element to a ``workouts`` row tuple (see WORKOUT_COLS)."""
    start = parse_apple_ts(el.get("startDate"))
    end = parse_apple_ts(el.get("endDate"))
    created_raw = el.get("creationDate")
    stats = [dict(c.attrib) for c in el if c.tag == "WorkoutStatistics"]
    meta = _metadata(el)

    # Older exports carry totals as attributes; newer ones only as WorkoutStatistics.
    dist, dist_u = _float_or_none(el.get("totalDistance")), el.get("totalDistanceUnit")
    energy, energy_u = _float_or_none(el.get("totalEnergyBurned")), el.get("totalEnergyBurnedUnit")
    for s in stats:
        t = s.get("type", "")
        if energy is None and t.endswith("ActiveEnergyBurned"):
            energy, energy_u = _float_or_none(s.get("sum")), s.get("unit")
        if dist is None and "Distance" in t:
            dist, dist_u = _float_or_none(s.get("sum")), s.get("unit")

    activity = el.get("workoutActivityType", "")
    source = _clean(el.get("sourceName"))
    start_utc, end_utc = to_utc_str(start), to_utc_str(end)
    h = content_hash(activity, source, start_utc, end_utc, el.get("duration"))
    return (
        activity, source, _clean(el.get("sourceVersion")), el.get("device"),
        _float_or_none(el.get("duration")), el.get("durationUnit"),
        dist, dist_u, energy, energy_u,
        start_utc, end_utc, to_utc_str(parse_apple_ts(created_raw)) if created_raw else None,
        offset_minutes(start), local_date(start, cfg.local_date_mode, cfg.timezone).isoformat(),
        json.dumps(stats) if stats else None,
        json.dumps(meta, ensure_ascii=False) if meta else None, h, run_id,
    )


# ---------------------------------------------------------------------------
# Main ingest loop
# ---------------------------------------------------------------------------

def _insert(con: sqlite3.Connection, table: str, cols: tuple[str, ...], rows: list[tuple]) -> int:
    if not rows:
        return 0
    sql = f"INSERT OR IGNORE INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"
    before = con.total_changes
    con.executemany(sql, rows)
    return con.total_changes - before


def ingest_export(path: str | Path, cfg: Config, db_path: str | Path | None = None,
                  con: sqlite3.Connection | None = None) -> IngestStats:
    """Stream an Apple Health export into the database.

    Args:
        path: ``export.zip`` or ``export.xml``.
        cfg: configuration (time zone, batch size).
        db_path: target database; defaults to ``cfg.db_path``.
        con: an existing connection (takes precedence over ``db_path``).

    Returns:
        Counters for the run. The run is also logged to ``import_runs``.
    """
    own = con is None
    con = con or connect(db_path or cfg.db_path)
    batch_size = int(cfg.section("ingest").get("batch_size", 10000))
    stats = IngestStats()

    started = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    cur = con.execute(
        "INSERT INTO import_runs (source_file, file_sha1, started_at) VALUES (?, ?, ?)",
        (str(path), file_sha1(path), started),
    )
    stats.run_id = cur.lastrowid
    con.commit()

    rec_batch: list[tuple] = []
    wk_batch: list[tuple] = []
    try:
        with open_export(path) as fh:
            depth = 0
            root: Element | None = None
            for event, el in iterparse(fh, events=("start", "end")):
                if event == "start":
                    depth += 1
                    if root is None:
                        root = el
                    continue
                depth -= 1
                tag = el.tag
                if tag == "Record":
                    stats.records_seen += 1
                    try:
                        rec_batch.append(record_to_row(el, cfg, stats.run_id))
                    except (ValueError, TypeError) as exc:
                        stats.parse_errors += 1
                        log.debug("skipping unparsable record: %s", exc)
                    if len(rec_batch) >= batch_size:
                        stats.records_inserted += _insert(con, "records", RECORD_COLS, rec_batch)
                        con.commit()
                        rec_batch.clear()
                elif tag == "Workout":
                    stats.workouts_seen += 1
                    try:
                        wk_batch.append(workout_to_row(el, cfg, stats.run_id))
                    except (ValueError, TypeError) as exc:
                        stats.parse_errors += 1
                        log.debug("skipping unparsable workout: %s", exc)
                elif tag == "ExportDate":
                    v = el.get("value")
                    stats.export_date = to_utc_str(parse_apple_ts(v)) if v else None
                elif depth == 1 and tag not in ("MetadataEntry",):
                    stats.other_elements[tag] += 1

                # Free memory: once a direct child of the root is complete, drop it.
                # (Records nested inside Correlation elements are kept until the
                # Correlation closes; content hashing dedups them against top-level copies.)
                if depth == 1 and root is not None:
                    el.clear()
                    root.clear()

        stats.records_inserted += _insert(con, "records", RECORD_COLS, rec_batch)
        stats.workouts_inserted += _insert(con, "workouts", WORKOUT_COLS, wk_batch)
        _finish_run(con, stats, "ok", None)
    except Exception as exc:
        con.rollback()
        _finish_run(con, stats, "failed", repr(exc))
        raise
    finally:
        if own:
            con.close()
    return stats


def _finish_run(con: sqlite3.Connection, s: IngestStats, status: str, message: str | None) -> None:
    con.execute(
        """UPDATE import_runs SET finished_at=?, status=?, export_date=?, n_records_seen=?,
           n_records_inserted=?, n_workouts_seen=?, n_workouts_inserted=?, n_parse_errors=?,
           other_elements=?, message=? WHERE id=?""",
        (
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), status, s.export_date,
            s.records_seen, s.records_inserted, s.workouts_seen, s.workouts_inserted,
            s.parse_errors, json.dumps(dict(s.other_elements)), message, s.run_id,
        ),
    )
    con.commit()
