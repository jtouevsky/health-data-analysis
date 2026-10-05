"""SQLite connection helpers and schema."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS import_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file   TEXT NOT NULL,
    file_sha1     TEXT,
    export_date   TEXT,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    status        TEXT NOT NULL DEFAULT 'running',   -- running | ok | failed
    n_records_seen     INTEGER DEFAULT 0,
    n_records_inserted INTEGER DEFAULT 0,
    n_workouts_seen    INTEGER DEFAULT 0,
    n_workouts_inserted INTEGER DEFAULT 0,
    n_parse_errors     INTEGER DEFAULT 0,
    other_elements     TEXT,          -- JSON {tag: count} for elements not ingested
    message       TEXT
);

CREATE TABLE IF NOT EXISTS records (
    id             INTEGER PRIMARY KEY,
    type           TEXT NOT NULL,
    source_name    TEXT,
    source_version TEXT,
    device         TEXT,
    unit           TEXT,
    value          REAL,              -- numeric value (quantity types)
    value_text     TEXT,              -- categorical value (category types) or unparsable value
    start_ts       TEXT NOT NULL,     -- UTC 'YYYY-MM-DD HH:MM:SS'
    end_ts         TEXT NOT NULL,
    creation_ts    TEXT,
    utc_offset     INTEGER,           -- minutes, as written in the export
    local_date     TEXT NOT NULL,     -- local date of start_ts
    local_end_date TEXT NOT NULL,     -- local date of end_ts
    metadata_json  TEXT,
    content_hash   TEXT NOT NULL UNIQUE,
    import_run_id  INTEGER REFERENCES import_runs(id)
);
CREATE INDEX IF NOT EXISTS ix_records_type_date ON records(type, local_date);

CREATE TABLE IF NOT EXISTS workouts (
    id                   INTEGER PRIMARY KEY,
    activity_type        TEXT NOT NULL,
    source_name          TEXT,
    source_version       TEXT,
    device               TEXT,
    duration             REAL,
    duration_unit        TEXT,
    total_distance       REAL,
    total_distance_unit  TEXT,
    total_energy         REAL,
    total_energy_unit    TEXT,
    start_ts             TEXT NOT NULL,
    end_ts               TEXT NOT NULL,
    creation_ts          TEXT,
    utc_offset           INTEGER,
    local_date           TEXT NOT NULL,
    statistics_json      TEXT,        -- WorkoutStatistics children
    metadata_json        TEXT,
    content_hash         TEXT NOT NULL UNIQUE,
    import_run_id        INTEGER REFERENCES import_runs(id)
);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    """Open (creating if needed) a HealthLab database and ensure the schema exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript(SCHEMA)
    return con


def table_exists(con: sqlite3.Connection, name: str) -> bool:
    """True if a table with this name exists."""
    row = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
    return row is not None
