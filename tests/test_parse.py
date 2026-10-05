"""Ingest parsing: generic record handling, metadata, workouts, idempotency."""

import json
import sqlite3

from healthlab.db import connect
from healthlab.ingest import ingest_export
from xmlfactory import rec, workout, write_export

STEPS = "HKQuantityTypeIdentifierStepCount"
SPEED = "HKQuantityTypeIdentifierWalkingSpeed"
SLEEP = "HKCategoryTypeIdentifierSleepAnalysis"
NEW_TYPE = "HKQuantityTypeIdentifierSomethingApplesInventsNext"


def _body() -> str:
    return (
        rec(STEPS, 120, "2024-05-01 08:00:00 -0700", "2024-05-01 08:05:00 -0700", source="Phone\n")
        + rec(STEPS, 80, "2024-05-01 09:00:00 -0700", "2024-05-01 09:05:00 -0700")
        + rec(SPEED, 4.5, "2024-05-01 09:00:00 -0700", unit="km/hr", meta={"HKMetadataKeyDevicePlacementSide": "Left"})
        + rec(SLEEP, "HKCategoryValueSleepAnalysisAsleepCore", "2024-05-01 23:00:00 -0700",
              "2024-05-02 06:30:00 -0700", unit=None)
        + rec(NEW_TYPE, 3.14, "2024-05-01 10:00:00 -0700", unit="widgets")
        + workout("HKWorkoutActivityTypeWalking", "2024-05-01 18:00:00 -0700", "2024-05-01 18:30:00 -0700", 30, kcal=120)
    )


def test_ingest_generic_records(tmp_path, cfg):
    exp = write_export(tmp_path / "export.zip", _body())
    s = ingest_export(exp, cfg, db_path=tmp_path / "h.db")
    assert (s.records_seen, s.records_inserted, s.workouts_inserted, s.parse_errors) == (5, 5, 1, 0)
    assert s.export_date == "2026-10-05 23:13:47"
    assert s.other_elements["Me"] == 1

    con = sqlite3.connect(tmp_path / "h.db")
    types = {r[0] for r in con.execute("SELECT DISTINCT type FROM records")}
    assert NEW_TYPE in types  # unknown types are stored, nothing is hardcoded

    src = con.execute("SELECT DISTINCT source_name FROM records WHERE type=?", (STEPS,)).fetchall()
    assert src == [("Phone",)]  # trailing newline stripped

    v, vt, start, off, ld = con.execute(
        "SELECT value, value_text, start_ts, utc_offset, local_date FROM records WHERE type=?", (SLEEP,)).fetchone()
    assert v is None and vt == "HKCategoryValueSleepAnalysisAsleepCore"
    assert start == "2024-05-02 06:00:00" and off == -420 and ld == "2024-05-01"
    end_ld = con.execute("SELECT local_end_date FROM records WHERE type=?", (SLEEP,)).fetchone()[0]
    assert end_ld == "2024-05-02"

    meta = json.loads(con.execute("SELECT metadata_json FROM records WHERE type=?", (SPEED,)).fetchone()[0])
    assert meta == {"HKMetadataKeyDevicePlacementSide": "Left"}

    w = con.execute("SELECT activity_type, duration, total_energy, total_energy_unit FROM workouts").fetchone()
    assert w == ("HKWorkoutActivityTypeWalking", 30.0, 120.0, "Cal")

    run = con.execute("SELECT status, n_records_inserted FROM import_runs").fetchone()
    assert run == ("ok", 5)


def test_plain_xml_accepted(tmp_path, cfg):
    exp = write_export(tmp_path / "export.xml", _body(), as_zip=False)
    assert ingest_export(exp, cfg, db_path=tmp_path / "h.db").records_inserted == 5


def test_reimport_is_idempotent(tmp_path, cfg):
    exp = write_export(tmp_path / "export.zip", _body())
    db = tmp_path / "h.db"
    ingest_export(exp, cfg, db_path=db)
    s2 = ingest_export(exp, cfg, db_path=db)
    assert s2.records_inserted == 0 and s2.workouts_inserted == 0 and s2.records_duplicate == 5
    con = connect(db)
    assert con.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 5
    assert con.execute("SELECT COUNT(*) FROM import_runs WHERE status='ok'").fetchone()[0] == 2


def test_newer_export_is_incremental(tmp_path, cfg):
    db = tmp_path / "h.db"
    ingest_export(write_export(tmp_path / "a.zip", _body()), cfg, db_path=db)
    newer = _body() + rec(STEPS, 999, "2024-05-03 08:00:00 -0700", "2024-05-03 08:10:00 -0700")
    s = ingest_export(write_export(tmp_path / "b.zip", newer), cfg, db_path=db)
    assert s.records_inserted == 1
    assert connect(db).execute("SELECT COUNT(*) FROM records").fetchone()[0] == 6


def test_same_samples_rendered_in_other_offset_are_duplicates(tmp_path, cfg):
    # An export made while travelling renders identical instants with +0300.
    db = tmp_path / "h.db"
    ingest_export(write_export(tmp_path / "la.zip", rec(STEPS, 50, "2024-05-01 08:00:00 -0700", "2024-05-01 08:05:00 -0700")), cfg, db_path=db)
    s = ingest_export(write_export(tmp_path / "bg.zip", rec(STEPS, 50, "2024-05-01 18:00:00 +0300", "2024-05-01 18:05:00 +0300")), cfg, db_path=db)
    assert s.records_inserted == 0


def test_bad_record_counted_not_fatal(tmp_path, cfg):
    body = _body() + ' <Record type="X" sourceName="P" startDate="not a date" endDate="nope" value="1"/>\n'
    s = ingest_export(write_export(tmp_path / "e.zip", body), cfg, db_path=tmp_path / "h.db")
    assert s.parse_errors == 1 and s.records_inserted == 5


def test_nested_correlation_records_deduplicated(tmp_path, cfg):
    food = rec("HKQuantityTypeIdentifierDietaryEnergyConsumed", 500, "2024-05-01 12:00:00 -0700", unit="kcal")
    body = food + (' <Correlation type="HKCorrelationTypeIdentifierFood" sourceName="App" '
                   'startDate="2024-05-01 12:00:00 -0700" endDate="2024-05-01 12:00:00 -0700">\n'
                   f'{food} </Correlation>\n')
    s = ingest_export(write_export(tmp_path / "c.zip", body), cfg, db_path=tmp_path / "h.db")
    assert s.records_seen == 2 and s.records_inserted == 1
    assert s.other_elements["Correlation"] == 1


def test_batches_smaller_than_input(tmp_path, cfg):
    cfg.raw["ingest"]["batch_size"] = 2
    body = "".join(rec(STEPS, i, f"2024-05-01 10:{i:02d}:00 -0700") for i in range(7))
    s = ingest_export(write_export(tmp_path / "b.zip", body), cfg, db_path=tmp_path / "h.db")
    assert s.records_inserted == 7
