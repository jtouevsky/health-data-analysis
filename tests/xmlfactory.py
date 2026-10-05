"""Helpers for building tiny Apple Health exports in tests."""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.sax.saxutils import quoteattr

HEADER = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE HealthData [
<!ELEMENT HealthData (ExportDate,Me,(Record|Correlation|Workout|ActivitySummary|ClinicalRecord)*)>
<!ATTLIST HealthData locale CDATA #REQUIRED>
]>
<HealthData locale="en_US">
 <ExportDate value="2026-10-05 16:13:47 -0700"/>
 <Me HKCharacteristicTypeIdentifierBiologicalSex="HKBiologicalSexNotSet"/>
"""


def rec(type_: str, value, start: str, end: str | None = None, *, source: str = "Phone",
        unit: str | None = "count", meta: dict | None = None, created: str | None = None) -> str:
    """One <Record> element as XML text."""
    attrs = {"type": type_, "sourceName": source, "sourceVersion": "17.0",
             "creationDate": created or end or start, "startDate": start, "endDate": end or start,
             "value": str(value)}
    if unit is not None:
        attrs["unit"] = unit
    a = " ".join(f"{k}={quoteattr(v)}" for k, v in attrs.items())
    if not meta:
        return f" <Record {a}/>\n"
    inner = "".join(f'  <MetadataEntry key={quoteattr(k)} value={quoteattr(str(v))}/>\n' for k, v in meta.items())
    return f" <Record {a}>\n{inner} </Record>\n"


def workout(activity: str, start: str, end: str, minutes: float, kcal: float | None = None) -> str:
    inner = ""
    if kcal is not None:
        inner = (f'  <WorkoutStatistics type="HKQuantityTypeIdentifierActiveEnergyBurned" '
                 f'startDate="{start}" endDate="{end}" sum="{kcal}" unit="Cal"/>\n')
    return (f' <Workout workoutActivityType="{activity}" duration="{minutes}" durationUnit="min" '
            f'sourceName="Phone" sourceVersion="1" creationDate="{end}" startDate="{start}" endDate="{end}">\n'
            f'{inner} </Workout>\n')


def write_export(path: Path, body: str, as_zip: bool = True) -> Path:
    """Write HEADER + body + footer to export.zip (or export.xml)."""
    xml = HEADER + body + "</HealthData>\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if as_zip:
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("apple_health_export/export.xml", xml)
    else:
        path.write_text(xml, encoding="utf-8")
    return path
