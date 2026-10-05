"""Timestamp parsing and local-date assignment.

Apple Health exports timestamps as ``"2024-05-01 08:00:00 -0700"``. We store the
UTC instant (as ``"YYYY-MM-DD HH:MM:SS"`` text, which sorts correctly in SQLite)
plus the offset as written, and derive a *local date* for daily rollups.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

UTC_FMT = "%Y-%m-%d %H:%M:%S"


def parse_apple_ts(s: str) -> datetime:
    """Parse an Apple Health timestamp into an aware datetime.

    Accepts ``"YYYY-MM-DD HH:MM:SS ±HHMM"`` (the export format) and, leniently,
    ISO-8601 strings with or without the space before the offset.

    Raises:
        ValueError: if the string has no UTC offset.
    """
    s = s.strip()
    if len(s) >= 25 and s[19] == " ":
        s = s[:19] + s[20:]
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        raise ValueError(f"timestamp without UTC offset: {s!r}")
    return dt


def offset_minutes(dt: datetime) -> int:
    """UTC offset of an aware datetime, in minutes (e.g. -420 for -0700)."""
    off = dt.utcoffset()
    assert off is not None
    return int(off.total_seconds() // 60)


def to_utc_str(dt: datetime) -> str:
    """Aware datetime -> naive UTC text ``YYYY-MM-DD HH:MM:SS``."""
    return dt.astimezone(timezone.utc).strftime(UTC_FMT)


@lru_cache(maxsize=16)
def _zone(name: str) -> ZoneInfo:
    return ZoneInfo(name)


def local_date(dt: datetime, mode: str, tz_name: str) -> date:
    """Calendar date a timestamp belongs to.

    Args:
        dt: aware datetime as parsed from the export.
        mode: ``"zone"`` converts the instant into ``tz_name`` (DST-aware);
            ``"offset"`` uses the offset written in the export.
        tz_name: IANA zone name, e.g. ``"America/Los_Angeles"``.
    """
    if mode == "offset":
        return dt.date()
    if mode == "zone":
        return dt.astimezone(_zone(tz_name)).date()
    raise ValueError(f"unknown local_date_mode {mode!r}")
