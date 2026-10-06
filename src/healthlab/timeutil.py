"""Timestamp parsing and local-date assignment.

Apple Health exports timestamps as ``"2024-05-01 08:00:00 -0700"``. We store the
UTC instant (as ``"YYYY-MM-DD HH:MM:SS"`` text, which sorts correctly in SQLite)
plus the offset as written, and derive a *local date* for daily rollups.

Local dates use a default IANA zone (America/Los_Angeles) except inside
configured override periods (e.g. Europe/Sofia while in Bulgaria); see
:class:`ZoneResolver`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
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


# ---------------------------------------------------------------------------
# Time-zone override periods
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ZoneOverride:
    """Records whose instant falls in [start, end] (inclusive local dates in ``timezone``) use that zone."""

    start: date
    end: date
    timezone: str
    label: str | None = None

    @property
    def start_utc(self) -> datetime:
        """Local midnight at the start date, as a UTC instant."""
        return datetime(self.start.year, self.start.month, self.start.day, tzinfo=_zone(self.timezone)).astimezone(timezone.utc)

    @property
    def end_utc(self) -> datetime:
        """Local midnight after the end date, as a UTC instant (exclusive bound)."""
        nxt = self.end + timedelta(days=1)
        return datetime(nxt.year, nxt.month, nxt.day, tzinfo=_zone(self.timezone)).astimezone(timezone.utc)


class ZoneResolver:
    """Decides which IANA zone defines the local date of each instant.

    The default zone applies everywhere except inside configured override
    periods (e.g. Europe/Sofia for summers in Bulgaria). Overrides are matched on
    the UTC instant, so the result does not depend on the offset an export was
    rendered with.

    Raises:
        ValueError: if two override periods overlap.
    """

    def __init__(self, default_tz: str, overrides: list[ZoneOverride] | None = None, mode: str = "zone"):
        self.default_tz = default_tz
        self.mode = mode
        self.overrides = sorted(overrides or [], key=lambda o: o.start_utc)
        for a, b in zip(self.overrides, self.overrides[1:]):
            if b.start_utc < a.end_utc:
                raise ValueError(f"time-zone overrides overlap: {a.start}..{a.end} and {b.start}..{b.end}")

    def zone_for(self, dt: datetime) -> str:
        """IANA zone that applies to an aware datetime."""
        u = dt.astimezone(timezone.utc)
        for o in self.overrides:
            if o.start_utc <= u < o.end_utc:
                return o.timezone
        return self.default_tz

    def local_date(self, dt: datetime) -> date:
        """Local calendar date of an aware datetime (honours ``offset`` mode)."""
        if self.mode == "offset":
            return dt.date()
        if self.mode != "zone":
            raise ValueError(f"unknown local_date_mode {self.mode!r}")
        return dt.astimezone(_zone(self.zone_for(dt))).date()

    def localize(self, utc: "pd.Series", offsets_min: "pd.Series | None" = None) -> "pd.Series":
        """Vectorised: tz-aware UTC Series -> naive local wall-clock Series."""
        import pandas as pd

        if self.mode == "offset":
            off = pd.to_timedelta((offsets_min if offsets_min is not None else 0), unit="min")
            return utc.dt.tz_localize(None) + off
        out = utc.dt.tz_convert(self.default_tz).dt.tz_localize(None)
        for o in self.overrides:
            m = (utc >= pd.Timestamp(o.start_utc)) & (utc < pd.Timestamp(o.end_utc))
            if m.any():
                out = out.where(~m, utc[m].dt.tz_convert(o.timezone).dt.tz_localize(None))
        return out
