"""Timestamp parsing and local-date assignment edge cases."""

from datetime import date

import pytest

from healthlab.timeutil import local_date, offset_minutes, parse_apple_ts, to_utc_str

LA = "America/Los_Angeles"


def test_parse_apple_format_to_utc():
    dt = parse_apple_ts("2024-05-01 08:00:00 -0700")
    assert to_utc_str(dt) == "2024-05-01 15:00:00"
    assert offset_minutes(dt) == -420


@pytest.mark.parametrize("s,utc,off", [
    ("2024-07-01 08:00:00 +0300", "2024-07-01 05:00:00", 180),     # Sofia summer
    ("2024-12-01 08:00:00 +0200", "2024-12-01 06:00:00", 120),     # Sofia winter
    ("2024-01-01 00:15:00 +0530", "2023-12-31 18:45:00", 330),     # half-hour offset
    ("2024-05-01T08:00:00-07:00", "2024-05-01 15:00:00", -420),    # ISO variant
])
def test_parse_offsets(s, utc, off):
    dt = parse_apple_ts(s)
    assert to_utc_str(dt) == utc
    assert offset_minutes(dt) == off


def test_missing_offset_rejected():
    with pytest.raises(ValueError):
        parse_apple_ts("2024-05-01 08:00:00")


def test_zone_mode_ignores_export_render_offset():
    # The same instant rendered by an export made in Bulgaria vs in LA must land
    # on the same local day in zone mode.
    a = parse_apple_ts("2024-07-02 07:30:00 +0300")
    b = parse_apple_ts("2024-07-01 21:30:00 -0700")
    assert local_date(a, "zone", LA) == local_date(b, "zone", LA) == date(2024, 7, 1)
    # ...but offset mode takes the rendered wall-clock date at face value.
    assert local_date(a, "offset", LA) == date(2024, 7, 2)


def test_winter_record_rendered_in_summer_offset():
    # Apple renders a January sample with the export-time offset (-0700). The real
    # LA offset in January is -0800, so 00:30 -0700 is actually 23:30 the previous day.
    dt = parse_apple_ts("2024-01-15 00:30:00 -0700")
    assert local_date(dt, "zone", LA) == date(2024, 1, 14)
    assert local_date(dt, "offset", LA) == date(2024, 1, 15)


def test_dst_transition_days():
    # Spring forward 2024-03-10 02:00 PST -> 03:00 PDT.
    before = parse_apple_ts("2024-03-10 09:59:00 +0000")   # 01:59 PST
    after = parse_apple_ts("2024-03-10 10:01:00 +0000")    # 03:01 PDT
    assert local_date(before, "zone", LA) == local_date(after, "zone", LA) == date(2024, 3, 10)
    # Fall back 2024-11-03: 23:30 PST on Nov 3 is 07:30 UTC Nov 4.
    late = parse_apple_ts("2024-11-04 07:30:00 +0000")
    assert local_date(late, "zone", LA) == date(2024, 11, 3)


def test_unknown_mode():
    with pytest.raises(ValueError):
        local_date(parse_apple_ts("2024-05-01 08:00:00 -0700"), "bogus", LA)
