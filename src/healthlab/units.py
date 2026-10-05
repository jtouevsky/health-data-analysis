"""Unit conversion: raw HealthKit units -> canonical metric units -> display units."""

from __future__ import annotations

import numpy as np

# unit -> (dimension, factor to the dimension's base unit)
_UNITS: dict[str, tuple[str, float]] = {
    # length (base: m)
    "m": ("length", 1.0), "km": ("length", 1000.0), "cm": ("length", 0.01), "mm": ("length", 0.001),
    "mi": ("length", 1609.344), "ft": ("length", 0.3048), "in": ("length", 0.0254), "yd": ("length", 0.9144),
    # energy (base: kcal). Apple writes "Cal" (food calorie) == kcal; "cal" is the small calorie.
    "kcal": ("energy", 1.0), "Cal": ("energy", 1.0), "cal": ("energy", 0.001), "kJ": ("energy", 1 / 4.184),
    # mass (base: kg)
    "kg": ("mass", 1.0), "g": ("mass", 1e-3), "mg": ("mass", 1e-6), "mcg": ("mass", 1e-9),
    "lb": ("mass", 0.45359237), "oz": ("mass", 0.028349523125), "st": ("mass", 6.35029318),
    # speed (base: km/hr)
    "km/hr": ("speed", 1.0), "m/s": ("speed", 3.6), "mi/hr": ("speed", 1.609344),
    # volume (base: mL)
    "mL": ("volume", 1.0), "L": ("volume", 1000.0), "fl_oz_us": ("volume", 29.5735295625),
    # time (base: s)
    "s": ("time", 1.0), "min": ("time", 60.0), "hr": ("time", 3600.0), "d": ("time", 86400.0),
    # HealthKit stores percentages as a 0-1 fraction with unit "%"; canonical "%" is 0-100.
    "%": ("percent", 100.0),
}

# Preferred canonical unit for each dimension (used for types without a configured unit).
_CANONICAL = {"length": "km", "energy": "kcal", "mass": "kg", "speed": "km/hr",
              "volume": "mL", "time": "min", "percent": "%"}


def canonical_unit(unit: str | None) -> str | None:
    """Canonical unit for a raw unit (identity for unknown/dimensionless units)."""
    if unit is None or unit not in _UNITS:
        return unit
    return _CANONICAL[_UNITS[unit][0]]


def convert(values: np.ndarray | float, from_unit: str | None, to_unit: str | None) -> np.ndarray | float:
    """Convert values between compatible units.

    Unknown or identical units are returned unchanged. ``%`` is special-cased:
    raw HealthKit ``%`` values are fractions, so ``%`` -> ``%`` multiplies by 100.
    Mass-like nutrient units are handled within the mass dimension (g -> mg etc.).

    Raises:
        ValueError: if both units are known but of different dimensions.
    """
    if from_unit == "%" and to_unit == "%":
        return np.asarray(values, dtype=float) * 100.0
    if from_unit == to_unit or from_unit not in _UNITS or to_unit not in _UNITS:
        return values
    d1, f1 = _UNITS[from_unit]
    d2, f2 = _UNITS[to_unit]
    if d1 != d2:
        raise ValueError(f"cannot convert {from_unit!r} to {to_unit!r}")
    if d1 == "percent":
        return values
    return np.asarray(values, dtype=float) * (f1 / f2)


# ---------------------------------------------------------------------------
# Display units (the app's kg/lb and km/mi toggle)
# ---------------------------------------------------------------------------

DISPLAY = {
    "metric": {"kg": ("kg", 1.0), "km": ("km", 1.0), "km/hr": ("km/h", 1.0), "cm": ("cm", 1.0)},
    "imperial": {"kg": ("lb", 1 / 0.45359237), "km": ("mi", 1 / 1.609344),
                 "km/hr": ("mph", 1 / 1.609344), "cm": ("in", 1 / 2.54)},
}


def display(unit: str | None, system: str) -> tuple[str | None, float]:
    """(display unit label, multiplier) for a canonical unit under a unit system."""
    return DISPLAY.get(system, {}).get(unit or "", (unit, 1.0))
