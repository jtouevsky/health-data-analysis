"""Configuration loading.

All tunable values live in ``config/healthlab.yaml``. This module turns the YAML
into typed objects and resolves the database path, switching to the synthetic
demo database when ``DEMO_MODE`` is set.

Lookup order for the YAML file:
1. ``HEALTHLAB_CONFIG`` environment variable, if set;
2. ``config/healthlab.yaml`` in the project root (two levels above ``src/``).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "healthlab.yaml"

_TRUTHY = {"1", "true", "yes", "on"}
_HK_PREFIXES = (
    "HKQuantityTypeIdentifier",
    "HKCategoryTypeIdentifier",
    "HKCorrelationTypeIdentifier",
    "HKDataType",
)


@dataclass(frozen=True)
class MetricRule:
    """How one HealthKit type is rolled up into the daily table."""

    hk_type: str
    name: str
    agg: str
    unit: str | None
    kind: str
    label: str
    plausible: tuple[float, float] | None = None
    configured: bool = True  # False when derived from default_rules


@dataclass
class Config:
    """Typed view over the YAML configuration."""

    raw: dict[str, Any]
    project_root: Path
    demo_mode: bool
    metrics: dict[str, MetricRule] = field(default_factory=dict)

    # -- paths ---------------------------------------------------------------
    def path(self, key: str) -> Path:
        """Resolve a path from the ``paths`` section relative to the project root."""
        p = Path(self.raw["paths"][key])
        return p if p.is_absolute() else self.project_root / p

    @property
    def db_path(self) -> Path:
        """Database in use: the demo DB in demo mode, otherwise the real one."""
        return self.path("demo_db" if self.demo_mode else "db")

    # -- sections ------------------------------------------------------------
    def section(self, name: str) -> dict[str, Any]:
        """Return a raw config section (empty dict if absent)."""
        return dict(self.raw.get(name) or {})

    @property
    def timezone(self) -> str:
        return self.raw["time"]["timezone"]

    @property
    def local_date_mode(self) -> str:
        return self.raw["time"].get("local_date_mode", "zone")

    @property
    def periods(self) -> dict[str, list[tuple[str, str]]]:
        """Seasonal periods; the demo set is used in demo mode."""
        src = self.raw["demo"]["periods"] if self.demo_mode else self.raw["seasonal"]["periods"]
        return {k: [tuple(r) for r in (v or [])] for k, v in (src or {}).items()}

    # -- metric rules --------------------------------------------------------
    def rule_for(self, hk_type: str, unit: str | None, is_category: bool) -> MetricRule:
        """Return the rollup rule for a type, deriving one from defaults if unlisted."""
        if hk_type in self.metrics:
            return self.metrics[hk_type]
        d = self.raw["default_rules"]
        name = short_name(hk_type)
        if hk_type.startswith(d["dietary_prefix"]):
            return MetricRule(hk_type, name, "sum", unit, "nutrition", name, None, False)
        if is_category:
            return MetricRule(hk_type, name, d["category_agg"], "count", "other", name, None, False)
        agg = d["cumulative_agg"] if unit in set(d["cumulative_units"]) else d["sample_agg"]
        return MetricRule(hk_type, name, agg, unit, "other", name, None, False)


def short_name(hk_type: str) -> str:
    """``HKQuantityTypeIdentifierWalkingSpeed`` -> ``walking_speed``."""
    s = hk_type
    for p in _HK_PREFIXES:
        if s.startswith(p):
            s = s[len(p):]
            break
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()
    return re.sub(r"[^a-z0-9_]", "_", s)


def _parse_metrics(raw: dict[str, Any]) -> dict[str, MetricRule]:
    out: dict[str, MetricRule] = {}
    for hk_type, m in (raw.get("metrics") or {}).items():
        pl = m.get("plausible")
        out[hk_type] = MetricRule(
            hk_type=hk_type,
            name=m["name"],
            agg=m["agg"],
            unit=m.get("unit"),
            kind=m.get("kind", "other"),
            label=m.get("label", m["name"]),
            plausible=(float(pl[0]), float(pl[1])) if pl else None,
        )
    return out


def env_demo_mode() -> bool | None:
    """Read ``DEMO_MODE`` from the environment (None if unset)."""
    v = os.environ.get("DEMO_MODE")
    return None if v is None else v.strip().lower() in _TRUTHY


def load_config(path: str | Path | None = None, demo_mode: bool | None = None) -> Config:
    """Load configuration from YAML.

    Args:
        path: explicit YAML path; defaults to ``HEALTHLAB_CONFIG`` or the project file.
        demo_mode: force demo mode on/off; defaults to ``DEMO_MODE`` env, then YAML.
    """
    p = Path(path or os.environ.get("HEALTHLAB_CONFIG") or DEFAULT_CONFIG_PATH)
    with open(p, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if demo_mode is None:
        demo_mode = env_demo_mode()
    if demo_mode is None:
        demo_mode = bool(raw.get("demo_mode", False))
    return Config(raw=raw, project_root=PROJECT_ROOT, demo_mode=demo_mode, metrics=_parse_metrics(raw))


@lru_cache(maxsize=4)
def get_config(demo_mode: bool | None = None) -> Config:
    """Cached :func:`load_config` for application code."""
    return load_config(demo_mode=demo_mode)
