"""Command-line entry point: ``healthlab <command>``."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from .config import load_config


def _cmd_ingest(args: argparse.Namespace) -> int:
    from .ingest import ingest_export

    cfg = load_config(demo_mode=args.demo or None)
    db = Path(args.db) if args.db else cfg.db_path
    if args.rebuild and db.exists():
        for suffix in ("", "-wal", "-shm"):
            Path(str(db) + suffix).unlink(missing_ok=True)
    t0 = time.perf_counter()
    s = ingest_export(args.export, cfg, db_path=db)
    dt = time.perf_counter() - t0
    print(f"Import run #{s.run_id} -> {db}")
    print(f"  records:  {s.records_seen:>9,} seen  {s.records_inserted:>9,} new  "
          f"{s.records_duplicate:>9,} already present  {s.parse_errors:,} unparsable")
    print(f"  workouts: {s.workouts_seen:>9,} seen  {s.workouts_inserted:>9,} new")
    if s.other_elements:
        print("  not ingested: " + ", ".join(f"{k}={v}" for k, v in s.other_elements.items()))
    print(f"  {dt:.1f}s")
    if not args.no_rollup:
        from .rollup import rebuild_daily

        out = rebuild_daily(cfg, db_path=db)
        print(f"Daily rollup: {len(out)} days x {out.shape[1]} columns")
    return 0


def _cmd_rollup(args: argparse.Namespace) -> int:
    from .rollup import rebuild_daily

    cfg = load_config(demo_mode=args.demo or None)
    db = Path(args.db) if args.db else cfg.db_path
    out = rebuild_daily(cfg, db_path=db)
    print(f"Daily rollup: {len(out)} days x {out.shape[1]} columns -> {db}")
    return 0


def _cmd_synth(args: argparse.Namespace) -> int:
    from .synth import write_synthetic_export

    cfg = load_config(demo_mode=True)
    out = Path(args.out) if args.out else cfg.path("demo_export")
    truth = write_synthetic_export(out, cfg, days=args.days, seed=args.seed)
    print(f"Synthetic export written to {out} ({truth['n_records']:,} records, {args.days or cfg.raw['demo']['days']} days)")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    """Generate synthetic data, ingest it into the demo DB, then launch the app."""
    from .ingest import ingest_export
    from .rollup import rebuild_daily
    from .synth import write_synthetic_export

    cfg = load_config(demo_mode=True)
    export, db = cfg.path("demo_export"), cfg.db_path
    for suffix in ("", "-wal", "-shm"):
        Path(str(db) + suffix).unlink(missing_ok=True)
    write_synthetic_export(export, cfg)
    s = ingest_export(export, cfg, db_path=db)
    daily = rebuild_daily(cfg, db_path=db)
    print(f"Demo DB ready: {s.records_inserted:,} records, {len(daily)} days -> {db}")
    if args.no_app:
        return 0
    return _launch_app(demo=True)


def _launch_app(demo: bool) -> int:
    import os
    import subprocess

    app = Path(__file__).parent / "app" / "main.py"
    env = dict(os.environ)
    if demo:
        env["DEMO_MODE"] = "1"
    # Local only: bind to loopback and disable usage statistics.
    cmd = [sys.executable, "-m", "streamlit", "run", str(app),
           "--server.address", "127.0.0.1", "--browser.gatherUsageStats", "false"]
    return subprocess.call(cmd, env=env)


def _cmd_app(args: argparse.Namespace) -> int:
    return _launch_app(demo=args.demo)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="healthlab", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="import an Apple Health export.zip / export.xml")
    pi.add_argument("export", help="path to export.zip or export.xml")
    pi.add_argument("--db", help="database path (default from config)")
    pi.add_argument("--demo", action="store_true", help="target the demo database")
    pi.add_argument("--rebuild", action="store_true", help="delete the database first")
    pi.add_argument("--no-rollup", action="store_true", help="skip rebuilding the daily table")
    pi.set_defaults(func=_cmd_ingest)

    pr = sub.add_parser("rollup", help="recompute the daily table from raw records")
    pr.add_argument("--db")
    pr.add_argument("--demo", action="store_true")
    pr.set_defaults(func=_cmd_rollup)

    ps = sub.add_parser("synth", help="write a synthetic export.zip for demo mode")
    ps.add_argument("--out")
    ps.add_argument("--days", type=int)
    ps.add_argument("--seed", type=int)
    ps.set_defaults(func=_cmd_synth)

    pd_ = sub.add_parser("demo", help="synth + ingest + rollup + launch app on synthetic data")
    pd_.add_argument("--no-app", action="store_true")
    pd_.set_defaults(func=_cmd_demo)

    pa = sub.add_parser("app", help="launch the Streamlit app")
    pa.add_argument("--demo", action="store_true")
    pa.set_defaults(func=_cmd_app)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
