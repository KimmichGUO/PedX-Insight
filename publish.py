#!/usr/bin/env python3
"""Publish PedX-Insight results to the PedX-Visualizer website database, end to end.

Runs AFTER run.py has analyzed videos (analysis_results/ populated):
  1. Aggregate:  get_all_video_info.py, get_all_pede_info.py, get_all_video_locations.py
     (+ statistics_with_pdf_save.py unless --skip-stats)
  2. Sync:       copy summary_data/*.csv -> ../PedX-Visualizer/summary_data/
  3. Ingest:     node scripts/aggregate-csv-data.js        (cities/videos/pedestrians/analytics)
  4. Coordinates: node scripts/import-video-coordinates.js  (REAL video positions -> videos table)
  5. Views:      node scripts/refresh-views.js              (materialized views)
  6. Insights:   node scripts/generate-city-insights.js     (city insight rows)

Requirements (checked up front, with clear errors):
  - node on PATH (the Visualizer's DB layer is JavaScript)
  - PedX-Visualizer/.env.local containing DATABASE_URL=... (create it yourself — credentials)
  - one-time on a fresh DB: apply database/schema.sql; on an existing DB apply
    scripts/migrate-add-localization-fields.sql once (see --help of this script's README notes)

Usage:
  python publish.py                 # full pipeline
  python publish.py --skip-stats    # skip the *_stats.csv/PDF regeneration
  python publish.py --dry-run       # show what would run, execute nothing
"""

import argparse
import os
import shutil
import subprocess
import sys

INSIGHT = os.path.dirname(os.path.abspath(__file__))
VISUALIZER = os.path.abspath(os.path.join(INSIGHT, "..", "PedX-Visualizer"))

AGGREGATORS = ["get_all_video_info.py", "get_all_pede_info.py", "get_all_video_locations.py"]
STATS = "statistics_with_pdf_save.py"
NODE_STEPS = [
    ("Ingest summary CSVs into Postgres", ["node", "scripts/aggregate-csv-data.js"]),
    ("Import real video coordinates", ["node", "scripts/import-video-coordinates.js"]),
    ("Refresh materialized views", ["node", "scripts/refresh-views.js"]),
    ("Generate city insights", ["node", "scripts/generate-city-insights.js"]),
]


def run(cmd, cwd, dry):
    print(f"\n>>> {' '.join(cmd)}   (cwd={os.path.relpath(cwd, INSIGHT) or '.'})")
    if dry:
        return 0
    return subprocess.run(cmd, cwd=cwd).returncode


def main():
    ap = argparse.ArgumentParser(description="Publish analysis results to the Visualizer website DB")
    ap.add_argument("--skip-stats", action="store_true", help="skip statistics_with_pdf_save.py")
    ap.add_argument("--dry-run", action="store_true", help="print steps without executing")
    args = ap.parse_args()

    # ---- Preconditions ----
    # Local half (aggregate + sync) only needs the repos; the DB half additionally
    # needs node and DATABASE_URL. Run what we can, stop cleanly before what we can't.
    if not os.path.isdir(VISUALIZER):
        print(f"✗ PedX-Visualizer not found at {VISUALIZER}")
        sys.exit(1)
    if not os.path.isdir(os.path.join(INSIGHT, "analysis_results")) and not args.dry_run:
        print("✗ analysis_results/ is empty — run `python run.py [--localize]` first")
        sys.exit(1)

    db_problems = []
    if shutil.which("node") is None:
        db_problems.append("node is not on PATH — install Node.js (the Visualizer's DB layer is JavaScript)")
    if not (os.path.exists(os.path.join(VISUALIZER, ".env.local"))
            or os.path.exists(os.path.join(VISUALIZER, ".env"))):
        db_problems.append("PedX-Visualizer/.env.local is missing — create it yourself with a single line: "
                           "DATABASE_URL=postgresql://... (the website's Postgres; credentials, so not created by tooling)")

    # ---- 1. Aggregate in Insight ----
    for script in AGGREGATORS:
        if run([sys.executable, script], INSIGHT, args.dry_run) != 0:
            print(f"[FAIL] {script} — stopping before DB ingest")
            sys.exit(1)
    if not args.skip_stats:
        if run([sys.executable, STATS], INSIGHT, args.dry_run) != 0:
            print(f"[WARN] {STATS} failed — continuing (stats CSVs may be stale/absent)")

    # ---- 2. Sync summary_data -> Visualizer ----
    src = os.path.join(INSIGHT, "summary_data")
    dst = os.path.join(VISUALIZER, "summary_data")
    print(f"\n>>> sync {src} -> {dst} (*.csv)")
    if not args.dry_run:
        os.makedirs(dst, exist_ok=True)
        copied = 0
        for name in sorted(os.listdir(src)):
            if name.lower().endswith(".csv"):
                shutil.copy2(os.path.join(src, name), os.path.join(dst, name))
                copied += 1
        print(f"    copied {copied} CSV files")

    # ---- 3-6. Node steps in Visualizer ----
    if db_problems and not args.dry_run:
        print("\n⏸ Aggregation + sync done. Database publishing needs:")
        for p in db_problems:
            print(f"  ✗ {p}")
        print("Fix the above and re-run `python publish.py` — the local half is idempotent.")
        sys.exit(2)
    elif db_problems:
        print("\n(dry-run) DB steps would be blocked by:")
        for p in db_problems:
            print(f"  ! {p}")

    for label, cmd in NODE_STEPS:
        print(f"\n=== {label} ===")
        if run(cmd, VISUALIZER, args.dry_run) != 0:
            print(f"[FAIL] {label} — fix and re-run publish.py (steps are idempotent)")
            sys.exit(1)

    print("\n✓ Publish complete. The website reads this database directly — refresh the app to see the update.")


if __name__ == "__main__":
    main()
