"""Backfill route_latlon / route_length_m / trajectory_source into existing [L1] CSVs.

Monocular-OSM-Localization has always written the camera's estimated route into
`result.json` as `position.route_latlon`, but `modules/localization/localize.py` only
carried the single chosen point into `[L1]localization.csv`. Videos localized before that
was fixed therefore have the route sitting on disk but missing from the CSV — and so
missing from summary_data/all_video_locations.csv and the Visualizer.

This rewrites those CSVs in place from the existing result.json files. It does NOT re-run
localization (which is the expensive part); it only re-reads what is already there.

    python scripts/backfill_localization_routes.py [--dry-run]
"""
import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from modules.localization.localize import _extract_position  # noqa: E402

FIELDNAMES = ["video_name", "city", "lat", "lon", "confidence_level", "confidence_spread_m",
              "street_names", "source", "status", "result_json", "candidates",
              "route_latlon", "route_length_m", "trajectory_source"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    updated = skipped = 0
    for csv_path in sorted(glob.glob(os.path.join("analysis_results", "*", "[[]L1[]]localization.csv"))):
        video_dir = os.path.dirname(csv_path)
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            skipped += 1
            continue
        row = rows[0]

        if row.get("route_latlon"):
            skipped += 1
            continue

        # Prefer the result.json the CSV itself points at; fall back to the newest one
        # under this video's osm_localization/ directory.
        candidates = []
        rj = (row.get("result_json") or "").strip()
        if rj and os.path.exists(rj):
            candidates.append(rj)
        candidates += sorted(
            glob.glob(os.path.join(video_dir, "osm_localization", "**", "result.json"), recursive=True),
            key=os.path.getmtime, reverse=True)
        if not candidates:
            print(f"  no result.json  {os.path.basename(video_dir)} (status={row.get('status')})")
            skipped += 1
            continue

        try:
            pos = _extract_position(candidates[0])
        except Exception as e:
            print(f"  UNREADABLE      {os.path.basename(video_dir)}: {type(e).__name__}: {e}")
            skipped += 1
            continue

        if not pos["route_latlon"]:
            print(f"  no route        {os.path.basename(video_dir)}")
            skipped += 1
            continue

        import json
        row["route_latlon"] = json.dumps(pos["route_latlon"], ensure_ascii=False)
        row["route_length_m"] = pos["route_length_m"] if pos["route_length_m"] is not None else ""
        row["trajectory_source"] = pos["trajectory_source"] or ""
        for k in FIELDNAMES:
            row.setdefault(k, "")

        print(f"  backfilled      {os.path.basename(video_dir)}: "
              f"{len(pos['route_latlon'])} pts, {pos['route_length_m']} m, src={pos['trajectory_source']}")
        if not args.dry_run:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
                w.writeheader()
                w.writerow(row)
        updated += 1

    print(f"\n{'Would update' if args.dry_run else 'Updated'} {updated} file(s); {skipped} skipped.")


if __name__ == "__main__":
    main()
