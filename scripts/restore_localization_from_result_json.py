"""Rebuild [L1]localization.csv rows from the result.json a previous run left on disk.

Localization failures used to overwrite a good CSV row with an empty placeholder (fixed in
modules/localization/localize.py), so a transient Overpass outage could wipe a result that
took ~40 minutes to compute. The tool's own result.json survives in
analysis_results/<video>/osm_localization/<slug>/, so the row can be rebuilt without
re-running anything.

    python scripts/restore_localization_from_result_json.py [--dry-run] [VIDEO ...]

With no VIDEO arguments it restores every video whose CSV lacks a lat but whose
result.json has one.
"""
import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from modules.localization.localize import _extract_position  # noqa: E402

FIELDNAMES = ["video_name", "city", "lat", "lon", "confidence_level", "confidence_spread_m",
              "street_names", "source", "status", "result_json", "candidates",
              "route_latlon", "route_length_m", "trajectory_source", "error"]


def restore(video_dir, dry_run):
    name = os.path.basename(video_dir)
    csv_path = os.path.join(video_dir, "[L1]localization.csv")
    if not os.path.exists(csv_path):
        return None

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    row = rows[0] if rows else {}
    if (row.get("lat") or "").strip():
        return None  # already has a position; nothing to restore

    results = sorted(
        glob.glob(os.path.join(video_dir, "osm_localization", "**", "result.json"), recursive=True),
        key=os.path.getmtime, reverse=True)
    if not results:
        return None

    try:
        pos = _extract_position(results[0])
    except Exception as e:
        print(f"  UNREADABLE  {name}: {type(e).__name__}: {e}")
        return None
    if pos["lat"] is None or pos["lon"] is None:
        return None

    restored = {
        "video_name": name,
        "city": row.get("city") or "",
        "lat": pos["lat"],
        "lon": pos["lon"],
        "confidence_level": pos["confidence_level"],
        "confidence_spread_m": pos["confidence_spread_m"],
        "street_names": pos["street_names"],
        "source": "monocular_osm_localization",
        "status": "ok",
        "result_json": os.path.relpath(results[0]),
        "candidates": json.dumps(pos["hypotheses"], ensure_ascii=False),
        "route_latlon": json.dumps(pos["route_latlon"], ensure_ascii=False) if pos["route_latlon"] else "",
        "route_length_m": pos["route_length_m"] if pos["route_length_m"] is not None else "",
        "trajectory_source": pos["trajectory_source"] or "",
        "error": "",
    }
    print(f"  restored    {name}: lat={pos['lat']}, lon={pos['lon']}, "
          f"{len(pos['route_latlon'])} route pts, from {os.path.basename(os.path.dirname(results[0]))}")
    if not dry_run:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
            w.writeheader()
            w.writerow(restored)
    return restored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*", help="video folder names; default = all")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    targets = ([os.path.join("analysis_results", v) for v in args.videos]
               if args.videos
               else sorted(p for p in glob.glob(os.path.join("analysis_results", "*"))
                           if os.path.isdir(p)))
    n = sum(1 for d in targets if restore(d, args.dry_run) is not None)
    print(f"\n{'Would restore' if args.dry_run else 'Restored'} {n} row(s).")


if __name__ == "__main__":
    main()
