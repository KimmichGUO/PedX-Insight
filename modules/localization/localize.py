"""
[L1] Video geolocation via the Monocular-OSM-Localization companion tool.

Given a video + its city, this estimates where the video was filmed (WGS84 lat/lon)
by running the `monocular-osm-localization` package and emitting a per-video CSV that
the Visualizer can plot alongside PedX's other summaries.

The tool is now a pip-installable package (distribution `monocular-osm-localization`,
import name `monocular_osm`, console script `osm-localize`). PedX runs it as a
SUBPROCESS via `python -m monocular_osm.cli` -- never imported -- so its heavy CV/geo
pipeline (osmnx, open3d, pycolmap, ...) stays in its own process and never loads into
PedX's interpreter. The interpreter that runs it is resolved in priority order from:
    1. the `osm_python` argument,
    2. the `OSM_LOCALIZATION_PYTHON` env var,
    3. PedX's own interpreter, if the `monocular_osm` package is importable there.

It is an OPTIONAL companion, deliberately kept out of PedX's core `requirements.txt`
because of that heavy stack (most PedX runs never localize). Install it once -- easiest
straight into PedX's own venv, so `--mode localize` needs no separate interpreter:
    pip install "git+https://github.com/M-Colley/Monocular-OSM-Localization.git@0.1.1"
    # (once published to PyPI:  pip install monocular-osm-localization)
    # ffmpeg must also be on PATH
See requirements-localize.txt for the pinned optional-dependency declaration.
"""

import os
import sys
import csv
import glob
import json
import time
import subprocess
import importlib.util

# The companion tool as a pip package: import name (find_spec / -m target) and the
# one-line install hint surfaced when it is missing. Not on PyPI yet, so install is
# from git pinned to the 0.1.1 tag; switch to the bare name once it is published.
OSM_PACKAGE = "monocular_osm"
OSM_INSTALL_HINT = ('pip install '
                    '"git+https://github.com/M-Colley/Monocular-OSM-Localization.git@0.1.1"')

# The tool caches working data (OSM graphs, metadata, OCR/geocoding results) under its
# --data-dir and may make other CWD-relative writes. We hand it a stable, gitignored
# directory so runs share that cache and nothing litters the PedX repo root.
OSM_CACHE_DIR = os.path.abspath("osm_localization_cache")


def _resolve_osm_python(osm_python=None):
    """Locate a Python interpreter that has the `monocular_osm` package installed.

    Priority: the `osm_python` argument, then `$OSM_LOCALIZATION_PYTHON`, then PedX's
    own interpreter when the package is importable there (the common case: the package
    was `pip install`ed into PedX's venv). Returns None when none is found, so the
    caller can point the user at the one-line pip install instead of failing cryptically
    inside the subprocess.
    """
    if osm_python:
        return osm_python
    env_python = os.environ.get("OSM_LOCALIZATION_PYTHON")
    if env_python:
        return env_python
    # find_spec() only checks importability -- it does not import the (heavy) package.
    if importlib.util.find_spec(OSM_PACKAGE) is not None:
        return sys.executable
    return None


def _infer_city_from_mapping(video_path, mapping_csv="mapping.csv"):
    """Best-effort: map the video id embedded in the filename to a 'City, Country' string.

    PedX names files `<name>_<video_id>.mp4` where <name> never contains an underscore,
    so the video id is everything after the FIRST underscore — the same convention the
    aggregators use to parse analysis folder names. (YouTube ids can themselves contain
    underscores, e.g. mtz_eM73GS0, so splitting at the LAST underscore would truncate
    them.) mapping.csv lists each city's `videos` (a list of ids). Returns None if
    pandas/mapping is unavailable or no row matches — the caller then requires --city.
    """
    if not os.path.exists(mapping_csv):
        return None
    stem = os.path.splitext(os.path.basename(video_path))[0]
    video_id = stem.split("_", 1)[1] if "_" in stem else stem
    try:
        import pandas as pd
        df = pd.read_csv(mapping_csv)
    except Exception:
        return None
    if "city" not in df.columns or "videos" not in df.columns:
        return None
    for _, row in df.iterrows():
        raw = row.get("videos")
        if not isinstance(raw, str):
            continue
        ids = [v.strip().strip("'\"") for v in raw.strip("[]").split(",")]
        if video_id in ids:
            city = str(row.get("city", "")).strip()
            country = str(row.get("country", "")).strip()
            if city and country and country.lower() != "nan":
                return f"{city}, {country}"
            return city or None
    return None


def _extract_position(result_json_path):
    """Pull the headline lat/lon (and supporting fields) out of the tool's result.json."""
    with open(result_json_path, encoding="utf-8") as f:
        data = json.load(f)
    pos = data.get("position") or {}
    hyps = pos.get("hypotheses") or []
    street_names = pos.get("street_names")
    if not street_names and hyps:
        street_names = hyps[0].get("street_names")
    if isinstance(street_names, (list, tuple)):
        street_names = "; ".join(str(s) for s in street_names)
    # spatial_confidence is a dict {level, concentration, spread_m}; flatten to scalar
    # CSV columns instead of dumping a Python-dict repr into one cell.
    conf = pos.get("spatial_confidence") or {}
    if not isinstance(conf, dict):
        conf = {"level": conf}
    # The tool also estimates the camera's ROUTE through the city (visual odometry snapped
    # to the OSM graph) as position.route_latlon = [[lat, lon], ...]. These are walking-tour
    # videos, so that polyline is the path actually walked. Everything but the single chosen
    # point used to be dropped here, leaving the route computed but discarded.
    clean_route = []
    for p in pos.get("route_latlon") or []:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            try:
                lat_p, lon_p = float(p[0]), float(p[1])
            except (TypeError, ValueError):
                continue
            if abs(lat_p) <= 90 and abs(lon_p) <= 180:
                clean_route.append([lat_p, lon_p])
    return {
        "lat": pos.get("latitude"),
        "lon": pos.get("longitude"),
        "confidence_level": conf.get("level"),
        "confidence_spread_m": conf.get("spread_m"),
        "street_names": street_names,
        "hypotheses": hyps,
        "route_latlon": clean_route,
        # Length and provenance live at the top level of result.json, not under position.
        "route_length_m": data.get("estimated_length_m"),
        "trajectory_source": data.get("trajectory_source"),
    }


def run_localization(video_path, city=None, osm_python=None, output_csv_path=None,
                     mapping_csv="mapping.csv", extra_args=None, timeout=None):
    """Estimate the video's geographic location and write [L1]localization.csv.

    Args:
        video_path: path to the local video file.
        city: "City, Country" (e.g. "Ulm, Germany"). If omitted, inferred from
            mapping.csv via the video id; a clear error is raised if that fails.
        osm_python: interpreter of the companion tool's env (see module docstring).
        output_csv_path: override for the output CSV location.
        extra_args: extra CLI flags forwarded to the monocular_osm CLI.
        timeout: optional subprocess timeout (seconds).

    Returns:
        dict with keys video_name, city, lat, lon, confidence_level, street_names.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path} (localization needs the video file present)")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join("analysis_results", video_name)
    os.makedirs(output_dir, exist_ok=True)
    if output_csv_path is None:
        output_csv_path = os.path.join(output_dir, "[L1]localization.csv")

    fieldnames = ["video_name", "city", "lat", "lon", "confidence_level", "confidence_spread_m",
                  "street_names", "source", "status", "result_json", "candidates",
                  "route_latlon", "route_length_m", "trajectory_source"]

    def _write(row):
        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerow(row)

    def _placeholder(status):
        return {
            "video_name": video_name, "city": city, "lat": "", "lon": "",
            "confidence_level": "", "confidence_spread_m": "", "street_names": "",
            "source": "monocular_osm_localization", "status": status,
            "result_json": "", "candidates": "",
            "route_latlon": "", "route_length_m": "", "trajectory_source": "",
        }

    # Resolve city.
    if not city:
        city = _infer_city_from_mapping(video_path, mapping_csv)
    if not city:
        raise ValueError(
            "Could not determine the city for localization. Pass --city 'City, Country' "
            "(e.g. --city 'Ulm, Germany'); it is required for the --mode localize step."
        )

    # Resolve the interpreter that has the monocular_osm package (see _resolve_osm_python).
    py = _resolve_osm_python(osm_python)
    if py is None:
        _write(_placeholder("osm_env_not_configured"))
        raise RuntimeError(
            "The `monocular_osm` package is not installed in this environment.\n"
            "It is an OPTIONAL companion, kept out of PedX's core requirements because of "
            "its heavy geo/CV stack (osmnx, open3d, pycolmap, ...).\n"
            "Install it into PedX's venv (then no --osm_python is needed):\n"
            f"  {OSM_INSTALL_HINT}\n"
            "  # ffmpeg must also be on PATH\n"
            "or point --osm_python / $OSM_LOCALIZATION_PYTHON at an interpreter that has it.\n"
            f"(wrote a placeholder {output_csv_path} with status=osm_env_not_configured)"
        )

    # The tool writes <output-dir>/<slug>/result.json; give it a per-video output dir we scan.
    osm_output_dir = os.path.abspath(os.path.join(output_dir, "osm_localization"))
    os.makedirs(osm_output_dir, exist_ok=True)
    os.makedirs(OSM_CACHE_DIR, exist_ok=True)

    # `-m monocular_osm.cli` is the installed package's entry point (identical to the
    # `osm-localize` console script) -- no source checkout or cwd-relative imports needed.
    cmd = [
        py, "-m", "monocular_osm.cli",
        "--video", os.path.abspath(video_path),
        "--city", city,
        "--output-dir", osm_output_dir,
        "--data-dir", OSM_CACHE_DIR,
    ]
    if extra_args:
        cmd += list(extra_args)

    print(f"[localize] {video_name}: running monocular_osm for city '{city}' ...")
    # Record the start time so we only accept a result.json written by THIS run — the
    # newest-mtime glob below would otherwise silently reuse a stale result from a
    # previous run (e.g. after the tool errors out early this time).
    run_started = time.time()
    # cwd = a stable, gitignored cache dir so the tool's default ./data cache and any
    # other relative writes never litter the repo root; imports resolve from the pkg.
    try:
        subprocess.run(cmd, cwd=OSM_CACHE_DIR, check=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        _write(_placeholder("timeout"))
        raise
    except subprocess.CalledProcessError:
        _write(_placeholder("subprocess_failed"))
        raise

    result_files = sorted(
        (p for p in glob.glob(os.path.join(osm_output_dir, "**", "result.json"), recursive=True)
         if os.path.getmtime(p) >= run_started - 1.0),  # 1s slack for coarse filesystems
        key=os.path.getmtime,
    )
    if not result_files:
        _write(_placeholder("no_result_json"))
        raise RuntimeError(
            f"Localization ran but produced no fresh result.json under {osm_output_dir}. "
            "Check the tool's logs above."
        )

    result_json = result_files[-1]
    pos = _extract_position(result_json)

    row = {
        "video_name": video_name,
        "city": city,
        "lat": pos["lat"],
        "lon": pos["lon"],
        "confidence_level": pos["confidence_level"],
        "confidence_spread_m": pos["confidence_spread_m"],
        "street_names": pos["street_names"],
        "source": "monocular_osm_localization",
        "status": "ok" if pos["lat"] is not None and pos["lon"] is not None else "no_position",
        "result_json": os.path.relpath(result_json),
        "candidates": json.dumps(pos["hypotheses"], ensure_ascii=False),
        # Empty string (not "[]") when there is no route, so the Visualizer importer can
        # tell "no route estimated" from "route estimated as empty".
        "route_latlon": json.dumps(pos["route_latlon"], ensure_ascii=False) if pos["route_latlon"] else "",
        "route_length_m": pos["route_length_m"] if pos["route_length_m"] is not None else "",
        "trajectory_source": pos["trajectory_source"] or "",
    }
    _write(row)
    print(f"[localize] {video_name}: lat={pos['lat']}, lon={pos['lon']}, "
          f"confidence={pos['confidence_level']}, route={len(pos['route_latlon'])} pts "
          f"-> {output_csv_path}")
    return {k: row[k] for k in ("video_name", "city", "lat", "lon", "confidence_level", "street_names")}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Geolocate a video via Monocular-OSM-Localization.")
    p.add_argument("--source_video_path", required=True)
    p.add_argument("--city", default=None, help="'City, Country'; inferred from mapping.csv if omitted")
    p.add_argument("--osm_python", default=None,
                   help="Interpreter that has the monocular_osm package installed "
                        "(defaults to PedX's own if importable there)")
    args, unknown = p.parse_known_args()
    # Any unrecognized flags are forwarded to the monocular_osm CLI.
    run_localization(video_path=args.source_video_path, city=args.city,
                     osm_python=args.osm_python, extra_args=unknown)
