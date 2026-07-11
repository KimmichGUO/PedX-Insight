"""
[L1] Video geolocation via the Monocular-OSM-Localization companion tool.

Given a video + its city, this estimates where the video was filmed (WGS84 lat/lon)
by running the vendored `external/Monocular-OSM-Localization` project and emitting a
per-video CSV that the Visualizer can plot alongside PedX's other summaries.

It is invoked as a SUBPROCESS (never imported) using a configurable Python interpreter,
resolved in priority order from: the `osm_python` argument, the `OSM_LOCALIZATION_PYTHON`
env var, or the submodule's own venv (external/Monocular-OSM-Localization/.venv). With
paddlex removed from PedX, the tool's pins (numpy>=2.5, opencv-python>=4.13) are compatible
with PedX's env, so `osm_python` MAY be the same interpreter running PedX (once the tool's
own requirements are installed there). It is still kept out of subprocess-vs-import mainly
because it is source-only (no setup.py) and pulls heavy extra deps (osmnx/open3d/gsplat).

Set up an environment for it once (its own venv, or reuse PedX's after installing its reqs):
    git submodule update --init external/Monocular-OSM-Localization
    python -m venv external/Monocular-OSM-Localization/.venv
    external/Monocular-OSM-Localization/.venv/Scripts/pip install -r \
        external/Monocular-OSM-Localization/requirements.txt
    # (ffmpeg must also be on PATH)
"""

import os
import csv
import glob
import json
import subprocess

# Repo-root-relative path to the vendored companion tool (a git submodule).
OSM_REPO = os.path.join("external", "Monocular-OSM-Localization")


def _resolve_osm_python(osm_python=None):
    """Locate the Python interpreter of the companion tool's own environment."""
    if osm_python:
        return osm_python
    env_python = os.environ.get("OSM_LOCALIZATION_PYTHON")
    if env_python:
        return env_python
    for rel in (os.path.join(".venv", "Scripts", "python.exe"),  # Windows
                os.path.join(".venv", "bin", "python")):          # POSIX
        candidate = os.path.join(OSM_REPO, rel)
        if os.path.exists(candidate):
            return candidate
    return None


def _infer_city_from_mapping(video_path, mapping_csv="mapping.csv"):
    """Best-effort: map the video id embedded in the filename to a 'City, Country' string.

    PedX names files `<name>_<video_id>.mp4`, so the video id is the segment after the
    last underscore. mapping.csv lists each city's `videos` (a list of ids). Returns None
    if pandas/mapping is unavailable or no row matches — the caller then requires --city.
    """
    if not os.path.exists(mapping_csv):
        return None
    stem = os.path.splitext(os.path.basename(video_path))[0]
    video_id = stem.rsplit("_", 1)[-1] if "_" in stem else stem
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
    return {
        "lat": pos.get("latitude"),
        "lon": pos.get("longitude"),
        "confidence": pos.get("spatial_confidence"),
        "street_names": street_names,
        "hypotheses": hyps,
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
        extra_args: extra CLI flags forwarded to the companion tool's main.py.
        timeout: optional subprocess timeout (seconds).

    Returns:
        dict with keys video_name, city, lat, lon, confidence, street_names.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path} (localization needs the video file present)")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join("analysis_results", video_name)
    os.makedirs(output_dir, exist_ok=True)
    if output_csv_path is None:
        output_csv_path = os.path.join(output_dir, "[L1]localization.csv")

    fieldnames = ["video_name", "city", "lat", "lon", "confidence",
                  "street_names", "source", "status", "result_json", "candidates"]

    def _write(row):
        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerow(row)

    # Resolve city.
    if not city:
        city = _infer_city_from_mapping(video_path, mapping_csv)
    if not city:
        raise ValueError(
            "Could not determine the city for localization. Pass --city 'City, Country' "
            "(e.g. --city 'Ulm, Germany'); it is required for the --mode localize step."
        )

    # Resolve the companion tool's interpreter (must be a separate env — deps conflict).
    py = _resolve_osm_python(osm_python)
    if py is None or not os.path.exists(OSM_REPO):
        _write({
            "video_name": video_name, "city": city, "lat": "", "lon": "",
            "confidence": "", "street_names": "", "source": "monocular_osm_localization",
            "status": "osm_env_not_configured", "result_json": "", "candidates": "",
        })
        raise RuntimeError(
            "Monocular-OSM-Localization environment not found.\n"
            "Set it up once (its deps conflict with PedX, so it needs its own venv):\n"
            "  git submodule update --init external/Monocular-OSM-Localization\n"
            "  python -m venv external/Monocular-OSM-Localization/.venv\n"
            "  external/Monocular-OSM-Localization/.venv/Scripts/pip install -r "
            "external/Monocular-OSM-Localization/requirements.txt\n"
            "Then re-run, or pass its interpreter via --osm_python / OSM_LOCALIZATION_PYTHON.\n"
            f"(wrote a placeholder {output_csv_path} with status=osm_env_not_configured)"
        )

    # The tool writes output/<slug>/result.json; give it a per-video output dir we can scan.
    osm_output_dir = os.path.abspath(os.path.join(output_dir, "osm_localization"))
    os.makedirs(osm_output_dir, exist_ok=True)

    cmd = [
        py, "main.py",
        "--video", os.path.abspath(video_path),
        "--city", city,
        "--output-dir", osm_output_dir,
    ]
    if extra_args:
        cmd += list(extra_args)

    print(f"[localize] {video_name}: running Monocular-OSM-Localization for city '{city}' ...")
    # cwd = submodule root so its relative imports (src/), data/, and yolov8s.pt resolve.
    subprocess.run(cmd, cwd=os.path.abspath(OSM_REPO), check=True, timeout=timeout)

    result_files = sorted(
        glob.glob(os.path.join(osm_output_dir, "**", "result.json"), recursive=True),
        key=os.path.getmtime,
    )
    if not result_files:
        _write({
            "video_name": video_name, "city": city, "lat": "", "lon": "",
            "confidence": "", "street_names": "", "source": "monocular_osm_localization",
            "status": "no_result_json", "result_json": "", "candidates": "",
        })
        raise RuntimeError(
            f"Localization ran but produced no result.json under {osm_output_dir}. "
            "Check the tool's logs above."
        )

    result_json = result_files[-1]
    pos = _extract_position(result_json)

    row = {
        "video_name": video_name,
        "city": city,
        "lat": pos["lat"],
        "lon": pos["lon"],
        "confidence": pos["confidence"],
        "street_names": pos["street_names"],
        "source": "monocular_osm_localization",
        "status": "ok" if pos["lat"] is not None and pos["lon"] is not None else "no_position",
        "result_json": os.path.relpath(result_json),
        "candidates": json.dumps(pos["hypotheses"], ensure_ascii=False),
    }
    _write(row)
    print(f"[localize] {video_name}: lat={pos['lat']}, lon={pos['lon']}, "
          f"confidence={pos['confidence']} -> {output_csv_path}")
    return {k: row[k] for k in ("video_name", "city", "lat", "lon", "confidence", "street_names")}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Geolocate a video via Monocular-OSM-Localization.")
    p.add_argument("--source_video_path", required=True)
    p.add_argument("--city", default=None, help="'City, Country'; inferred from mapping.csv if omitted")
    p.add_argument("--osm_python", default=None, help="Interpreter of the companion tool's env")
    args, unknown = p.parse_known_args()
    # Any unrecognized flags are forwarded to the companion tool's main.py.
    run_localization(video_path=args.source_video_path, city=args.city,
                     osm_python=args.osm_python, extra_args=unknown)
