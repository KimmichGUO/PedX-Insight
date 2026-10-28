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
import subprocess
import importlib.util
import math
import re
import datetime
import tempfile
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from modules.paths import RESULTS_ROOT
from modules.pipeline import file_hash, read_manifest

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

LOCALIZATION_METADATA_FIELDS = [
    "localization_result_source", "localization_answer_mode", "localization_answer_method",
    "localization_experimental", "localization_confidence_calibrated",
    "localization_tool_version", "localization_tool_git_sha", "localization_tool_git_dirty",
    "localization_python_version",
]


def localization_succeeded(row):
    """A process exit alone is insufficient: require a usable geographic result."""
    try:
        lat, lon = float(row.get("lat")), float(row.get("lon"))
        return (str(row.get("status", "")).strip().lower() == "ok" and math.isfinite(lat)
                and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180)
    except (AttributeError, TypeError, ValueError):
        return False


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
    """Read the selected answer; older tools exposed only a position headline."""
    with open(result_json_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Localization result.json must contain an object")
    provenance = data.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    git = provenance.get("git")
    git = git if isinstance(git, dict) else {}
    config = provenance.get("config")
    config = config if isinstance(config, dict) else {}
    version = re.match(r"^(\d+)\.(\d+)\.(\d+)", str(provenance.get("package_version", "")))
    selected_schema = ("answer" in data or bool(config.get("answer")) or
                       (version is not None and tuple(map(int, version.groups())) >= (0, 1, 2)))
    selection_error = None
    if selected_schema:
        answer = data.get("answer")
        pos = answer if isinstance(answer, dict) else {}
        status = str(pos.get("status", "")).strip().lower()
        success = str(pos.get("success", "")).strip().lower()
        if (not pos or status not in {"", "ok", "success", "succeeded", "complete", "completed"}
                or success in {"false", "0", "no"}):
            selection_error = "Selected localization answer is missing, malformed, or unsuccessful"
    else:
        legacy = data.get("position")
        pos = legacy if isinstance(legacy, dict) else {}
    hyps = pos.get("hypotheses") or []
    hyps = [hyp for hyp in hyps if isinstance(hyp, dict)] if isinstance(hyps, list) else []
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
    # Never attach diagnostic geometry/confidence to a different selected route.
    clean_route = []
    for p in pos.get("route_latlon") or []:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            try:
                lat_p, lon_p = float(p[0]), float(p[1])
            except (TypeError, ValueError):
                continue
            if abs(lat_p) <= 90 and abs(lon_p) <= 180:
                clean_route.append([lat_p, lon_p])
    result = {
        "lat": None if selection_error else pos.get("latitude"),
        "lon": None if selection_error else pos.get("longitude"),
        "confidence_level": conf.get("level"),
        "confidence_spread_m": conf.get("spread_m"),
        "street_names": street_names,
        "hypotheses": hyps,
        "route_latlon": clean_route,
        "route_length_m": pos.get("route_length_m", pos.get("estimated_length_m")) if selected_schema else data.get("estimated_length_m"),
        "trajectory_source": pos.get("source") or pos.get("method") if selected_schema else data.get("trajectory_source"),
        "selection_error": selection_error,
        "localization_result_source": "answer" if selected_schema else "position",
        "localization_answer_mode": pos.get("mode") if selected_schema else None,
        "localization_answer_method": pos.get("method") if selected_schema else None,
        "localization_experimental": pos.get("experimental") if selected_schema else None,
        "localization_confidence_calibrated": pos.get("confidence_calibrated") if selected_schema else None,
        "localization_tool_version": provenance.get("package_version"),
        "localization_tool_git_sha": git.get("commit"),
        "localization_tool_git_dirty": git.get("dirty"),
        "localization_python_version": provenance.get("python"),
    }
    if selected_schema and not localization_succeeded({"status": "ok", "lat": result["lat"], "lon": result["lon"]}):
        result.update(lat=None, lon=None, selection_error=selection_error or "Selected localization answer has no valid finite position")
    return result


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
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(RESULTS_ROOT, video_name)
    os.makedirs(output_dir, exist_ok=True)
    if output_csv_path is None:
        output_csv_path = os.path.join(output_dir, "[L1]localization.csv")

    fieldnames = ["video_name", "city", "lat", "lon", "confidence_level", "confidence_spread_m",
                  "street_names", "source", "status", "result_json", "candidates",
                  "route_latlon", "route_length_m", "trajectory_source", "error",
                  "source_video_sha256", "analysis_run_id", "localized_at"] + LOCALIZATION_METADATA_FIELDS
    source_video_sha256, analysis_run_id = "", ""
    result_metadata = {}

    attempt_path = os.path.join(os.path.dirname(os.path.abspath(output_csv_path)), "[L1]localization_attempt.csv")

    def _write(row, path=output_csv_path, columns=fieldnames):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8",
                                             dir=os.path.dirname(os.path.abspath(path)),
                                             delete=False) as f:
                temp_path = f.name
                w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
                w.writeheader()
                w.writerow(row)
            os.replace(temp_path, path)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    def _attempt(row):
        _write({**row, "attempted_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()},
               attempt_path, fieldnames + ["attempted_utc"])

    def _existing_row():
        """The row already on disk, if any. Used to avoid destroying a good result."""
        if not os.path.exists(output_csv_path):
            return None
        try:
            with open(output_csv_path, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            return rows[0] if rows else None
        except Exception:
            return None

    def _placeholder(status, error=""):
        return {
            "video_name": video_name, "city": city, "lat": "", "lon": "",
            "confidence_level": "", "confidence_spread_m": "", "street_names": "",
            "source": "monocular_osm_localization", "status": status,
            "result_json": "", "candidates": "",
            "route_latlon": "", "route_length_m": "", "trajectory_source": "",
            # Why it failed. Without this the CSV said only "subprocess_failed" and the
            # reason was lost to the console, so diagnosing a batch run meant reproducing
            # each failure by hand.
            "error": error,
            "source_video_sha256": source_video_sha256,
            "analysis_run_id": analysis_run_id,
            "localized_at": "",
            **result_metadata,
        }

    # Failure output is written to a full log next to the CSV; `error` in the CSV holds a
    # trimmed tail of it so the reason is visible without opening another file.
    log_path = os.path.join(output_dir, "[L1]localization.log")

    def _record_failure(status, output):
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        text = (output or "").strip()
        if text:
            try:
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(text)
            except OSError:
                pass
        tail = " | ".join(line.strip() for line in text.splitlines()[-6:] if line.strip())
        failure = _placeholder(status, tail[:1000])
        _attempt(failure)

        # NEVER overwrite a previously SUCCESSFUL localization with a failure placeholder.
        # The usual failure here is a transient Overpass timeout, and clobbering turns
        # "this retry failed" into "this video has no location" — silently destroying the
        # lat/lon, candidates and route polyline of a result that took ~40 min to compute.
        # (Observed: a retry of Ulm1/Ulm2 during an Overpass outage wiped both good rows.)
        # The failure is still fully recorded in the .log next to the CSV.
        prior = _existing_row()
        if localization_succeeded(prior):
            print(f"[localize] {video_name}: {status} — KEEPING the previous successful "
                  f"result; this attempt's output is in {log_path}")
            if text:
                print(text[-2000:])
            return

        _write(failure)
        if text:
            print(f"[localize] {video_name}: {status} — full output in {log_path}")
            print(text[-2000:])

    if not os.path.exists(video_path):
        _record_failure("video_missing", f"Video not found: {video_path}")
        raise FileNotFoundError(f"Video not found: {video_path} (localization needs the video file present)")

    # A filename/YouTube ID can be reused for a different interval. Bind the result
    # to the actual analyzed bytes rather than attaching an older clip's geography.
    source_video_sha256 = file_hash(video_path)
    analysis_manifest = read_manifest(output_dir) or {}
    configuration = analysis_manifest.get("configuration") if isinstance(analysis_manifest, dict) else None
    if isinstance(configuration, dict) and configuration.get("source_sha256") == source_video_sha256:
        analysis_run_id = analysis_manifest.get("run_id") or ""

    # Resolve city.
    if not city:
        _record_failure("city_unknown", "Could not determine the city from mapping.csv")
        city = _infer_city_from_mapping(video_path, mapping_csv)
    if not city:
        raise ValueError(
            "Could not determine the city for localization. Pass --city 'City, Country' "
            "(e.g. --city 'Ulm, Germany'); it is required for the --mode localize step."
        )

    # Resolve the interpreter that has the monocular_osm package (see _resolve_osm_python).
    py = _resolve_osm_python(osm_python)
    if py is None:
        _record_failure("osm_env_not_configured", "The monocular_osm package is not configured")
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

    # Keep every invocation's result tree separate. A shared tree plus an mtime
    # tolerance can accept an earlier retry's result and relabel it with new clip
    # bytes, even when this invocation exited without producing a result.
    osm_history_dir = os.path.abspath(os.path.join(output_dir, "osm_localization"))
    os.makedirs(osm_history_dir, exist_ok=True)
    osm_output_dir = tempfile.mkdtemp(prefix="attempt_", dir=osm_history_dir)
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
    # cwd = a stable, gitignored cache dir so the tool's default ./data cache and any
    # other relative writes never litter the repo root; imports resolve from the pkg.
    # Capture the tool's output (merged stdout+stderr) instead of letting it stream to the
    # console and vanish. It is still echoed on failure, but now it also lands in the CSV's
    # `error` column and a per-video .log — otherwise a batch run reports only
    # "subprocess_failed" and every diagnosis means reproducing the failure by hand.
    try:
        subprocess.run(cmd, cwd=OSM_CACHE_DIR, check=True, timeout=timeout,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired as e:
        output = e.output.decode("utf-8", errors="replace") if isinstance(e.output, bytes) else (e.output or "")
        _record_failure("timeout", output + f"\n[timed out after {timeout}s]")
        raise
    except subprocess.CalledProcessError as e:
        _record_failure("subprocess_failed", e.output)
        raise
    except OSError as e:
        _record_failure("subprocess_failed", str(e))
        raise

    result_files = sorted(
        glob.glob(os.path.join(osm_output_dir, "**", "result.json"), recursive=True),
        key=os.path.getmtime,
    )
    if not result_files:
        _record_failure("no_result_json", f"no fresh result.json under {osm_output_dir}")
        raise RuntimeError(
            f"Localization ran but produced no fresh result.json under {osm_output_dir}. "
            "Check the tool's logs above."
        )

    result_json = result_files[-1]
    try:
        pos = _extract_position(result_json)
        result_metadata = {name: pos.get(name) for name in LOCALIZATION_METADATA_FIELDS}
    except (OSError, ValueError, TypeError, AttributeError) as e:
        _record_failure("invalid_result_json", str(e))
        raise

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
        "error": "",
        "source_video_sha256": source_video_sha256,
        "analysis_run_id": analysis_run_id,
        "localized_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **result_metadata,
    }
    if not localization_succeeded(row):
        _record_failure("no_position", pos.get("selection_error") or "Localization produced no finite, valid latitude/longitude")
        raise RuntimeError("Localization completed without a usable position")
    try:
        unchanged = file_hash(video_path) == source_video_sha256
    except OSError:
        unchanged = False
    if not unchanged:
        _record_failure("source_changed", "Video bytes changed while localization was running")
        raise RuntimeError("Source video changed during localization; result was not accepted")
    _write(row)
    _attempt(row)
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
