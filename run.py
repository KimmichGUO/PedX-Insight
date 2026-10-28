"""Resumable download, analysis, optional localization, and source retention."""
import argparse
import csv
import datetime
import math
import os
import subprocess
import sys
import time

import pandas as pd

from modules.paths import RESULTS_ROOT
from modules.localization.localize import localization_succeeded
from modules.pipeline import file_hash
from get_all_video_locations import bind_localization_to_analysis


def _true(value):
    return pd.notna(value) and str(value).strip().lower() in {"true", "1", "1.0"}


def _window(row, clip_seconds=0):
    """Mapping times are absolute source seconds; a budget starts at the mapped start."""
    def number(value, default):
        if pd.isna(value) or str(value).strip() == "":
            return default
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("Clip times must be finite and nonnegative")
        return value

    start = number(row.get("start_time"), 0.0)
    end = number(row.get("end_time"), None)
    if end is not None and end <= start:
        raise ValueError("end_time must be greater than start_time")
    if clip_seconds:
        end = min(end, start + clip_seconds) if end is not None else start + clip_seconds
    return start, end


def _localization_status(folder, video_path=None, previous_status=None):
    """Require the analyzed bytes; confidence describes an estimate, not retry success."""
    def read(name):
        try:
            with open(os.path.join(RESULTS_ROOT, folder, name), encoding="utf-8") as f:
                return next(csv.DictReader(f), {})
        except (OSError, ValueError):
            return {}
    latest = read("[L1]localization_attempt.csv")
    if latest:
        latest = bind_localization_to_analysis(latest, os.path.join(RESULTS_ROOT, folder))
        if not localization_succeeded(latest):
            return latest.get("status") or "failed"
    result = bind_localization_to_analysis(read("[L1]localization.csv"), os.path.join(RESULTS_ROOT, folder))
    status = "ok" if localization_succeeded(result) else result.get("status") or "pending"
    if status == "ok" and video_path is not None and os.path.isfile(video_path):
        try:
            if file_hash(video_path) != result["analysis_source_video_sha256"]:
                return "source_mismatch"
        except OSError:
            return "source_unverified"
    prior = str(previous_status or "").strip().lower()
    if status == "ok" and prior not in {"", "nan", "none", "ok", "not_requested", "pending"}:
        return prior
    return status


def _write_analysis_time(video_file_name, analysis_seconds, start=0.0, end=None):
    folder = os.path.splitext(video_file_name)[0]
    out_dir = os.path.join(RESULTS_ROOT, folder)
    os.makedirs(out_dir, exist_ok=True)
    link = folder.split("_", 1)[1] if "_" in folder else folder
    from modules.pipeline import read_manifest
    manifest = read_manifest(out_dir) or {}
    duration = None
    try:
        vi = pd.read_csv(os.path.join(out_dir, "[A1]video_info.csv"))
        values = vi.loc[vi["metric"] == "duration_seconds", "value"]
        if not values.empty:
            duration = values.iloc[0]
    except (OSError, ValueError, KeyError):
        pass
    pd.DataFrame([{
        "link": link, "video_name": folder, "duration_seconds": duration,
        "analysis_seconds": round(analysis_seconds, 1),
        "clip_start_seconds": start, "clip_end_seconds": end,
        "analysis_run_id": manifest.get("run_id") if manifest.get("status") == "complete" else None,
        "analyzed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }]).to_csv(os.path.join(out_dir, "[A3]analysis_time.csv"), index=False)


def run(start_row=1, start_step=1, csv_file="mapping_one_each.csv", localize=False,
        limit=0, clip_seconds=0):
    """Retry pending stages; delete media only after all requested stages succeed.

    ``finished`` remains analysis success. Localization has independent requested/status
    columns. ``start_step`` applies to the first pending row, including location retries.
    """
    if start_row < 1 or start_step not in (1, 2, 3) or limit < 0 or not math.isfinite(clip_seconds) or clip_seconds < 0:
        raise ValueError("Invalid resume, limit, or clip options")
    df = pd.read_csv(csv_file)
    for col in ("finished", "downloaded", "analysis_status", "localization_requested",
                "localization_status", "downloaded_start_time", "downloaded_end_time"):
        if col not in df:
            df[col] = None
        df[col] = df[col].astype("object")
    os.makedirs("videos", exist_ok=True)

    def save():
        temp = str(csv_file) + ".tmp"
        df.to_csv(temp, index=False, encoding="utf-8")
        os.replace(temp, csv_file)

    processed = 0
    for idx, row in df.iloc[start_row - 1:].iterrows():
        if limit and processed >= limit:
            break
        name, video_id = str(row["name"]), str(row["video"])
        folder = f"{name}_{video_id}"
        video_name = folder + ".mp4"
        video_path = os.path.join("videos", video_name)
        finished = _true(row["finished"])
        wants_location = localize or _true(row["localization_requested"])
        prior_status = str(row.get("localization_status", "")).strip().lower()
        location_status = _localization_status(folder, video_path, prior_status) if wants_location else "not_requested"
        location_needed = wants_location and location_status != "ok"
        exists = os.path.isfile(video_path) and os.path.getsize(video_path) > 0
        if finished and not location_needed and not exists:
            if _true(row["downloaded"]):
                df.loc[idx, "downloaded"] = "FALSE"
                save()
            continue
        effective_step = start_step if processed == 0 else 1
        processed += 1
        df.loc[idx, "localization_requested"] = "TRUE" if wants_location else "FALSE"
        df.loc[idx, "localization_status"] = location_status
        df.loc[idx, "downloaded"] = "TRUE" if exists else "FALSE"
        try:
            # Localization retries use the analyzed clip even without the old budget flag.
            if (finished or (effective_step == 2 and not clip_seconds)) and pd.notna(row.get("downloaded_start_time")):
                start, end = _window({"start_time": row["downloaded_start_time"],
                                      "end_time": row.get("downloaded_end_time")})
            else:
                start, end = _window(row, clip_seconds)
        except (TypeError, ValueError) as e:
            print(f"[FAIL] Invalid window for {video_name}: {e}")
            save()
            continue

        needs_media = not finished or location_needed
        prior_start, prior_end = row.get("downloaded_start_time"), row.get("downloaded_end_time")
        known_window = pd.notna(prior_start)
        try:
            window_matches = ((not known_window and (finished or (start == 0 and end is None))) or
                              (known_window and float(prior_start) == start and
                               ((pd.isna(prior_end) and end is None) or
                                (pd.notna(prior_end) and end is not None and float(prior_end) == end))))
        except (TypeError, ValueError):
            window_matches = False
        if needs_media and (not exists or not window_matches):
            if effective_step > 1:
                print(f"[KEEP] {video_name}: source missing or window differs; resume from step 1")
                save()
                continue
            staging = video_path + ".download.mp4"
            command = ["yt-dlp", "--cookies", "www.youtube.com_cookies.txt", "--js-runtimes", "node",
                       "--user-agent", "Mozilla/5.0", "--referer", "https://www.youtube.com/",
                       "-f", "bestvideo[height<=720]/bestvideo", "--force-overwrites", "-o", staging]
            if start > 0 or end is not None:
                finish = f"{end:g}" if end is not None else "inf"
                command += ["--download-sections", f"*{start:g}-{finish}", "--force-keyframes-at-cuts"]
            command.append(f"https://www.youtube.com/watch?v={video_id}")
            try:
                subprocess.run(command, check=True)
                if not os.path.isfile(staging) or os.path.getsize(staging) == 0:
                    raise OSError("Downloader returned without a nonempty media file")
                os.replace(staging, video_path)
                exists = True
                df.loc[idx, "downloaded"] = "TRUE"
                df.loc[idx, "downloaded_start_time"] = start
                df.loc[idx, "downloaded_end_time"] = end
            except (subprocess.CalledProcessError, OSError) as e:
                print(f"[FAIL] Download {video_name}: {e}")
                save()
                continue

        if not finished and effective_step <= 2:
            df.loc[idx, "analysis_status"] = "running"
            save()
            try:
                began = time.monotonic()
                subprocess.run([sys.executable, "main.py", "--mode", "single_all",
                                "--source_video_path", video_path], check=True,
                               env=dict(os.environ, PEDX_CLIP_START_SECONDS=f"{start:g}",
                                        PEDX_CLIP_END_SECONDS=f"{end:g}" if end is not None else ""))
                _write_analysis_time(video_name, time.monotonic() - began, start, end)
                finished = True
                df.loc[idx, "finished"] = "TRUE"
                df.loc[idx, "analysis_status"] = "ok"
                if wants_location:
                    # Reanalysis can change clip bytes without changing the folder/link.
                    location_status = _localization_status(folder, video_path, prior_status)
                    location_needed = location_status != "ok"
                    df.loc[idx, "localization_status"] = location_status
            except (subprocess.CalledProcessError, OSError) as e:
                df.loc[idx, "analysis_status"] = "failed"
                print(f"[FAIL] Analysis {video_name}: {e}")
                save()
                continue

        if location_needed and finished and effective_step <= 2:
            df.loc[idx, "localization_status"] = "running"
            save()
            try:
                result = subprocess.run([
                    sys.executable, "-m", "modules.localization.localize",
                    "--source_video_path", video_path, "--no-splat", "--no-aerial",
                    "--frame-stride", "20", "--max-frames", "600", "--vo-segment", "60:900",
                ], check=False)
                location_status = _localization_status(folder, video_path)
                if result.returncode and location_status == "ok":
                    location_status = "subprocess_failed"
            except OSError as e:
                print(f"[FAIL] Localization {video_name}: {e}")
                location_status = "subprocess_failed"
            df.loc[idx, "localization_status"] = location_status

        if finished and (not wants_location or location_status == "ok"):
            if os.path.exists(video_path):
                os.remove(video_path)
            df.loc[idx, "downloaded"] = "FALSE"
        elif exists:
            print(f"[KEEP] {video_name}: requested stages remain incomplete")
        save()
    return processed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start_row", type=int, default=1)
    parser.add_argument("--start_step", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--csv", default="mapping_one_each.csv")
    parser.add_argument("--localize", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--clip-seconds", type=int, default=0,
                        help="Bound the mapped interval to N seconds from start_time (0 = full mapped interval)")
    args = parser.parse_args()
    run(args.start_row, args.start_step, args.csv, args.localize, args.limit, args.clip_seconds)
