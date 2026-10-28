"""Durable, private archive reanalysis. Never publishes or verifies filming location."""
import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[1]
READY = "ready_for_reanalysis_location_pending"
MIN_FREE_BYTES = 20 * 1024 ** 3


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def batch_lock(path):
    """An OS lock releases on process death; a leftover filename never blocks resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    if stream.tell() == 0:
        stream.write(b"0")
        stream.flush()
    stream.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise RuntimeError("Another archive worker holds this work-root lock") from None
    try:
        yield
    finally:
        stream.seek(0)
        if os.name == "nt":
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def number(value):
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError("Clip bounds must be finite and nonnegative")
    return parsed


def select_jobs(args):
    with args.queue.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    eligible = [row for row in rows if row.get("queue_status") == READY]
    counts = {row.get("video"): sum(other.get("video") == row.get("video") for other in eligible)
              for row in eligible}
    jobs, skipped = [], []
    for index, row in enumerate(rows, 2):
        link = row.get("video", "").strip()
        if args.video_id and link != args.video_id:
            continue
        try:
            if row.get("queue_status") != READY:
                raise ValueError("queue_status is blocked")
            if not re.fullmatch(r"[A-Za-z0-9_-]{11}", link) or counts[link] != 1:
                raise ValueError("Video ID must be a unique bare 11-character YouTube ID")
            name = row.get("name", "").strip()
            if (not name or "_" in name or name in {".", ".."} or
                    re.search(r'[<>:"/\\|?*\x00-\x1f]', name) or name.endswith((".", " "))):
                raise ValueError("Existing analysis name must be a safe underscore-free filename")
            start, end = number(row.get("start_time")), number(row.get("end_time"))
            if end <= start:
                raise ValueError("An explicit end greater than start is required")
            job = {"video_id": link, "name": name, "city": row.get("city", "").strip(),
                   "country": row.get("country", "").strip(), "queue_row": index,
                   "historical_window": {"start": start, "end": end},
                   "requested_window": {"start": start, "end": end},
                   "scope": "pilot" if args.clip_seconds is not None else "full_mapped_interval"}
            if not job["city"] or not job["country"]:
                raise ValueError("An explicit searched city and country are required")
            if args.clip_seconds is not None:
                job["requested_window"]["end"] = min(end, start + args.clip_seconds)
            token = {key: job[key] for key in ("video_id", "name", "city", "country",
                                              "historical_window", "requested_window", "scope")}
            job["key"] = link + "-" + hashlib.sha256(json.dumps(token, sort_keys=True).encode()).hexdigest()[:16]
            jobs.append(job)
        except (ValueError, TypeError) as error:
            skipped.append({"queue_row": index, "video_id": link, "reason": str(error)})
    limit = 1 if args.prepare_only else args.limit
    return jobs[:limit] if limit else jobs, skipped


def run_command(command, *, cwd, env, log_path=None, timeout=None):
    options = dict(cwd=cwd, env=env, text=True, encoding="utf-8", errors="replace")

    def execute(stdout, stderr):
        if timeout is None:
            # Long-running CV/localization retain their existing unbounded runtime.
            return subprocess.run(command, stdout=stdout, stderr=stderr, check=True, **options)
        group = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
                 if os.name == "nt" else {"start_new_session": True})
        with subprocess.Popen(command, stdout=stdout, stderr=stderr, **options, **group) as process:
            try:
                output, errors = process.communicate(timeout=timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                # Only this invocation's child tree is terminated, including its ffmpeg.
                if os.name == "nt":
                    taskkill = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "taskkill.exe"
                    try:
                        subprocess.run([str(taskkill), "/PID", str(process.pid), "/T", "/F"],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
                    except (OSError, subprocess.TimeoutExpired):
                        process.kill()
                else:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=10)
                raise
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command, output=output, stderr=errors)
            return subprocess.CompletedProcess(command, process.returncode, output, errors)
    if log_path is None:
        return execute(subprocess.PIPE, subprocess.PIPE)
    with Path(log_path).open("a", encoding="utf-8") as stream:
        stream.write("\n[archive worker stage started " + now() + "]\n")
        stream.flush()
        return execute(stream, subprocess.STDOUT)


def disk_guard(root):
    if shutil.disk_usage(root).free < MIN_FREE_BYTES:
        raise RuntimeError("Work disk has less than the required 20 GiB free")


def wait_for_gpu(args, env, runner, sleeper):
    if args.device == "cpu":
        return dict(env, CUDA_VISIBLE_DEVICES="")
    began = time.monotonic()
    while True:
        result = runner(["nvidia-smi", "--id=" + str(args.gpu_index),
                         "--query-gpu=memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
                        cwd=args.work_root, env=env, timeout=15)
        fields = result.stdout.strip().splitlines()[0].split(",")
        free, utilization = map(float, fields)
        if math.isfinite(free) and math.isfinite(utilization) and free >= 8192 and 0 <= utilization <= 25:
            return dict(env, CUDA_VISIBLE_DEVICES=str(args.gpu_index))
        if args.gpu_wait_seconds and time.monotonic() - began >= args.gpu_wait_seconds:
            raise RuntimeError("GPU remained busy; no inference was started")
        print("[archive] waiting for 8 GiB free VRAM and GPU utilization <= 25%", flush=True)
        sleeper(30)


def inspect_download(media, info_path, job, args, env, runner):
    if media.is_symlink() or info_path.is_symlink() or not media.is_file() or media.stat().st_size == 0:
        raise ValueError("Acquisition did not produce a regular nonempty media file")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("id") != job["video_id"]:
        raise ValueError("Downloaded metadata ID differs from the queued YouTube ID")
    window = job["requested_window"]
    for field, key in (("section_start", "start"), ("section_end", "end")):
        if info.get(field) is None or abs(number(info[field]) - window[key]) > 0.001:
            raise ValueError("Downloader section metadata differs from the requested bounds")
    result = runner([args.ffprobe, "-v", "error", "-show_entries",
                     "format=duration,start_time:stream=codec_type,codec_name,width,height,avg_frame_rate",
                     "-of", "json", str(media)], cwd=args.work_root, env=env, timeout=45)
    probe = json.loads(result.stdout)
    duration = number(probe.get("format", {}).get("duration"))
    if duration <= 0 or not any(s.get("codec_type") == "video" for s in probe.get("streams", [])):
        raise ValueError("Acquired media has no usable video stream")
    if abs(duration - (window["end"] - window["start"])) > 3:
        raise ValueError("Acquired duration differs from the requested interval by more than 3 seconds")
    return {"status": "complete", "source_video_sha256": digest(media), "info_sha256": digest(info_path),
            "requested_window": window, "ffprobe": probe, "acquired_at": now(),
            "uploader_metadata": {key: info.get(key) for key in
                ("id", "title", "channel_id", "upload_date", "duration", "section_start", "section_end", "format_id", "vcodec")}}


def reusable_media(state, media, info, job):
    previous = state.get("acquisition", {})
    try:
        return (previous.get("status") == "complete" and previous.get("requested_window") == job["requested_window"]
                and not media.is_symlink() and not info.is_symlink()
                and previous.get("source_video_sha256") == digest(media)
                and previous.get("info_sha256") == digest(info)
                and json.loads(info.read_text(encoding="utf-8")).get("id") == job["video_id"])
    except (OSError, ValueError, TypeError):
        return False


def current_analysis(folder, source_hash, window):
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from modules.completeness import publication_issues
    from modules.pipeline import read_manifest
    manifest = read_manifest(folder) or {}
    config = manifest.get("configuration", {})
    if (publication_issues(folder, allow_legacy=False) or not manifest.get("run_id") or
            config.get("source_sha256") != source_hash or
            config.get("clip_start_seconds") != window["start"] or config.get("clip_end_seconds") != window["end"]):
        raise RuntimeError("Analysis returned without a completed, source-bound output manifest")
    return manifest


def current_localization(folder, source_hash):
    try:
        with (folder / "[L1]localization.csv").open(newline="", encoding="utf-8-sig") as stream:
            row = next(csv.DictReader(stream))
        lat, lon = float(row["lat"]), float(row["lon"])
        if (row.get("status") == "ok" and row.get("source_video_sha256") == source_hash and
                math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
            return row
    except (OSError, ValueError, KeyError, StopIteration, TypeError):
        pass
    return None


def run_worker(args, runner=run_command, sleeper=time.sleep):
    args.queue, args.work_root = Path(args.queue).resolve(), Path(args.work_root).resolve()
    if args.work_root == REPO or REPO in args.work_root.parents:
        raise ValueError("Work root must be outside the source repository")
    for name in ("media", "results", "state", "logs"):
        path = args.work_root / name
        path.mkdir(parents=True, exist_ok=True)
        if not path.resolve().is_relative_to(args.work_root):
            raise ValueError("Work directories must not redirect outside the work root")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    if args.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    with batch_lock(args.work_root / "state" / "batch.lock"):
        jobs, skipped = select_jobs(args)
        batch_path = args.work_root / "state" / "batch.json"
        batch = {"schema_version": 1, "queue_path": str(args.queue), "queue_sha256": digest(args.queue),
                 "started_at": now(), "pid": os.getpid(), "skipped": skipped, "jobs": {},
                 "publication_performed": False, "location_verification_performed": False}
        # Per-video state is canonical; recover the summary after any interrupted two-file update.
        for path in (args.work_root / "state").glob("video-*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                batch["jobs"][record["request"]["key"]] = {"status": record.get("status"), "state_path": str(path)}
            except (OSError, ValueError, TypeError, KeyError):
                batch["jobs"][path.stem] = {"status": "state_unreadable", "state_path": str(path)}
        atomic_json(batch_path, batch)
        if not jobs:
            raise ValueError("No selected rows meet the ready queue contract")
        failures = 0
        for job in jobs:
            state_path = args.work_root / "state" / ("video-" + job["key"] + ".json")
            try:
                state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {
                    "schema_version": 1, "request": job, "created_at": now(), "status": "pending",
                    "recorded_at": None, "recording_date_source": "unknown", "eligibility_status": "unverified",
                    "location_verification_status": "pending", "verified_city_id": None, "attempts": []}
                # Queue row ordering may change; it is not part of the source request.
                if ({k: v for k, v in state["request"].items() if k != "queue_row"} !=
                        {k: v for k, v in job.items() if k != "queue_row"} or not isinstance(state["attempts"], list)):
                    raise ValueError("Existing state belongs to a different source request")
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                # Preserve damaged evidence for review, and continue other rows.
                batch["jobs"][job["key"]] = {"status": "state_unreadable", "state_path": str(state_path)}
                atomic_json(batch_path, batch)
                failures += 1
                continue
            def save():
                state["updated_at"] = now()
                atomic_json(state_path, state)
                batch["jobs"][job["key"]] = {"status": state["status"], "state_path": str(state_path)}
                batch["updated_at"] = now()
                atomic_json(batch_path, batch)
            media_dir = args.work_root / "media" / job["key"]
            media = media_dir / (job["name"] + "_" + job["video_id"] + ".mp4")
            info = media.with_name(media.name + ".info.json")
            results_root = args.work_root / "results" / job["key"]
            folder = results_root / media.stem
            log_path = args.work_root / "logs" / (job["key"] + ".log")
            state["attempts"].append({"started_at": now(), "device": args.device, "prepare_only": args.prepare_only})
            try:
                disk_guard(args.work_root)
                for path in (media_dir, results_root, folder, state_path, log_path):
                    if not path.resolve().is_relative_to(args.work_root):
                        raise ValueError("Per-video paths must stay inside the work root")
                media_dir.mkdir(parents=True, exist_ok=True)
                if not reusable_media(state, media, info, job):
                    state.update(status="acquiring", analysis={"status": "pending"}, localization={"status": "pending"})
                    save()
                    staging = media_dir / ("download-" + uuid.uuid4().hex)
                    staging.mkdir()
                    cookies = staging / "cookies.txt"
                    try:
                        shutil.copyfile(args.cookies, cookies)
                        target = staging / media.name
                        window = job["requested_window"]
                        download_python = getattr(args, "download_python", None) or args.python
                        download_timeout = getattr(args, "download_timeout_seconds", 3600)
                        command = [download_python, "-m", "yt_dlp", "--ignore-config", "--no-playlist", "--no-progress", "--write-info-json",
                            "--cookies", str(cookies), "--js-runtimes", "node", "--force-overwrites",
                            "--socket-timeout", "30", "--retries", "3", "--fragment-retries", "3",
                            "--extractor-retries", "3", "--file-access-retries", "3", "--abort-on-unavailable-fragments",
                            "--retry-sleep", "http:exp=1:8", "--retry-sleep", "fragment:exp=1:8",
                            "--downloader-args", "ffmpeg_i:-rw_timeout 30000000",
                            "-f", "bestvideo[height<=720][vcodec^=avc1]/bestvideo[height<=720]/bestvideo",
                            "--remux-video", "mp4", "--download-sections", f"*{window['start']:g}-{window['end']:g}",
                            "--force-keyframes-at-cuts", "-o", str(target),
                            "https://www.youtube.com/watch?v=" + job["video_id"]]
                        runner(command, cwd=REPO, env=env, log_path=log_path, timeout=download_timeout)
                        info_candidates = [target.with_name(target.name + ".info.json"), target.with_suffix(".info.json")]
                        staged_info = next((path for path in info_candidates if path.is_file()), info_candidates[0])
                        acquisition = inspect_download(target, staged_info, job, args, env, runner)
                        acquisition.update(download_python=download_python, download_timeout_seconds=download_timeout)
                        os.replace(target, media)
                        os.replace(staged_info, info)
                        state["acquisition"] = acquisition
                        atomic_json(media.with_suffix(".source.json"), {"request": job, **acquisition,
                            "recorded_at": None, "recording_date_source": "unknown"})
                    finally:
                        cookies.unlink(missing_ok=True)
                state["status"] = "media_ready"
                save()
                if args.prepare_only:
                    continue
                disk_guard(args.work_root)
                state["status"] = "waiting_for_gpu" if args.device == "gpu" else "analyzing"
                save()
                child_env = wait_for_gpu(args, env, runner, sleeper)
                window = job["requested_window"]
                child_env.update(PEDX_RESULTS_ROOT=str(results_root), PEDX_CLIP_START_SECONDS=f"{window['start']:g}",
                                 PEDX_CLIP_END_SECONDS=f"{window['end']:g}")
                state.update(status="analyzing", analysis={"status": "running"})
                save()
                runner([args.python, "-B", str(REPO / "main.py"), "--mode", "single_all", "--resume-failed",
                        "--source_video_path", str(media)], cwd=REPO, env=child_env, log_path=log_path)
                source_hash = state["acquisition"]["source_video_sha256"]
                if digest(media) != source_hash:
                    raise RuntimeError("Source changed during analysis")
                manifest = current_analysis(folder, source_hash, window)
                state["analysis"] = {"status": "complete", "analysis_run_id": manifest["run_id"], "results_root": str(results_root)}
                state["status"] = "localizing"
                save()
                location = current_localization(folder, source_hash)
                if location is None:
                    disk_guard(args.work_root)
                    child_env = wait_for_gpu(args, child_env, runner, sleeper)
                    runner([args.python, "-B", str(REPO / "modules" / "localization" / "localize.py"),
                            "--source_video_path", str(media), "--city", job["city"] + ", " + job["country"],
                            "--osm_python", args.osm_python, "--no-splat", "--no-aerial"],
                           cwd=args.work_root / "media", env=child_env, log_path=log_path)
                    location = current_localization(folder, source_hash)
                if location is None or digest(media) != source_hash:
                    raise RuntimeError("Localization returned without a usable source-bound position")
                manifest = current_analysis(folder, source_hash, window)
                state["localization"] = {"status": "complete", "origin_run_id": location.get("analysis_run_id"),
                    "target_analysis_run_id": manifest["run_id"], "source_video_sha256": source_hash,
                    "localized_at": location.get("localized_at")}
                state["status"] = "complete_review_pending"
                save()
            except KeyboardInterrupt:
                state["status"] = "interrupted"
                save()
                return 130
            except Exception as error:
                state.update(error={"stage": state["status"], "type": type(error).__name__, "message": str(error)}, status="failed")
                save()
                failures += 1
                print("[archive] failed " + job["video_id"] + "; details in private state/log", flush=True)
        batch["finished_at"] = now()
        atomic_json(batch_path, batch)
        return 1 if failures else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--python", required=True, help="CV Python interpreter")
    parser.add_argument("--download-python", help="Isolated yt-dlp[default] interpreter (default: --python)")
    parser.add_argument("--download-timeout-seconds", type=float, default=3600,
                        help="Total acquisition deadline, including ffmpeg; does not limit inference")
    parser.add_argument("--osm-python", help="Explicit companion-tool Python interpreter")
    parser.add_argument("--cookies", type=Path, default=REPO / "www.youtube.com_cookies.txt")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--limit", type=int, default=0, help="0 processes all ready rows")
    parser.add_argument("--video-id")
    parser.add_argument("--clip-seconds", type=float, help="Explicit pilot cap; requires --video-id")
    parser.add_argument("--prepare-only", action="store_true", help="Acquire at most one clip; run no inference")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--gpu-wait-seconds", type=float, default=0, help="0 waits until the selected GPU is free")
    args = parser.parse_args(argv)
    if args.limit < 0 or args.gpu_index < 0 or not math.isfinite(args.gpu_wait_seconds) or args.gpu_wait_seconds < 0:
        parser.error("Limits and GPU index/wait must be nonnegative")
    if args.clip_seconds is not None and (not args.video_id or not math.isfinite(args.clip_seconds) or args.clip_seconds <= 0):
        parser.error("--clip-seconds requires --video-id and a finite positive duration")
    if not math.isfinite(args.download_timeout_seconds) or args.download_timeout_seconds <= 0:
        parser.error("--download-timeout-seconds must be finite and positive")
    if not args.prepare_only and not args.osm_python:
        parser.error("--osm-python is required for analysis/localization")
    for attr in ("python", "osm_python", "download_python"):
        if getattr(args, attr):
            value = Path(getattr(args, attr)).resolve()
            if not value.is_file():
                parser.error("An explicit existing interpreter path is required for --" + attr.replace("_", "-"))
            setattr(args, attr, str(value))
    try:
        return run_worker(args)
    except (OSError, ValueError, RuntimeError) as error:
        print("[archive] " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
