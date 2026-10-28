"""Dependency-aware execution with durable, verifiable per-video manifests."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from importlib import metadata
import socket
import math
import warnings

from modules.paths import RESULTS_ROOT
from modules.provenance import git_state, utc_now

MANIFEST_NAME = "[A0]run_manifest.json"
LOCK_NAME = "[A0]run.lock"
OWNER_NAME = "[A0]run_owner.json"
# mode: (prerequisites, outputs). Empty CSVs are valid; missing headers are not.
STAGES = {
    "id_img": ((), ("[B0]video_meta.csv", "[B1]tracked_pedestrians.csv", "[B2]dense_tracks.csv", "pedestrian_img")),
    "ego": (("id_img",), ("[B3]ego_motion.csv",)),
    "waiting": (("id_img", "ego"), ("[P3]waiting_time.csv",)),
    "phone": (("id_img",), ("[P5]phone_usage.csv",)),
    "ag": (("id_img",), ("[P6]age_gender.csv",)),
    "clothing": (("id_img",), ("[P8]clothing.csv",)),
    "belongings": (("id_img",), ("[P9]pedestrian_belongings.csv",)),
    "vehicle_type": ((), ("[V1]vehicle_type.csv",)),
    "lane": ((), ("[V5]lane_detection.csv",)),
    "count_vehicle": ((), ("[V6]vehicle_count.csv", "[V7]vehicle_tracks.csv", "[V10]line_crossing_events.csv")),
    "weather": ((), ("[E1]weather.csv",)),
    "light": ((), ("[E2]traffic_light.csv",)),
    "traffic_sign": ((), ("[E3]traffic_sign.csv",)),
    "road_condition": ((), ("[E4]road_condition.csv",)),
    "width": (("lane",), ("[E5]road_width.csv",)),
    "daytime": ((), ("[E6]daytime.csv",)),
    "crosswalk": ((), ("[E7]crosswalk_detection.csv", "[E10]scene_segmentation.csv")),
    "accident": ((), ("[E8]accident_detection.csv",)),
    "sidewalk": ((), ("[E9]sidewalk_detection.csv",)),
    "cross_pede": (("id_img", "sidewalk"), ("[C3]crossing_judge.csv",)),
    "scale": (("crosswalk",), ("[S2]scale_calibration.csv",)),
    "speed": (("id_img", "ego", "cross_pede", "scale", "lane"), ("[S1]pedestrian_speed.csv",)),
    "pose": (("id_img", "cross_pede"), ("[P12]pose_behavior.csv",)),
    "risky": (("cross_pede", "light", "traffic_sign", "crosswalk"), ("[C1]risky_crossing.csv",)),
    "crosswalk_usage": (("cross_pede", "crosswalk"), ("[C4]crosswalk_usage.csv",)),
    "run_red": (("cross_pede", "light"), ("[C5]red_light_runner.csv",)),
    "crossing_vehicle_count": (("cross_pede", "count_vehicle"), ("[C6]crossing_ve_count.csv",)),
    "on_lane": (("id_img", "lane"), ("[C8]pedestrian_on_lane.csv",)),
    "nearby": (("id_img", "cross_pede"), ("[C10]nearby_count.csv",)),
    "vehicle_speed": (("count_vehicle", "ego", "scale", "lane"), ("[V8]vehicle_speed.csv",)),
    "pet": (("vehicle_speed", "cross_pede", "speed"), ("[I1]pet_conflicts.csv",)),
    "headway": (("count_vehicle",), ("[V11]headway_stats.csv",)),
    "signal_timing": (("cross_pede", "light"), ("[P10]signal_timing.csv",)),
    "micro_events": (("speed", "risky", "crossing_vehicle_count"), ("[P11]micro_events.csv",)),
    "groups": (("speed", "cross_pede"), ("[I2]pedestrian_groups.csv", "[I3]group_crossings.csv")),
    "personal_info": (("nearby", "risky", "run_red", "crosswalk_usage", "ag", "phone", "clothing", "belongings", "speed"), ("[C7]crossing_pe_info.csv",)),
    "env_info": (("cross_pede", "weather", "daytime", "traffic_sign", "width", "road_condition", "crosswalk", "accident", "sidewalk", "vehicle_type"), ("[C9]crossing_env_info.csv",)),
}
_ANALYSIS = tuple(STAGES)
STAGES["sum_video"] = (_ANALYSIS, ("[A1]video_info.csv",))
STAGES["sum_pede"] = (("personal_info", "env_info"), ("[A2]pedestrian_info.csv",))
PROFILES = {
    "single_all": tuple(STAGES), "mul_all": tuple(STAGES),
    "pedestrian": ("waiting", "speed", "phone", "ag", "clothing", "belongings"),
    "vehicle": ("vehicle_type", "lane", "count_vehicle"),
    "environment": ("weather", "light", "traffic_sign", "road_condition", "width", "daytime", "crosswalk", "accident", "sidewalk"),
}


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _os_lock(stream, *, release=False):
    stream.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK if release else msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN if release else fcntl.LOCK_EX | fcntl.LOCK_NB)


def _check_child_leases(folder):
    leases = Path(folder) / ".pipeline_children"
    if not leases.exists():
        return
    for child in leases.iterdir():
        path = child / LOCK_NAME
        if child.is_symlink() or not path.resolve().is_relative_to(Path(folder).resolve()):
            raise ValueError("Child lease path escapes the result folder")
        if not path.is_file():
            continue
        with open(path, "r+b", buffering=0) as stream:
            try:
                _os_lock(stream)
            except OSError as error:
                raise RuntimeError("Video still has an active pipeline child; wait for its PID to exit") from error
            else:
                _os_lock(stream, release=True)


@contextmanager
def video_lock(folder, run_id, *, check_children=True):
    """OS-owned lock; crashed PIDs release automatically. Keep the inode after unlock."""
    path = Path(folder) / LOCK_NAME
    owner_path = Path(folder) / OWNER_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("Video lock must not be a symlink")
    stream = open(path, "a+b", buffering=0)
    acquired, owner = False, {}
    try:
        if path.stat().st_size == 0:
            stream.write(b"\n")
        stream.seek(0)
        try:
            _os_lock(stream)
            acquired = True
        except OSError as error:
            raise RuntimeError(f"Video analysis is already owned by another process: {path}") from error
        try:
            previous = json.loads(owner_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError):
            previous = {}
        owner = {"pid": os.getpid(), "host": socket.gethostname(), "run_id": run_id,
                 "status": "active", "started_at": utc_now()}
        if isinstance(previous, dict) and previous.get("status") == "active":
            owner["recovered_owner"] = {key: previous.get(key) for key in ("pid", "host", "run_id")}
        atomic_json(owner_path, owner)
        if check_children:
            _check_child_leases(folder)
        yield owner
    finally:
        try:
            if acquired:
                owner.update(status="released", released_at=utc_now())
                atomic_json(owner_path, owner)
        finally:
            try:
                if acquired:
                    _os_lock(stream, release=True)
            finally:
                stream.close()


def _runtime_versions():
    names = ("numpy", "pandas", "opencv-python", "torch", "torchvision", "ultralytics",
             "transformers", "supervision", "scipy", "Pillow", "statsmodels", "scikit-image",
             "onnxruntime", "filterpy", "lap", "Shapely", "scikit-learn")
    versions = {"python": sys.version.split()[0], "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    for name in names:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Implicit None on return values.*", category=DeprecationWarning)
                versions[name] = metadata.version(name)
        except (metadata.PackageNotFoundError, KeyError):
            versions[name] = None
    return versions


def _clip_bound(name):
    value = os.environ.get(name)
    if not value:
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def _directory_record(path):
    path = Path(path)
    digest, files = hashlib.sha256(), 0
    for child in sorted(path.rglob("*"), key=lambda value: value.relative_to(path).as_posix()):
        if child.is_symlink():
            raise ValueError(f"Output directory contains a symlink: {child}")
        if child.is_file():
            digest.update(json.dumps([child.relative_to(path).as_posix(), file_hash(child)]).encode("utf-8"))
            files += 1
    return {"kind": "directory", "sha256": digest.hexdigest(), "files": files}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def read_manifest(folder):
    path = Path(folder) / MANIFEST_NAME
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        return result if isinstance(result, dict) else {"status": "invalid", "error": "run manifest is not an object"}
    except (ValueError, OSError):
        return {"status": "invalid", "error": "unreadable run manifest"}


def build_plan(profile):
    result, visiting = [], set()
    def visit(mode):
        if mode in result:
            return
        if mode in visiting or mode not in STAGES:
            raise ValueError(f"Invalid pipeline dependency: {mode}")
        visiting.add(mode)
        for dependency in STAGES[mode][0]:
            visit(dependency)
        visiting.remove(mode)
        result.append(mode)
    for mode in PROFILES.get(profile, (profile,)):
        visit(mode)
    return result


def output_record(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"Output must not be a symlink: {path}")
    if path.name == "pedestrian_img" and not path.is_dir():
        raise ValueError("Pedestrian crop output must be a directory")
    if path.is_dir():
        if path.suffix.lower() == ".csv":
            raise ValueError(f"CSV output is a directory: {path}")
        return _directory_record(path)
    with open(path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        columns = next(reader, None)
        if not columns or not any(columns):
            raise ValueError(f"CSV has no header: {path}")
        required = {
            "[B0]video_meta.csv": {"fps", "total_frames", "width", "height"},
            "[B1]tracked_pedestrians.csv": {"track_id", "frame_id", "timestamp", "x1", "y1", "x2", "y2"},
            "[B2]dense_tracks.csv": {"track_id", "frame_id", "timestamp", "x1", "y1", "x2", "y2"},
            "[C3]crossing_judge.csv": {"track_id", "crossed", "started_frame", "ended_frame"},
            "[E1]weather.csv": {"frame_id", "weather_label"},
            "[E2]traffic_light.csv": {"frame_id", "main_light_color"},
            "[E7]crosswalk_detection.csv": {"frame_id", "crosswalk_detected", "crosswalk_boxes"},
            "[V6]vehicle_count.csv": {"Vehicle_Type", "Count"},
            "[A1]video_info.csv": {"metric", "value"},
            "[A2]pedestrian_info.csv": {"track_id", "crossed"},
        }.get(Path(path).name, set())
        if not required.issubset(columns):
            raise ValueError(f"Missing columns in {path}: {sorted(required - set(columns))}")
        rows = sum(1 for _ in reader)
    return {"sha256": file_hash(path), "columns": columns, "rows": rows}


def stage_current(folder, stage):
    if not isinstance(stage, dict) or stage.get("status") != "complete" or not isinstance(stage.get("outputs"), dict) or not stage["outputs"]:
        return False
    outputs = stage["outputs"]
    owners = [name for name, (_, expected) in STAGES.items() if set(outputs).intersection(expected)]
    if len(owners) != 1 or set(outputs) != set(STAGES[owners[0]][1]):
        return False
    for name, record in outputs.items():
        if not isinstance(record, dict):
            return False
        path = Path(folder) / name
        if path.is_symlink():
            return False
        if name == "pedestrian_img" and record.get("kind") != "directory":
            return False
        try:
            actual = _directory_record(path) if record.get("kind") == "directory" and path.is_dir() else {"sha256": file_hash(path)}
            if actual["sha256"] != record.get("sha256"):
                return False
        except (OSError, ValueError):
            return False
    return True


def source_digest(repo):
    digest = hashlib.sha256()
    paths = []
    skip = {".git", ".venv", "venv", "external", "analysis_results", "analysis_results_legacy_archive", "osm_localization_cache", "SGDepth", "__pycache__"}
    for directory, dirs, names in os.walk(repo):
        dirs[:] = [name for name in dirs if name not in skip]
        paths.extend(Path(directory) / name for name in names if name.endswith(".py"))
    for path in sorted(paths):
        relative = path.relative_to(repo)
        digest.update(relative.as_posix().encode())
        digest.update(path.read_bytes())
    for name in ("mybotsort.yaml", "mybytetrack.yaml", "requirements.txt"):
        path = Path(repo) / name
        if path.exists():
            digest.update(name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def invalidate_standalone_run(video_path, mode):
    if mode == "localize":
        return
    folder = Path(RESULTS_ROOT) / Path(video_path).stem
    manifest = read_manifest(folder)
    child_id = os.environ.get("PEDX_PIPELINE_RUN_ID")
    if child_id:
        try:
            owner = json.loads((folder / OWNER_NAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            owner = {}
        if not manifest or not isinstance(owner, dict) or manifest.get("run_id") != child_id or manifest.get("status") != "running" or owner.get("run_id") != child_id or owner.get("status") != "active":
            raise RuntimeError("Pipeline child does not own the active video run")
        return
    if manifest:
        manifest.update(status="stale", invalidated_at=utc_now(), invalidated_by=mode)
        atomic_json(folder / MANIFEST_NAME, manifest)


@contextmanager
def standalone_run(video_path, mode):
    if mode == "localize":
        invalidate_standalone_run(video_path, mode)
        yield
        return
    folder = Path(RESULTS_ROOT) / Path(video_path).stem
    child_id = os.environ.get("PEDX_PIPELINE_RUN_ID")
    if child_id:
        invalidate_standalone_run(video_path, mode)
        # Keep an OS lease in the child too: killing only the parent cannot allow a
        # new analysis to overlap a still-running orphan detector subprocess.
        lease = folder / ".pipeline_children" / f"{child_id}_{os.getpid()}"
        with video_lock(lease, child_id, check_children=False):
            invalidate_standalone_run(video_path, mode)
            yield
        return
    standalone_id = uuid.uuid4().hex
    with video_lock(folder, standalone_id):
        invalidate_standalone_run(video_path, mode)
        if mode == "id_img":
            archive = folder / ".previous" / standalone_id
            if not archive.resolve().is_relative_to(folder.resolve()):
                raise ValueError("Archive path escapes the video result folder")
            for name in STAGES["id_img"][1]:
                old_output = folder / name
                if old_output.exists() or old_output.is_symlink():
                    if old_output.is_symlink() or not old_output.resolve().is_relative_to(folder.resolve()):
                        raise ValueError("Output archive target escapes the video result folder")
                    archive.mkdir(parents=True, exist_ok=True)
                    os.replace(old_output, archive / name)
        yield


def execute_pipeline(video_path, profile, interval=1.0, weights="yolo11n.pt", *,
                     resume=False, runner=None, results_root=None, repo=None):
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("Analysis interval must be finite and positive")
    video_path = Path(video_path).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    folder = Path(results_root or RESULTS_ROOT) / video_path.stem
    run_id = uuid.uuid4().hex
    with video_lock(folder, run_id):
        return _execute_pipeline(video_path, profile, interval, weights, resume=resume,
                                 runner=runner, results_root=results_root, repo=repo, run_id=run_id)


def _execute_pipeline(video_path, profile, interval=1.0, weights="yolo11n.pt", *,
                      resume=False, runner=None, results_root=None, repo=None, run_id=None):
    runner = runner or subprocess.run
    repo = Path(repo or Path(__file__).resolve().parents[1])
    video_path = Path(video_path).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    folder = Path(results_root or RESULTS_ROOT) / video_path.stem
    folder.mkdir(parents=True, exist_ok=True)
    plan = build_plan(profile)
    config = {"analysis_interval": interval, "weights_yolo": weights,
              "source_sha256": file_hash(video_path), "source_code_sha256": source_digest(repo),
              "mapping_sha256": file_hash(repo / "mapping.csv") if (repo / "mapping.csv").is_file() else None,
              "runtime_versions": _runtime_versions(),
              "clip_start_seconds": _clip_bound("PEDX_CLIP_START_SECONDS"),
              "clip_end_seconds": _clip_bound("PEDX_CLIP_END_SECONDS")}
    if config["clip_start_seconds"] is not None and config["clip_end_seconds"] is not None and config["clip_end_seconds"] <= config["clip_start_seconds"]:
        raise ValueError("Clip end must be later than clip start")
    weight_path = Path(weights).expanduser()
    if not weight_path.is_absolute():
        weight_path = repo / weight_path
    config["explicit_weights_sha256"] = file_hash(weight_path) if weight_path.is_file() else None
    assets_path = folder / "[A0]model_assets.json"
    config["model_assets_sha256"] = file_hash(assets_path) if assets_path.is_file() else None
    from modules.provenance import weights_manifest, manifest_digest
    initial_weights = weights_manifest()
    config["weights_digest"] = manifest_digest(initial_weights)
    config["scene_model_revision"] = os.environ.get("PEDX_SCENE_MODEL_REVISION") or "4772b6bf101d91f2534c106dc524d906aeb3c68a"
    if len(config["scene_model_revision"]) != 40 or any(char not in "0123456789abcdefABCDEF" for char in config["scene_model_revision"]):
        raise ValueError("PEDX_SCENE_MODEL_REVISION must identify an immutable 40-character commit")
    previous = read_manifest(folder) or {}
    reusable = previous.get("stages", {}) if resume and previous.get("configuration") == config else {}
    manifest = {"schema_version": 1, "run_id": run_id, "status": "running",
                "profile": profile, "configuration": config, "source_video": str(video_path),
                "started_at": utc_now(), "git_sha": git_state()[0], "stages": {}}
    atomic_json(folder / MANIFEST_NAME, manifest)
    started = time.monotonic()
    archived_timing = False
    unchanged_resume = False
    try:
        for mode in plan:
            dependencies = STAGES[mode][0]
            blocked = [dep for dep in dependencies if manifest["stages"].get(dep, {}).get("status") != "complete"]
            if blocked:
                manifest["stages"][mode] = {"status": "blocked", "dependencies": blocked}
                atomic_json(folder / MANIFEST_NAME, manifest)
                continue
            old = reusable.get(mode, {})
            deps_reused = all(manifest["stages"][dep].get("reused") for dep in dependencies)
            if deps_reused and stage_current(folder, old):
                manifest["stages"][mode] = {**old, "reused": True}
                atomic_json(folder / MANIFEST_NAME, manifest)
                continue
            stage = {"status": "running", "started_at": utc_now()}
            manifest["stages"][mode] = stage
            atomic_json(folder / MANIFEST_NAME, manifest)
            stage_started = time.monotonic()
            env = dict(os.environ, PEDX_PIPELINE_RUN_ID=manifest["run_id"],
                       PEDX_RESULTS_ROOT=str(folder.parent.resolve()), PEDX_SOURCE_SHA256=config["source_sha256"])
            command = [sys.executable, str(repo / "main.py"), "--mode", mode,
                       "--source_video_path", str(video_path), "--analysis_interval", str(interval),
                       "--weights_yolo", weights]
            print(f"[pipeline] {mode}", flush=True)
            try:
                # Keep the previous files for audit, but never mistake them for this attempt's outputs.
                archive = folder / ".previous" / manifest["run_id"]
                if not archive.resolve().is_relative_to(folder.resolve()):
                    raise ValueError("Archive path escapes the video result folder")
                # [A3] is written by the batch after analysis. It cannot describe a new
                # actual execution; retain it only for a completely unchanged resume.
                old_timing = folder / "[A3]analysis_time.csv"
                if not archived_timing and old_timing.is_file():
                    if old_timing.is_symlink() or not old_timing.resolve().is_relative_to(folder.resolve()):
                        raise ValueError("Timing archive target escapes the video result folder")
                    archive.mkdir(parents=True, exist_ok=True)
                    os.replace(old_timing, archive / old_timing.name)
                    archived_timing = True
                for name in STAGES[mode][1]:
                    old_output = folder / name
                    if old_output.exists() or old_output.is_symlink():
                        if not old_output.resolve().is_relative_to(folder.resolve()) or old_output.is_symlink():
                            raise ValueError("Output archive target escapes the video result folder")
                        archive.mkdir(parents=True, exist_ok=True)
                        os.replace(old_output, archive / name)
                result = runner(command, cwd=str(repo), env=env)
                current = read_manifest(folder) or {}
                if current.get("run_id") != run_id or current.get("status") != "running":
                    raise RuntimeError("Pipeline ownership changed while a stage was executing")
                if result.returncode:
                    raise RuntimeError(f"exited with status {result.returncode}")
                outputs = {name: output_record(folder / name) for name in STAGES[mode][1]}
                stage.update(status="complete", outputs=outputs)
            except Exception as error:
                stage.update(status="failed", error=str(error))
                print(f"[FAIL] {mode}: {error}", flush=True)
            stage.update(finished_at=utc_now(), seconds=round(time.monotonic() - stage_started, 3))
            atomic_json(folder / MANIFEST_NAME, manifest)
        failed = [name for name, stage in manifest["stages"].items() if stage["status"] != "complete"]
        manifest["status"] = "failed" if failed else ("complete" if set(plan) == set(STAGES) else "partial")
        manifest["failed_stages"] = failed
        final_mapping = file_hash(repo / "mapping.csv") if (repo / "mapping.csv").is_file() else None
        final_weights = weights_manifest()
        changed_inputs = (file_hash(video_path) != config["source_sha256"]
            or source_digest(repo) != config["source_code_sha256"]
            or final_mapping != config["mapping_sha256"] or _runtime_versions() != config["runtime_versions"]
            or (os.environ.get("PEDX_SCENE_MODEL_REVISION") or "4772b6bf101d91f2534c106dc524d906aeb3c68a") != config["scene_model_revision"]
            or any(final_weights.get(name) != value for name, value in initial_weights.items())
            or (config["explicit_weights_sha256"] is not None and file_hash(weight_path) != config["explicit_weights_sha256"]))
        if changed_inputs:
            manifest.update(status="failed", error="Source, code, mapping, runtime, or existing weights changed during this run")
        else:
            config["weights_digest"] = manifest_digest(final_weights)
            config["explicit_weights_sha256"] = file_hash(weight_path) if weight_path.is_file() else None
            config["model_assets_sha256"] = file_hash(assets_path) if assets_path.is_file() else None
            unchanged_resume = (previous.get("status") == "complete" and bool(plan)
                                and all(stage.get("reused") for stage in manifest["stages"].values()))
            if unchanged_resume:
                # A verification-only resume did not create new observations/timings.
                for key in ("run_id", "started_at", "finished_at", "seconds"):
                    manifest[key] = previous.get(key)
    except BaseException:
        manifest["status"] = "interrupted"
        raise
    finally:
        if unchanged_resume:
            manifest.update(last_verified_at=utc_now(), verification_run_id=run_id,
                            verification_seconds=round(time.monotonic() - started, 3))
        else:
            manifest.update(finished_at=utc_now(), seconds=round(time.monotonic() - started, 3))
        atomic_json(folder / MANIFEST_NAME, manifest)
    return manifest
