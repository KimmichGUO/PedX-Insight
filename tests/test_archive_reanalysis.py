"""Worker lifecycle with fake external tools; no downloads, models, GPU or database."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from scripts import reanalyze_archive as worker
from modules import completeness, pipeline


def row(link="ULl8s4qydrk", **extra):
    return {"video": link, "name": "Ulm2", "city": "Ulm", "country": "Germany",
            "start_time": "0", "end_time": "1800", "queue_status": worker.READY, **extra}


def arguments(tmp_path, rows=None, **extra):
    queue = tmp_path / "queue.csv"
    rows = rows or [row()]
    with queue.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cookies = tmp_path / "original-cookies.txt"
    cookies.write_text("test cookie jar; retain unchanged", encoding="utf-8")
    return argparse.Namespace(queue=queue, work_root=tmp_path / "work", python=sys.executable,
        osm_python=sys.executable, cookies=cookies, ffprobe="ffprobe", limit=0, video_id=None,
        clip_seconds=None, prepare_only=False, device="cpu", gpu_index=0, gpu_wait_seconds=0, **extra)


@pytest.fixture
def fake_contract(monkeypatch):
    # Retain the real manifest/hash verification while shrinking the stage inventory.
    monkeypatch.setattr(pipeline, "STAGES", {"fake": ((), ("result.csv",))})
    monkeypatch.setattr(completeness, "REQUIRED_OUTPUTS", ["result.csv"])


class Tools:
    def __init__(self, *, fail_analysis=None, zero_output=False, wrong_id=False,
                 short_media=False, wrong_location=False, change_source=False, interrupt=False,
                 missing_sections=False, shifted_section=False):
        self.calls, self.windows = [], {}
        self.fail_analysis, self.zero_output, self.wrong_id = fail_analysis, zero_output, wrong_id
        self.short_media, self.wrong_location = short_media, wrong_location
        self.change_source, self.interrupt = change_source, interrupt
        self.missing_sections, self.shifted_section = missing_sections, shifted_section
        self.gpu = ["9000, 0"]

    def __call__(self, command, *, cwd, env, log_path=None, timeout=None):
        self.calls.append({"command": list(command), "env": dict(env), "cwd": Path(cwd), "timeout": timeout})
        if "yt_dlp" in command:
            if self.interrupt:
                self.interrupt = False
                raise KeyboardInterrupt()
            target = Path(command[command.index("-o") + 1])
            start, end = map(float, command[command.index("--download-sections") + 1][1:].split("-"))
            link = command[-1].split("=")[-1]
            target.write_bytes((link + str(start) + str(end)).encode())
            self.windows[str(target)] = (start, end)
            info = {"id": "XXXXXXXXXXX" if self.wrong_id else link,
                    "upload_date": "20200102", "format_id": "avc-test", "vcodec": "avc1"}
            if not self.missing_sections:
                info.update(section_start=start + (10 if self.shifted_section else 0), section_end=end)
            target.with_name(target.name + ".info.json").write_text(json.dumps(info), encoding="utf-8")
        elif command[0] == "ffprobe":
            start, end = self.windows[command[-1]]
            return subprocess.CompletedProcess(command, 0, json.dumps({
                "format": {"duration": str(end - start - (10 if self.short_media else 0))},
                "streams": [{"codec_type": "video", "codec_name": "h264"}]}), "")
        elif command[0] == "nvidia-smi":
            value = self.gpu.pop(0) if len(self.gpu) > 1 else self.gpu[0]
            return subprocess.CompletedProcess(command, 0, value, "")
        elif any(Path(part).name == "main.py" for part in command):
            source = Path(command[command.index("--source_video_path") + 1])
            if self.fail_analysis and self.fail_analysis in source.name:
                raise subprocess.CalledProcessError(1, command)
            if self.zero_output:
                return subprocess.CompletedProcess(command, 0, "", "")
            folder = Path(env["PEDX_RESULTS_ROOT"]) / source.stem
            folder.mkdir(parents=True, exist_ok=True)
            result = folder / "result.csv"
            result.write_text("observed\n1\n", encoding="utf-8")
            worker.atomic_json(folder / pipeline.MANIFEST_NAME, {"status": "complete", "run_id": "fake-real-contract",
                "configuration": {"source_sha256": worker.digest(source),
                    "clip_start_seconds": float(env["PEDX_CLIP_START_SECONDS"]),
                    "clip_end_seconds": float(env["PEDX_CLIP_END_SECONDS"])},
                "stages": {"fake": {"status": "complete", "outputs": {"result.csv": pipeline.output_record(result)}}}})
            if self.change_source:
                source.write_bytes(b"changed during inference")
        elif any(Path(part).name == "localize.py" for part in command):
            source = Path(command[command.index("--source_video_path") + 1])
            folder = Path(env["PEDX_RESULTS_ROOT"]) / source.stem
            with (folder / "[L1]localization.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["status", "lat", "lon", "source_video_sha256", "analysis_run_id", "localized_at"])
                writer.writeheader()
                writer.writerow({"status": "ok", "lat": "48.4", "lon": "10.0", "source_video_sha256":
                    "f" * 64 if self.wrong_location else worker.digest(source),
                    "analysis_run_id": "fake-real-contract", "localized_at": "2026-10-28T00:00:00Z"})
        else:
            raise AssertionError("Unexpected external tool")
        return subprocess.CompletedProcess(command, 0, "", "")

    def count(self, token):
        return sum(any(token == part or Path(part).name == token for part in call["command"]) for call in self.calls)


def states(args):
    return [json.loads(path.read_text(encoding="utf-8")) for path in (args.work_root / "state").glob("video-*.json")]


def test_selection_never_promotes_blocked_invalid_or_ambiguous_rows(tmp_path):
    rows = [row(), row("bad", name="Bad"), row("ABCDEFGHIJK", name="Unsafe_name"),
            row("LMNOPQRSTUV", end_time="nan"), row("abcdefghijk", queue_status="needs_metadata_review"),
            row("12345678901"), row("12345678901")]
    args = arguments(tmp_path, rows)
    jobs, skipped = worker.select_jobs(args)
    assert [job["video_id"] for job in jobs] == ["ULl8s4qydrk"]
    assert len(skipped) == 6


def test_prepare_only_is_one_bounded_pilot_without_inference(tmp_path):
    args = arguments(tmp_path, [row(), row("Qh1n6VVKgn0", name="Ulm1")])
    args.prepare_only, args.video_id, args.clip_seconds = True, "ULl8s4qydrk", 300
    tools = Tools()
    assert worker.run_worker(args, tools) == 0
    state, = states(args)
    assert state["request"]["scope"] == "pilot"
    assert state["request"]["historical_window"]["end"] == 1800
    assert state["acquisition"]["requested_window"] == {"start": 0, "end": 300}
    assert state["status"] == "media_ready" and state["recorded_at"] is None
    assert state["recording_date_source"] == "unknown" and state["verified_city_id"] is None
    assert tools.count("yt_dlp") == 1 and tools.count("main.py") == tools.count("localize.py") == tools.count("nvidia-smi") == 0
    assert args.cookies.read_text() == "test cookie jar; retain unchanged"
    assert not list(args.work_root.rglob("cookies.txt"))


def test_media_resume_requires_both_original_hashes(tmp_path):
    args = arguments(tmp_path)
    args.prepare_only = True
    tools = Tools()
    assert worker.run_worker(args, tools) == worker.run_worker(args, tools) == 0
    assert tools.count("yt_dlp") == 1
    media, = args.work_root.glob("media/*/*.mp4")
    media.write_bytes(b"unexpected replacement")
    assert worker.run_worker(args, tools) == 0
    assert tools.count("yt_dlp") == 2
    info = media.with_name(media.name + ".info.json")
    info.write_text('{"id":"XXXXXXXXXXX"}')
    assert worker.run_worker(args, tools) == 0
    assert tools.count("yt_dlp") == 3


@pytest.mark.parametrize("option", ["wrong_id", "short_media", "missing_sections", "shifted_section"])
def test_invalid_acquisition_never_reaches_inference(tmp_path, option):
    args, tools = arguments(tmp_path), Tools(**{option: True})
    assert worker.run_worker(args, tools) == 1
    state, = states(args)
    assert state["status"] == "failed"
    assert tools.count("main.py") == 0
    assert not list(args.work_root.glob("media/*/*.mp4"))


def test_failed_row_is_durable_and_other_rows_continue(tmp_path, fake_contract):
    args = arguments(tmp_path, [row(), row("Qh1n6VVKgn0", name="Ulm1")])
    tools = Tools(fail_analysis="ULl8s4qydrk")
    assert worker.run_worker(args, tools) == 1
    records = {state["request"]["video_id"]: state for state in states(args)}
    assert records["ULl8s4qydrk"]["status"] == "failed"
    assert records["Qh1n6VVKgn0"]["status"] == "complete_review_pending"
    assert len(list(args.work_root.glob("media/*/*.mp4"))) == 2
    assert all(state["eligibility_status"] == "unverified" for state in records.values())


def test_exit_zero_does_not_substitute_for_complete_manifest(tmp_path, fake_contract):
    args, tools = arguments(tmp_path), Tools(zero_output=True)
    assert worker.run_worker(args, tools) == 1
    assert tools.count("localize.py") == 0
    assert "completed, source-bound" in states(args)[0]["error"]["message"]


@pytest.mark.parametrize("option", ["change_source", "wrong_location"])
def test_source_change_or_mismatched_location_fails_closed(tmp_path, fake_contract, option):
    args, tools = arguments(tmp_path), Tools(**{option: True})
    assert worker.run_worker(args, tools) == 1
    assert states(args)[0]["status"] == "failed"
    assert len(list(args.work_root.glob("media/*/*.mp4"))) == 1


def test_resume_uses_pipeline_resume_and_same_source_location(tmp_path, fake_contract):
    args, tools = arguments(tmp_path), Tools()
    assert worker.run_worker(args, tools) == worker.run_worker(args, tools) == 0
    assert tools.count("yt_dlp") == tools.count("localize.py") == 1
    assert tools.count("main.py") == 2
    for call in tools.calls:
        assert call["env"]["CUDA_VISIBLE_DEVICES"] == ""
        if any(Path(part).name == "main.py" for part in call["command"]):
            assert "--resume-failed" in call["command"]
            assert Path(call["env"]["PEDX_RESULTS_ROOT"]).is_relative_to(args.work_root)
    assert states(args)[0]["status"] == "complete_review_pending"


def test_pilot_and_full_interval_never_overwrite_each_other(tmp_path):
    args, tools = arguments(tmp_path), Tools()
    args.prepare_only, args.video_id, args.clip_seconds = True, "ULl8s4qydrk", 300
    assert worker.run_worker(args, tools) == 0
    args.clip_seconds = None
    assert worker.run_worker(args, tools) == 0
    assert len(states(args)) == len(list(args.work_root.glob("media/*/*.mp4"))) == 2
    assert {state["request"]["scope"] for state in states(args)} == {"pilot", "full_mapped_interval"}


def test_gpu_waits_for_both_memory_and_utilization(tmp_path, fake_contract):
    args, tools = arguments(tmp_path), Tools()
    args.device = "gpu"
    tools.gpu = ["4000, 0", "9000, 90", "9000, 25"]
    waits = []
    assert worker.run_worker(args, tools, waits.append) == 0
    assert waits == [30, 30]
    first_analysis = next(i for i, call in enumerate(tools.calls) if any(Path(part).name == "main.py" for part in call["command"]))
    assert sum(call["command"][0] == "nvidia-smi" for call in tools.calls[:first_analysis]) == 3


def test_insufficient_disk_stops_before_download(tmp_path, monkeypatch):
    args, tools = arguments(tmp_path), Tools()
    monkeypatch.setattr(worker.shutil, "disk_usage", lambda path: argparse.Namespace(free=1))
    assert worker.run_worker(args, tools) == 1
    assert not tools.calls and "20 GiB" in states(args)[0]["error"]["message"]


def test_os_lock_prevents_concurrent_worker_and_releases(tmp_path):
    path = tmp_path / "batch.lock"
    with worker.batch_lock(path):
        with pytest.raises(RuntimeError, match="Another archive worker"):
            with worker.batch_lock(path):
                pass
    with worker.batch_lock(path):
        pass


def test_interruption_is_durable_and_resume_repairs_batch_summary(tmp_path):
    args, tools = arguments(tmp_path), Tools(interrupt=True)
    args.prepare_only = True
    assert worker.run_worker(args, tools) == 130
    assert states(args)[0]["status"] == "interrupted"
    (args.work_root / "state" / "batch.json").unlink()
    assert worker.run_worker(args, tools) == 0
    batch = json.loads((args.work_root / "state" / "batch.json").read_text())
    assert len(batch["jobs"]) == 1 and next(iter(batch["jobs"].values()))["status"] == "media_ready"


def test_corrupt_state_blocks_only_its_row_and_preserves_evidence(tmp_path, fake_contract):
    args, tools = arguments(tmp_path, [row(), row("Qh1n6VVKgn0", name="Ulm1")]), Tools()
    assert worker.run_worker(args, tools) == 0
    corrupt = next((args.work_root / "state").glob("video-ULl8s4qydrk-*.json"))
    corrupt.write_text("interrupted state write", encoding="utf-8")
    assert worker.run_worker(args, tools) == 1
    assert corrupt.read_text() == "interrupted state write"
    assert tools.count("main.py") == 3


def test_prepare_only_without_explicit_id_selects_at_most_one(tmp_path):
    args = arguments(tmp_path, [row(), row("Qh1n6VVKgn0", name="Ulm1")])
    args.prepare_only = True
    jobs, _ = worker.select_jobs(args)
    assert len(jobs) == 1


def test_per_job_redirect_cannot_write_outside_work_root(tmp_path):
    import os
    args, tools = arguments(tmp_path), Tools()
    args.prepare_only = True
    jobs, _ = worker.select_jobs(args)
    redirect = args.work_root / "media" / jobs[0]["key"]
    redirect.parent.mkdir(parents=True)
    outside = tmp_path / "unrelated-data"
    outside.mkdir()
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(redirect), str(outside)],
                       check=True, capture_output=True)
    else:
        redirect.symlink_to(outside, target_is_directory=True)
    assert worker.run_worker(args, tools) == 1
    assert not tools.calls and not list(outside.iterdir())


def test_actual_manifest_gate_checks_stage_hash_and_clip_bounds(tmp_path, fake_contract):
    folder = tmp_path / "analysis"
    folder.mkdir()
    result = folder / "result.csv"
    result.write_text("observed\n1\n")
    manifest = {"status": "complete", "run_id": "bound", "configuration": {
        "source_sha256": "a" * 64, "clip_start_seconds": 0, "clip_end_seconds": 300},
        "stages": {"fake": {"status": "complete", "outputs": {"result.csv": pipeline.output_record(result)}}}}
    worker.atomic_json(folder / pipeline.MANIFEST_NAME, manifest)
    assert worker.current_analysis(folder, "a" * 64, {"start": 0, "end": 300})["run_id"] == "bound"
    with pytest.raises(RuntimeError):
        worker.current_analysis(folder, "a" * 64, {"start": 0, "end": 301})
    result.write_text("observed\n2\n")
    with pytest.raises(RuntimeError):
        worker.current_analysis(folder, "a" * 64, {"start": 0, "end": 300})


def test_cli_rejects_unbounded_pilot_without_launching(tmp_path):
    with pytest.raises(SystemExit):
        worker.main(["--queue", str(tmp_path / "missing.csv"), "--work-root", str(tmp_path),
                     "--python", sys.executable, "--prepare-only", "--clip-seconds", "300"])


def test_isolated_download_interpreter_and_deadline_leave_cv_unbounded(tmp_path, fake_contract):
    args, tools = arguments(tmp_path), Tools()
    args.download_python = str(tmp_path / "download-env" / "python.exe")
    args.download_timeout_seconds = 123
    assert worker.run_worker(args, tools) == 0
    download = next(call for call in tools.calls if "yt_dlp" in call["command"])
    command = download["command"]
    assert command[0] == args.download_python and download["timeout"] == 123
    for flag in ("--retries", "--fragment-retries", "--extractor-retries", "--file-access-retries"):
        assert command[command.index(flag) + 1] == "3"
    assert command[command.index("--socket-timeout") + 1] == "30"
    assert command[command.index("--downloader-args") + 1] == "ffmpeg_i:-rw_timeout 30000000"
    for call in tools.calls:
        if any(Path(part).name in {"main.py", "localize.py"} for part in call["command"]):
            assert call["command"][0] == args.python and call["timeout"] is None


def test_acquisition_deadline_is_durable_and_resume_can_retry(tmp_path):
    args, tools = arguments(tmp_path), Tools()
    args.prepare_only = True

    def timeout(command, **options):
        if "yt_dlp" in command:
            raise subprocess.TimeoutExpired(command, options["timeout"])
        return tools(command, **options)
    assert worker.run_worker(args, timeout) == 1
    state, = states(args)
    assert state["status"] == "failed" and state["error"]["type"] == "TimeoutExpired"
    assert state["error"]["stage"] == "acquiring"
    assert not list(args.work_root.rglob("cookies.txt"))
    assert worker.run_worker(args, tools) == 0 and states(args)[0]["status"] == "media_ready"


def test_deadline_terminates_only_the_owned_process_tree(tmp_path):
    started, leaked = tmp_path / "child-started.txt", tmp_path / "child-leaked.txt"
    child = ("from pathlib import Path; import time; "
             f"Path({str(started)!r}).write_text('started'); time.sleep(2); "
             f"Path({str(leaked)!r}).write_text('leaked')")
    parent = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(60)"
    began = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        worker.run_command([sys.executable, "-c", parent], cwd=tmp_path, env=dict(worker.os.environ), timeout=0.8)
    assert time.monotonic() - began < 10 and started.exists()
    time.sleep(1.5)
    assert not leaked.exists()


@pytest.mark.parametrize("duration", ["0", "nan", "inf"])
def test_cli_rejects_nonfinite_or_disabled_download_deadline(tmp_path, duration):
    with pytest.raises(SystemExit):
        worker.main(["--queue", str(tmp_path / "missing.csv"), "--work-root", str(tmp_path),
                     "--python", sys.executable, "--prepare-only", "--download-timeout-seconds", duration])
