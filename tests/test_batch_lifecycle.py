"""Offline regressions for clip identity, independent retries, and source retention."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run as batch


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(batch, "RESULTS_ROOT", str(tmp_path / "results"))
    (tmp_path / "videos").mkdir()
    return tmp_path


def queue(root, rows):
    path = root / "queue.csv"
    pd.DataFrame([{"name": "City1", "video": "abc_DEF", "finished": False,
                   "downloaded": False, **row} for row in rows]).to_csv(path, index=False)
    return path


def media(root, name="City1_abc_DEF"):
    path = root / "videos" / (name + ".mp4")
    path.write_bytes(b"fixture")
    return path


def completed_analysis(root, source_bytes=None, run_id="fixture-run"):
    directory = root / "results" / "City1_abc_DEF"
    directory.mkdir(parents=True, exist_ok=True)
    if source_bytes is None:
        source_bytes = (root / "videos" / "City1_abc_DEF.mp4").read_bytes()
    manifest = {"status": "complete", "run_id": run_id,
                "configuration": {"source_sha256": hashlib.sha256(source_bytes).hexdigest()}}
    (directory / "[A0]run_manifest.json").write_text(json.dumps(manifest))
    return manifest


def location(root, status, bound=True):
    directory = root / "results" / "City1_abc_DEF"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((directory / "[A0]run_manifest.json").read_text()) if bound else {}
    row = {"status": status, "lat": 48 if status == "ok" else None,
           "lon": 9 if status == "ok" else None, "confidence_level": "low", "confidence_spread_m": 4000,
           "source_video_sha256": manifest.get("configuration", {}).get("source_sha256"),
           "analysis_run_id": manifest.get("run_id")}
    pd.DataFrame([row]).to_csv(directory / "[L1]localization.csv", index=False)
    pd.DataFrame([row]).to_csv(directory / "[L1]localization_attempt.csv", index=False)


@pytest.mark.parametrize("row,budget,expected", [
    ({}, 0, (0, None)), ({"start_time": 27, "end_time": 3250}, 60, (27, 87)),
    ({"start_time": 27, "end_time": 50}, 60, (27, 50)),
    ({"start_time": 27}, 0, (27, None)), ({"end_time": 30}, 0, (0, 30)),
])
def test_mapped_window_and_budget(row, budget, expected):
    assert batch._window(row, budget) == expected


@pytest.mark.parametrize("row", [{"start_time": -1}, {"end_time": "inf"},
                                   {"start_time": 10, "end_time": 10}])
def test_invalid_mapping_window(row):
    with pytest.raises(ValueError):
        batch._window(row)


def test_download_reconciles_stale_flag_and_records_actual_window(workspace, monkeypatch):
    path = queue(workspace, [{"downloaded": True, "start_time": 27, "end_time": 3250,
                              "city_id": "geonames:123"}])
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        if command[0] == "yt-dlp":
            Path(command[command.index("-o") + 1]).write_bytes(b"downloaded")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(batch.subprocess, "run", execute)
    batch.run(csv_file=path, clip_seconds=60)
    assert calls[0][calls[0].index("--download-sections") + 1] == "*27-87"
    assert calls[1][0] == sys.executable
    out = pd.read_csv(path).iloc[0]
    assert bool(out.finished) and not bool(out.downloaded)
    assert out.city_id == "geonames:123"
    timing = pd.read_csv(workspace / "results/City1_abc_DEF/[A3]analysis_time.csv").iloc[0]
    assert timing.link == "abc_DEF" and timing.clip_start_seconds == 27 and timing.clip_end_seconds == 87
    assert pd.notna(timing.analyzed_at)
    assert not (workspace / "analysis_results").exists()


def test_resume_step_applies_first_pending_row(workspace, monkeypatch):
    path = queue(workspace, [{"finished": True, "name": "Done"}, {}, {"name": "Next"}])
    media(workspace)
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        if command[0] == "yt-dlp":
            Path(command[command.index("-o") + 1]).write_bytes(b"downloaded")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(batch.subprocess, "run", execute)
    batch.run(csv_file=path, start_step=2)
    assert calls[0][1] == "main.py"
    assert calls[1][0] == "yt-dlp"


def test_finished_analysis_retries_location_and_retains_until_valid(workspace, monkeypatch):
    path = queue(workspace, [{"finished": True}])
    source = media(workspace)
    completed_analysis(workspace)
    calls = []
    status = ["no_position"]
    def execute(command, **kwargs):
        calls.append(command)
        assert command[1:3] == ["-m", "modules.localization.localize"]
        location(workspace, status[0])
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(batch.subprocess, "run", execute)
    batch.run(csv_file=path, localize=True)
    assert source.exists()
    assert pd.read_csv(path).iloc[0].localization_status == "no_position"
    status[0] = "ok"
    batch.run(csv_file=path)  # requested stage persists without repeating --localize
    assert len(calls) == 2 and not source.exists()
    assert pd.read_csv(path).iloc[0].localization_status == "ok"


def test_failed_analysis_and_step_three_never_delete_unfinished_source(workspace, monkeypatch):
    path = queue(workspace, [{}])
    source = media(workspace)
    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(batch.subprocess, "run", fail)
    batch.run(csv_file=path)
    assert source.exists() and pd.read_csv(path).iloc[0].analysis_status == "failed"
    batch.run(csv_file=path, start_step=3)
    assert source.exists()


def test_known_wrong_window_is_redownloaded(workspace, monkeypatch):
    path = queue(workspace, [{"start_time": 20, "end_time": 40,
                              "downloaded_start_time": 0, "downloaded_end_time": 10}])
    media(workspace)
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        if command[0] == "yt-dlp":
            Path(command[command.index("-o") + 1]).write_bytes(b"correct window")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(batch.subprocess, "run", execute)
    batch.run(csv_file=path)
    assert calls[0][calls[0].index("--download-sections") + 1] == "*20-40"


def test_deletion_reconciles_downloaded_flag(workspace, monkeypatch):
    path = queue(workspace, [{"finished": True, "downloaded": True}])
    source = media(workspace)
    monkeypatch.setattr(batch.subprocess, "run", lambda *a, **k: pytest.fail("No stages pending"))
    batch.run(csv_file=path, start_step=3)
    assert not source.exists() and not bool(pd.read_csv(path).iloc[0].downloaded)


def test_localization_redownload_keeps_analyzed_window_without_repeating_budget(workspace, monkeypatch):
    path = queue(workspace, [{"finished": True, "start_time": 27, "end_time": 3250,
                              "downloaded_start_time": 27, "downloaded_end_time": 87}])
    completed_analysis(workspace, source_bytes=b"original analyzed clip")
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        if command[0] == "yt-dlp":
            Path(command[command.index("-o") + 1]).write_bytes(b"original analyzed clip")
        else:
            assert command[1:3] == ["-m", "modules.localization.localize"]
            location(workspace, "ok")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(batch.subprocess, "run", execute)
    batch.run(csv_file=path, localize=True)
    assert calls[0][calls[0].index("--download-sections") + 1] == "*27-87"


def test_completed_missing_media_flag_is_reconciled_without_consuming_resume_row(workspace, monkeypatch):
    path = queue(workspace, [{"finished": True, "downloaded": True}])
    monkeypatch.setattr(batch.subprocess, "run", lambda *a, **k: pytest.fail("No stages pending"))
    assert batch.run(csv_file=path) == 0
    assert not bool(pd.read_csv(path).iloc[0].downloaded)


def test_analysis_resume_uses_retained_clip_without_repeating_budget(workspace, monkeypatch):
    path = queue(workspace, [{"start_time": 27, "end_time": 3250,
                              "downloaded_start_time": 27, "downloaded_end_time": 87}])
    media(workspace)
    def execute(command, **kwargs):
        assert command[1] == "main.py"
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(batch.subprocess, "run", execute)
    batch.run(csv_file=path, start_step=2)
    timing = pd.read_csv(workspace / "results/City1_abc_DEF/[A3]analysis_time.csv").iloc[0]
    assert timing.clip_start_seconds == 27 and timing.clip_end_seconds == 87


def test_analysis_bounds_and_timing_are_joined_to_completed_run(workspace, monkeypatch):
    import json
    path = queue(workspace, [{"start_time": 27, "end_time": 3250}])
    def execute(command, **kwargs):
        if command[0] == "yt-dlp":
            Path(command[command.index("-o") + 1]).write_bytes(b"clip")
        else:
            assert kwargs['env']['PEDX_CLIP_START_SECONDS'] == '27'
            assert kwargs['env']['PEDX_CLIP_END_SECONDS'] == '87'
            folder = workspace / 'results' / 'City1_abc_DEF'
            folder.mkdir(parents=True)
            (folder / '[A0]run_manifest.json').write_text(json.dumps({'status': 'complete', 'run_id': 'matching-run'}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(batch.subprocess, "run", execute)
    batch.run(csv_file=path, clip_seconds=60)
    timing = pd.read_csv(workspace / 'results/City1_abc_DEF/[A3]analysis_time.csv').iloc[0]
    assert timing.analysis_run_id == 'matching-run'


def test_changed_source_reanalysis_retries_old_location_and_keeps_failed_media(workspace, monkeypatch):
    path = queue(workspace, [{}])
    source = media(workspace)
    completed_analysis(workspace, run_id="old-run")
    location(workspace, "ok")
    source.write_bytes(b"different analyzed clip")
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        if command[1] == "main.py":
            completed_analysis(workspace, run_id="new-run")
            return SimpleNamespace(returncode=0)
        assert command[1:3] == ["-m", "modules.localization.localize"]
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(batch.subprocess, "run", execute)
    batch.run(csv_file=path, start_step=2, localize=True)
    assert len(calls) == 2 and source.exists()
    assert pd.read_csv(path).iloc[0].localization_status == "source_mismatch"


def test_same_bytes_reanalysis_reuses_bound_low_confidence_estimate(workspace, monkeypatch):
    path = queue(workspace, [{}])
    source = media(workspace)
    completed_analysis(workspace, run_id="old-run")
    location(workspace, "ok")
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        assert command[1] == "main.py"
        completed_analysis(workspace, run_id="new-run")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(batch.subprocess, "run", execute)
    batch.run(csv_file=path, start_step=2, localize=True)
    assert len(calls) == 1 and not source.exists()
    assert pd.read_csv(path).iloc[0].localization_status == "ok"
    raw = pd.read_csv(workspace / "results/City1_abc_DEF/[L1]localization.csv").iloc[0]
    assert raw.analysis_run_id == "old-run" and raw.confidence_level == "low"


def test_legacy_unbound_location_retries_with_current_source_provenance(workspace, monkeypatch):
    path = queue(workspace, [{"finished": True}])
    source = media(workspace)
    completed_analysis(workspace)
    location(workspace, "ok", bound=False)
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        assert command[1:3] == ["-m", "modules.localization.localize"]
        location(workspace, "ok")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(batch.subprocess, "run", execute)
    batch.run(csv_file=path, start_step=2, localize=True)
    assert len(calls) == 1 and not source.exists()


def test_completed_manifest_does_not_validate_reencoded_redownload_for_deletion(workspace, monkeypatch):
    path = queue(workspace, [{"finished": True}])
    source = media(workspace)
    completed_analysis(workspace, source_bytes=b"original encoding")
    location(workspace, "ok")
    def execute(command, **kwargs):
        assert command[1:3] == ["-m", "modules.localization.localize"]
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(batch.subprocess, "run", execute)
    batch.run(csv_file=path, start_step=2, localize=True)
    assert source.exists()
    assert pd.read_csv(path).iloc[0].localization_status == "source_mismatch"
