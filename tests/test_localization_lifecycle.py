"""No companion package/model/network is loaded: subprocesses write tiny result fixtures."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.localization import localize
from modules.pipeline import atomic_json, file_hash, MANIFEST_NAME
from get_all_video_locations import bind_localization_to_analysis, main as aggregate_locations


@pytest.fixture
def setup_localizer(tmp_path, monkeypatch):
    monkeypatch.setattr(localize, "RESULTS_ROOT", str(tmp_path / "results"))
    monkeypatch.setattr(localize, "OSM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(localize, "_resolve_osm_python", lambda value: "fixture-python")
    video = tmp_path / "City1_abc_DEF.mp4"
    video.write_bytes(b"fixture")
    out = tmp_path / "results/City1_abc_DEF/[L1]localization.csv"
    out.parent.mkdir(parents=True)
    out.write_text("status,lat,lon,candidates\nok,48,9,previous\n", encoding="utf-8")
    return video, out


def read(path):
    with path.open(encoding="utf-8") as f:
        return next(csv.DictReader(f))


@pytest.mark.parametrize("failure", ["environment", "city", "video", "process", "timeout",
                                      "no_result", "invalid_json", "no_position", "invalid_position"])
def test_failed_retry_preserves_good_result_and_records_attempt(setup_localizer, monkeypatch, failure):
    video, out = setup_localizer
    original = out.read_bytes()
    city = "City, Country"
    if failure == "environment":
        monkeypatch.setattr(localize, "_resolve_osm_python", lambda value: None)
    if failure == "city":
        city = None
        monkeypatch.setattr(localize, "_infer_city_from_mapping", lambda *a: None)
    if failure == "video":
        video.unlink()
    def execute(command, **kwargs):
        if failure == "process":
            raise subprocess.CalledProcessError(1, command, output="fixture failure")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 1, output=b"partial output")
        if failure != "no_result":
            result = Path(command[command.index("--output-dir") + 1]) / "result.json"
            if failure == "invalid_json":
                result.write_text("{invalid", encoding="utf-8")
            else:
                result.write_text(json.dumps({"position": {"latitude": 95} if failure == "invalid_position" else {}}), encoding="utf-8")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(localize.subprocess, "run", execute)
    with pytest.raises((OSError, RuntimeError, ValueError, subprocess.SubprocessError)):
        localize.run_localization(str(video), city=city)
    assert out.read_bytes() == original
    attempt = read(out.with_name("[L1]localization_attempt.csv"))
    assert attempt["status"] != "ok" and attempt["attempted_utc"]


def test_success_replaces_prior_and_marks_attempt(setup_localizer, monkeypatch):
    video, out = setup_localizer
    atomic_json(out.parent / MANIFEST_NAME, {'status': 'complete', 'run_id': 'original-run',
        'configuration': {'source_sha256': file_hash(video)}})
    def execute(command, **kwargs):
        result = Path(command[command.index("--output-dir") + 1]) / "result.json"
        result.write_text(json.dumps({"position": {"latitude": 49, "longitude": 10}}), encoding="utf-8")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(localize.subprocess, "run", execute)
    localize.run_localization(str(video), city="City, Country")
    assert read(out)["lat"] == "49"
    assert read(out)["source_video_sha256"] == file_hash(video)
    assert read(out)["analysis_run_id"] == "original-run"
    assert read(out)["localized_at"]
    assert read(out.with_name("[L1]localization_attempt.csv"))["status"] == "ok"


@pytest.mark.parametrize("lat,lon", [("nan", 1), (0, "inf"), (91, 0), (0, 181), (None, 0)])
def test_invalid_coordinates_are_not_success(lat, lon):
    assert not localize.localization_succeeded({"status": "ok", "lat": lat, "lon": lon})


def good_bound_row(video, run_id='old-run'):
    return {'status': 'ok', 'lat': 48, 'lon': 9, 'confidence_level': 'high',
        'confidence_spread_m': 10, 'street_names': 'Synthetic street',
        'candidates': '[{"lat":48,"lon":9}]', 'route_latlon': '[[48,9],[48.1,9.1]]',
        'route_length_m': 100, 'trajectory_source': 'synthetic',
        'source_video_sha256': file_hash(video), 'analysis_run_id': run_id,
        'localized_at': '2026-10-27T12:00:00Z'}


def test_changed_clip_failed_retry_retains_raw_but_aggregate_clears_location(setup_localizer, monkeypatch, tmp_path):
    video, out = setup_localizer
    old = good_bound_row(video)
    with out.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(old))
        writer.writeheader()
        writer.writerow(old)
    prior = out.read_bytes()
    video.write_bytes(b'a different downloaded interval of the same YouTube video')
    atomic_json(out.parent / MANIFEST_NAME, {'status': 'complete', 'run_id': 'new-clip-run',
        'configuration': {'source_sha256': file_hash(video)}})
    monkeypatch.setattr(localize, '_resolve_osm_python', lambda _: None)
    with pytest.raises(RuntimeError):
        localize.run_localization(str(video), city='City, Country')
    assert out.read_bytes() == prior, 'retain the older expensive result for audit'
    attempt = read(out.with_name('[L1]localization_attempt.csv'))
    assert attempt['source_video_sha256'] == file_hash(video)
    assert attempt['analysis_run_id'] == 'new-clip-run'
    aggregate_locations(results_root=out.parent.parent, mapping_path=tmp_path / 'absent.csv', output_root=tmp_path / 'summary')
    aggregate = read(tmp_path / 'summary/all_video_locations.csv')
    assert aggregate['status'] == 'source_mismatch'
    assert aggregate['analysis_run_id'] == 'new-clip-run'
    assert aggregate['localization_origin_run_id'] == 'old-run'
    for name in ('lat', 'lon', 'street_names', 'confidence_level', 'confidence_spread_m',
                 'candidates', 'route_latlon', 'route_length_m', 'trajectory_source'):
        assert aggregate[name] == '', f'{name} leaked from another source clip'
    assert out.read_bytes() == prior


def test_same_source_new_analysis_run_reuses_bound_coordinates(setup_localizer, tmp_path):
    video, out = setup_localizer
    row = good_bound_row(video)
    atomic_json(out.parent / MANIFEST_NAME, {'status': 'complete', 'run_id': 'new-code-run',
        'configuration': {'source_sha256': file_hash(video)}})
    result = bind_localization_to_analysis(row, out.parent)
    assert result['status'] == 'ok' and result['lat'] == 48
    assert result['source_video_sha256'] == result['analysis_source_video_sha256'] == file_hash(video)
    assert result['analysis_run_id'] == 'new-code-run'
    assert result['localization_origin_run_id'] == 'old-run'
    assert result['localized_at'] == row['localized_at']
    with out.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    aggregate_locations(results_root=out.parent.parent, mapping_path=tmp_path / 'absent.csv', output_root=tmp_path / 'summary')
    aggregate = read(tmp_path / 'summary/all_video_locations.csv')
    assert aggregate['status'] == 'ok' and float(aggregate['lat']) == 48
    assert aggregate['analysis_run_id'] == 'new-code-run'
    assert aggregate['localization_origin_run_id'] == 'old-run'
    assert aggregate['source_video_sha256'] == file_hash(video)


@pytest.mark.parametrize('case', ['legacy-result', 'legacy-analysis', 'incomplete-analysis', 'malformed-analysis'])
def test_unknown_source_binding_fails_closed_without_modifying_raw(setup_localizer, case):
    video, out = setup_localizer
    row = good_bound_row(video)
    if case != 'legacy-analysis':
        atomic_json(out.parent / MANIFEST_NAME, {'status': 'running' if case == 'incomplete-analysis' else 'complete',
            'run_id': 'current-run', 'configuration': None if case == 'malformed-analysis' else {'source_sha256': file_hash(video)}})
    if case == 'legacy-result':
        row.pop('source_video_sha256')
    original = out.read_bytes()
    result = bind_localization_to_analysis(row, out.parent)
    assert result['status'] == 'source_unverified'
    assert result['lat'] is None and result['lon'] is None and result['route_latlon'] is None
    assert out.read_bytes() == original


def test_mutating_source_during_localization_does_not_replace_prior_result(setup_localizer, monkeypatch):
    video, out = setup_localizer
    original = out.read_bytes()

    def execute(command, **kwargs):
        result = Path(command[command.index('--output-dir') + 1]) / 'result.json'
        result.write_text(json.dumps({'position': {'latitude': 49, 'longitude': 10}}), encoding='utf-8')
        video.write_bytes(b'changed during the external process')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(localize.subprocess, 'run', execute)
    with pytest.raises(RuntimeError, match='Source video changed'):
        localize.run_localization(str(video), city='City, Country')
    assert out.read_bytes() == original
    assert read(out.with_name('[L1]localization_attempt.csv'))['status'] == 'source_changed'


def test_rapid_retry_cannot_rebind_previous_tool_result_to_changed_clip(setup_localizer, monkeypatch):
    """A zero-exit retry without output must not consume a recent older result."""
    video, out = setup_localizer
    output_directories = []

    def execute(command, **kwargs):
        directory = Path(command[command.index('--output-dir') + 1])
        output_directories.append(directory)
        if len(output_directories) == 1:
            result = directory / 'result.json'
            result.write_text(json.dumps({'position': {'latitude': 49, 'longitude': 10}}), encoding='utf-8')
            # Keep it inside the old shared-tree freshness gate even on slow CI.
            recent = result.stat().st_mtime + 60
            os.utime(result, (recent, recent))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(localize.subprocess, 'run', execute)
    localize.run_localization(str(video), city='City, Country')
    previous = out.read_bytes()
    video.write_bytes(b'different clip using the same video name')

    with pytest.raises(RuntimeError, match='no fresh result.json'):
        localize.run_localization(str(video), city='City, Country')

    assert output_directories[0] != output_directories[1]
    assert out.read_bytes() == previous
    attempt = read(out.with_name('[L1]localization_attempt.csv'))
    assert attempt['status'] == 'no_result_json'
    assert attempt['source_video_sha256'] == file_hash(video)
