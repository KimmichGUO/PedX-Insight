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


def selected_answer_fixture():
    """Match the installed 0.1.2 report schema, with disagreeing diagnostics."""
    return {
        'answer': {'mode': 'track', 'method': 'vpr_track_only', 'source': 'mapillary',
                   'latitude': 50, 'longitude': 11, 'route_latlon': [[50, 11], [50.1, 11.2]]},
        'position': {'latitude': 48, 'longitude': 9, 'route_latlon': [[48, 9], [48.2, 9.3]],
                     'street_names': ['Diagnostic street'],
                     'spatial_confidence': {'level': 'high', 'spread_m': 12}},
        'estimated_length_m': 900, 'trajectory_source': 'diagnostic_visual_odometry',
        'provenance': {'package_version': '0.1.2', 'python': '3.12.7',
                       'git': {'commit': 'b' * 40, 'dirty': True}, 'config': {'answer': 'track'}},
    }


def test_selected_answer_supplies_headline_and_never_inherits_diagnostic_fields(tmp_path):
    payload = selected_answer_fixture()
    result_json = tmp_path / 'result.json'
    result_json.write_text(json.dumps(payload), encoding='utf-8')
    result = localize._extract_position(result_json)
    assert (result['lat'], result['lon']) == (50, 11)
    assert result['route_latlon'] == payload['answer']['route_latlon']
    assert result['trajectory_source'] == 'mapillary'
    assert result['route_length_m'] is None
    assert result['confidence_level'] is None and result['confidence_spread_m'] is None
    assert result['street_names'] is None and result['hypotheses'] == []
    assert result['localization_result_source'] == 'answer'
    assert result['localization_answer_method'] == 'vpr_track_only'
    assert result['localization_tool_version'] == '0.1.2'
    assert result['localization_tool_git_sha'] == 'b' * 40
    assert result['localization_tool_git_dirty'] is True
    assert result['localization_python_version'] == '3.12.7'


@pytest.mark.parametrize('answer', [None, {}, [], 'invalid',
    {'latitude': 50, 'longitude': 11, 'status': 'failed'},
    {'latitude': 50, 'longitude': 11, 'status': 'pending'},
    {'latitude': 50, 'longitude': 11, 'success': False},
    {'latitude': 50, 'longitude': 'nan'}, {'latitude': 95, 'longitude': 11}])
def test_unsuccessful_selected_answer_never_falls_back_to_valid_diagnostic(setup_localizer, monkeypatch, answer):
    video, out = setup_localizer
    previous = out.read_bytes()
    payload = selected_answer_fixture()
    payload['answer'] = answer

    def execute(command, **kwargs):
        result = Path(command[command.index('--output-dir') + 1]) / 'result.json'
        result.write_text(json.dumps(payload), encoding='utf-8')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(localize.subprocess, 'run', execute)
    with pytest.raises(RuntimeError, match='without a usable position'):
        localize.run_localization(str(video), city='City, Country')
    assert out.read_bytes() == previous
    attempt = read(out.with_name('[L1]localization_attempt.csv'))
    assert attempt['status'] == 'no_position' and attempt['lat'] == '' and attempt['lon'] == ''
    assert attempt['localization_result_source'] == 'answer'
    assert attempt['localization_tool_version'] == '0.1.2'
    assert 'Selected localization answer' in attempt['error']


@pytest.mark.parametrize('schema_marker', ['version', 'config'])
def test_modern_result_without_answer_does_not_promote_diagnostic(tmp_path, schema_marker):
    payload = selected_answer_fixture()
    del payload['answer']
    payload['provenance'].pop('package_version' if schema_marker == 'config' else 'config')
    result_json = tmp_path / 'result.json'
    result_json.write_text(json.dumps(payload), encoding='utf-8')
    result = localize._extract_position(result_json)
    assert result['lat'] is None and result['lon'] is None
    assert result['localization_result_source'] == 'answer' and result['selection_error']


def test_legacy_position_remains_supported_with_its_original_metadata(tmp_path):
    payload = selected_answer_fixture()
    del payload['answer']
    payload['provenance'] = {'package_version': '0.1.1'}
    payload['position']['hypotheses'] = [{'street_names': ['Legacy street']}]
    payload['position'].pop('street_names')
    result_json = tmp_path / 'result.json'
    result_json.write_text(json.dumps(payload), encoding='utf-8')
    result = localize._extract_position(result_json)
    assert (result['lat'], result['lon']) == (48, 9)
    assert result['route_latlon'] == payload['position']['route_latlon']
    assert result['street_names'] == 'Legacy street'
    assert result['confidence_level'] == 'high' and result['confidence_spread_m'] == 12
    assert result['route_length_m'] == 900
    assert result['trajectory_source'] == 'diagnostic_visual_odometry'
    assert result['localization_result_source'] == 'position'


def test_successful_selected_answer_persists_provenance_in_result_and_attempt(setup_localizer, monkeypatch):
    video, out = setup_localizer
    payload = selected_answer_fixture()
    payload['answer'].update(experimental=True, confidence_calibrated=False)

    def execute(command, **kwargs):
        result = Path(command[command.index('--output-dir') + 1]) / 'result.json'
        result.write_text(json.dumps(payload), encoding='utf-8')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(localize.subprocess, 'run', execute)
    localize.run_localization(str(video), city='City, Country')
    for path in [out, out.with_name('[L1]localization_attempt.csv')]:
        row = read(path)
        assert (row['lat'], row['lon']) == ('50', '11')
        assert row['localization_result_source'] == 'answer'
        assert row['localization_tool_version'] == '0.1.2'
        assert row['localization_tool_git_sha'] == 'b' * 40
        assert row['localization_python_version'] == '3.12.7'
        assert row['localization_experimental'] == 'True'
        assert row['localization_confidence_calibrated'] == 'False'
        assert row['confidence_level'] == '' and row['route_length_m'] == ''
