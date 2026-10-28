"""Offline pipeline/release contracts, including the Visualizer's pre-DB validator.

No detector, video decoder, network, model fit, or database is used. Every generated
artifact lives under pytest's temporary directory rather than the published tree.
"""

import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from modules import pipeline, quality  # noqa: E402
from scripts.build_release import build_release  # noqa: E402


def write_csv(path, columns, rows=()):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def toy_pipeline(tmp_path, monkeypatch):
    stages = {
        'seed': ((), ('seed.csv',)),
        'child': (('seed',), ('child.csv',)),
        'independent': ((), ('independent.csv',)),
        'summary': (('child', 'independent'), ('summary.csv',)),
    }
    monkeypatch.setattr(pipeline, 'STAGES', stages)
    monkeypatch.setattr(pipeline, 'PROFILES', {'toy': tuple(stages)})
    monkeypatch.setattr(pipeline, 'git_state', lambda: ('synthetic-commit', False))
    import modules.provenance as provenance
    monkeypatch.setattr(provenance, 'weights_manifest', lambda: {})
    repo = tmp_path / 'source-code'
    repo.mkdir()
    (repo / 'main.py').write_text('# offline runner fixture\n', encoding='utf-8')
    video = tmp_path / 'Toy1_abcdefghijk.mp4'
    video.write_bytes(b'synthetic source identity, never decoded')
    results = tmp_path / 'results'
    calls = []

    def runner(command, *, cwd, env):
        mode = command[command.index('--mode') + 1]
        calls.append(mode)
        assert Path(cwd) == repo
        assert env['PEDX_PIPELINE_RUN_ID']
        assert Path(env['PEDX_RESULTS_ROOT']) == results
        assert env['PEDX_SOURCE_SHA256'] == pipeline.file_hash(video)
        for name in stages[mode][1]:
            write_csv(results / video.stem / name, ['value'], [{'value': len(calls)}])
        return SimpleNamespace(returncode=0)

    def execute(**kwargs):
        return pipeline.execute_pipeline(video, 'toy', runner=runner,
                                         results_root=results, repo=repo, **kwargs)

    return SimpleNamespace(stages=stages, repo=repo, video=video, results=results,
                           folder=results / video.stem, calls=calls, runner=runner, execute=execute)


def test_real_dag_places_every_dependency_before_consumer():
    plan = pipeline.build_plan('single_all')
    assert len(plan) == len(set(plan)) == 39
    for mode in plan:
        assert all(plan.index(dependency) < plan.index(mode) for dependency in pipeline.STAGES[mode][0])
    assert plan.index('vehicle_speed') < plan.index('pet') < plan.index('sum_video')


def test_failed_stage_blocks_dependents_but_runs_independent_branch(toy_pipeline):
    fixture = toy_pipeline

    def runner(command, **kwargs):
        mode = command[command.index('--mode') + 1]
        if mode == 'seed':
            fixture.calls.append(mode)
            return SimpleNamespace(returncode=7)
        return fixture.runner(command, **kwargs)

    manifest = pipeline.execute_pipeline(fixture.video, 'toy', runner=runner,
        results_root=fixture.results, repo=fixture.repo)
    assert manifest['status'] == 'failed'
    assert fixture.calls == ['seed', 'independent']
    assert manifest['stages']['seed']['status'] == 'failed'
    assert manifest['stages']['child']['dependencies'] == ['seed']
    assert manifest['stages']['summary']['status'] == 'blocked'
    assert manifest['stages']['independent']['status'] == 'complete'


def test_resume_requires_output_hash_and_source_configuration(toy_pipeline):
    fixture = toy_pipeline
    first = fixture.execute()
    assert first['status'] == 'complete'
    assert fixture.calls == list(fixture.stages)
    fixture.calls.clear()
    reused = fixture.execute(resume=True)
    assert fixture.calls == []
    assert all(stage['reused'] for stage in reused['stages'].values())
    write_csv(fixture.folder / 'child.csv', ['value'], [{'value': 'tampered'}])
    fixture.execute(resume=True)
    assert fixture.calls == ['child', 'summary']
    fixture.calls.clear()
    fixture.video.write_bytes(b'changed source bytes')
    fixture.execute(resume=True)
    assert fixture.calls == list(fixture.stages)
    fixture.calls.clear()
    fixture.execute(resume=True, interval=2.0)
    assert fixture.calls == list(fixture.stages)
    fixture.calls.clear()
    (fixture.repo / 'main.py').write_text('# changed implementation\n', encoding='utf-8')
    fixture.execute(resume=True, interval=2.0)
    assert fixture.calls == list(fixture.stages)
    fixture.calls.clear()
    fixture.execute(resume=True, interval=2.0, weights='different.pt')
    assert fixture.calls == list(fixture.stages)


def test_zero_exit_does_not_reuse_archived_output(toy_pipeline):
    fixture = toy_pipeline
    fixture.folder.mkdir(parents=True)
    write_csv(fixture.folder / 'seed.csv', ['value'], [{'value': 'old result'}])
    manifest = pipeline.execute_pipeline(fixture.video, 'toy',
        runner=lambda *a, **kw: SimpleNamespace(returncode=0),
        results_root=fixture.results, repo=fixture.repo)
    assert manifest['status'] == 'failed'
    assert manifest['stages']['seed']['status'] == 'failed'
    assert not (fixture.folder / 'seed.csv').exists()
    archived = fixture.folder / '.previous' / manifest['run_id'] / 'seed.csv'
    assert 'old result' in archived.read_text(encoding='utf-8')
    assert manifest['stages']['child']['status'] == 'blocked'


@pytest.mark.parametrize('name,content', [
    ('empty.csv', ''), ('[B1]tracked_pedestrians.csv', 'track_id\n1\n'),
    ('[A1]video_info.csv', 'wrong,value\nx,1\n'),
    ('[A2]pedestrian_info.csv', 'track_id\n1\n'),
])
def test_outputs_require_real_schema_headers(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding='utf-8')
    with pytest.raises(ValueError):
        pipeline.output_record(path)


def test_invalid_metrics_never_enter_experimental_comparisons():
    rows = pd.DataFrame([
        {'city': 'London', 'link': 'abcdefghijk', 'eligible_for_comparison': True,
         'weather': 'rain', 'age': 40, 'risky_crossing': 1, 'phone_using': 0,
         'metric_validity': json.dumps({'weather': {'status': 'invalid'}, 'age': {'status': 'invalid'},
            'risky_crossing': {'status': 'experimental'}, 'phone_using': {'status': 'validated'}})},
        {'city': 'London', 'link': 'abcdefghijk', 'eligible_for_comparison': False,
         'weather': 'dry', 'age': 22, 'risky_crossing': 0, 'phone_using': 1,
         'metric_validity': json.dumps({'phone_using': {'status': 'validated'}})},
    ])
    strict = quality.apply_metric_policy(rows)
    exploratory = quality.apply_metric_policy(rows, include_experimental=True)
    assert strict['weather'].isna().all() and exploratory['weather'].isna().all()
    assert strict['age'].isna().all() and exploratory['age'].isna().all()
    assert pd.isna(strict.loc[0, 'risky_crossing']) and exploratory.loc[0, 'risky_crossing'] == 1
    assert exploratory.loc[0, 'phone_using'] == 0
    assert exploratory.loc[1, ['weather', 'age', 'risky_crossing', 'phone_using']].isna().all()


def test_city_identity_unicode_and_unknown_filming_date(tmp_path):
    assert quality.stable_city_id(' São-Paulo ', 'BR') == 'place:br:unknown:sao-paulo'
    assert quality.stable_city_id('London', 'GB') != quality.stable_city_id('London', 'CA')
    assert quality.stable_city_id('Portland', 'US', 'ME') != quality.stable_city_id('Portland', 'US', 'OR')
    assert quality.stable_city_id('London', geonameid=2643743) == 'geonames:2643743'
    folder = tmp_path / 'London1_abcdefghijk'
    folder.mkdir()
    metadata = quality.video_metadata(folder, 'London', 'abcdefghijk',
        {'published_at': '2020-01-01T00:00:00Z', 'discovered_at': '2026-10-27T12:00:00Z'})
    assert metadata['recorded_at'] is None and metadata['recording_date_source'] == 'unknown'
    assert metadata['analyzed_at'] is None
    assert metadata['eligible_for_comparison'] is False


SCHEMAS = {
    '[B0]video_meta.csv': ['fps', 'total_frames', 'width', 'height'],
    '[B1]tracked_pedestrians.csv': ['track_id', 'frame_id', 'timestamp', 'x1', 'y1', 'x2', 'y2'],
    '[B2]dense_tracks.csv': ['track_id', 'frame_id', 'timestamp', 'x1', 'y1', 'x2', 'y2'],
    '[C3]crossing_judge.csv': ['track_id', 'crossed', 'started_frame', 'ended_frame'],
    '[E1]weather.csv': ['frame_id', 'weather_label'],
    '[E2]traffic_light.csv': ['frame_id', 'main_light_color'],
    '[E7]crosswalk_detection.csv': ['frame_id', 'crosswalk_detected', 'crosswalk_boxes'],
    '[V6]vehicle_count.csv': ['Vehicle_Type', 'Count'],
}


def complete_results_tree(root, *, legacy=False):
    """Produce schema-valid CSV artifacts only; the synthetic source is never decoded."""
    from get_all_pede_info import PEDESTRIAN_COLUMNS
    link = 'abcdefghijk'
    results = root / 'results'
    folder = results / f'London1_{link}'
    folder.mkdir(parents=True)
    for _, outputs in pipeline.STAGES.values():
        for name in outputs:
            if name == 'pedestrian_img':
                (folder / name).mkdir()
            else:
                write_csv(folder / name, SCHEMAS.get(name, ['frame_id', 'value']))
    write_csv(folder / '[B0]video_meta.csv', SCHEMAS['[B0]video_meta.csv'],
              [{'fps': 1, 'total_frames': 4, 'width': 4, 'height': 4}])
    write_csv(folder / '[V6]vehicle_count.csv', SCHEMAS['[V6]vehicle_count.csv'],
              [{'Vehicle_Type': 'car', 'Count': 2}, {'Vehicle_Type': 'Total', 'Count': 2}])
    metrics = {'video_name': folder.name, 'duration_seconds': 4, 'duration_source': 'video_metadata',
        'total_frames': 4, 'total_pedestrians': 1, 'total_crossed_pedestrians': 1,
        'measured_crossing_speed_n': 1, 'pipeline_version': 'synthetic-fixture',
        'average_age': 30, 'risky_crossing_ratio': 1, 'risky_crossing_coverage': 1,
        'phone_usage_ratio': 0, 'run_red_light_ratio': 0, 'crosswalk_usage_ratio': 1,
        'total_vehicles': 2, 'top3_vehicles': 'car', 'main_weather': 'rain'}
    write_csv(folder / '[A1]video_info.csv', ['metric', 'value'],
              [{'metric': key, 'value': value} for key, value in metrics.items()])
    person = {key: None for key in PEDESTRIAN_COLUMNS}
    person.update(track_id=1, crossed=True, risky_crossing=1, run_red_light=0,
                  crosswalk_use_or_not=1, phone_using=0, age=30, gender='unknown',
                  weather='rain', daytime=0, walking_speed_mps=1.2, crossing_speed_mps=1.1)
    write_csv(folder / '[A2]pedestrian_info.csv', PEDESTRIAN_COLUMNS, [person])
    write_csv(folder / '[A3]analysis_time.csv', ['link', 'video_name', 'duration_seconds',
        'analysis_seconds', 'clip_start_seconds', 'clip_end_seconds', 'analysis_run_id'],
        [{'link': link, 'video_name': folder.name, 'duration_seconds': 4,
          'analysis_seconds': 12, 'clip_start_seconds': 10, 'clip_end_seconds': 14,
          'analysis_run_id': 'synthetic-full-run'}])
    write_csv(folder / '[L1]localization.csv', ['video_name', 'city', 'lat', 'lon', 'status',
        'confidence_level', 'confidence_spread_m', 'source', 'source_video_sha256', 'analysis_run_id', 'localized_at'],
        [{'video_name': folder.name, 'city': 'London, United Kingdom', 'lat': 51.5, 'lon': -0.1,
          'status': 'ok', 'confidence_level': 'low', 'confidence_spread_m': 2000, 'source': 'synthetic_fixture',
          'source_video_sha256': 'a' * 64, 'analysis_run_id': 'synthetic-location-origin',
          'localized_at': '2026-10-26T11:00:00Z'}])
    if not legacy:
        stages = {mode: {'status': 'complete', 'outputs': {name: pipeline.output_record(folder / name)
                  for name in outputs}} for mode, (_, outputs) in pipeline.STAGES.items()}
        pipeline.atomic_json(folder / pipeline.MANIFEST_NAME, {'schema_version': 1,
            'run_id': 'synthetic-full-run', 'status': 'complete', 'profile': 'single_all',
            'configuration': {'source_sha256': 'a' * 64},
            'started_at': '2026-10-26T12:00:00Z', 'finished_at': '2026-10-26T12:00:12Z', 'stages': stages})
    mapping = root / 'mapping.csv'
    evidence = {'city_id': 'geonames:2643743', 'searched_city_id': 'geonames:2643743',
        'verified_city_id': 'geonames:2643743', 'eligibility_status': 'verified',
        'location_verification_source': 'human:synthetic-review', 'discovered_at': '2026-10-25T12:00:00Z',
        'discovery_run_id': 'synthetic-discovery', 'recording_date_source': 'unknown'}
    row = {'city': 'London', 'country': 'United Kingdom', 'country_code': 'GB', 'iso3': 'GBR',
        'continent': 'Europe', 'lat': 51.5, 'lon': -0.1, 'videos': f'[{link}]',
        'city_id': 'geonames:2643743', 'video_provenance': json.dumps({link: evidence})}
    write_csv(mapping, list(row), [row])
    return SimpleNamespace(results=results, folder=folder, mapping=mapping, link=link)


def test_complete_release_passes_visualizer_validator_without_database(tmp_path, monkeypatch):
    fixture = complete_results_tree(tmp_path)
    monkeypatch.chdir(tmp_path)  # External city-reference CSVs are intentionally absent.
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    output = tmp_path / 'release'
    manifest = build_release(output, results_root=fixture.results,
                             mapping_path=fixture.mapping, skip_stats=True)
    assert manifest['n_videos'] == manifest['n_pedestrians'] == manifest['n_eligible_videos'] == 1
    assert manifest['legacy_audit_only'] is False
    inputs = json.loads((output / 'analysis_runs.json').read_text(encoding='utf-8'))
    assert inputs['runs'][0]['configuration']['source_sha256'] == 'a' * 64
    videos = pd.read_csv(output / 'all_video_info.csv')
    people = pd.read_csv(output / 'all_pedestrian_info.csv')
    assert videos.loc[0, 'city_id'] == 'geonames:2643743'
    assert videos.loc[0, 'analysis_run_id'] == people.loc[0, 'analysis_run_id'] == 'synthetic-full-run'
    assert str(videos.loc[0, 'analyzed_at']).startswith('2026-10-26T12:00:12')
    assert pd.isna(videos.loc[0, 'recorded_at']) and videos.loc[0, 'recording_date_source'] == 'unknown'
    assert videos.loc[0, 'clip_start_seconds'] == 10 and videos.loc[0, 'clip_end_seconds'] == 14
    assert json.loads(videos.loc[0, 'metric_validity'])['weather']['status'] == 'invalid'
    locations = pd.read_csv(output / 'all_video_locations.csv')
    assert locations.loc[0, 'status'] == 'ok'
    assert locations.loc[0, 'source_video_sha256'] == 'a' * 64
    assert locations.loc[0, 'analysis_run_id'] == 'synthetic-full-run'
    assert locations.loc[0, 'localization_origin_run_id'] == 'synthetic-location-origin'
    visualizer = REPO.parent / 'PedX-Visualizer'
    node = shutil.which('node')
    if not node or not (visualizer / 'scripts' / 'publish-release.js').is_file():
        pytest.skip('Sibling Visualizer and Node are needed for cross-repository validation')
    result = subprocess.run([node, str(visualizer / 'scripts' / 'publish-release.js'),
        '--csv-dir', str(output), '--validate-only'], capture_output=True, text=True,
        encoding='utf-8', timeout=30, env=dict(os.environ, DATABASE_URL='invalid://must-not-connect'))
    assert result.returncode == 0, result.stdout + result.stderr
    validated = json.loads(result.stdout.strip().splitlines()[-1])
    assert validated['validated'] is True and validated['dataset_release_id'] == manifest['dataset_release_id']
    assert validated['videos'] == validated['pedestrians'] == 1


def test_legacy_only_release_fails_without_activation_manifest(tmp_path, monkeypatch):
    fixture = complete_results_tree(tmp_path, legacy=True)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / 'release'
    with pytest.raises(ValueError, match='No completed videos'):
        build_release(output, results_root=fixture.results, mapping_path=fixture.mapping, skip_stats=True)
    assert (output / '.failed.json').is_file()
    assert not (output / 'release_manifest.json').exists()


def test_malformed_policy_and_missing_admin_fail_closed(tmp_path):
    import pandas as pd
    from modules.quality import apply_metric_policy, stable_city_id, metric_validity
    assert stable_city_id('London', 'GB', pd.NA) == 'place:gb:unknown:london'
    frame = pd.DataFrame([{'eligible_for_comparison': True, 'metric_validity': raw,
                           'risky_crossing': 1} for raw in ('[]', '{"risky_crossing":"validated"}')])
    assert apply_metric_policy(frame)['risky_crossing'].isna().all()
    assert metric_validity(tmp_path)['avg_age']['status'] == 'invalid'


def test_manual_tracking_rerun_cannot_reuse_old_person_crops(tmp_path, monkeypatch):
    monkeypatch.delenv('PEDX_PIPELINE_RUN_ID', raising=False)
    monkeypatch.setattr(pipeline, 'RESULTS_ROOT', str(tmp_path))
    folder = tmp_path / 'clip'
    stale = folder / 'pedestrian_img' / '1' / 'old.png'
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b'old tracking identity')
    with pipeline.standalone_run('clip.mp4', 'id_img'):
        assert not (folder / 'pedestrian_img').exists()
    archived = list((folder / '.previous').rglob('old.png'))
    assert len(archived) == 1 and archived[0].read_bytes() == b'old tracking identity'


def test_independent_localization_does_not_invalidate_analysis(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'RESULTS_ROOT', str(tmp_path))
    folder = tmp_path / 'City1_video'
    pipeline.atomic_json(folder / pipeline.MANIFEST_NAME, {'status': 'complete', 'run_id': 'valid'})
    with pipeline.standalone_run(tmp_path / 'City1_video.mp4', 'localize'):
        pass
    assert pipeline.read_manifest(folder)['status'] == 'complete'


def test_exclusive_lock_releases_after_error_and_recovers_stale_pid(tmp_path):
    folder = tmp_path / 'locked'
    folder.mkdir()
    (folder / pipeline.OWNER_NAME).write_text(json.dumps({'pid': 99999999, 'run_id': 'crashed', 'status': 'active'}))
    with pytest.raises(ValueError, match='fixture abort'):
        with pipeline.video_lock(folder, 'first') as owner:
            assert owner['recovered_owner']['pid'] == 99999999
            with pytest.raises(RuntimeError, match='already owned'):
                with pipeline.video_lock(folder, 'concurrent'):
                    pytest.fail('Concurrent owner acquired the video')
            raise ValueError('fixture abort')
    with pipeline.video_lock(folder, 'after-abort'):
        assert json.loads((folder / pipeline.OWNER_NAME).read_text())['run_id'] == 'after-abort'
    assert json.loads((folder / pipeline.OWNER_NAME).read_text())['status'] == 'released'


def test_concurrent_pipeline_cannot_certify_another_configuration(toy_pipeline):
    fixture = toy_pipeline
    def runner(command, **kwargs):
        with pytest.raises(RuntimeError, match='already owned'):
            pipeline.execute_pipeline(fixture.video, 'toy', interval=2.0,
                runner=fixture.runner, results_root=fixture.results, repo=fixture.repo)
        return fixture.runner(command, **kwargs)
    result = pipeline.execute_pipeline(fixture.video, 'toy', runner=runner,
        results_root=fixture.results, repo=fixture.repo)
    assert result['status'] == 'complete' and result['configuration']['analysis_interval'] == 1


def test_pipeline_child_requires_active_run_ownership(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'RESULTS_ROOT', str(tmp_path))
    monkeypatch.setenv('PEDX_PIPELINE_RUN_ID', 'child-owner')
    folder = tmp_path / 'City1_video'
    pipeline.atomic_json(folder / pipeline.MANIFEST_NAME, {'status': 'running', 'run_id': 'child-owner'})
    with pytest.raises(RuntimeError, match='does not own'):
        pipeline.invalidate_standalone_run('City1_video.mp4', 'weather')
    with pipeline.video_lock(folder, 'child-owner'):
        pipeline.invalidate_standalone_run('City1_video.mp4', 'weather')
        monkeypatch.setenv('PEDX_PIPELINE_RUN_ID', 'other-owner')
        with pytest.raises(RuntimeError, match='does not own'):
            pipeline.invalidate_standalone_run('City1_video.mp4', 'weather')


def test_orphan_child_lease_blocks_new_owner_until_child_exits(tmp_path, monkeypatch):
    folder = tmp_path / 'City1_video'
    monkeypatch.setattr(pipeline, 'RESULTS_ROOT', str(tmp_path))
    monkeypatch.setenv('PEDX_PIPELINE_RUN_ID', 'orphan')
    pipeline.atomic_json(folder / pipeline.MANIFEST_NAME, {'status': 'running', 'run_id': 'orphan'})
    pipeline.atomic_json(folder / pipeline.OWNER_NAME, {'status': 'active', 'run_id': 'orphan', 'pid': 9999999})
    with pipeline.standalone_run('City1_video.mp4', 'weather'):
        with pytest.raises(RuntimeError, match='active pipeline child'):
            with pipeline.video_lock(folder, 'new-owner'):
                pytest.fail('New owner overlapped an orphan child')
    with pipeline.video_lock(folder, 'after-child'):
        pass


def test_lock_is_released_even_when_metadata_write_fails(tmp_path, monkeypatch):
    original = pipeline.atomic_json
    def failing(path, value):
        if Path(path).name == pipeline.OWNER_NAME and value.get('status') == 'released':
            raise OSError('fixture metadata failure')
        return original(path, value)
    monkeypatch.setattr(pipeline, 'atomic_json', failing)
    with pytest.raises(OSError, match='metadata failure'):
        with pipeline.video_lock(tmp_path, 'first'):
            pass
    monkeypatch.setattr(pipeline, 'atomic_json', original)
    with pipeline.video_lock(tmp_path, 'second'):
        pass


def test_mapping_external_weights_runtime_and_model_revision_invalidate_resume(toy_pipeline, monkeypatch):
    fixture = toy_pipeline
    weights = fixture.repo.parent / 'external.pt'
    weights.write_bytes(b'first model')
    fixture.execute(weights=str(weights))
    changes = [lambda: (fixture.repo / 'mapping.csv').write_text('city,avg_height\nToy,170\n'),
               lambda: weights.write_bytes(b'changed model'),
               lambda: monkeypatch.setattr(pipeline, '_runtime_versions', lambda: {'python': 'changed runtime'}),
               lambda: monkeypatch.setenv('PEDX_SCENE_MODEL_REVISION', 'a' * 40)]
    for change in changes:
        fixture.calls.clear()
        change()
        fixture.execute(resume=True, weights=str(weights))
        assert fixture.calls == list(fixture.stages)


@pytest.mark.parametrize('target', ['mapping', 'code', 'video'])
def test_changed_inputs_during_execution_cannot_be_complete(toy_pipeline, target):
    fixture = toy_pipeline
    def runner(command, **kwargs):
        result = fixture.runner(command, **kwargs)
        if command[command.index('--mode') + 1] == 'summary':
            path = {'mapping': fixture.repo / 'mapping.csv', 'code': fixture.repo / 'main.py',
                    'video': fixture.video}[target]
            path.write_bytes(b'changed while running')
        return result
    manifest = pipeline.execute_pipeline(fixture.video, 'toy', runner=runner,
        results_root=fixture.results, repo=fixture.repo)
    assert manifest['status'] == 'failed' and 'changed during' in manifest['error']


def test_crop_directory_is_archived_and_hash_verified(tmp_path, monkeypatch):
    stages = {'id_img': ((), ('pedestrian_img',))}
    monkeypatch.setattr(pipeline, 'STAGES', stages)
    monkeypatch.setattr(pipeline, 'PROFILES', {'toy-crops': ('id_img',)})
    import modules.provenance as provenance
    monkeypatch.setattr(provenance, 'weights_manifest', lambda: {})
    repo = tmp_path / 'repo'; repo.mkdir()
    video = tmp_path / 'City1_video.mp4'; video.write_bytes(b'fixture')
    folder = tmp_path / 'results' / video.stem
    crops = folder / 'pedestrian_img'; crops.mkdir(parents=True)
    (crops / 'obsolete.png').write_bytes(b'old person')
    def runner(*args, **kwargs):
        crops.mkdir()
        (crops / 'new.png').write_bytes(b'new person')
        return SimpleNamespace(returncode=0)
    manifest = pipeline.execute_pipeline(video, 'toy-crops', runner=runner,
        repo=repo, results_root=folder.parent)
    assert manifest['status'] == 'complete' and not (crops / 'obsolete.png').exists()
    assert (folder / '.previous' / manifest['run_id'] / 'pedestrian_img' / 'obsolete.png').exists()
    assert pipeline.stage_current(folder, manifest['stages']['id_img'])
    (crops / 'new.png').write_bytes(b'tampered crop')
    assert not pipeline.stage_current(folder, manifest['stages']['id_img'])


def test_unchanged_resume_preserves_analysis_identity_and_timing(toy_pipeline):
    fixture = toy_pipeline
    first = fixture.execute()
    timing = fixture.folder / '[A3]analysis_time.csv'
    timing.write_text('analysis_run_id\n' + first['run_id'] + '\n')
    resumed = fixture.execute(resume=True)
    assert resumed['run_id'] == first['run_id'] and resumed['finished_at'] == first['finished_at']
    assert timing.exists() and resumed['last_verified_at']
    fixture.execute(resume=True, interval=2)
    assert not timing.exists()


def test_release_uses_one_private_mapping_snapshot(tmp_path, monkeypatch):
    import get_all_video_info
    fixture = complete_results_tree(tmp_path)
    original_hash = pipeline.file_hash(fixture.mapping)
    original = get_all_video_info.main
    def aggregate(*args, **kwargs):
        assert Path(kwargs['mapping_path']) != fixture.mapping
        result = original(*args, **kwargs)
        fixture.mapping.write_text('city,country,videos\nOther,Elsewhere,[unrelated]\n', encoding='utf-8')
        return result
    monkeypatch.setattr(get_all_video_info, 'main', aggregate)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / 'release'
    release = build_release(output, results_root=fixture.results, mapping_path=fixture.mapping, skip_stats=True)
    assert release['producer']['mapping_sha256'] == original_hash
    assert pd.read_csv(output / 'all_pedestrian_info.csv').iloc[0]['city'] == 'London'
    assert not (output / 'mapping.csv').exists()


def model_asset_fixture(tmp_path, *, windows_paths=False):
    """Tiny asset references only; no model loader or network is involved."""
    fixture = complete_results_tree(tmp_path)
    private = tmp_path / 'private-cache' / 'account-name'
    private.mkdir(parents=True)
    checkpoint = private / 'tracker.pt'
    checkpoint.write_bytes(b'synthetic checkpoint identity, never loaded')
    weights = str(checkpoint) if not windows_paths else r'C:\private\account-name\weights\tracker.pt'
    config_path = str(private / 'config.json') if not windows_paths else r'C:\private\account-name\hub\config.json'
    assets = {
        'pedestrian_tracking': {'model_id': weights, 'revision': None,
            'files': {weights: pipeline.file_hash(checkpoint)},
            'parameters': {'tracking_fps': 15, 'analysis_interval': 1},
            'recorded_at': '2026-10-26T12:00:01Z'},
        'scene_segmentation': {'model_id': 'facebook/synthetic-model', 'revision': 'd' * 40,
            'files': {config_path: 'c' * 64}, 'parameters': {'short_side': 384},
            'recorded_at': '2026-10-26T12:00:02Z'},
    }
    assets_path = fixture.folder / '[A0]model_assets.json'
    pipeline.atomic_json(assets_path, assets)
    manifest = pipeline.read_manifest(fixture.folder)
    manifest['configuration'].update(weights_yolo=weights, model_assets_sha256=pipeline.file_hash(assets_path))
    pipeline.atomic_json(fixture.folder / pipeline.MANIFEST_NAME, manifest)
    return fixture, assets_path, assets, weights, config_path


@pytest.mark.parametrize('windows_paths', [False, True])
def test_analysis_runs_sanitizes_model_references_and_keeps_exact_evidence(tmp_path, monkeypatch, windows_paths):
    fixture, assets_path, original, weights, config_path = model_asset_fixture(tmp_path, windows_paths=windows_paths)
    original_bytes = assets_path.read_bytes()
    original_hash = pipeline.file_hash(assets_path)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / 'release'
    release = build_release(output, results_root=fixture.results, mapping_path=fixture.mapping, skip_stats=True)
    artifact = output / 'analysis_runs.json'
    published = json.loads(artifact.read_text(encoding='utf-8'))['runs'][0]
    assert published['analysis_run_id'] == 'synthetic-full-run'
    assert published['configuration']['source_sha256'] == 'a' * 64
    assert published['configuration']['model_assets_sha256'] == original_hash
    assert published['configuration']['weights_yolo'] == 'tracker.pt'
    tracking = published['model_assets']['pedestrian_tracking']
    assert tracking['model_id'] == 'tracker.pt'
    assert tracking['files'] == [{'name': 'tracker.pt', 'sha256': original['pedestrian_tracking']['files'][weights]}]
    assert tracking['parameters'] == original['pedestrian_tracking']['parameters']
    scene = published['model_assets']['scene_segmentation']
    assert scene['model_id'] == 'facebook/synthetic-model' and scene['revision'] == 'd' * 40
    assert scene['files'] == [{'name': 'config.json', 'sha256': 'c' * 64}]
    text = artifact.read_text(encoding='utf-8')
    assert 'account-name' not in text and weights not in text and config_path not in text
    assert assets_path.read_bytes() == original_bytes, 'Sanitizing a release must not rewrite source provenance'
    record = next(item for item in release['files'] if item['path'] == 'analysis_runs.json')
    assert record['sha256'] == pipeline.file_hash(artifact)


@pytest.mark.parametrize('mutation', ['changed', 'deleted'])
def test_bound_model_provenance_tampering_fails_without_release_manifest(tmp_path, monkeypatch, mutation):
    fixture, assets_path, assets, _, _ = model_asset_fixture(tmp_path)
    if mutation == 'changed':
        assets['scene_segmentation']['revision'] = 'e' * 40
        pipeline.atomic_json(assets_path, assets)
    else:
        assets_path.unlink()
    monkeypatch.chdir(tmp_path)
    output = tmp_path / 'release'
    with pytest.raises(ValueError, match='Model provenance'):
        build_release(output, results_root=fixture.results, mapping_path=fixture.mapping, skip_stats=True)
    assert (output / '.failed.json').is_file()
    assert not (output / 'release_manifest.json').exists()
