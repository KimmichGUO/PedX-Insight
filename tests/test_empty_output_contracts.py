"""Empty/no-event outputs remain readable CSVs without loading detector models."""

import ast
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from collections import Counter, defaultdict

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def isolated_function(relative_path, name, namespace):
    """Compile the real function while excluding optional detector imports."""
    filename = REPO / relative_path
    tree = ast.parse(filename.read_text(encoding='utf-8'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    code = compile(ast.Module(body=[function], type_ignores=[]), str(filename), 'exec')
    exec(code, namespace)
    return namespace[name]


@pytest.mark.parametrize('has_tracks', [False, True])
def test_no_lane_observations_write_declared_header(tmp_path, monkeypatch, has_tracks):
    from modules.pede_on_lane import pede_on_lane as module
    monkeypatch.setattr(module, 'RESULTS_ROOT', str(tmp_path / 'results'))
    tracks, lanes, output = tmp_path / 'tracks.csv', tmp_path / 'lanes.csv', tmp_path / 'output.csv'
    pd.DataFrame([{'track_id': 1, 'frame_id': 1, 'x1': 0, 'y1': 0, 'x2': 2, 'y2': 3}]
        if has_tracks else [], columns=['track_id', 'frame_id', 'x1', 'y1', 'x2', 'y2']).to_csv(tracks, index=False)
    pd.DataFrame(columns=['frame', 'left_x1', 'left_y1', 'left_x2', 'left_y2',
        'right_x1', 'right_y1', 'right_x2', 'right_y2']).to_csv(lanes, index=False)
    module.pedestrian_on_lane('not-opened.mp4', tracking_csv_path=str(tracks),
                             lane_csv_path=str(lanes), output_csv_path=str(output))
    result = pd.read_csv(output)
    assert result.empty
    assert list(result.columns) == ['track_id', 'entered_lane', 'start_frame', 'end_frame']


def test_empty_strict_crossing_preserves_consumer_schema(tmp_path, monkeypatch):
    from modules.crossing_judge import crossing_strict as module
    monkeypatch.setattr(module, 'get_video_mid_x', lambda _: 100)
    tracks, sidewalks, output = tmp_path / 'tracks.csv', tmp_path / 'sidewalk.csv', tmp_path / 'strict.csv'
    pd.DataFrame(columns=['track_id', 'frame_id', 'x1', 'y1', 'x2', 'y2']).to_csv(tracks, index=False)
    pd.DataFrame(columns=['frame_id', 'polygons']).to_csv(sidewalks, index=False)
    module.detect_crossing_strict('not-opened.mp4', tracked_csv_path=str(tracks),
        sidewalk_csv_path=str(sidewalks), output_csv_path=str(output))
    result = pd.read_csv(output)
    assert result.empty
    assert list(result.columns) == ['track_id', 'crossed', 'started_frame', 'ended_frame']


def test_clothing_no_usable_frame_ids_keeps_all_attribute_columns(tmp_path):
    tracks, output = tmp_path / 'tracks.csv', tmp_path / 'clothing.csv'
    pd.DataFrame([{'frame_id': None, 'track_id': 1, 'x1': 0, 'y1': 0, 'x2': 2, 'y2': 3}]).to_csv(tracks, index=False)
    namespace = {'os': os, 'pd': pd, 'math': math, 'RESULTS_ROOT': str(tmp_path),
        'cv2': SimpleNamespace(VideoCapture=lambda _: SimpleNamespace(get=lambda _: 30, release=lambda: None),
                               CAP_PROP_FPS=5, CAP_PROP_POS_FRAMES=7),
        'YOLO': lambda _: SimpleNamespace(to=lambda _: None),
        'torch': SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))}
    run = isolated_function('modules/clothing/clothing.py', 'run_clothing_detection', namespace)
    run('not-opened.mp4', tracking_csv_path=str(tracks), output_csv_path=str(output))
    result = pd.read_csv(output)
    assert result.empty
    assert list(result.columns) == ['frame_id', 'track_id', 'short_sleeved_shirt',
        'long_sleeved_shirt', 'short_sleeved_outwear', 'long_sleeved_outwear', 'vest', 'sling',
        'shorts', 'trousers', 'skirt', 'short_sleeved_dress', 'long_sleeved_dress', 'vest_dress', 'sling_dress']


def test_vehicle_stream_with_no_frames_still_has_dynamic_class_header(tmp_path):
    output = tmp_path / 'vehicles.csv'
    capture = SimpleNamespace(isOpened=lambda: True, read=lambda: (False, None),
                              get=lambda _: 30, release=lambda: None)
    namespace = {'os': os, 'pd': pd, 'math': math, 'Counter': Counter, 'defaultdict': defaultdict,
        'RESULTS_ROOT': str(tmp_path), 'cv2': SimpleNamespace(VideoCapture=lambda _: capture, CAP_PROP_FPS=5),
        'YOLO': lambda _: SimpleNamespace(to=lambda _: None, model=SimpleNamespace(names={0: 'car', 1: 'bus'})),
        'torch': SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
        'sv': SimpleNamespace(BoxAnnotator=lambda: None)}
    run = isolated_function('modules/type_vehicle/type_vehicle.py', 'run_vehicle_frame_analysis', namespace)
    run('not-opened.mp4', output_csv_path=str(output))
    result = pd.read_csv(output)
    assert result.empty and list(result.columns) == ['frame_id', 'bus', 'car', 'total']
