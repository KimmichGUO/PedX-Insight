"""Synthetic video/CSV coverage for OpenCV 4/5 road-width Hough layouts."""
import csv
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.road_width import road_width


OUTPUT_COLUMNS = ['Frame Index', 'Time (s)', 'Road Width (px)', 'Road Width (m)']
SEGMENTS = np.array([[80, 180, 100, 132], [240, 180, 220, 132]], dtype=np.int32)


@pytest.fixture
def video_and_output(tmp_path):
    video = tmp_path / 'City_video.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 4, (320, 240))
    assert writer.isOpened(), 'Synthetic MJPG video writer is unavailable'
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.line(image, (32, 228), (144, 144), (255, 255, 255), 3)
    cv2.line(image, (288, 228), (176, 144), (255, 255, 255), 3)
    for _ in range(8):
        writer.write(image)
    writer.release()
    output = tmp_path / 'results' / '[E5]road_width.csv'
    output.parent.mkdir()
    return video, output


def write_lane_scale(directory):
    columns = ['left_x1', 'left_y1', 'left_x2', 'left_y2',
               'right_x1', 'right_y1', 'right_x2', 'right_y2']
    with (directory / '[V5]lane_detection.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerow([80, 240, 144, 144, 240, 240, 176, 144])


@pytest.mark.parametrize('shape', [(2, 4), (2, 1, 4)])
def test_road_width_video_csv_supports_both_hough_layouts(video_and_output, monkeypatch, shape):
    video, output = video_and_output
    write_lane_scale(output.parent)
    monkeypatch.setattr(road_width.cv2, 'HoughLinesP', lambda *args, **kwargs: SEGMENTS.reshape(shape))
    road_width.run_road_width_analysis(str(video), output_csv_path=str(output))
    result = pd.read_csv(output)
    assert list(result.columns) == OUTPUT_COLUMNS
    assert result['Frame Index'].tolist() == [4, 8]
    assert result['Time (s)'].tolist() == [1.0, 2.0]
    # The known projective fixture maps 140 birdseye pixels to 64.9 native
    # pixels; the adjacent lane CSV supplies 160 native pixels per 3.5 metres.
    np.testing.assert_allclose(result['Road Width (px)'], [64.9, 64.9], atol=0.1)
    np.testing.assert_allclose(result['Road Width (m)'], [1.42, 1.42], atol=0.01)


def test_road_width_no_hough_lines_keeps_unknown_widths(video_and_output, monkeypatch):
    video, output = video_and_output
    write_lane_scale(output.parent)
    monkeypatch.setattr(road_width.cv2, 'HoughLinesP', lambda *args, **kwargs: None)
    road_width.run_road_width_analysis(str(video), output_csv_path=str(output))
    result = pd.read_csv(output)
    assert list(result.columns) == OUTPUT_COLUMNS and len(result) == 2
    assert result[['Road Width (px)', 'Road Width (m)']].isna().all().all()


def test_road_width_detected_pixels_without_lane_scale_do_not_fabricate_metres(video_and_output, monkeypatch):
    video, output = video_and_output
    monkeypatch.setattr(road_width.cv2, 'HoughLinesP', lambda *args, **kwargs: SEGMENTS)
    road_width.run_road_width_analysis(str(video), output_csv_path=str(output))
    result = pd.read_csv(output)
    assert len(result) == 2 and result['Road Width (px)'].notna().all()
    assert result['Road Width (m)'].isna().all()
