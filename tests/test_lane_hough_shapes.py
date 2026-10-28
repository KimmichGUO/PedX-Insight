"""CPU geometry regressions for OpenCV's flat and singleton Hough layouts."""
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.lane_detection import lane_detection as lane


def road_image():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.line(image, (100, 719), (600, 432), (255, 255, 255), 6)
    cv2.line(image, (1180, 719), (680, 432), (255, 255, 255), 6)
    return image


@pytest.mark.parametrize("shape", [(2, 4), (2, 1, 4)])
def test_pipeline_accepts_both_hough_segment_layouts(monkeypatch, shape):
    segments = np.array([[100, 720, 600, 432], [1180, 720, 680, 432]], dtype=np.int32).reshape(shape)
    monkeypatch.setattr(lane.cv2, "HoughLinesP", lambda *args, **kwargs: segments)
    image = road_image()
    overlay, left, right = lane.pipeline(image)
    np.testing.assert_allclose(left, [100, 720, 600, 432], atol=1)
    np.testing.assert_allclose(right, [1180, 720, 680, 432], atol=1)
    assert overlay.shape == image.shape and overlay.dtype == image.dtype
    assert np.any(overlay != image)


def test_pipeline_no_hough_lines_keeps_unknown_lanes_and_image(monkeypatch):
    monkeypatch.setattr(lane.cv2, "HoughLinesP", lambda *args, **kwargs: None)
    image = road_image()
    overlay, left, right = lane.pipeline(image)
    assert overlay is image and left is None and right is None


def test_pipeline_one_sided_flat_hough_result_keeps_other_side_unknown(monkeypatch):
    segments = np.array([[100, 720, 600, 432]], dtype=np.int32)
    monkeypatch.setattr(lane.cv2, "HoughLinesP", lambda *args, **kwargs: segments)
    image = road_image()
    overlay, left, right = lane.pipeline(image)
    np.testing.assert_allclose(left, [100, 720, 600, 432], atol=1)
    assert right is None and overlay is image


def test_real_opencv_hough_pipeline_finds_synthetic_road_lanes():
    """Use actual Canny/ROI/Hough output, without models or inference."""
    image = road_image()
    overlay, left, right = lane.pipeline(image)
    assert left is not None and right is not None
    assert left[0] < 640 < right[0]
    assert left[1] == right[1] == 720
    assert left[3] == right[3] == 432
    assert overlay.shape == image.shape and np.any(overlay != image)
