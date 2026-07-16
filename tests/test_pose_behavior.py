"""Tests for modules/insights/pose_behavior.py ([P12] head-scanning + gait).

Plain asserts, no pytest needed:  python tests/test_pose_behavior.py

Covers the spec's test strategy (pure core, no video / GPU / ultralytics):
  * scripted head turn left then right          => counts exact (1 left, 1 right)
  * ear/eye visibility asymmetry tiebreak       => resolves a nose-less look
  * brief (<0.3 s) glance                       => not counted
  * synthetic 2.0 Hz ankle sinusoid at 15 Hz    => cadence within 0.1 Hz
  * noise-only keypoints                        => not reliable, no cadence
  * reliability gate boundaries (frames / conf / bbox height)
  * IoU matching greedy uniqueness
  * missing video                               => header-only CSV
"""

import os
import sys
import shutil
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np
import pandas as pd

from modules.insights.pose_behavior import (
    OUTPUT_COLUMNS,
    KP_NOSE, KP_L_EYE, KP_R_EYE, KP_L_EAR, KP_R_EAR,
    KP_L_SHOULDER, KP_R_SHOULDER, KP_L_ANKLE, KP_R_ANKLE,
    YAW_THR,
    head_yaw_series,
    count_sustained_looks,
    ankle_y_series,
    gait_cadence,
    compute_track_metrics,
    match_tracks_to_detections,
    run_pose_behavior,
)

FS = 15.0                       # pose sampling rate used throughout
SHOULDER_W = 20.0               # px


# ---------------------------------------------------------------- helpers ---

def make_frames(n, conf=0.9, bbox_h=120.0):
    """Neutral upright pose repeated n times.

    Shoulders at x = 90/110 (width 20), nose centered at 100; ankles at
    y = 300.  Returns (kps (n,17,2), confs (n,17), bbox_heights (n,))."""
    kps = np.zeros((n, 17, 2), dtype=float)
    kps[:, :, 0] = 100.0
    kps[:, :, 1] = 100.0
    kps[:, KP_L_SHOULDER] = [110.0, 150.0]
    kps[:, KP_R_SHOULDER] = [90.0, 150.0]
    kps[:, KP_NOSE] = [100.0, 100.0]
    kps[:, KP_L_EYE] = [104.0, 98.0]
    kps[:, KP_R_EYE] = [96.0, 98.0]
    kps[:, KP_L_EAR] = [108.0, 100.0]
    kps[:, KP_R_EAR] = [92.0, 100.0]
    kps[:, KP_L_ANKLE] = [102.0, 300.0]
    kps[:, KP_R_ANKLE] = [98.0, 300.0]
    confs = np.full((n, 17), conf, dtype=float)
    bh = np.full(n, bbox_h, dtype=float)
    return kps, confs, bh


def set_nose_offset(kps, i0, i1, frac):
    """Nose x offset = frac * shoulder width on frames [i0, i1)."""
    kps[i0:i1, KP_NOSE, 0] = 100.0 + frac * SHOULDER_W


def times(n, fs=FS, t0=0.0):
    return t0 + np.arange(n) / fs


# ------------------------------------------------------------------ tests ---

def test_head_turn_left_then_right():
    # 6 s at 15 Hz: neutral 1 s, LEFT 0.6 s, neutral 0.5 s, RIGHT 0.6 s, neutral
    n = 90
    kps, confs, bh = make_frames(n)
    set_nose_offset(kps, 15, 24, -0.5)      # 0.6 s image-left look
    set_nose_offset(kps, 32, 41, +0.5)      # 0.6 s image-right look
    t = times(n)
    yaw = head_yaw_series(kps, confs)
    n_left, n_right = count_sustained_looks(t, yaw)
    assert n_left == 1, n_left
    assert n_right == 1, n_right

    # same script through compute_track_metrics: turns live in the pre-window
    # of a crossing that starts at t = 6 s
    stats = compute_track_metrics(t, kps, confs, bh, window_s=(6.0, 10.0))
    assert stats["looked_left"] is True, stats
    assert stats["looked_right"] is True, stats
    assert stats["n_head_turns"] == 2, stats
    assert stats["n_pose_frames"] == n, stats
    assert stats["reliable"] is True, stats
    print("ok  test_head_turn_left_then_right")


def test_brief_glance_not_counted():
    n = 45
    kps, confs, bh = make_frames(n)
    set_nose_offset(kps, 20, 22, -0.5)      # 2 frames ~0.2 s incl. dt padding
    yaw = head_yaw_series(kps, confs)
    n_left, n_right = count_sustained_looks(times(n), yaw)
    assert n_left == 0, n_left
    assert n_right == 0, n_right
    print("ok  test_brief_glance_not_counted")


def test_ear_eye_tiebreak():
    # nose invisible; left-side facial keypoints clearly more confident
    # => image-LEFT look under the documented frontal-view convention
    n = 30
    kps, confs, bh = make_frames(n)
    confs[:, KP_NOSE] = 0.0
    confs[10:20, KP_L_EYE] = 0.9
    confs[10:20, KP_L_EAR] = 0.9
    confs[10:20, KP_R_EYE] = 0.1
    confs[10:20, KP_R_EAR] = 0.1
    # outside the look: symmetric visibility, nose invisible -> NaN yaw
    confs[:10, KP_L_EYE] = confs[:10, KP_R_EYE] = 0.5
    confs[:10, KP_L_EAR] = confs[:10, KP_R_EAR] = 0.5
    confs[20:, KP_L_EYE] = confs[20:, KP_R_EYE] = 0.5
    confs[20:, KP_L_EAR] = confs[20:, KP_R_EAR] = 0.5
    yaw = head_yaw_series(kps, confs)
    assert np.all(np.isnan(yaw[:10])), yaw[:10]
    assert np.allclose(yaw[10:20], -YAW_THR), yaw[10:20]
    n_left, n_right = count_sustained_looks(times(n), yaw)
    assert n_left == 1 and n_right == 0, (n_left, n_right)
    print("ok  test_ear_eye_tiebreak")


def test_gap_breaks_look():
    # two 0.25 s left bursts separated by a 0.5 s neutral gap: neither run is
    # sustained on its own and the gap prevents merging -> 0 turns
    n = 60
    kps, confs, bh = make_frames(n)
    set_nose_offset(kps, 10, 13, -0.5)
    set_nose_offset(kps, 21, 24, -0.5)
    yaw = head_yaw_series(kps, confs)
    n_left, n_right = count_sustained_looks(times(n), yaw)
    assert n_left == 0 and n_right == 0, (n_left, n_right)
    print("ok  test_gap_breaks_look")


def test_cadence_two_hz():
    # 6 s crossing at 15 Hz, ankle-y = 300 + 3 sin(2 pi 2.0 t) + linear drift
    n = 90
    kps, confs, bh = make_frames(n)
    t = times(n)
    drift = 5.0 * t                          # camera / walking trend
    osc = 3.0 * np.sin(2 * np.pi * 2.0 * t)
    kps[:, KP_L_ANKLE, 1] = 300.0 + osc + drift
    kps[:, KP_R_ANKLE, 1] = 300.0 + osc + drift
    ankle = ankle_y_series(kps, confs)
    cadence, steps = gait_cadence(t, ankle)
    assert abs(cadence - 2.0) <= 0.1, cadence
    span = t[-1] - t[0] + 1.0 / FS
    assert steps == int(round(cadence * span)), (steps, cadence, span)
    assert 10 <= steps <= 14, steps          # ~2 steps/s * 6 s

    stats = compute_track_metrics(t, kps, confs, bh, window_s=(0.0, t[-1]))
    assert stats["cadence_hz"] is not None and abs(stats["cadence_hz"] - 2.0) <= 0.1, stats
    assert stats["step_count"] == steps, stats
    print("ok  test_cadence_two_hz")


def test_noise_only_not_reliable():
    # 30 frames of low-confidence noise keypoints: reliability gate fails on
    # median conf, ankles are below the conf floor -> no cadence either
    rng = np.random.default_rng(42)
    n = 30
    kps, confs, bh = make_frames(n, conf=0.1)
    kps += rng.normal(0, 20, size=kps.shape)
    t = times(n)
    stats = compute_track_metrics(t, kps, confs, bh, window_s=(0.0, t[-1]))
    assert stats["reliable"] is False, stats
    assert stats["cadence_hz"] is None, stats
    assert stats["step_count"] is None, stats
    assert stats["n_head_turns"] == 0, stats
    print("ok  test_noise_only_not_reliable")


def test_noise_ankle_no_cadence_peak():
    # confident but aperiodic ankle motion: the peak-prominence gate rejects it
    rng = np.random.default_rng(7)
    n = 90
    t = times(n)
    ankle = 300.0 + rng.normal(0, 3.0, size=n)
    cadence, steps = gait_cadence(t, ankle)
    assert cadence != cadence, cadence       # NaN
    assert steps is None, steps
    print("ok  test_noise_ankle_no_cadence_peak")


def test_reliable_gate_boundaries():
    t20 = times(20)

    kps, confs, bh = make_frames(19)
    s = compute_track_metrics(times(19), kps, confs, bh, window_s=(0.0, 1.0))
    assert s["reliable"] is False, s         # 19 frames < 20

    kps, confs, bh = make_frames(20)
    s = compute_track_metrics(t20, kps, confs, bh, window_s=(0.0, 1.0))
    assert s["reliable"] is True, s          # exactly at all thresholds

    kps, confs, bh = make_frames(20, conf=0.29)
    s = compute_track_metrics(t20, kps, confs, bh, window_s=(0.0, 1.0))
    assert s["reliable"] is False, s         # conf below 0.3

    kps, confs, bh = make_frames(20, bbox_h=79.0)
    s = compute_track_metrics(t20, kps, confs, bh, window_s=(0.0, 1.0))
    assert s["reliable"] is False, s         # bbox height below 80 px

    kps, confs, bh = make_frames(20, bbox_h=80.0)
    s = compute_track_metrics(t20, kps, confs, bh, window_s=(0.0, 1.0))
    assert s["reliable"] is True, s
    print("ok  test_reliable_gate_boundaries")


def test_iou_matching_greedy_unique():
    tracks = {
        1: [0.0, 0.0, 10.0, 10.0],
        2: [8.0, 0.0, 18.0, 10.0],
    }
    dets = [
        [1.0, 0.0, 11.0, 10.0],      # best for track 1
        [8.5, 0.0, 18.5, 10.0],      # best for track 2
        [100.0, 100.0, 110.0, 110.0]  # matches nothing
    ]
    assign = match_tracks_to_detections(tracks, dets, iou_thr=0.3)
    assert assign == {1: 0, 2: 1}, assign
    # one detection cannot serve two tracks
    assign = match_tracks_to_detections(tracks, [dets[0]], iou_thr=0.05)
    assert len(assign) == 1, assign
    print("ok  test_iou_matching_greedy_unique")


def test_header_only_when_video_missing():
    work = tempfile.mkdtemp(prefix="pose_behavior_novideo_")
    old_cwd = os.getcwd()
    try:
        os.chdir(work)
        out_csv = run_pose_behavior(os.path.join("videos", "Nowhere_zzz.mp4"))
        res = pd.read_csv(out_csv)
        assert list(res.columns) == OUTPUT_COLUMNS, list(res.columns)
        assert res.empty
        # heavy deps must not have been touched on this path
        assert "ultralytics" not in sys.modules
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(work, ignore_errors=True)
    print("ok  test_header_only_when_video_missing")


if __name__ == "__main__":
    test_head_turn_left_then_right()
    test_brief_glance_not_counted()
    test_ear_eye_tiebreak()
    test_gap_breaks_look()
    test_cadence_two_hz()
    test_noise_only_not_reliable()
    test_noise_ankle_no_cadence_peak()
    test_reliable_gate_boundaries()
    test_iou_matching_greedy_unique()
    test_header_only_when_video_missing()
    print("ALL pose_behavior TESTS PASSED")
