"""Tests for modules/insights/pose_behavior.py ([P12] head-scanning + gait).

Plain asserts, no pytest needed:  python tests/test_pose_behavior.py

Covers the spec's test strategy (pure core, no video / GPU / ultralytics):
  * scripted head turn left then right          => counts exact (1 left, 1 right)
  * ear/eye visibility asymmetry tiebreak       => resolves a nose-less look
  * brief (<0.3 s) glance                       => not counted
  * synthetic 2.0 Hz ankle sinusoid at 15 Hz    => cadence within 0.1 Hz
  * 1.4/1.8/2.2/2.6 Hz sweep                    => each recovered within 0.1 Hz
  * moderately noisy 2.0 Hz gait                => still recovered (no over-reject)
  * too-short window / sparse sampling / gaps   => NaN
  * dominant 0.5 Hz sway, 1.0 Hz stride-only    => NaN (outside the accept band)
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
    YAW_THR, CADENCE_BAND_HZ,
    head_yaw_series,
    count_sustained_looks,
    ankle_y_series,
    cadence_signal,
    gait_cadence,
    gait_cadence_detail,
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


def gait_signal(freq_hz, duration_s, fs=FS, amp=3.0, noise_sd=0.0, drift=5.0,
                seed=0):
    """Synthetic ankle-y trace: sinusoid at ``freq_hz`` + linear drift + noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration_s, 1.0 / fs)
    y = (300.0 + amp * np.sin(2 * np.pi * freq_hz * t) + drift * t
         + rng.normal(0.0, noise_sd, size=len(t)))
    return t, y


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


def test_cadence_recovers_plausible_band():
    # a sweep across the whole accepted band must come back within 0.1 Hz
    for f in (1.4, 1.8, 2.0, 2.2, 2.6, 2.9):
        t, y = gait_signal(f, 6.0)
        cadence, steps, reason = gait_cadence_detail(t, y)
        assert reason == "", (f, reason)
        assert abs(cadence - f) <= 0.1, (f, cadence)
        assert steps is not None and steps > 0, (f, steps)
    print("ok  test_cadence_recovers_plausible_band")


def test_cadence_survives_moderate_noise():
    # the gates must not be so tight that ordinary keypoint jitter kills a real
    # cadence: 3 px oscillation with 1-2 px keypoint noise is still recovered
    for sd in (0.5, 1.0, 1.5, 2.0):
        for seed in (0, 1, 2):
            t, y = gait_signal(2.0, 6.0, noise_sd=sd, seed=seed)
            cadence, _steps, reason = gait_cadence_detail(t, y)
            assert reason == "", (sd, seed, reason)
            assert abs(cadence - 2.0) <= 0.15, (sd, seed, cadence)
    # a longer window rescues even a 1:1 amplitude-to-noise trace
    t, y = gait_signal(2.0, 12.0, noise_sd=3.0, seed=5)
    cadence, _steps, reason = gait_cadence_detail(t, y)
    assert reason == "" and abs(cadence - 2.0) <= 0.15, (cadence, reason)
    print("ok  test_cadence_survives_moderate_noise")


def test_cadence_short_window_is_nan():
    # [P12] defect: short windows leak into neighbouring bins.  A 2 s window
    # cannot resolve the band at all (df = 0.5 Hz) -> must refuse to answer,
    # even though the underlying signal is a perfect 2.0 Hz gait.
    for dur in (1.5, 2.0, 3.0):
        t, y = gait_signal(2.0, dur)
        cadence, steps, reason = gait_cadence_detail(t, y)
        assert cadence != cadence, (dur, cadence)      # NaN
        assert steps is None, (dur, steps)
        assert reason in ("too_few_samples", "window_too_short"), (dur, reason)
    # ... and the same signal over a long-enough window IS accepted
    cadence, _steps, reason = gait_cadence_detail(*gait_signal(2.0, 6.0))
    assert reason == "" and abs(cadence - 2.0) <= 0.1, (cadence, reason)
    print("ok  test_cadence_short_window_is_nan")


def test_cadence_out_of_physiological_band_is_nan():
    # 0.5 Hz body sway (the old code's favourite artifact: it clipped exactly
    # onto the 0.50 Hz search-band edge) and a 1.0 Hz single-ankle stride
    # signal are both outside the accepted step-frequency band.
    for f in (0.5, 0.8, 1.0, 3.6):
        t, y = gait_signal(f, 10.0)
        cadence, steps, reason = gait_cadence_detail(t, y)
        assert cadence != cadence, (f, cadence)
        assert steps is None, (f, steps)
        assert reason in ("out_of_physiological_band", "peak_not_dominant",
                          "peak_not_concentrated"), (f, reason)
    print("ok  test_cadence_out_of_physiological_band_is_nan")


def test_cadence_sparse_sampling_is_nan():
    # 5 Hz pose sampling -> Nyquist 2.5 Hz < band top: 2.0 Hz is unverifiable
    # against its aliases, so no estimate may be emitted.
    for fs in (3.0, 5.0):
        t, y = gait_signal(2.0, 10.0, fs=fs)
        cadence, _steps, reason = gait_cadence_detail(t, y)
        assert cadence != cadence, (fs, cadence)
        assert reason in ("sampling_too_sparse", "too_few_samples"), (fs, reason)
    print("ok  test_cadence_sparse_sampling_is_nan")


def test_cadence_large_dropout_is_nan():
    # 6 s of the 10 s window is missing: interpolating across it would invent
    # a signal, so the coverage gate must refuse.
    t, y = gait_signal(2.0, 10.0)
    keep = (t < 2.0) | (t > 8.0)
    cadence, _steps, reason = gait_cadence_detail(t[keep], y[keep])
    assert cadence != cadence, cadence
    assert reason == "sparse_coverage", reason
    print("ok  test_cadence_large_dropout_is_nan")


def test_cadence_pure_noise_is_nan():
    # no periodic content at all, over a long, densely sampled window
    for seed in (0, 1, 2, 3, 4):
        rng = np.random.default_rng(seed)
        t = np.arange(0.0, 8.0, 1.0 / FS)
        y = 300.0 + rng.normal(0.0, 3.0, size=len(t))
        cadence, steps, reason = gait_cadence_detail(t, y)
        assert cadence != cadence, (seed, cadence)
        assert steps is None, (seed, steps)
        assert reason != "", (seed, reason)
    print("ok  test_cadence_pure_noise_is_nan")


def test_cadence_signal_prefers_both_ankles():
    # both ankles confident -> strict (step-frequency) series is used
    n = 90
    kps, confs, _bh = make_frames(n)
    sig = cadence_signal(kps, confs)
    assert np.isfinite(sig).all()
    # right ankle dropped on most frames -> strict series too sparse, the
    # mixed series is used instead (and still has every frame)
    confs[:, KP_R_ANKLE] = 0.0
    sig = cadence_signal(kps, confs)
    assert np.isfinite(sig).sum() == n, np.isfinite(sig).sum()
    strict = ankle_y_series(kps, confs, require_both=True)
    assert np.isfinite(strict).sum() == 0, np.isfinite(strict).sum()
    print("ok  test_cadence_signal_prefers_both_ankles")


def test_compute_track_metrics_short_crossing_no_cadence():
    # a 2 s crossing window is too short for cadence but head scanning in the
    # pre-window is unaffected
    n = 120
    kps, confs, bh = make_frames(n)
    t = times(n)
    osc = 3.0 * np.sin(2 * np.pi * 2.0 * t)
    kps[:, KP_L_ANKLE, 1] = 300.0 + osc
    kps[:, KP_R_ANKLE, 1] = 300.0 + osc
    set_nose_offset(kps, 15, 24, -0.5)          # 0.6 s look, pre-window
    stats = compute_track_metrics(t, kps, confs, bh, window_s=(6.0, 8.0))
    assert stats["cadence_hz"] is None, stats
    assert stats["step_count"] is None, stats
    assert stats["looked_left"] is True, stats
    assert stats["n_head_turns"] == 1, stats
    print("ok  test_compute_track_metrics_short_crossing_no_cadence")


def test_emitted_cadence_always_in_band():
    # nothing outside CADENCE_BAND_HZ may ever reach the output
    rng = np.random.default_rng(3)
    lo, hi = CADENCE_BAND_HZ
    for _ in range(60):
        f = float(rng.uniform(0.2, 6.0))
        dur = float(rng.uniform(1.0, 12.0))
        sd = float(rng.uniform(0.0, 6.0))
        t, y = gait_signal(f, dur, noise_sd=sd, seed=int(f * 1000))
        cadence, steps, _r = gait_cadence_detail(t, y)
        if cadence == cadence:
            assert lo <= cadence <= hi, (f, dur, cadence)
            assert steps is not None
    print("ok  test_emitted_cadence_always_in_band")


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
    test_cadence_recovers_plausible_band()
    test_cadence_survives_moderate_noise()
    test_cadence_short_window_is_nan()
    test_cadence_out_of_physiological_band_is_nan()
    test_cadence_sparse_sampling_is_nan()
    test_cadence_large_dropout_is_nan()
    test_cadence_pure_noise_is_nan()
    test_cadence_signal_prefers_both_ankles()
    test_compute_track_metrics_short_crossing_no_cadence()
    test_emitted_cadence_always_in_band()
    test_noise_only_not_reliable()
    test_noise_ankle_no_cadence_peak()
    test_reliable_gate_boundaries()
    test_iou_matching_greedy_unique()
    test_header_only_when_video_missing()
    print("ALL pose_behavior TESTS PASSED")
