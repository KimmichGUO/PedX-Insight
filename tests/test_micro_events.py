"""Tests for modules/insights/micro_events.py ([P11] micro-event profiler).

Plain asserts, no pytest needed:  python tests/test_micro_events.py

Covers the spec's test strategy:
  * scripted stop -> lunge -> stop -> cross  => exactly 1 aborted start
  * mid-road 2 s freeze                      => exactly 1 mid-crossing stop + progress frac
  * clean crossing with 2.5 m/s burst at 60% => exactly 1 evasive event at ~0.6 progress
  * +/-2 px foot jitter on a pure STOP trace => zero events (dwell threshold)
  * hysteresis boundaries at 0.3 / 0.8 m/s and the 0.4 s dwell rule
  * reliable-gating of evasive events, empty-input guard, [C1]/[C6] join
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

from modules.insights.micro_events import (
    OUTPUT_COLUMNS,
    segment_speed_states,
    detect_events,
    run_micro_events,
)

FPS = 15.0
H_PX = 170.0                 # with assumed_height 1.70 m -> scale = 100 px/m
SCALE_PX_PER_M = H_PX / 1.70


# ---------------------------------------------------------------- helpers ---

def build_track(track_id, segments, x0=100.0, y_foot=500.0, noise_px=0.0, seed=None):
    """Synthetic 15 Hz [B2] rows for one track.

    ``segments``: list of (duration_s, v_mps) or (duration_s, v0, v1) linear ramps.
    frame_id is 1-based; timestamp = frame_id / 15 (so time T <-> frame 15*T).
    """
    dt = 1.0 / FPS
    speeds = []
    for seg in segments:
        if len(seg) == 2:
            dur, v0 = seg
            v1 = v0
        else:
            dur, v0, v1 = seg
        n = int(round(dur * FPS))
        for k in range(n):
            speeds.append(v0 + (v1 - v0) * (k / n))
    xs = [x0]
    for v in speeds:
        xs.append(xs[-1] + v * SCALE_PX_PER_M * dt)
    rng = np.random.default_rng(seed)
    rows = []
    for i, x in enumerate(xs):
        nx = x + (rng.uniform(-noise_px, noise_px) if noise_px else 0.0)
        fr = i + 1
        rows.append((fr, fr / FPS, track_id,
                     nx - 20.0, y_foot - H_PX, nx + 20.0, y_foot))
    return pd.DataFrame(rows, columns=["frame_id", "timestamp", "track_id",
                                       "x1", "y1", "x2", "y2"])


def constant_speed_states(v, n_steps=30):
    t = np.arange(n_steps + 1) / FPS
    return segment_speed_states(t, [v] * n_steps)


# ------------------------------------------------- unit tests: hysteresis ---

def test_hysteresis_boundaries():
    # STOP is strict v < 0.3; WALK is strict v > 0.8; SLOW in between
    assert [s["label"] for s in constant_speed_states(0.29)] == ["STOP"]
    assert [s["label"] for s in constant_speed_states(0.30)] == ["SLOW"]
    assert [s["label"] for s in constant_speed_states(0.80)] == ["SLOW"]
    assert [s["label"] for s in constant_speed_states(0.81)] == ["WALK"]
    print("ok  test_hysteresis_boundaries")


def test_min_dwell_rule():
    # 1 s STOP + 0.2 s WALK spike + 1 s STOP  -> spike merged away, 1 STOP state
    speeds = [0.0] * 15 + [1.5] * 3 + [0.0] * 15
    t = np.arange(len(speeds) + 1) / FPS
    states = segment_speed_states(t, speeds)
    assert len(states) == 1 and states[0]["label"] == "STOP", states
    # a WALK run of exactly 0.4 s (6 steps) survives (merge rule is dur < 0.4)
    speeds = [0.0] * 15 + [1.5] * 6 + [0.0] * 15
    t = np.arange(len(speeds) + 1) / FPS
    states = segment_speed_states(t, speeds)
    assert [s["label"] for s in states] == ["STOP", "WALK", "STOP"], states
    assert abs(states[1]["dur"] - 0.4) < 1e-9
    print("ok  test_min_dwell_rule")


def test_gap_breaks_states():
    # a None (invalid) step splits the trace; runs are never merged across it
    speeds = [0.0] * 15 + [None] + [0.0] * 15
    t = np.arange(len(speeds) + 1) / FPS
    states = segment_speed_states(t, speeds)
    assert [s["label"] for s in states] == ["STOP", "STOP"], states
    print("ok  test_gap_breaks_states")


# ------------------------------------------- unit tests: event detection ----

def test_evasive_reliable_gate():
    # 6 s at 1.2, 0.6 s ramp to 2.5 (a ~= 2.2 m/s^2), hold 2.5 -> one evasive
    speeds = [1.2] * 90 + [1.2 + 1.3 * k / 9 for k in range(9)] + [2.5] * 51
    t = np.arange(len(speeds) + 1) / FPS
    states = segment_speed_states(t, speeds)
    r = detect_events(states, t, speeds, window_s=(0.0, 10.0), reliable=True)
    assert r["n_evasive_events"] == 1, r
    assert abs(r["first_event_progress_frac"] - 0.6) < 0.05, r
    assert abs(r["max_burst_speed_mps"] - 2.5) < 0.05, r
    # identical trace, unreliable speed measurement -> gated to zero
    r0 = detect_events(states, t, speeds, window_s=(0.0, 10.0), reliable=False)
    assert r0["n_evasive_events"] == 0 and r0["max_burst_speed_mps"] is None, r0
    print("ok  test_evasive_reliable_gate")


def test_no_window_means_no_anchored_events():
    speeds = [0.0] * 30 + [1.2] * 10 + [0.0] * 30       # a lunge, but no crossing
    t = np.arange(len(speeds) + 1) / FPS
    states = segment_speed_states(t, speeds)
    r = detect_events(states, t, speeds, window_s=None, reliable=True)
    assert r["n_aborted_starts"] == 0 and r["n_midcross_stops"] == 0
    assert r["n_evasive_events"] == 0 and r["first_event_progress_frac"] is None
    assert r["total_stop_time_s"] > 3.5                  # stop time still measured
    print("ok  test_no_window_means_no_anchored_events")


# ------------------------------------------- end-to-end fixture scenarios ---

def run_end_to_end():
    """Four scripted tracks through the full CSV pipeline in a temp cwd."""
    work = tempfile.mkdtemp(prefix="micro_events_test_")
    video_name = "TestCity_abc123XYZ"
    out_dir = os.path.join(work, "analysis_results", video_name)
    os.makedirs(out_dir, exist_ok=True)

    # Track 1: curb dance. stop 3 s -> 1 s lunge -> stop 1.6 s -> cross 8 s -> stop
    t1 = build_track(1, [(3, 0.0), (1, 1.2), (1.6, 0.0), (8, 1.3), (1.6, 0.0)])
    # crossing = the 8 s walk: steps 84..203 -> rows 85..204
    win1 = (85, 203)
    # Track 2: mid-road freeze. cross 3 s -> freeze 2 s -> cross 3 s
    t2 = build_track(2, [(3, 1.3), (2, 0.0), (3, 1.3)])
    win2 = (1, 121)
    # Track 3: evasive burst at 60% of a 10 s crossing (1.2 -> 2.5 m/s over 0.6 s)
    t3 = build_track(3, [(6, 1.2), (0.6, 1.2, 2.5), (3.4, 2.5)])
    win3 = (1, 151)
    # Track 4: pure STOP trace with +/-2 px foot jitter, never crosses
    t4 = build_track(4, [(12, 0.0)], noise_px=2.0, seed=42)

    pd.concat([t1, t2, t3, t4]).to_csv(
        os.path.join(out_dir, "[B2]dense_tracks.csv"), index=False)
    pd.DataFrame(
        [(1, True, win1[0], win1[1], "cross"),
         (2, True, win2[0], win2[1], "cross"),
         (3, True, win3[0], win3[1], "cross"),
         (4, False, None, None, None)],
        columns=["track_id", "crossed", "started_frame", "ended_frame",
                 "movement_type"]).to_csv(
        os.path.join(out_dir, "[C3]crossing_judge.csv"), index=False)
    # degenerate-but-'good' [S2] (a=0, b=4 px/m implies 42 m tall pedestrians):
    # the per-track plausibility guard must reject it and fall back to the prior.
    pd.DataFrame([(0.0, 4.0, 50, 0.0, 1.0, 4.0, "good")],
                 columns=["a", "b", "n_samples", "fit_residual_px",
                          "stripe_period_m", "median_scale_px_per_m",
                          "quality"]).to_csv(
        os.path.join(out_dir, "[S2]scale_calibration.csv"), index=False)
    # validation-join sources for track 3
    pd.DataFrame([(3, True, 1, 151, 150, 30, 0.2, "risky")],
                 columns=["track_id", "crossed", "started_frame", "ended_frame",
                          "total_frames", "risky_frames", "risky_ratio",
                          "risk"]).to_csv(
        os.path.join(out_dir, "[C1]risky_crossing.csv"), index=False)
    pd.DataFrame([(3, True, 5)],
                 columns=["track_id", "crossed", "total_vehicle_count"]).to_csv(
        os.path.join(out_dir, "[C6]crossing_ve_count.csv"), index=False)

    old_cwd = os.getcwd()
    try:
        os.chdir(work)
        out_csv = run_micro_events(os.path.join("videos", video_name + ".mp4"),
                                   assumed_height_m=1.70)
        res = pd.read_csv(out_csv)
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(work, ignore_errors=True)
    return res


def test_end_to_end_scenarios():
    res = run_end_to_end()
    assert list(res.columns) == OUTPUT_COLUMNS, res.columns.tolist()
    assert set(res["track_id"]) == {1, 2, 3, 4}
    r = {int(row["track_id"]): row for _, row in res.iterrows()}

    # Track 1: exactly one aborted start, nothing else
    assert r[1]["n_aborted_starts"] == 1, dict(r[1])
    assert r[1]["n_midcross_stops"] == 0, dict(r[1])
    assert r[1]["n_evasive_events"] == 0, dict(r[1])
    assert r[1]["n_states"] == 5, dict(r[1])                # S W S W S
    assert 5.0 < r[1]["total_stop_time_s"] < 7.0, dict(r[1])
    assert pd.isna(r[1]["first_event_progress_frac"]), dict(r[1])
    assert r[1]["hesitation_score"] > 1.0, dict(r[1])

    # Track 2: exactly one mid-crossing stop at ~3/8 progress
    assert r[2]["n_midcross_stops"] == 1, dict(r[2])
    assert r[2]["n_aborted_starts"] == 0, dict(r[2])
    assert r[2]["n_evasive_events"] == 0, dict(r[2])
    assert 1.5 < r[2]["total_stop_time_s"] < 2.5, dict(r[2])
    assert 0.30 < r[2]["first_event_progress_frac"] < 0.45, dict(r[2])

    # Track 3: exactly one evasive burst at ~60% progress, ~2.5 m/s terminal
    assert r[3]["n_evasive_events"] == 1, dict(r[3])
    assert r[3]["n_aborted_starts"] == 0 and r[3]["n_midcross_stops"] == 0, dict(r[3])
    assert 0.55 < r[3]["first_event_progress_frac"] < 0.66, dict(r[3])
    assert 2.3 < r[3]["max_burst_speed_mps"] < 2.7, dict(r[3])
    assert bool(r[3]["reliable"]) is True, dict(r[3])
    assert r[3]["c1_risk"] == "risky", dict(r[3])
    assert r[3]["c6_total_vehicle_count"] == 5, dict(r[3])

    # Track 4: +/-2 px jitter on a STOP trace -> zero events, one merged STOP state
    assert r[4]["n_aborted_starts"] == 0, dict(r[4])
    assert r[4]["n_midcross_stops"] == 0, dict(r[4])
    assert r[4]["n_evasive_events"] == 0, dict(r[4])
    assert r[4]["n_states"] == 1, dict(r[4])
    assert r[4]["total_stop_time_s"] > 8.0, dict(r[4])
    print("ok  test_end_to_end_scenarios")


def test_empty_input_guard():
    work = tempfile.mkdtemp(prefix="micro_events_empty_")
    old_cwd = os.getcwd()
    try:
        os.chdir(work)
        out_csv = run_micro_events(os.path.join("videos", "Nowhere_zzz.mp4"))
        res = pd.read_csv(out_csv)
        assert list(res.columns) == OUTPUT_COLUMNS
        assert res.empty
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(work, ignore_errors=True)
    print("ok  test_empty_input_guard")


if __name__ == "__main__":
    test_hysteresis_boundaries()
    test_min_dwell_rule()
    test_gap_breaks_states()
    test_evasive_reliable_gate()
    test_no_window_means_no_anchored_events()
    test_end_to_end_scenarios()
    test_empty_input_guard()
    print("ALL micro_events TESTS PASSED")
