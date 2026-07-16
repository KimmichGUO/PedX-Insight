"""Tests for modules.insights.vehicle_speed ([V8], insight Rank 2).

Plain asserts, no pytest. Run from the repo root (or anywhere):
    python tests/test_vehicle_speed.py

Covers the spec's test strategy:
  1. exact 10 m/s fixture under scale(y) = 0*y + 10 px/m
  2. ego-compensation invariance (linear camera drift + matching [B3])
  3. scale-priority fallback matrix (S2 good / lane / length prior / none)
  4. crosswalk-band vs mid-block split
plus the pan-corruption reliability gate, the max-step-speed outlier filter,
the [E7] box loader, and header-only outputs on missing inputs.
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from modules.insights.vehicle_speed import (
    OUTPUT_COLUMNS, _load_crosswalk_boxes, compute_vehicle_speeds, run_vehicle_speed,
)

FPS = 15.0
DT = 1.0 / FPS


def make_track(track_id=1, n=20, x0=100.0, speed_px_s=100.0, y2=500.0,
               width_px=40.0, height_px=60.0, vtype="car", axis="x",
               frame_step=2, frame0=1):
    """Straight constant-velocity [V7]-shaped track sampled at 15 Hz."""
    rows = []
    for i in range(n):
        pos = x0 + speed_px_s * DT * i
        if axis == "x":
            cx, ybot = pos, y2
        else:
            cx, ybot = x0, pos
        rows.append({
            "frame_id": frame0 + frame_step * i,
            "timestamp": i * DT,
            "track_id": track_id,
            "vtype": vtype,
            "conf": 0.9,
            "x1": cx - width_px / 2, "y1": ybot - height_px,
            "x2": cx + width_px / 2, "y2": ybot,
        })
    return pd.DataFrame(rows)


def test_exact_10_mps():
    """100 px/s under scale(y)=0*y+10 px/m -> exactly 10 m/s everywhere."""
    veh = make_track(n=20, speed_px_s=100.0)
    rows = compute_vehicle_speeds(veh, stripe_ab=(0.0, 10.0))
    assert len(rows) == 1, rows
    r = rows[0]
    assert r["n_valid_steps"] == 19, r
    assert abs(r["median_speed_mps"] - 10.0) < 1e-9, r
    assert abs(r["p85_speed_mps"] - 10.0) < 1e-9, r
    assert abs(r["max_speed_mps"] - 10.0) < 1e-9, r
    assert r["scale_source"] == "stripe_ground_plane", r
    assert r["camera_moving"] is False, r
    assert r["reliable"] is True, r
    print("ok  test_exact_10_mps")


def test_ego_comp_invariance():
    """Linear camera drift + matching [B3] -> identical speeds to the static case."""
    static = make_track(n=20, speed_px_s=100.0)
    base = compute_vehicle_speeds(static, stripe_ab=(0.0, 10.0))[0]

    drift_per_frame = 3.0  # px/frame in x; cumulative 3*40 = 120 px < 200 px pan limit
    drifted = static.copy()
    drifted["x1"] = drifted["x1"] + drift_per_frame * drifted["frame_id"]
    drifted["x2"] = drifted["x2"] + drift_per_frame * drifted["frame_id"]
    max_f = int(static["frame_id"].max())
    ego = pd.DataFrame({
        "frame_id": np.arange(0, max_f + 1),
        "timestamp": np.arange(0, max_f + 1) / 30.0,
        "cam_x": drift_per_frame * np.arange(0, max_f + 1),
        "cam_y": np.zeros(max_f + 1),
        "step_px": np.full(max_f + 1, drift_per_frame),
        "n_bg_points": np.full(max_f + 1, 200),
    })
    got = compute_vehicle_speeds(drifted, ego_df=ego, stripe_ab=(0.0, 10.0))[0]
    assert got["camera_moving"] is True, got
    for k in ("n_valid_steps", "median_speed_mps", "p85_speed_mps", "max_speed_mps"):
        assert abs(got[k] - base[k]) < 1e-6, (k, got[k], base[k])
    assert got["reliable"] is True, got
    print("ok  test_ego_comp_invariance")


def test_scale_priority_fallback_matrix():
    lateral_car = make_track(n=20, speed_px_s=100.0, width_px=45.0, vtype="car")

    # (a) S2 good beats lane scale
    r = compute_vehicle_speeds(lateral_car, stripe_ab=(0.0, 10.0),
                               lane_scale_px_per_m=999.0)[0]
    assert r["scale_source"] == "stripe_ground_plane", r
    assert abs(r["median_speed_mps"] - 10.0) < 1e-9, r

    # (b) no S2 -> lane-width scale, still reliable
    r = compute_vehicle_speeds(lateral_car, stripe_ab=None, lane_scale_px_per_m=10.0)[0]
    assert r["scale_source"] == "lane_width", r
    assert abs(r["median_speed_mps"] - 10.0) < 1e-9, r
    assert r["reliable"] is True, r

    # (c) neither -> car-length prior for a near-lateral car: 45 px / 4.5 m = 10 px/m,
    # correct speed but NEVER reliable
    r = compute_vehicle_speeds(lateral_car)[0]
    assert r["scale_source"] == "length_prior", r
    assert abs(r["median_speed_mps"] - 10.0) < 1e-9, r
    assert r["reliable"] is False, r

    # (d) neither + non-lateral (vertical) track -> no scale, NaN speeds
    vertical = make_track(n=20, speed_px_s=100.0, axis="y", x0=400.0)
    r = compute_vehicle_speeds(vertical)[0]
    assert r["scale_source"] == "none", r
    assert r["n_valid_steps"] == 0, r
    assert r["median_speed_mps"] is None, r
    assert r["reliable"] is False, r

    # (e) length prior is cars/taxis only
    truck = make_track(n=20, speed_px_s=100.0, vtype="truck")
    r = compute_vehicle_speeds(truck)[0]
    assert r["scale_source"] == "none", r
    print("ok  test_scale_priority_fallback_matrix")


def test_crosswalk_band_split():
    """10 steps at 10 m/s mid-block, then 10 steps at 5 m/s inside the band."""
    rows = []
    x = 0.0
    for i in range(21):
        rows.append({
            "frame_id": 1 + 2 * i, "timestamp": i * DT, "track_id": 7,
            "vtype": "car", "conf": 0.9,
            "x1": x - 20, "y1": 440, "x2": x + 20, "y2": 500,
        })
        x += (100.0 if i < 10 else 50.0) * DT
    veh = pd.DataFrame(rows)
    # pre-padded band covering only the slow segment (step mids >= 68.3 px)
    band = [[66.8, 400.0, 200.0, 600.0]]
    r = compute_vehicle_speeds(veh, stripe_ab=(0.0, 10.0), crosswalk_boxes=band,
                               smooth_window=1)[0]
    assert r["n_valid_steps"] == 20, r
    assert abs(r["speed_at_crosswalk_mps"] - 5.0) < 1e-9, r
    assert abs(r["midblock_speed_mps"] - 10.0) < 1e-9, r
    assert abs(r["median_speed_mps"] - 7.5) < 1e-9, r
    assert abs(r["max_speed_mps"] - 10.0) < 1e-9, r
    # no boxes -> crosswalk median NaN, everything mid-block
    r2 = compute_vehicle_speeds(veh, stripe_ab=(0.0, 10.0), smooth_window=1)[0]
    assert r2["speed_at_crosswalk_mps"] is None, r2
    assert abs(r2["midblock_speed_mps"] - 7.5) < 1e-9, r2
    print("ok  test_crosswalk_band_split")


def test_pan_corruption_gate():
    """Cumulative camera displacement > 200 px -> row kept but never reliable."""
    veh = make_track(n=20, speed_px_s=100.0)
    max_f = int(veh["frame_id"].max())
    ego = pd.DataFrame({
        "frame_id": np.arange(0, max_f + 1),
        "timestamp": np.arange(0, max_f + 1) / 30.0,
        "cam_x": 10.0 * np.arange(0, max_f + 1),   # 390 px cumulative > 200
        "cam_y": np.zeros(max_f + 1),
        "step_px": np.full(max_f + 1, 10.0),
        "n_bg_points": np.full(max_f + 1, 200),
    })
    veh = veh.copy()
    veh["x1"] = veh["x1"] + 10.0 * veh["frame_id"]
    veh["x2"] = veh["x2"] + 10.0 * veh["frame_id"]
    r = compute_vehicle_speeds(veh, ego_df=ego, stripe_ab=(0.0, 10.0))[0]
    assert r["camera_moving"] is True, r
    assert r["reliable"] is False, r
    assert r["n_valid_steps"] > 0, r  # speeds still computed, just gated
    print("ok  test_pan_corruption_gate")


def test_max_step_speed_filter():
    """A single teleport step (> 50 m/s) is discarded, not averaged in."""
    veh = make_track(n=20, speed_px_s=100.0)
    veh = veh.copy()
    # teleport the last row 5000 px away -> that step alone is implausible
    veh.loc[veh.index[-1], ["x1", "x2"]] += 5000.0
    r = compute_vehicle_speeds(veh, stripe_ab=(0.0, 10.0), smooth_window=1)[0]
    assert r["n_valid_steps"] == 18, r
    assert abs(r["median_speed_mps"] - 10.0) < 1e-9, r
    assert r["max_speed_mps"] <= 50.0, r
    print("ok  test_max_step_speed_filter")


def test_load_crosswalk_boxes():
    tmp = tempfile.mkdtemp(prefix="v8_e7_")
    try:
        p = os.path.join(tmp, "[E7]crosswalk_detection.csv")
        pd.DataFrame({
            "frame_id": [0, 1, 2],
            "crosswalk_detected": ["No", "Yes", "Yes"],
            "crosswalk_boxes": ["[]", "[[100.0, 200.0, 200.0, 300.0]]",
                                "[[100.0, 200.0, 200.0, 300.0]]"],  # duplicate -> one box
        }).to_csv(p, index=False)
        boxes = _load_crosswalk_boxes(p)
        assert len(boxes) == 1, boxes
        x1, y1, x2, y2 = boxes[0]
        assert abs(x1 - 85.0) < 1e-9 and abs(x2 - 215.0) < 1e-9, boxes
        assert abs(y1 - 185.0) < 1e-9 and abs(y2 - 315.0) < 1e-9, boxes
        assert _load_crosswalk_boxes(os.path.join(tmp, "missing.csv")) == []
        bad = os.path.join(tmp, "bad.csv")
        with open(bad, "w") as f:
            f.write("frame_id,crosswalk_detected,crosswalk_boxes\n0,Yes,not-a-list\n")
        assert _load_crosswalk_boxes(bad) == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok  test_load_crosswalk_boxes")


def test_run_vehicle_speed_end_to_end_and_empty():
    """Entry point with per-video CSV layout; missing [V7] -> header-only output."""
    tmp = tempfile.mkdtemp(prefix="v8_run_")
    old_cwd = os.getcwd()
    try:
        os.chdir(tmp)
        vdir = os.path.join("analysis_results", "testvid")
        os.makedirs(vdir, exist_ok=True)
        make_track(n=20, speed_px_s=100.0).to_csv(
            os.path.join(vdir, "[V7]vehicle_tracks.csv"), index=False)
        pd.DataFrame([{"a": 0.0, "b": 10.0, "n_samples": 50, "fit_residual_px": 0.0,
                       "stripe_period_m": 1.0, "median_scale_px_per_m": 10.0,
                       "quality": "good"}]).to_csv(
            os.path.join(vdir, "[S2]scale_calibration.csv"), index=False)
        out = run_vehicle_speed(os.path.join("videos", "testvid.mp4"))
        df = pd.read_csv(out)
        assert list(df.columns) == OUTPUT_COLUMNS, df.columns
        assert len(df) == 1, df
        assert abs(df.iloc[0]["median_speed_mps"] - 10.0) < 1e-6, df
        assert df.iloc[0]["scale_source"] == "stripe_ground_plane", df
        assert bool(df.iloc[0]["reliable"]) is True, df
        assert np.isnan(df.iloc[0]["speed_at_crosswalk_mps"]), df  # no [E7] present

        # S2 quality != good must be ignored -> falls through (no V5 here, car is
        # lateral -> length prior)
        pd.DataFrame([{"a": 0.0, "b": 10.0, "n_samples": 5, "fit_residual_px": 9.0,
                       "stripe_period_m": 1.0, "median_scale_px_per_m": 10.0,
                       "quality": "poor"}]).to_csv(
            os.path.join(vdir, "[S2]scale_calibration.csv"), index=False)
        df = pd.read_csv(run_vehicle_speed(os.path.join("videos", "testvid.mp4")))
        assert df.iloc[0]["scale_source"] == "length_prior", df

        # missing [V7] -> valid header-only CSV, no crash
        out2 = run_vehicle_speed(os.path.join("videos", "novid.mp4"))
        df2 = pd.read_csv(out2)
        assert list(df2.columns) == OUTPUT_COLUMNS, df2.columns
        assert len(df2) == 0, df2

        # header-only [V7] -> header-only output
        vdir3 = os.path.join("analysis_results", "hdrvid")
        os.makedirs(vdir3, exist_ok=True)
        pd.DataFrame(columns=["frame_id", "timestamp", "track_id", "vtype", "conf",
                              "x1", "y1", "x2", "y2"]).to_csv(
            os.path.join(vdir3, "[V7]vehicle_tracks.csv"), index=False)
        df3 = pd.read_csv(run_vehicle_speed(os.path.join("videos", "hdrvid.mp4")))
        assert list(df3.columns) == OUTPUT_COLUMNS and len(df3) == 0, df3
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok  test_run_vehicle_speed_end_to_end_and_empty")


if __name__ == "__main__":
    test_exact_10_mps()
    test_ego_comp_invariance()
    test_scale_priority_fallback_matrix()
    test_crosswalk_band_split()
    test_pan_corruption_gate()
    test_max_step_speed_filter()
    test_load_crosswalk_boxes()
    test_run_vehicle_speed_end_to_end_and_empty()
    print("\nAll vehicle_speed tests passed.")
