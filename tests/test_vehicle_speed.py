"""Tests for modules.insights.vehicle_speed ([V8], insight Rank 2).

Plain asserts, no pytest. Run from the repo root (or anywhere):
    python tests/test_vehicle_speed.py

Covers the spec's test strategy:
  1. exact 10 m/s fixture under scale(y) = 0*y + 10 px/m
  2. ego-compensation invariance (linear camera drift + matching [B3])
  3. scale-priority fallback matrix (S2 good / ped plane / lane plane / lane const /
     length prior / none)
  4. crosswalk-band vs mid-block split
plus the audit regressions:
  * the reliability gate must NOT depend on video length (cumulative cam_x/cam_y)
  * the pedestrian-height plane fit and the lane/ped scale cross-check
  * the per-step camera-still gate and ego_static_frac
and the max-step-speed outlier filter, the [E7] box loader, and header-only
outputs on missing inputs.
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from modules.insights.vehicle_speed import (
    OUTPUT_COLUMNS, _load_crosswalk_boxes, compute_vehicle_speeds,
    fit_lane_width_plane, fit_ped_height_plane, run_vehicle_speed,
)

FPS = 15.0
DT = 1.0 / FPS

# A camera-still [B3] stub: the per-step gate defaults to [S1]'s 0.5 px, so fixtures
# that mean "the camera was not moving" must say so with step_px, not by omission.
OPEN_GATE = dict(ego_max_step_px=1e9, ego_max_expansion=1e9)


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


def make_ego(max_frame, cam_x_per_frame=0.0, cam_y_per_frame=0.0, step_px=0.0):
    f = np.arange(0, max_frame + 1)
    return pd.DataFrame({
        "frame_id": f,
        "timestamp": f / 30.0,
        "cam_x": cam_x_per_frame * f,
        "cam_y": cam_y_per_frame * f,
        "step_px": np.full(f.size, float(step_px)),
        "n_bg_points": np.full(f.size, 200),
    })


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
    assert abs(r["running_speed_mps"] - 10.0) < 1e-9, r
    assert r["stopped_frac"] == 0.0, r
    assert abs(r["scale_px_per_m_median"] - 10.0) < 1e-9, r
    assert r["ego_static_frac"] == 1.0, r
    assert r["local_pan_px"] == 0.0, r
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
    ego = make_ego(max_f, cam_x_per_frame=drift_per_frame, step_px=drift_per_frame)
    # The gate is opened deliberately: this test isolates the ARITHMETIC of ego
    # compensation. Whether such a step is trustworthy is test_ego_static_gate's job.
    got = compute_vehicle_speeds(drifted, ego_df=ego, stripe_ab=(0.0, 10.0), **OPEN_GATE)[0]
    assert got["camera_moving"] is True, got
    for k in ("n_valid_steps", "median_speed_mps", "p85_speed_mps", "max_speed_mps"):
        assert abs(got[k] - base[k]) < 1e-6, (k, got[k], base[k])
    assert got["reliable"] is True, got
    print("ok  test_ego_comp_invariance")


def test_ego_static_gate_drops_moving_steps():
    """With the default gate, steps taken while [B3] reports pan are discarded."""
    veh = make_track(n=20, speed_px_s=100.0)
    max_f = int(veh["frame_id"].max())
    veh = veh.copy()
    veh["x1"] = veh["x1"] + 3.0 * veh["frame_id"]
    veh["x2"] = veh["x2"] + 3.0 * veh["frame_id"]

    moving = make_ego(max_f, cam_x_per_frame=3.0, step_px=3.0)   # > 0.5 px -> not still
    r = compute_vehicle_speeds(veh, ego_df=moving, stripe_ab=(0.0, 10.0))[0]
    assert r["n_valid_steps"] == 0, r
    assert r["ego_static_frac"] == 0.0, r
    assert r["reliable"] is False, r

    still = make_ego(max_f, cam_x_per_frame=3.0, step_px=0.1)    # <= 0.5 px -> still
    r2 = compute_vehicle_speeds(veh, ego_df=still, stripe_ab=(0.0, 10.0))[0]
    assert r2["n_valid_steps"] == 19, r2
    assert r2["ego_static_frac"] == 1.0, r2
    assert r2["reliable"] is True, r2
    print("ok  test_ego_static_gate_drops_moving_steps")


def test_reliability_is_length_independent():
    """REGRESSION: the pan gate must test LOCAL pan, not cumulative cam_x/cam_y.

    [B3] integrates camera position, so cam_x/cam_y are a random walk reaching tens
    of thousands of px on a full-length clip. The old gate compared that cumulative
    value with 200 px and so forced reliable = False on every long video (0 reliable
    tracks out of 138,830 across 11 cities). A 20-frame track measured while the
    camera is still must stay reliable no matter how long the video around it is.
    """
    veh = make_track(n=20, speed_px_s=100.0)
    max_f = int(veh["frame_id"].max())

    # short clip, camera still the whole time
    short = make_ego(max_f, step_px=0.1)
    r_short = compute_vehicle_speeds(veh, ego_df=short, stripe_ab=(0.0, 10.0))[0]
    assert r_short["reliable"] is True, r_short

    # SAME track, but [B3] now covers 200k frames and has wandered 50,000 px away
    # long before this track starts, exactly as on the real clips.
    n = 200000
    f = np.arange(0, n)
    walk = np.zeros(n)
    walk[max_f + 1:] = np.linspace(0, 50000.0, n - max_f - 1)   # drift AFTER the track
    long_ego = pd.DataFrame({
        "frame_id": f, "timestamp": f / 30.0,
        "cam_x": walk + 22808.0,        # huge constant offset, as on a real clip
        "cam_y": np.zeros(n),
        "step_px": np.full(n, 0.1),
        "n_bg_points": np.full(n, 200),
    })
    r_long = compute_vehicle_speeds(veh, ego_df=long_ego, stripe_ab=(0.0, 10.0))[0]
    assert r_long["reliable"] is True, r_long
    assert r_long["local_pan_px"] < 1.0, r_long
    for k in ("n_valid_steps", "median_speed_mps", "max_speed_mps"):
        assert abs(r_long[k] - r_short[k]) < 1e-6, (k, r_long[k], r_short[k])
    print("ok  test_reliability_is_length_independent")


def test_local_pan_gate_still_bites():
    """A track whose OWN window contains a > 200 px pan is still gated unreliable."""
    veh = make_track(n=20, speed_px_s=100.0)
    max_f = int(veh["frame_id"].max())
    veh = veh.copy()
    veh["x1"] = veh["x1"] + 10.0 * veh["frame_id"]
    veh["x2"] = veh["x2"] + 10.0 * veh["frame_id"]
    # 10 px/frame over the track's own 39 frames = 390 px local pan > 200,
    # but step_px is kept small so the step gate is not what rejects it.
    ego = make_ego(max_f, cam_x_per_frame=10.0, step_px=0.1)
    r = compute_vehicle_speeds(veh, ego_df=ego, stripe_ab=(0.0, 10.0))[0]
    assert r["camera_moving"] is False, r          # step_px median 0.1 < 1.0
    assert r["local_pan_px"] > 200.0, r
    assert r["reliable"] is False, r
    assert r["n_valid_steps"] > 0, r               # speeds still computed, just gated
    print("ok  test_local_pan_gate_still_bites")


def test_fit_ped_height_plane():
    """Recover a known ground plane from synthetic pedestrian boxes."""
    a_true, b_true, H = 0.4, -180.0, 1.7
    rows = []
    for tid in range(40):
        y = 500.0 + 4.0 * tid
        h = (a_true * y + b_true) * H
        for k in range(8):
            rows.append({"track_id": tid, "frame_id": k, "timestamp": k * DT,
                         "x1": 100.0, "y1": y - h, "x2": 100.0 + h / 2.5, "y2": y})
    a, b, n = fit_ped_height_plane(pd.DataFrame(rows), assumed_height_m=H)
    assert n == 40, n
    assert abs(a - a_true) < 1e-6, (a, b)
    assert abs(b - b_true) < 1e-3, (a, b)

    # a FLAT fit (a <= 0) carries no depth information and must be refused: px/m on a
    # ground plane always grows with the image row.
    flat = [dict(r, y1=r["y2"] - 34.0, x2=r["x1"] + 13.6) for r in rows]
    assert fit_ped_height_plane(pd.DataFrame(flat)) == (None, None, 0)

    # too few tracks / empty / missing columns -> no fit, no crash
    assert fit_ped_height_plane(pd.DataFrame(rows[:16])) == (None, None, 0)
    assert fit_ped_height_plane(pd.DataFrame()) == (None, None, 0)
    assert fit_ped_height_plane(None) == (None, None, 0)
    assert fit_ped_height_plane(pd.DataFrame({"a": [1]})) == (None, None, 0)
    print("ok  test_fit_ped_height_plane")


def test_fit_lane_width_plane():
    """Lane separation is linear in y, so the plane fit must recover it exactly."""
    # left line (200,700)->(500,400); right line (1000,700)->(600,400)
    # width(700)=800, width(400)=100 -> slope 700/300 px per row, /3.5 m
    d = pd.DataFrame([{"left_x1": 200, "left_y1": 700, "left_x2": 500, "left_y2": 400,
                       "right_x1": 1000, "right_y1": 700, "right_x2": 600, "right_y2": 400}])
    tmp = tempfile.mkdtemp(prefix="v8_v5_")
    try:
        p = os.path.join(tmp, "[V5]lane_detection.csv")
        pd.concat([d] * 30, ignore_index=True).assign(
            frame=range(30)).to_csv(p, index=False)
        fit = fit_lane_width_plane(p)
        assert fit is not None, fit
        a, b = fit
        assert abs((a * 700 + b) - 800.0 / 3.5) < 1e-6, fit
        assert abs((a * 400 + b) - 100.0 / 3.5) < 1e-6, fit
        # one-sided detections (zero coords) contribute nothing
        zero = os.path.join(tmp, "zeros.csv")
        d0 = d.copy(); d0[["right_x1", "right_x2"]] = 0
        pd.concat([d0] * 30, ignore_index=True).to_csv(zero, index=False)
        assert fit_lane_width_plane(zero) is None
        assert fit_lane_width_plane(os.path.join(tmp, "missing.csv")) is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok  test_fit_lane_width_plane")


def test_scale_priority_fallback_matrix():
    lateral_car = make_track(n=20, speed_px_s=100.0, width_px=45.0, vtype="car")
    flat10 = (0.0, 10.0)      # a plane that is 10 px/m at every row

    # (a) S2 good beats everything
    r = compute_vehicle_speeds(lateral_car, stripe_ab=flat10, ped_plane_ab=(0.0, 999.0),
                               lane_scale_px_per_m=999.0)[0]
    assert r["scale_source"] == "stripe_ground_plane", r
    assert abs(r["median_speed_mps"] - 10.0) < 1e-9, r

    # (b) no S2 -> pedestrian-height plane beats both lane paths, and is reliable
    r = compute_vehicle_speeds(lateral_car, ped_plane_ab=flat10,
                               lane_plane_ab=(0.0, 999.0), lane_scale_px_per_m=999.0)[0]
    assert r["scale_source"] == "ped_height_plane", r
    assert abs(r["median_speed_mps"] - 10.0) < 1e-9, r
    assert r["reliable"] is True, r
    # the disagreement with the lane scale is recorded, not silently dropped
    assert abs(r["scale_cross_check_ratio"] - 99.9) < 1e-6, r

    # (c) no S2, no pedestrians -> lane PLANE (y-aware), not the frame-bottom constant
    r = compute_vehicle_speeds(lateral_car, lane_plane_ab=flat10,
                               lane_scale_px_per_m=999.0)[0]
    assert r["scale_source"] == "lane_width_plane", r
    assert abs(r["median_speed_mps"] - 10.0) < 1e-9, r
    assert r["reliable"] is True, r            # no ped scale exists -> nothing contradicts it

    # (d) only the legacy global constant -> unchanged behaviour, still reliable
    r = compute_vehicle_speeds(lateral_car, lane_scale_px_per_m=10.0)[0]
    assert r["scale_source"] == "lane_width", r
    assert abs(r["median_speed_mps"] - 10.0) < 1e-9, r
    assert r["reliable"] is True, r

    # (e) neither -> car-length prior for a near-lateral car: 45 px / 4.5 m = 10 px/m,
    # correct speed but NEVER reliable
    r = compute_vehicle_speeds(lateral_car)[0]
    assert r["scale_source"] == "length_prior", r
    assert abs(r["median_speed_mps"] - 10.0) < 1e-9, r
    assert r["reliable"] is False, r

    # (f) neither + non-lateral (vertical) track -> no scale, NaN speeds
    vertical = make_track(n=20, speed_px_s=100.0, axis="y", x0=400.0)
    r = compute_vehicle_speeds(vertical)[0]
    assert r["scale_source"] == "none", r
    assert r["n_valid_steps"] == 0, r
    assert r["median_speed_mps"] is None, r
    assert r["running_speed_mps"] is None, r
    assert r["reliable"] is False, r

    # (g) length prior is cars/taxis only
    truck = make_track(n=20, speed_px_s=100.0, vtype="truck")
    r = compute_vehicle_speeds(truck)[0]
    assert r["scale_source"] == "none", r
    print("ok  test_scale_priority_fallback_matrix")


def test_lane_scale_rejected_when_pedestrians_disagree():
    """A lane scale that contradicts the pedestrian-height scale is never reliable.

    This is the audited failure: on all 11 cities the [V5] frame-bottom constant was
    4.3x - 7.4x the pedestrian-height scale at the vehicles' own image rows.
    """
    car = make_track(n=20, speed_px_s=100.0, width_px=45.0, vtype="car")
    # ped plane says 10 px/m; the lane plane claims 70 px/m -> ratio 7.0
    r = compute_vehicle_speeds(car, ped_plane_ab=(0.0, 10.0), lane_plane_ab=(0.0, 70.0))[0]
    assert r["scale_source"] == "ped_height_plane", r      # ped wins outright
    assert abs(r["scale_cross_check_ratio"] - 7.0) < 1e-6, r
    assert abs(r["median_speed_mps"] - 10.0) < 1e-9, r     # 10 m/s, not 1.43 m/s
    assert r["reliable"] is True, r

    # Same 7x disagreement, but with the lane scale forced to be the one in use:
    # it must be reported unreliable rather than passed off as a measurement.
    rows = compute_vehicle_speeds(car, ped_plane_ab=(0.0, 10.0), lane_plane_ab=(0.0, 70.0),
                                  stripe_ab=None)
    r2 = rows[0]
    assert r2["scale_source"] != "lane_width_plane", r2
    # and when the two DO agree the lane path stays usable
    r3 = compute_vehicle_speeds(car, lane_plane_ab=(0.0, 10.0),
                                lane_scale_px_per_m=11.0)[0]
    assert r3["scale_source"] == "lane_width_plane", r3
    assert r3["reliable"] is True, r3
    print("ok  test_lane_scale_rejected_when_pedestrians_disagree")


def test_running_speed_and_stopped_frac():
    """A vehicle that queues then departs: median mixes the two, running_speed does not."""
    rows = []
    x = 0.0
    for i in range(21):
        rows.append({"frame_id": 1 + 2 * i, "timestamp": i * DT, "track_id": 3,
                     "vtype": "car", "conf": 0.9,
                     "x1": x - 20, "y1": 440, "x2": x + 20, "y2": 500})
        x += (0.0 if i < 10 else 100.0) * DT      # stopped, then 10 m/s at 10 px/m
    r = compute_vehicle_speeds(pd.DataFrame(rows), stripe_ab=(0.0, 10.0),
                               smooth_window=1)[0]
    assert r["n_valid_steps"] == 20, r
    assert abs(r["stopped_frac"] - 0.5) < 1e-9, r
    assert abs(r["running_speed_mps"] - 10.0) < 1e-9, r
    assert r["median_speed_mps"] < r["running_speed_mps"], r
    print("ok  test_running_speed_and_stopped_frac")


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
    assert r["n_crosswalk_boxes"] == 1, r
    # no boxes -> crosswalk median NaN, everything mid-block, and the appended
    # count says WHY it is NaN (no band existed, as in 10 of the 11 audited cities)
    r2 = compute_vehicle_speeds(veh, stripe_ab=(0.0, 10.0), smooth_window=1)[0]
    assert r2["speed_at_crosswalk_mps"] is None, r2
    assert r2["n_crosswalk_boxes"] == 0, r2
    assert abs(r2["midblock_speed_mps"] - 7.5) < 1e-9, r2
    print("ok  test_crosswalk_band_split")


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
        # a clip where [E7] found nothing -> no band at all (the honest-NaN case)
        none = os.path.join(tmp, "none.csv")
        pd.DataFrame({"frame_id": [0, 1], "crosswalk_detected": ["No", "No"],
                      "crosswalk_boxes": ["[]", "[]"]}).to_csv(none, index=False)
        assert _load_crosswalk_boxes(none) == []
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
        assert df.iloc[0]["n_crosswalk_boxes"] == 0, df

        # S2 quality != good must be ignored -> falls through. With [B2] pedestrians
        # present the next source is the pedestrian-height plane.
        pd.DataFrame([{"a": 0.0, "b": 10.0, "n_samples": 5, "fit_residual_px": 9.0,
                       "stripe_period_m": 1.0, "median_scale_px_per_m": 10.0,
                       "quality": "poor"}]).to_csv(
            os.path.join(vdir, "[S2]scale_calibration.csv"), index=False)
        df = pd.read_csv(run_vehicle_speed(os.path.join("videos", "testvid.mp4")))
        assert df.iloc[0]["scale_source"] == "length_prior", df   # no [B2] yet

        # A real ground plane must get STEEPER downward (a > 0); a flat fit carries no
        # depth information and fit_ped_height_plane deliberately refuses it.
        ped_rows = []
        for tid in range(40):
            y = 500.0 + 4.0 * tid
            h = (0.05 * y - 5.0) * 1.7          # 20 px/m at y=500, the vehicle's row
            for k in range(8):
                ped_rows.append({"frame_id": k, "timestamp": k * DT, "track_id": tid,
                                 "x1": 100.0, "y1": y - h, "x2": 100.0 + h / 2.5, "y2": y})
        pd.DataFrame(ped_rows).to_csv(os.path.join(vdir, "[B2]dense_tracks.csv"), index=False)
        df = pd.read_csv(run_vehicle_speed(os.path.join("videos", "testvid.mp4")))
        assert df.iloc[0]["scale_source"] == "ped_height_plane", df
        # 100 px/s at 20 px/m = 5 m/s (the [S2] fixture above is now "poor" and ignored)
        assert abs(df.iloc[0]["median_speed_mps"] - 5.0) < 1e-6, df
        assert abs(df.iloc[0]["scale_px_per_m_median"] - 20.0) < 1e-6, df

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

        # corrupt/garbage companion CSVs must not crash the entry point
        vdir4 = os.path.join("analysis_results", "junkvid")
        os.makedirs(vdir4, exist_ok=True)
        make_track(n=20, speed_px_s=100.0).to_csv(
            os.path.join(vdir4, "[V7]vehicle_tracks.csv"), index=False)
        for name in ("[B3]ego_motion.csv", "[S2]scale_calibration.csv",
                     "[V5]lane_detection.csv", "[E7]crosswalk_detection.csv",
                     "[B2]dense_tracks.csv"):
            with open(os.path.join(vdir4, name), "w") as f:
                f.write("not,a,valid\n1,2\n")
        df4 = pd.read_csv(run_vehicle_speed(os.path.join("videos", "junkvid.mp4")))
        assert list(df4.columns) == OUTPUT_COLUMNS and len(df4) == 1, df4
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok  test_run_vehicle_speed_end_to_end_and_empty")


if __name__ == "__main__":
    test_exact_10_mps()
    test_ego_comp_invariance()
    test_ego_static_gate_drops_moving_steps()
    test_reliability_is_length_independent()
    test_local_pan_gate_still_bites()
    test_fit_ped_height_plane()
    test_fit_lane_width_plane()
    test_scale_priority_fallback_matrix()
    test_lane_scale_rejected_when_pedestrians_disagree()
    test_running_speed_and_stopped_frac()
    test_crosswalk_band_split()
    test_max_step_speed_filter()
    test_load_crosswalk_boxes()
    test_run_vehicle_speed_end_to_end_and_empty()
    print("\nAll vehicle_speed tests passed.")
