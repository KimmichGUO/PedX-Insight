"""Tests for modules/insights/pet_conflicts.py ([I1] PET conflicts).

Plain asserts, no pytest. Run from the repo root:
    python tests/test_pet_conflicts.py

Fixtures (per the review spec):
  (a) pedestrian clears a ground cell 2.0 s before a vehicle, both at known
      constant velocities -> min_pet_s == 2.0 exactly;
  (b) disjoint paths -> single NaN-PET row;
  (c) linear camera drift injected into BOTH pixel tracks plus a matching
      synthetic [B3] -> PET identical to the static case (ego-comp invariance);
  (d) FAST local pan (50 px/s) -> row flagged camera_pan_ok=False, no PET.
Plus: missing [V7] -> header-only output (empty-input guard).

Camera-gate regressions (the [I1]-degenerate bug):
  (i) a long-video [B3] whose CUMULATIVE cam_x/cam_y has already random-walked to
      ~30,000 px before the crossing, but which drifts slowly (5 px/s) during it
      -> camera_pan_ok stays True and the PET is still 2.0 s. The old gate
      (max |cam - cam[0]| <= 200 px) declared every full-length video corrupted
      and emitted 0 conflicts across all 11 new cities;
  (j) local_camera_stats() directly: a constant offset added to the whole
      cumulative series changes nothing, and the rate does not grow with the
      window length;
  (k) forward-driving camera (all bboxes growing together) -> ego_regime
      'forward_motion' -> gated, even though step_px is ~0.

Speed-gating fixtures (severe_timing shifts the vehicle 1 s earlier -> PET 1.0 s):
  (f) moving [V8] row (median 5.0 m/s) -> stays 'severe', speed_gated=True;
  (g) same geometry, stationary [V8] row (median 0.4 m/s) -> 'queued';
  (h) no [V8] -> old ungated 'severe', speed_gated=False, NaN veh speed.

Geometry (scale = 10 px/m everywhere via an [S2] with a=0, b=10, quality=good):
  Pedestrian track 1: y2=105 (cell row 10), x = 10*t px for t in [0, 3.0] at 10 Hz;
    occupies cell (2, 10) for t in [2.0, 2.9] -> exits at t=2.9. bbox height 17 px
    -> implied stature 1.7 m (passes the [S2] sanity guard).
  Vehicle track 7: cx=25 (cell col 2), y2 = 10*t + 51 px for t in [4.0, 6.0] at
    10 Hz; enters cell row 10 (y2 >= 100) exactly at t=4.9 -> PET = 4.9 - 2.9 = 2.0,
    pedestrian first.
"""

import math
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from modules.insights.pet_conflicts import (OUTPUT_COLUMNS, local_camera_stats,
                                            pet_from_tracks, run_pet_conflicts)

FPS = 10.0


def _build_case(tmp, drift=None, far_vehicle=False, severe_timing=False, v8=None,
                b3_prehistory_s=0.0, expanding=False):
    """Write synthetic [B2]/[V7]/[C3]/[S2] (and [B3] when drift, [V8] when v8) into tmp.

    severe_timing: shift the vehicle 1 s earlier (y2 = 10*t + 61, t in [3, 5]) so it
    enters the shared cell at t=3.9 -> PET = 3.9 - 2.9 = 1.0 s (< 1.5 s, severe).
    v8: None (no [V8] file) | 'moving' (median 5.0 m/s) | 'stationary' (0.4 m/s).
    b3_prehistory_s: prepend this many seconds of the SAME slow drift at negative
        timestamps, so [B3]'s cumulative cam_x/cam_y is already huge when the
        crossing starts (what a full-length video looks like). The crossing tracks
        are untouched: only the camera's integrated history grows.
    expanding: add 4 extra [B2] tracks whose bbox heights all grow together, the
        signature of a forward-driving camera.
    """
    if drift is None:
        cam_x = lambda t: 0.0
        cam_y = lambda t: 0.0
    else:
        cam_x, cam_y = drift

    # [B2] pedestrian: 10 Hz, t in [0, 3.0]
    b2 = []
    for i in range(31):
        t = i / 10.0
        x = 10.0 * t + cam_x(t)
        y2 = 105.0 + cam_y(t)
        b2.append({"frame_id": i + 1, "timestamp": t, "track_id": 1,
                   "x1": x - 5.0, "y1": y2 - 17.0, "x2": x + 5.0, "y2": y2})
    if expanding:
        for k in range(4):
            for i in range(31):
                t = i / 10.0
                h = 20.0 * (1.0 + 0.25 * t)      # ~0.25/s relative growth >> 0.05/s
                y2 = 300.0 + 40.0 * k
                b2.append({"frame_id": i + 1, "timestamp": t, "track_id": 100 + k,
                           "x1": 200.0 + 40.0 * k, "y1": y2 - h,
                           "x2": 200.0 + 40.0 * k + h / 2.0, "y2": y2})
    b2_path = os.path.join(tmp, "[B2]dense_tracks.csv")
    pd.DataFrame(b2).to_csv(b2_path, index=False)

    # [V7] vehicle: 10 Hz, t in [4.0, 6.0] (or [3.0, 5.0] when severe_timing)
    v7 = []
    i0 = 30 if severe_timing else 40
    for i in range(i0, i0 + 21):
        t = i / 10.0
        if far_vehicle:
            cx, y2 = 500.0 + cam_x(t), 500.0 + cam_y(t)
        elif severe_timing:
            cx, y2 = 25.0 + cam_x(t), 10.0 * t + 61.0 + cam_y(t)
        else:
            cx, y2 = 25.0 + cam_x(t), 10.0 * t + 51.0 + cam_y(t)
        v7.append({"frame_id": i + 1, "timestamp": t, "track_id": 7, "vtype": "car",
                   "conf": 0.9, "x1": cx - 10.0, "y1": y2 - 30.0, "x2": cx + 10.0, "y2": y2})
    v7_path = os.path.join(tmp, "[V7]vehicle_tracks.csv")
    pd.DataFrame(v7).to_csv(v7_path, index=False)

    c3_path = os.path.join(tmp, "[C3]crossing_judge.csv")
    pd.DataFrame([{"track_id": 1, "crossed": True, "started_frame": 1,
                   "ended_frame": 31, "movement_type": "cross"}]).to_csv(c3_path, index=False)

    s2_path = os.path.join(tmp, "[S2]scale_calibration.csv")
    pd.DataFrame([{"a": 0.0, "b": 10.0, "n_samples": 50, "fit_residual_px": 0.0,
                   "stripe_period_m": 1.0, "median_scale_px_per_m": 10.0,
                   "quality": "good"}]).to_csv(s2_path, index=False)

    b3_path = os.path.join(tmp, "[B3]ego_motion.csv")
    if drift is not None:
        i_start = -int(round(b3_prehistory_s * 10))
        b3, prev = [], (cam_x((i_start - 1) / 10.0), cam_y((i_start - 1) / 10.0))
        for i in range(i_start, 61):
            t = i / 10.0
            cx, cy = cam_x(t), cam_y(t)
            b3.append({"frame_id": i + 1, "timestamp": t, "cam_x": cx, "cam_y": cy,
                       "step_px": math.hypot(cx - prev[0], cy - prev[1]),
                       "n_bg_points": 200})
            prev = (cx, cy)
        pd.DataFrame(b3).to_csv(b3_path, index=False)

    v8_path = os.path.join(tmp, "[V8]vehicle_speed.csv")
    if v8 is not None:
        median = 5.0 if v8 == "moving" else 0.4
        pd.DataFrame([{"track_id": 7, "veh_type": "car", "n_valid_steps": 20,
                       "median_speed_mps": median, "p85_speed_mps": median,
                       "max_speed_mps": median, "speed_at_crosswalk_mps": None,
                       "midblock_speed_mps": median, "scale_source": "lane_width",
                       "camera_moving": False, "reliable": True}]).to_csv(v8_path, index=False)
    else:
        v8_path = os.path.join(tmp, "missing_v8.csv")

    out_path = os.path.join(tmp, "[I1]pet_conflicts.csv")
    return {"tracks_csv": b2_path, "vehicle_csv": v7_path, "crossing_csv": c3_path,
            "ego_csv": b3_path, "scale_csv": s2_path,
            "speed_csv": os.path.join(tmp, "missing_s1.csv"),
            "video_meta_csv": os.path.join(tmp, "missing_b0.csv"),
            "vehicle_speed_csv": v8_path,
            "output_csv": out_path}


def _run(tmp, **kw):
    paths = _build_case(tmp, **kw)
    run_pet_conflicts(os.path.join(tmp, "TestCity1_fixture.mp4"), fps=FPS, **paths)
    return pd.read_csv(paths["output_csv"])


def test_exact_pet():
    tmp = tempfile.mkdtemp(prefix="pet_a_")
    try:
        out = _run(tmp)
        assert list(out.columns) == OUTPUT_COLUMNS, out.columns
        assert len(out) == 1, out
        r = out.iloc[0]
        assert r["track_id"] == 1
        assert r["veh_track_id"] == 7
        assert r["veh_type"] == "car"
        assert r["min_pet_s"] == 2.0, r["min_pet_s"]
        assert r["first_agent"] == "ped"
        assert r["n_shared_cells"] == 1, r["n_shared_cells"]
        assert r["severity"] == "moderate"          # 1.5 <= 2.0 < 3.0
        assert r["scale_source"] == "stripe_ground_plane"
        assert bool(r["camera_pan_ok"]) is True
        assert bool(r["reliable"]) is True
        print("test_exact_pet OK (min_pet_s == 2.0)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_disjoint_paths_nan():
    tmp = tempfile.mkdtemp(prefix="pet_b_")
    try:
        out = _run(tmp, far_vehicle=True)
        assert len(out) == 1, out
        r = out.iloc[0]
        assert r["track_id"] == 1
        assert pd.isna(r["veh_track_id"])
        assert pd.isna(r["min_pet_s"])
        assert r["severity"] == "none"
        assert r["n_shared_cells"] == 0
        print("test_disjoint_paths_nan OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_ego_comp_invariance():
    # linear drift: median step_px = hypot(1, 0.5) ~= 1.118 >= 1.0 (moving gate on),
    # cumulative displacement hypot(60, 30) ~= 67 px <= 200 (pan ok)
    tmp_a = tempfile.mkdtemp(prefix="pet_c0_")
    tmp_c = tempfile.mkdtemp(prefix="pet_c1_")
    try:
        static = _run(tmp_a)
        drifted = _run(tmp_c, drift=(lambda t: 10.0 * t, lambda t: 5.0 * t))
        assert len(drifted) == 1, drifted
        ra, rc = static.iloc[0], drifted.iloc[0]
        assert rc["min_pet_s"] == ra["min_pet_s"] == 2.0, (ra["min_pet_s"], rc["min_pet_s"])
        assert rc["first_agent"] == ra["first_agent"] == "ped"
        assert rc["n_shared_cells"] == ra["n_shared_cells"] == 1
        assert rc["cell_y_px"] == ra["cell_y_px"]
        assert bool(rc["camera_pan_ok"]) is True
        print("test_ego_comp_invariance OK (drifted PET == static PET == 2.0)")
    finally:
        shutil.rmtree(tmp_a, ignore_errors=True)
        shutil.rmtree(tmp_c, ignore_errors=True)


def test_fast_local_pan_flagged():
    # cam_x = 50*t -> a LOCAL translation rate of 50 px/s, above the 30 px/s gate
    tmp = tempfile.mkdtemp(prefix="pet_d_")
    try:
        out = _run(tmp, drift=(lambda t: 50.0 * t, lambda t: 0.0))
        assert len(out) == 1, out
        r = out.iloc[0]
        assert bool(r["camera_pan_ok"]) is False
        assert pd.isna(r["min_pet_s"])
        assert bool(r["reliable"]) is False
        assert r["local_step_rate_px_s"] > 30.0, r["local_step_rate_px_s"]
        print("test_fast_local_pan_flagged OK (50 px/s local rate -> gated)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_long_video_cumulative_pan_not_flagged():
    """REGRESSION: huge CUMULATIVE cam_x must not gate a locally-calm crossing.

    100 minutes of 5 px/s drift before the crossing puts [B3]'s integrated cam_x at
    ~-30,000 px, exactly like the real 11-city archives (22k-89k px). The camera is
    still only moving 5 px/s while the pedestrian crosses, so the PET must survive
    untouched. The old max(|cam - cam[0]|) <= 200 px test rejected this outright.
    """
    tmp = tempfile.mkdtemp(prefix="pet_i_")
    try:
        out = _run(tmp, drift=(lambda t: 5.0 * t, lambda t: 0.0), b3_prehistory_s=6000.0)
        b3 = pd.read_csv(os.path.join(tmp, "[B3]ego_motion.csv"))
        span = float(b3["cam_x"].max() - b3["cam_x"].min())
        assert span > 20000.0, span            # the file really is a long random walk
        assert len(out) == 1, out
        r = out.iloc[0]
        assert bool(r["camera_pan_ok"]) is True, r.to_dict()
        assert r["min_pet_s"] == 2.0, r["min_pet_s"]
        assert r["first_agent"] == "ped"
        assert r["local_step_rate_px_s"] <= 30.0, r["local_step_rate_px_s"]
        assert r["local_pan_px"] < 200.0, r["local_pan_px"]   # local excursion is tiny
        assert bool(r["reliable"]) is True
        print(f"test_long_video_cumulative_pan_not_flagged OK "
              f"(cam_x spans {span:.0f} px, local rate "
              f"{r['local_step_rate_px_s']:.1f} px/s, PET 2.0)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_forward_motion_gated():
    # every bbox grows together -> forward-driving camera; [B3]'s median translation
    # cannot remove radial flow, so the crossing must be gated on expansion alone
    tmp = tempfile.mkdtemp(prefix="pet_k_")
    try:
        out = _run(tmp, expanding=True)
        assert len(out) == 1, out
        r = out.iloc[0]
        assert r["ego_regime"] == "forward_motion", r["ego_regime"]
        assert r["ego_expansion"] > 0.05, r["ego_expansion"]
        assert bool(r["camera_pan_ok"]) is False
        assert pd.isna(r["min_pet_s"])
        print("test_forward_motion_gated OK (expansion %.3f/s -> gated)" % r["ego_expansion"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_local_camera_stats_direct():
    """The gate must read local DIFFERENCES, not the absolute cumulative position."""
    import numpy as np
    t = np.arange(0.0, 100.0, 0.1)
    rate = np.full(t.shape, 20.0)                      # 20 px/s everywhere
    for offset in (0.0, 50000.0):                      # cumulative random-walk offset
        cam = (t, offset + 20.0 * t, offset - 20.0 * t, rate)
        r_short, p_short = local_camera_stats(cam, 10.0, 12.0)
        r_long, p_long = local_camera_stats(cam, 10.0, 90.0)
        assert r_short == r_long == 20.0, (r_short, r_long)   # rate: length-invariant
        assert abs(p_short - math.hypot(40.0, 40.0)) < 1.0, p_short
        assert p_long > p_short                                # excursion does grow
    # no [B3] and an empty window -> NaN, which must never gate
    assert all(v != v for v in local_camera_stats(None, 0.0, 1.0))
    print("test_local_camera_stats_direct OK")


def test_missing_v7_header_only():
    tmp = tempfile.mkdtemp(prefix="pet_e_")
    try:
        paths = _build_case(tmp)
        os.remove(paths["vehicle_csv"])
        run_pet_conflicts(os.path.join(tmp, "TestCity1_fixture.mp4"), fps=FPS, **paths)
        out = pd.read_csv(paths["output_csv"])
        assert list(out.columns) == OUTPUT_COLUMNS
        assert out.empty
        print("test_missing_v7_header_only OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_speed_gate_moving_stays_severe():
    tmp = tempfile.mkdtemp(prefix="pet_f_")
    try:
        out = _run(tmp, severe_timing=True, v8="moving")
        assert list(out.columns) == OUTPUT_COLUMNS, out.columns
        assert len(out) == 1, out
        r = out.iloc[0]
        assert r["min_pet_s"] == 1.0, r["min_pet_s"]
        assert r["severity"] == "severe", r["severity"]
        assert bool(r["speed_gated"]) is True
        assert r["veh_median_speed_mps"] == 5.0, r["veh_median_speed_mps"]
        print("test_speed_gate_moving_stays_severe OK (PET 1.0, moving -> severe)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_speed_gate_stationary_queued():
    # identical geometry to the severe case; only the [V8] speed differs
    tmp = tempfile.mkdtemp(prefix="pet_g_")
    try:
        out = _run(tmp, severe_timing=True, v8="stationary")
        assert len(out) == 1, out
        r = out.iloc[0]
        assert r["min_pet_s"] == 1.0, r["min_pet_s"]
        assert r["severity"] == "queued", r["severity"]
        assert bool(r["speed_gated"]) is True
        assert r["veh_median_speed_mps"] == 0.4, r["veh_median_speed_mps"]
        print("test_speed_gate_stationary_queued OK (PET 1.0, stationary -> queued)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_missing_v8_old_behavior():
    tmp = tempfile.mkdtemp(prefix="pet_h_")
    try:
        out = _run(tmp, severe_timing=True)  # v8=None -> no [V8] file
        assert len(out) == 1, out
        r = out.iloc[0]
        assert r["min_pet_s"] == 1.0, r["min_pet_s"]
        assert r["severity"] == "severe", r["severity"]        # OLD ungated behavior
        assert bool(r["speed_gated"]) is False
        assert pd.isna(r["veh_median_speed_mps"])
        print("test_missing_v8_old_behavior OK (no [V8] -> ungated severe)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_gated_severity_direct():
    # speed_at_crosswalk overrides a stationary median; 'none' never becomes 'queued'
    from modules.insights.pet_conflicts import _gated_severity
    speeds = {7: (0.2, 3.0),   # queued median but moving at the crosswalk
              8: (0.2, 0.2),   # stationary
              9: (None, None)}  # [V8] row with NaN speeds
    assert _gated_severity(1.0, 1.5, 3.0, speeds, 7) == ("severe", 0.2, True)
    assert _gated_severity(1.0, 1.5, 3.0, speeds, 8) == ("queued", 0.2, True)
    assert _gated_severity(2.0, 1.5, 3.0, speeds, 8) == ("queued", 0.2, True)
    assert _gated_severity(5.0, 1.5, 3.0, speeds, 8) == ("none", 0.2, True)
    assert _gated_severity(1.0, 1.5, 3.0, speeds, 9) == ("severe", None, False)
    assert _gated_severity(1.0, 1.5, 3.0, speeds, 99) == ("severe", None, False)
    assert _gated_severity(1.0, 1.5, 3.0, None, 7) == ("severe", None, False)
    print("test_gated_severity_direct OK")


def test_core_direct():
    # pure-core sanity: overlap -> PET 0 / severe path; empty vehicle df -> no rows
    ped = pd.DataFrame({"t": [0.0, 0.1, 0.2], "x": [25.0, 25.0, 25.0],
                        "y": [105.0, 105.0, 105.0]})
    veh = pd.DataFrame({"t": [0.1, 0.2, 0.3], "x": [25.0, 25.0, 25.0],
                        "y": [105.0, 105.0, 105.0],
                        "veh_track_id": [9, 9, 9], "veh_type": ["car"] * 3})
    rows = pet_from_tracks(ped, veh, lambda y: 10.0)
    assert len(rows) == 1 and rows[0]["min_pet_s"] == 0.0 and rows[0]["first_agent"] == "ped"
    assert pet_from_tracks(ped, veh.iloc[0:0], lambda y: 10.0) == []
    print("test_core_direct OK")


if __name__ == "__main__":
    test_exact_pet()
    test_disjoint_paths_nan()
    test_ego_comp_invariance()
    test_fast_local_pan_flagged()
    test_long_video_cumulative_pan_not_flagged()
    test_forward_motion_gated()
    test_local_camera_stats_direct()
    test_missing_v7_header_only()
    test_speed_gate_moving_stays_severe()
    test_speed_gate_stationary_queued()
    test_missing_v8_old_behavior()
    test_gated_severity_direct()
    test_core_direct()
    print("ALL PET TESTS PASSED")
