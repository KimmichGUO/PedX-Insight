"""Tests for the occlusion-truncation guard in modules.speed.speed_estimation ([S1]).

Plain asserts, no pytest. Run from the repo root (or anywhere):
    python tests/test_speed_reliability.py

A pedestrian whose lower body is occluded gets a SHORT bbox -> inflated px/m
scale -> inflated speed. A standing/walking adult's bbox aspect (h/w) is
~1.6-4.5; occlusion-truncated or merged boxes fall outside. [S1] now computes
median_bbox_aspect per track and the reliable gate additionally requires
1.4 <= median_aspect <= 5.0. Covers:

  1. squat track (h/w ~1.0, lower body occluded)  -> reliable False
  2. normal track (h/w ~2.6)                      -> reliable True
  3. regression: dense 1.0 m/s fixture still measures ~1.0 m/s, stays reliable
  4. degenerate width (w <= 0) guard              -> no crash, reliable False
  5. gate boundaries (1.4 / 5.0 in, outside out)
  6. output column order: median_bbox_aspect appended before 'traj_source'
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from modules.speed.speed_estimation import OUTPUT_COLUMNS, run_speed_estimation

FPS = 15.0
DT = 1.0 / FPS


def make_ped_track(track_id=1, n=20, x0=300.0, speed_px_s=100.0, y2=600.0,
                   h_px=170.0, w_px=65.0):
    """Dense [B2]-shaped constant-velocity track sampled at 15 Hz."""
    rows = []
    for i in range(n):
        cx = x0 + speed_px_s * DT * i
        rows.append({
            "frame_id": 1 + 2 * i,
            "timestamp": i * DT,
            "track_id": track_id,
            "x1": cx - w_px / 2.0, "y1": y2 - h_px,
            "x2": cx + w_px / 2.0, "y2": y2,
        })
    return pd.DataFrame(rows)


def run_fixture(tracks, video_name="testvid"):
    """Write tracks as [B2]dense_tracks.csv in a temp per-video layout and run [S1]."""
    tmp = tempfile.mkdtemp(prefix="s1_aspect_")
    old_cwd = os.getcwd()
    try:
        os.chdir(tmp)
        vdir = os.path.join("analysis_results", video_name)
        os.makedirs(vdir, exist_ok=True)
        tracks.to_csv(os.path.join(vdir, "[B2]dense_tracks.csv"), index=False)
        out = run_speed_estimation(os.path.join("videos", f"{video_name}.mp4"),
                                   assumed_height_m=1.70)
        return pd.read_csv(out)
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(tmp, ignore_errors=True)


def test_squat_track_not_reliable():
    """Lower-body-occluded box (h/w ~1.0) fails the aspect gate -> reliable False."""
    df = run_fixture(make_ped_track(h_px=100.0, w_px=100.0))
    assert len(df) == 1, df
    r = df.iloc[0]
    assert abs(r["median_bbox_aspect"] - 1.0) < 1e-9, r["median_bbox_aspect"]
    # every other gate input is healthy (dense dt, >=3 steps, tall enough, stable
    # height), so the False verdict is attributable to the aspect guard alone
    assert r["n_valid_steps"] >= 3, r
    assert r["median_bbox_h_px"] >= 40, r
    assert r["height_cv"] < 0.35, r
    assert bool(r["reliable"]) is False, r
    print("ok  test_squat_track_not_reliable")


def test_normal_track_reliable():
    """Full-body walking box (h/w ~2.6) passes the gate -> reliable True."""
    df = run_fixture(make_ped_track(h_px=170.0, w_px=65.0))
    assert len(df) == 1, df
    r = df.iloc[0]
    # column is rounded to 3 decimals -> tolerance 5e-4
    assert abs(r["median_bbox_aspect"] - 170.0 / 65.0) < 5e-4, r["median_bbox_aspect"]
    assert bool(r["reliable"]) is True, r
    print("ok  test_normal_track_reliable")


def test_regression_dense_1mps_still_1mps_and_reliable():
    """Speed math untouched: 100 px/s at 170 px / 1.70 m (100 px/m) -> ~1.0 m/s."""
    df = run_fixture(make_ped_track(h_px=170.0, w_px=65.0, speed_px_s=100.0))
    r = df.iloc[0]
    assert abs(r["walking_speed_mps"] - 1.0) < 1e-3, r["walking_speed_mps"]
    assert abs(r["mean_speed_mps"] - 1.0) < 1e-3, r["mean_speed_mps"]
    assert abs(r["scale_px_per_m_median"] - 100.0) < 1e-6, r["scale_px_per_m_median"]
    assert bool(r["is_running"]) is False, r
    assert bool(r["reliable"]) is True, r
    print("ok  test_regression_dense_1mps_still_1mps_and_reliable")


def test_zero_width_guard():
    """Degenerate boxes (w <= 0) are excluded from the median; all-degenerate track
    has no computable aspect -> NaN column, reliable False, and no crash."""
    df = run_fixture(make_ped_track(h_px=170.0, w_px=0.0))
    assert len(df) == 1, df
    r = df.iloc[0]
    assert pd.isna(r["median_bbox_aspect"]), r["median_bbox_aspect"]
    assert bool(r["reliable"]) is False, r

    # mixed: one degenerate row among normal ones must not poison the median
    tr = make_ped_track(h_px=170.0, w_px=65.0)
    tr.loc[tr.index[0], "x1"] = tr.loc[tr.index[0], "x2"]  # w = 0 on first row
    r = run_fixture(tr).iloc[0]
    assert abs(r["median_bbox_aspect"] - 170.0 / 65.0) < 5e-4, r["median_bbox_aspect"]
    assert bool(r["reliable"]) is True, r
    print("ok  test_zero_width_guard")


def test_gate_boundaries():
    """1.4 and 5.0 are inclusive; just outside fails."""
    cases = [
        (140.0, 100.0, True),    # aspect 1.4  -> in
        (500.0, 100.0, True),    # aspect 5.0  -> in
        (139.0, 100.0, False),   # aspect 1.39 -> out (truncated/merged)
        (505.0, 100.0, False),   # aspect 5.05 -> out (sliver/fragment)
    ]
    for h, w, expect in cases:
        r = run_fixture(make_ped_track(h_px=h, w_px=w)).iloc[0]
        assert bool(r["reliable"]) is expect, (h, w, r["median_bbox_aspect"], r["reliable"])
    print("ok  test_gate_boundaries")


def test_output_column_order():
    """Column-order contract: the historical prefix keeps its order, median_bbox_aspect
    stays after camera_moving and before traj_source, and reliable stays last. Newer
    columns (e.g. the ego gate's ego_regime/ego_static_frac) may be appended in between —
    the module's contract permits ADDING columns, only renaming/removing is forbidden."""
    df = run_fixture(make_ped_track())
    assert list(df.columns) == OUTPUT_COLUMNS, df.columns
    i = OUTPUT_COLUMNS.index("median_bbox_aspect")
    assert OUTPUT_COLUMNS.index("camera_moving") < i, OUTPUT_COLUMNS
    assert i < OUTPUT_COLUMNS.index("traj_source"), OUTPUT_COLUMNS
    assert OUTPUT_COLUMNS[-1] == "reliable", OUTPUT_COLUMNS
    for col in ("track_id", "walking_speed_mps", "crossing_speed_mps", "reliable"):
        assert col in OUTPUT_COLUMNS, col
    print("ok  test_output_column_order")


if __name__ == "__main__":
    test_squat_track_not_reliable()
    test_normal_track_reliable()
    test_regression_dense_1mps_still_1mps_and_reliable()
    test_zero_width_guard()
    test_gate_boundaries()
    test_output_column_order()
    print("\nAll speed_reliability tests passed.")
