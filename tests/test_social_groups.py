"""Unit tests for modules/insights/social_groups.py (plain asserts, no video).

Run from the repo root:  python tests/test_social_groups.py
"""

import os
import sys
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from modules.insights.social_groups import (
    GROUPS_COLUMNS, CROSSINGS_COLUMNS,
    detect_group_edges, connected_components, build_group_rows,
    build_crossing_rows, run_social_groups,
)

SCALE = 20.0          # px per metre used by the synthetic fixtures
FPS = 15.0            # dense-track sampling rate
NATIVE_FPS = 60.0     # native video rate (frame ids)


def make_track(tid, t0=0.0, dur=10.0, x0=0.0, y0=400.0, vx=0.0, vy=0.0,
               h=40.0, w=14.0):
    """[B2]-shaped rows for one pedestrian with constant velocity (px/s)."""
    rows = []
    n = int(round(dur * FPS)) + 1
    for k in range(n):
        t = t0 + k / FPS
        fx = x0 + vx * (t - t0)
        fy = y0 + vy * (t - t0)
        rows.append({
            "frame_id": int(round(t * NATIVE_FPS)) + 1,
            "timestamp": round(t, 4),
            "track_id": tid,
            "x1": fx - w / 2, "y1": fy - h, "x2": fx + w / 2, "y2": fy,
        })
    return rows


def const_scale(value):
    return lambda y, ta, tb: np.full(len(np.asarray(y, dtype=float)), value)


def test_co_moving_pair():
    """Two tracks 1.0 m apart co-moving for 10 s -> exactly one 2-group."""
    df = pd.DataFrame(make_track(1, y0=400.0, vx=28.0)
                      + make_track(2, y0=420.0, vx=28.0))   # 20 px = 1.0 m gap
    edges = detect_group_edges(df, scale_fn=const_scale(SCALE))
    assert len(edges) == 1, f"expected 1 edge, got {edges}"
    e = edges[0]
    assert {e["a"], e["b"]} == {1, 2}
    assert abs(e["mean_gap_m"] - 1.0) < 0.01, e
    assert e["gap_std_m"] < 0.01, e
    assert e["mean_vel_cos"] > 0.99, e
    assert abs(e["co_duration_s"] - 10.0) < 0.2, e
    rows, comps = build_group_rows(edges)
    assert comps == [[1, 2]]
    assert len(rows) == 2 and all(r["n_members"] == 2 for r in rows)
    print("ok  test_co_moving_pair")


def test_passer_by_excluded():
    """A fast co-directional passer inside 1.5 m for only ~1 s is rejected
    by the sustain rule; the real pair still forms its group."""
    df = pd.DataFrame(
        make_track(1, y0=400.0, vx=28.0)
        + make_track(2, y0=420.0, vx=28.0)
        # 3: 0.5 m below A, 3x walking speed, starts 140 px behind ->
        # gap < 1.5 m only while |dx| < sqrt(30^2-10^2) = 28.3 px,
        # i.e. 56.6 px / 56 px/s ~ 1.0 s.
        + make_track(3, y0=410.0, x0=-140.0, vx=84.0))
    edges = detect_group_edges(df, scale_fn=const_scale(SCALE))
    pairs = {frozenset((e["a"], e["b"])) for e in edges}
    assert pairs == {frozenset((1, 2))}, f"passer-by not excluded: {pairs}"
    _rows, comps = build_group_rows(edges)
    assert comps == [[1, 2]]
    assert all(3 not in c for c in comps)
    print("ok  test_passer_by_excluded")


def test_chain_component():
    """A-B, B-C, C-D adjacency (1.2 m spacing) -> one 4-member component."""
    df = pd.DataFrame(
        make_track(1, y0=400.0, vx=28.0)
        + make_track(2, y0=424.0, vx=28.0)     # 1.2 m from 1
        + make_track(3, y0=448.0, vx=28.0)     # 1.2 m from 2, 2.4 m from 1
        + make_track(4, y0=472.0, vx=28.0))
    edges = detect_group_edges(df, scale_fn=const_scale(SCALE))
    pairs = {frozenset((e["a"], e["b"])) for e in edges}
    assert pairs == {frozenset((1, 2)), frozenset((2, 3)), frozenset((3, 4))}, pairs
    comps = connected_components(edges)
    assert comps == [[1, 2, 3, 4]], comps
    rows, _ = build_group_rows(edges)
    assert len(rows) == 4 and all(r["n_members"] == 4 for r in rows)
    print("ok  test_chain_component")


def test_staggered_lags():
    """Scripted started_frames -> exact leader/follower lags at known fps."""
    c3 = pd.DataFrame([
        {"track_id": 1, "crossed": True, "started_frame": 100.0,
         "ended_frame": 400.0, "movement_type": "road-to-road"},
        {"track_id": 2, "crossed": True, "started_frame": 130.0,
         "ended_frame": 430.0, "movement_type": "road-to-road"},
        {"track_id": 3, "crossed": True, "started_frame": 190.0,
         "ended_frame": 490.0, "movement_type": "road-to-road"},
        {"track_id": 4, "crossed": True, "started_frame": 250.0,
         "ended_frame": 550.0, "movement_type": "sidewalk-to-sidewalk"},
        {"track_id": 5, "crossed": False, "started_frame": None,
         "ended_frame": None, "movement_type": None},
    ])
    rows = build_crossing_rows([[1, 2, 3], [4, 5]], c3, fps=15.0)
    g1 = [r for r in rows if r["group_id"] == 1]
    assert len(g1) == 2
    assert all(r["leader_track_id"] == 1 and r["n_crossers"] == 3
               and r["n_members"] == 3 and r["movement_type_leader"] == "road-to-road"
               for r in g1)
    lags = {r["follower_track_id"]: r["follower_lag_s"] for r in g1}
    assert lags == {2: 2.0, 3: 6.0}, lags
    # single-crosser group still appears, with empty follower fields
    g2 = [r for r in rows if r["group_id"] == 2]
    assert len(g2) == 1
    assert g2[0]["n_crossers"] == 1 and g2[0]["leader_track_id"] == 4
    assert g2[0]["follower_track_id"] is None and g2[0]["follower_lag_s"] is None
    print("ok  test_staggered_lags")


def test_scale_invariance():
    """Doubling pixel geometry and scale(y) together -> identical grouping."""
    base = pd.DataFrame(make_track(1, y0=400.0, vx=28.0)
                        + make_track(2, y0=420.0, vx=28.0)
                        + make_track(3, y0=410.0, x0=-140.0, vx=84.0))
    doubled = base.copy()
    for c in ("x1", "y1", "x2", "y2"):
        doubled[c] = doubled[c] * 2.0
    e1 = detect_group_edges(base, scale_fn=const_scale(SCALE))
    e2 = detect_group_edges(doubled, scale_fn=const_scale(2 * SCALE))
    p1 = {frozenset((e["a"], e["b"])) for e in e1}
    p2 = {frozenset((e["a"], e["b"])) for e in e2}
    assert p1 == p2 == {frozenset((1, 2))}, (p1, p2)
    a, b = e1[0], e2[0]
    assert abs(a["mean_gap_m"] - b["mean_gap_m"]) < 1e-6
    assert abs(a["co_duration_s"] - b["co_duration_s"]) < 1e-6
    assert abs(a["mean_vel_cos"] - b["mean_vel_cos"]) < 1e-6
    print("ok  test_scale_invariance")


def test_end_to_end_and_empty_guard():
    """run_social_groups on real fixture files + header-only on missing input."""
    tmp = tempfile.mkdtemp(prefix="social_groups_test_")
    old_cwd = os.getcwd()
    try:
        os.chdir(tmp)

        # 1) no inputs at all -> valid header-only outputs, no crash
        g_csv, c_csv = run_social_groups("no_such_video.mp4")
        for path, cols in ((g_csv, GROUPS_COLUMNS), (c_csv, CROSSINGS_COLUMNS)):
            out = pd.read_csv(path)
            assert out.empty and list(out.columns) == cols, (path, list(out.columns))

        # 2) full fixture: pair (1,2) + 1 s passer-by (3); tracks 1,2 cross
        vdir = os.path.join("analysis_results", "vid1")
        os.makedirs(vdir, exist_ok=True)
        b2 = pd.DataFrame(make_track(1, y0=400.0, vx=28.0)
                          + make_track(2, y0=420.0, vx=28.0)
                          + make_track(3, y0=410.0, x0=-140.0, vx=84.0))
        b2.to_csv(os.path.join(vdir, "[B2]dense_tracks.csv"), index=False)
        pd.DataFrame([
            {"track_id": 1, "crossed": True, "started_frame": 300.0,
             "ended_frame": 500.0, "movement_type": "road-to-road"},
            {"track_id": 2, "crossed": True, "started_frame": 345.0,
             "ended_frame": 540.0, "movement_type": "road-to-road"},
            {"track_id": 3, "crossed": False, "started_frame": None,
             "ended_frame": None, "movement_type": None},
        ]).to_csv(os.path.join(vdir, "[C3]crossing_judge.csv"), index=False)
        pd.DataFrame([{"video_name": "vid1", "fps": 60.0, "width": 1280,
                       "height": 720, "total_frames": 3600}]
                     ).to_csv(os.path.join(vdir, "[B0]video_meta.csv"), index=False)

        g_csv, c_csv = run_social_groups("vid1.mp4")
        groups = pd.read_csv(g_csv)
        # no [S1]/[S2] -> height-prior scale = 40 px / 1.70 m = 23.5 px/m,
        # so the 20 px gap is 0.85 m -> a group; passer-by still excluded.
        assert sorted(groups["track_id"].tolist()) == [1, 2], groups
        assert (groups["n_members"] == 2).all()
        assert (groups["mean_gap_m"] - 0.85).abs().max() < 0.02, groups
        crossings = pd.read_csv(c_csv)
        assert len(crossings) == 1, crossings
        r = crossings.iloc[0]
        assert int(r["leader_track_id"]) == 1 and int(r["follower_track_id"]) == 2
        assert abs(r["follower_lag_s"] - 45.0 / 60.0) < 1e-6, r  # [B0] fps honoured
        assert int(r["n_crossers"]) == 2 and int(r["n_members"]) == 2
        assert r["movement_type_leader"] == "road-to-road"

        # 3) header-only [B2] -> header-only outputs
        vdir2 = os.path.join("analysis_results", "vid2")
        os.makedirs(vdir2, exist_ok=True)
        pd.DataFrame(columns=["frame_id", "timestamp", "track_id",
                              "x1", "y1", "x2", "y2"]
                     ).to_csv(os.path.join(vdir2, "[B2]dense_tracks.csv"), index=False)
        g_csv2, c_csv2 = run_social_groups("vid2.mp4")
        assert pd.read_csv(g_csv2).empty and pd.read_csv(c_csv2).empty
        print("ok  test_end_to_end_and_empty_guard")
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_co_moving_pair()
    test_passer_by_excluded()
    test_chain_component()
    test_staggered_lags()
    test_scale_invariance()
    test_end_to_end_and_empty_guard()
    print("ALL social_groups TESTS PASSED")
