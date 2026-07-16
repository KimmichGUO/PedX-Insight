"""Tests for modules/insights/signal_timing.py ([P10] signal-phase micro-timing).

Plain asserts, no pytest. Run from the repo root (or anywhere):
    python tests/test_signal_timing.py
"""

import math
import os
import sys
import tempfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd

from modules.insights.signal_timing import (
    OUTPUT_COLUMNS,
    PHASE_CONVENTION,
    build_sustained_segments,
    compute_signal_timing_rows,
    extract_color_samples,
    run_signal_timing,
)

FPS = 30.0


def _e2(frame_color_pairs):
    """Build an [E2]-shaped dataframe from (frame_id, color) pairs."""
    return pd.DataFrame({
        "frame_id": [f for f, _ in frame_color_pairs],
        "main_light_color": [c for _, c in frame_color_pairs],
        "other_lights": ["None"] * len(frame_color_pairs),
    })


def _e2_1hz(colors_per_second, fps=FPS, offset_frames=0):
    """1 Hz [E2] samples: colors_per_second[i] sampled at frame i*fps+offset."""
    return _e2([(int(i * fps) + offset_frames, c)
                for i, c in enumerate(colors_per_second)])


def _c3(rows):
    """rows: list of (track_id, crossed, started_frame, ended_frame)."""
    return pd.DataFrame(
        [{"track_id": t, "crossed": cr, "started_frame": sf,
          "ended_frame": ef, "movement_type": "walk"}
         for t, cr, sf, ef in rows],
        columns=["track_id", "crossed", "started_frame", "ended_frame",
                 "movement_type"],
    )


def _isnan(x):
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return False


def test_clean_phases_exact_deltas():
    # Vehicle signal at 1 Hz: red 0-9 s, green 10-19 s, red 20-29 s.
    colors = ["red"] * 10 + ["green"] * 10 + ["red"] * 10
    e2 = _e2_1hz(colors)

    samples = extract_color_samples(e2, FPS)
    segments, transitions = build_sustained_segments(samples)
    assert len(segments) == 3, segments
    assert [(round(t, 6), a, b) for t, a, b in transitions] == [
        (10.0, "red", "green"),
        (20.0, "green", "red"),
    ], transitions

    c3 = _c3([
        # A: starts 1 s AFTER vehicle green->red (ped-favorable): latency 1 s.
        (1, True, 630, 750),    # t_start=21, t_end=25
        # B: starts 1 s BEFORE green->red: anticipatory.
        (2, True, 570, 690),    # t_start=19, t_end=23
        # C: starts 1 s after red->green (vehicle green = ped-hostile).
        (3, True, 330, 420),    # t_start=11, t_end=14
        # D: not crossed -> no row.
        (4, False, np.nan, np.nan),
    ])

    rows = compute_signal_timing_rows(c3, e2, FPS)
    assert len(rows) == 3, rows
    by_id = {r["track_id"]: r for r in rows}

    a = by_id[1]
    assert a["nearest_transition"] == "green_to_red"
    assert abs(a["delta_start_s"] - 1.0) < 1e-9
    assert a["anticipatory"] is False
    assert abs(a["startup_latency_s"] - 1.0) < 1e-9
    # Window [21, 25] does not overlap green [10, 20).
    assert abs(a["red_exposure_s"] - 0.0) < 1e-9
    assert a["n_transitions_in_video"] == 2
    assert a["light_available"] is True
    assert a["phase_convention"] == PHASE_CONVENTION

    b = by_id[2]
    assert b["nearest_transition"] == "green_to_red"
    assert abs(b["delta_start_s"] - (-1.0)) < 1e-9
    assert b["anticipatory"] is True
    assert _isnan(b["startup_latency_s"])
    # Window [19, 23] overlaps green [10, 20) for exactly 1 s.
    assert abs(b["red_exposure_s"] - 1.0) < 1e-9

    c = by_id[3]
    assert c["nearest_transition"] == "red_to_green"
    assert abs(c["delta_start_s"] - 1.0) < 1e-9
    # Transition into vehicle-green is not pedestrian-favorable.
    assert c["anticipatory"] is False
    assert _isnan(c["startup_latency_s"])
    # Window [11, 14] fully inside green -> 3 s of red-clearance exposure.
    assert abs(c["red_exposure_s"] - 3.0) < 1e-9

    print("test_clean_phases_exact_deltas OK")


def test_one_sample_flicker_rejected():
    # red 0-9 s, ONE green sample at 10 s, red 11-29 s.
    colors = ["red"] * 10 + ["green"] + ["red"] * 19
    e2 = _e2_1hz(colors)

    samples = extract_color_samples(e2, FPS)
    segments, transitions = build_sustained_segments(samples)
    # The 1 s green run is below the 3 s sustain threshold: no transitions,
    # and the two red runs merge into one sustained red state.
    assert transitions == [], transitions
    assert len(segments) == 1 and segments[0][2] == "red", segments

    c3 = _c3([(1, True, 630, 750)])  # t_start=21
    rows = compute_signal_timing_rows(c3, e2, FPS)
    assert len(rows) == 1
    r = rows[0]
    assert r["nearest_transition"] == "none"
    assert _isnan(r["delta_start_s"])
    assert _isnan(r["anticipatory"])
    assert _isnan(r["startup_latency_s"])
    assert r["n_transitions_in_video"] == 0
    assert r["light_available"] is True
    # No green segment survives -> zero red exposure.
    assert abs(r["red_exposure_s"] - 0.0) < 1e-9
    print("test_one_sample_flicker_rejected OK")


def test_all_none_e2():
    e2 = _e2_1hz(["None"] * 30)
    c3 = _c3([(1, True, 630, 750), (2, True, 300, 450)])
    rows = compute_signal_timing_rows(c3, e2, FPS)
    assert len(rows) == 2
    for r in rows:
        assert r["light_available"] is False
        assert r["nearest_transition"] == "none"
        assert _isnan(r["delta_start_s"])
        assert _isnan(r["anticipatory"])
        assert _isnan(r["startup_latency_s"])
        assert _isnan(r["red_exposure_s"])
        assert r["n_transitions_in_video"] == 0
    print("test_all_none_e2 OK")


def test_sampling_offset_invariance():
    # Same underlying phase plan (vehicle red until 10 s, then green until
    # 20 s, then red), sampled at 1 Hz with different grid offsets. The
    # measured delta_start_s may move by at most one sampling interval (1 s).
    colors = ["red"] * 10 + ["green"] * 10 + ["red"] * 10
    c3 = _c3([(1, True, 630, 750)])  # t_start = 21 s

    deltas = []
    for offset_frames in (0, 6, 12, 20, 29):  # 0 .. ~0.97 s at 30 fps
        # Sample the *underlying* plan at t = i + offset (state at that time).
        pairs = []
        for i in range(30):
            t = i + offset_frames / FPS
            if t < 10.0:
                c = "red"
            elif t < 20.0:
                c = "green"
            else:
                c = "red"
            pairs.append((int(round(t * FPS)), c))
        rows = compute_signal_timing_rows(c3, _e2(pairs), FPS)
        assert len(rows) == 1
        assert rows[0]["nearest_transition"] == "green_to_red"
        deltas.append(rows[0]["delta_start_s"])

    ref = deltas[0]
    assert abs(ref - 1.0) < 1e-9, deltas
    for d in deltas[1:]:
        assert abs(d - ref) <= 1.0 + 1e-9, deltas
    print("test_sampling_offset_invariance OK  deltas=%s" % (deltas,))


def test_transition_outside_window_gives_none():
    colors = ["red"] * 10 + ["green"] * 30  # transition at 10 s only
    e2 = _e2_1hz(colors)
    # Start at 30 s: delta = +20 s, outside [-10, +15] -> no usable transition.
    c3 = _c3([(1, True, 900, 990)])
    rows = compute_signal_timing_rows(c3, e2, FPS)
    assert rows[0]["nearest_transition"] == "none"
    assert _isnan(rows[0]["delta_start_s"])
    assert rows[0]["n_transitions_in_video"] == 1
    # Red exposure still computed from segments: [30, 33] inside green.
    assert abs(rows[0]["red_exposure_s"] - 3.0) < 1e-9
    print("test_transition_outside_window_gives_none OK")


def test_run_signal_timing_missing_inputs_header_only():
    tmp = tempfile.mkdtemp(prefix="pedx_sigtim_")
    out = os.path.join(tmp, "[P10]signal_timing.csv")
    run_signal_timing(
        os.path.join(tmp, "NoSuchCity_abc123.mp4"),
        crossing_csv_path=os.path.join(tmp, "missing_c3.csv"),
        traffic_light_csv_path=os.path.join(tmp, "missing_e2.csv"),
        video_meta_csv_path=os.path.join(tmp, "missing_b0.csv"),
        dense_tracks_csv_path=os.path.join(tmp, "missing_b2.csv"),
        output_csv_path=out,
    )
    assert os.path.exists(out)
    df = pd.read_csv(out)
    assert list(df.columns) == OUTPUT_COLUMNS
    assert len(df) == 0
    print("test_run_signal_timing_missing_inputs_header_only OK")


def test_run_signal_timing_end_to_end_with_files():
    tmp = tempfile.mkdtemp(prefix="pedx_sigtim_e2e_")
    c3_path = os.path.join(tmp, "c3.csv")
    e2_path = os.path.join(tmp, "e2.csv")
    b0_path = os.path.join(tmp, "b0.csv")
    out = os.path.join(tmp, "[P10]signal_timing.csv")

    colors = ["red"] * 10 + ["green"] * 10 + ["red"] * 10
    _e2_1hz(colors).to_csv(e2_path, index=False)
    _c3([(7, True, 630, 750), (8, False, np.nan, np.nan)]).to_csv(
        c3_path, index=False)
    pd.DataFrame([{"video_name": "NoSuchCity_abc123", "fps": FPS,
                   "width": 1920, "height": 1080,
                   "total_frames": 900}]).to_csv(b0_path, index=False)

    run_signal_timing(
        os.path.join(tmp, "NoSuchCity_abc123.mp4"),
        crossing_csv_path=c3_path,
        traffic_light_csv_path=e2_path,
        video_meta_csv_path=b0_path,
        dense_tracks_csv_path=os.path.join(tmp, "missing_b2.csv"),
        output_csv_path=out,
    )
    df = pd.read_csv(out)
    assert list(df.columns) == OUTPUT_COLUMNS
    assert len(df) == 1
    row = df.iloc[0]
    assert int(row["track_id"]) == 7
    assert row["nearest_transition"] == "green_to_red"
    assert abs(float(row["delta_start_s"]) - 1.0) < 1e-9
    assert bool(row["anticipatory"]) is False
    assert abs(float(row["startup_latency_s"]) - 1.0) < 1e-9
    assert int(row["n_transitions_in_video"]) == 2
    assert row["phase_convention"] == PHASE_CONVENTION
    print("test_run_signal_timing_end_to_end_with_files OK")


def test_fps_fallback_from_b2_ratio():
    # No [B0]; fps must come from [B2] frame_id/timestamp median ratio (60).
    tmp = tempfile.mkdtemp(prefix="pedx_sigtim_fps_")
    b2_path = os.path.join(tmp, "b2.csv")
    frames = list(range(4, 400, 4))
    pd.DataFrame({
        "frame_id": frames,
        "timestamp": [round(f / 60.0, 3) for f in frames],
        "track_id": [1] * len(frames),
        "x1": 0, "y1": 0, "x2": 10, "y2": 10,
    }).to_csv(b2_path, index=False)

    fps60 = 60.0
    colors = ["red"] * 10 + ["green"] * 10 + ["red"] * 10
    e2_path = os.path.join(tmp, "e2.csv")
    _e2_1hz(colors, fps=fps60).to_csv(e2_path, index=False)
    c3_path = os.path.join(tmp, "c3.csv")
    _c3([(1, True, 1260, 1500)]).to_csv(c3_path, index=False)  # t_start=21 @60fps
    out = os.path.join(tmp, "[P10]signal_timing.csv")

    run_signal_timing(
        os.path.join(tmp, "X_y.mp4"),
        crossing_csv_path=c3_path,
        traffic_light_csv_path=e2_path,
        video_meta_csv_path=os.path.join(tmp, "missing_b0.csv"),
        dense_tracks_csv_path=b2_path,
        output_csv_path=out,
    )
    df = pd.read_csv(out)
    assert len(df) == 1
    # With correct 60 fps: transition at 20 s, start at 21 s -> delta ~ 1 s.
    assert abs(float(df.iloc[0]["delta_start_s"]) - 1.0) < 0.05, df
    print("test_fps_fallback_from_b2_ratio OK")


if __name__ == "__main__":
    test_clean_phases_exact_deltas()
    test_one_sample_flicker_rejected()
    test_all_none_e2()
    test_sampling_offset_invariance()
    test_transition_outside_window_gives_none()
    test_run_signal_timing_missing_inputs_header_only()
    test_run_signal_timing_end_to_end_with_files()
    test_fps_fallback_from_b2_ratio()
    print("ALL signal_timing TESTS PASSED")
