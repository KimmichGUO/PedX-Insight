"""Unit tests for modules/insights/headway_stats.py.

Plain asserts, no pytest needed. Run from the repo root:

    python tests/test_headway_stats.py
"""

import math
import os
import sys
import tempfile

import numpy as np
import pandas as pd

# Make the repo root importable no matter where the test is launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.insights.headway_stats import (  # noqa: E402
    OUTPUT_COLUMNS,
    headway_stats,
    run_headway_stats,
)


def _events(times, direction=1, cx=100.0):
    times = list(times)
    return pd.DataFrame(
        {
            "track_id": range(1, len(times) + 1),
            "frame_id": [int(t * 30) + 1 for t in times],
            "time_s": times,
            "cx": [cx] * len(times),
            "cy": [200.0] * len(times),
            "direction": [direction] * len(times),
            "axis": ["y"] * len(times),
            "veh_type": ["car"] * len(times),
        }
    )


def _row(rows, direction, lane_half):
    matches = [r for r in rows if r["direction"] == direction and r["lane_half"] == lane_half]
    assert len(matches) == 1, f"expected exactly one row for {direction}/{lane_half}, got {len(matches)}"
    return matches[0]


def test_deterministic_arrivals():
    """Arrivals every 3 s: mean 3, cv 0, no platooning, lambda undefined (NaN)."""
    df = _events([3.0 * i for i in range(11)])  # 11 events -> 10 gaps of 3 s
    rows = headway_stats(df)
    r = _row(rows, 1, "all")
    assert r["n_events"] == 11
    assert r["n_gaps"] == 10
    assert abs(r["mean_headway_s"] - 3.0) < 1e-12
    assert r["cv_headway"] == 0.0
    assert r["platoon_frac"] == 0.0
    assert abs(r["tau_s"] - 3.0) < 1e-12
    assert abs(r["flow_veh_per_min"] - 20.0) < 1e-9
    # mean == tau -> shifted-exponential rate is undefined, must be NaN not inf
    assert math.isnan(r["lambda_hz"])
    print("ok: deterministic 3 s arrivals")


def test_shifted_exponential_recovery():
    """Seeded gaps ~ tau + Exp(lambda): MLE recovers both within 10%."""
    tau_true, lambda_true, n = 1.0, 0.5, 500
    rng = np.random.default_rng(42)
    gaps = tau_true + rng.exponential(scale=1.0 / lambda_true, size=n)
    # Keep the fixture inside the flow-interruption cutoff so nothing is discarded.
    assert gaps.max() <= 20.0, "seed produced an interruption-length gap; pick another seed"
    times = np.concatenate([[0.0], np.cumsum(gaps)])
    rows = headway_stats(_events(times))
    r = _row(rows, 1, "all")
    assert r["n_gaps"] == n
    assert abs(r["tau_s"] - tau_true) / tau_true < 0.10, r["tau_s"]
    assert abs(r["lambda_hz"] - lambda_true) / lambda_true < 0.10, r["lambda_hz"]
    mean_true = tau_true + 1.0 / lambda_true
    assert abs(r["mean_headway_s"] - mean_true) / mean_true < 0.10
    print(f"ok: shifted-exponential recovery (tau={r['tau_s']:.3f}, lambda={r['lambda_hz']:.3f})")


def test_flow_interruption_gap_discarded():
    """A 60 s hole is a flow interruption, not a headway: it must be discarded."""
    first = [2.0 * i for i in range(10)]        # 0..18, nine 2 s gaps
    second = [78.0 + 2.0 * i for i in range(10)]  # 78..96, nine 2 s gaps; 60 s hole before
    rows = headway_stats(_events(first + second))
    r = _row(rows, 1, "all")
    assert r["n_events"] == 20
    assert r["n_gaps"] == 18  # 19 raw gaps, the 60 s one dropped
    assert abs(r["mean_headway_s"] - 2.0) < 1e-12  # hole did not poison the mean
    assert r["cv_headway"] == 0.0
    print("ok: 60 s flow-interruption gap discarded")


def test_directions_never_pooled():
    """Interleaved opposing flows must be split, never merged into one stream."""
    # Direction +1 crosses at even seconds, direction -1 at odd seconds.
    up = _events([0.0, 4.0, 8.0, 12.0], direction=1)
    down = _events([2.0, 6.0, 10.0, 14.0], direction=-1)
    mixed = pd.concat([up, down], ignore_index=True).sample(frac=1.0, random_state=7)
    rows = headway_stats(mixed)
    r_up = _row(rows, 1, "all")
    r_down = _row(rows, -1, "all")
    # Pooled they would show 2 s headways; separated each stream is 4 s.
    assert abs(r_up["mean_headway_s"] - 4.0) < 1e-12
    assert abs(r_down["mean_headway_s"] - 4.0) < 1e-12
    assert r_up["n_events"] == 4 and r_down["n_events"] == 4
    assert all(r["direction"] in (1, -1) for r in rows)
    print("ok: directions never pooled")


def test_lane_half_split():
    """>= 30 events in one direction additionally split into cx halves."""
    left = _events([3.0 * i for i in range(20)], cx=100.0)            # left lane, 3 s
    right = _events([1.0 + 2.0 * i for i in range(20)], cx=500.0)     # right lane, 2 s
    df = pd.concat([left, right], ignore_index=True)
    rows = headway_stats(df, frame_width=640)
    halves = {r["lane_half"] for r in rows if r["direction"] == 1}
    assert halves == {"all", "left", "right"}
    assert abs(_row(rows, 1, "left")["mean_headway_s"] - 3.0) < 1e-12
    assert abs(_row(rows, 1, "right")["mean_headway_s"] - 2.0) < 1e-12
    # Below the threshold no half rows appear.
    rows_small = headway_stats(df.head(10), frame_width=640)
    assert {r["lane_half"] for r in rows_small} == {"all"}
    print("ok: lane-half split at >= 30 events, suppressed below")


def test_empty_and_missing_inputs():
    """Missing [V10] and empty [V10] both yield a valid header-only [V11]."""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "[V11]headway_stats.csv")
        # Missing producer file.
        run_headway_stats(
            os.path.join(tmp, "some_video.mp4"),
            line_crossing_csv_path=os.path.join(tmp, "does_not_exist.csv"),
            video_meta_csv_path=os.path.join(tmp, "no_meta.csv"),
            output_csv_path=out,
        )
        df = pd.read_csv(out)
        assert list(df.columns) == OUTPUT_COLUMNS
        assert len(df) == 0
        # Header-only producer file.
        empty_v10 = os.path.join(tmp, "[V10]line_crossing_events.csv")
        pd.DataFrame(
            columns=["track_id", "frame_id", "time_s", "cx", "cy", "direction", "axis", "veh_type"]
        ).to_csv(empty_v10, index=False)
        run_headway_stats(
            os.path.join(tmp, "some_video.mp4"),
            line_crossing_csv_path=empty_v10,
            video_meta_csv_path=os.path.join(tmp, "no_meta.csv"),
            output_csv_path=out,
        )
        df = pd.read_csv(out)
        assert list(df.columns) == OUTPUT_COLUMNS
        assert len(df) == 0
    print("ok: missing/empty inputs -> header-only output")


def test_end_to_end_with_meta():
    """Full run_headway_stats round trip with a real-ish [V10] and [B0] meta."""
    with tempfile.TemporaryDirectory() as tmp:
        v10 = os.path.join(tmp, "[V10]line_crossing_events.csv")
        _events([2.5 * i for i in range(8)]).to_csv(v10, index=False)
        meta = os.path.join(tmp, "[B0]video_meta.csv")
        pd.DataFrame(
            [{"video_name": "vid", "fps": 30.0, "width": 1280, "height": 720, "total_frames": 900}]
        ).to_csv(meta, index=False)
        out = os.path.join(tmp, "[V11]headway_stats.csv")
        result = run_headway_stats(
            os.path.join(tmp, "vid.mp4"),
            line_crossing_csv_path=v10,
            video_meta_csv_path=meta,
            output_csv_path=out,
        )
        assert list(result.columns) == OUTPUT_COLUMNS
        df = pd.read_csv(out)
        assert len(df) == 1
        assert abs(df.loc[0, "mean_headway_s"] - 2.5) < 1e-9
    print("ok: end-to-end run with [B0] meta")


if __name__ == "__main__":
    test_deterministic_arrivals()
    test_shifted_exponential_recovery()
    test_flow_interruption_gap_discarded()
    test_directions_never_pooled()
    test_lane_half_split()
    test_empty_and_missing_inputs()
    test_end_to_end_with_meta()
    print("ALL TESTS PASSED")
