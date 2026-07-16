"""Pedestrian waiting-time detection (module [P3]).

Rewritten (fix #10). The previous version ran Lucas-Kanade optical flow between
frames sampled ~1 s apart — a large-displacement regime that violates LK's
small-motion assumption — and thresholded raw pixels (2.0 px), which is not
scale-aware, so far pedestrians were always "waiting" and near ones rarely were.
Camera motion also contaminated the estimate.

This version measures a metric velocity on the pedestrian's foot-point trajectory
(dense [B2] preferred, else 1 Hz [B1]), converts pixels to metres with the same
per-person height-prior scale used by [S1], and subtracts camera motion from
[B3]ego_motion.csv when the camera is moving. A pedestrian is "waiting" while their
metric speed is below `wait_speed_mps`; contiguous waiting spans of at least
`min_wait_seconds` are summed. Output column names (track_id, waiting_time) are
unchanged.
"""

import os
import math
import numpy as np
import pandas as pd

from modules.speed.speed_estimation import _resolve_assumed_height_m, _rolling_median

WAIT_SPEED_MPS = 0.3
MIN_WAIT_SECONDS = 1.0


def run_waiting_time_analysis(video_path, csv_path=None, output_csv=None,
                              mapping_csv="mapping.csv", wait_speed_mps=WAIT_SPEED_MPS,
                              min_wait_seconds=MIN_WAIT_SECONDS, smooth_window=3,
                              # accepted for backward compatibility; no longer used
                              move_thresh=None, frame_thresh=None, min_good_points=None):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join("analysis_results", video_name)
    os.makedirs(output_dir, exist_ok=True)
    if output_csv is None:
        output_csv = os.path.join(output_dir, "[P3]waiting_time.csv")

    dense_path = os.path.join(output_dir, "[B2]dense_tracks.csv")
    b1_path = os.path.join(output_dir, "[B1]tracked_pedestrians.csv")
    if csv_path is None:
        csv_path = dense_path if (os.path.exists(dense_path) and os.path.getsize(dense_path) > 0) else b1_path

    def _write_empty(msg):
        pd.DataFrame(columns=["track_id", "waiting_time"]).to_csv(output_csv, index=False)
        print(f"[waiting] {msg} Empty results saved to {output_csv}")

    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return _write_empty("Trajectory CSV missing/empty.")
    df = pd.read_csv(csv_path)
    # Header-only [B2] must not defeat the fallback to a populated [B1] (row count, not size).
    if df.empty and csv_path == dense_path and os.path.exists(b1_path) and os.path.getsize(b1_path) > 0:
        b1_df = pd.read_csv(b1_path)
        if not b1_df.empty:
            df = b1_df
    if df.empty or not {"frame_id", "timestamp", "track_id", "x1", "y1", "x2", "y2"}.issubset(df.columns):
        return _write_empty("Trajectory CSV empty or malformed.")

    assumed_height_m, _ = _resolve_assumed_height_m(video_name, mapping_csv)

    # Optional ego-motion: ALWAYS subtract when available (subtracting a near-zero series
    # on static clips is harmless; the old moving-only gate left partial pans uncompensated).
    ego_fx = ego_fy = None
    ego_path = os.path.join(output_dir, "[B3]ego_motion.csv")
    if os.path.exists(ego_path) and os.path.getsize(ego_path) > 0:
        try:
            e = pd.read_csv(ego_path)
            if not e.empty:
                ef = e.sort_values("frame_id")
                fr = ef["frame_id"].to_numpy(dtype=float)
                ego_fx = (fr, ef["cam_x"].to_numpy(dtype=float))
                ego_fy = (fr, ef["cam_y"].to_numpy(dtype=float))
        except Exception:
            pass

    def cam_at(frame_ids):
        if ego_fx is None:
            z = np.zeros_like(frame_ids, dtype=float)
            return z, z
        return (np.interp(frame_ids, ego_fx[0], ego_fx[1]),
                np.interp(frame_ids, ego_fy[0], ego_fy[1]))

    results = []
    for track_id, g in df.groupby("track_id"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        if len(g) < 2:
            continue
        fr = g["frame_id"].to_numpy(dtype=float)
        t = g["timestamp"].to_numpy(dtype=float)
        h_px = (g["y2"].to_numpy() - g["y1"].to_numpy()).astype(float)
        cx, cy = cam_at(fr)
        # Ego-compensate BEFORE smoothing (smoothed foot minus raw camera diffs
        # re-injected camera shake as fake pedestrian motion).
        foot_x = _rolling_median((g["x1"].to_numpy() + g["x2"].to_numpy()) / 2.0 - cx, smooth_window)
        foot_y = _rolling_median(g["y2"].to_numpy(dtype=float) - cy, smooth_window)

        waiting_total = 0.0
        seg = 0.0
        for i in range(len(g) - 1):
            dt = t[i + 1] - t[i]
            h_avg = 0.5 * (h_px[i] + h_px[i + 1])
            if dt <= 0 or h_avg <= 1:
                continue
            scale = h_avg / assumed_height_m
            dxp = foot_x[i + 1] - foot_x[i]      # foot series already ego-compensated
            dyp = foot_y[i + 1] - foot_y[i]
            v = (math.hypot(dxp, dyp) / scale) / dt
            if v < wait_speed_mps:
                seg += dt
            else:
                if seg >= min_wait_seconds:
                    waiting_total += seg
                seg = 0.0
        if seg >= min_wait_seconds:
            waiting_total += seg

        if waiting_total >= 1.0:
            results.append({"track_id": track_id, "waiting_time": round(waiting_total, 2)})

    pd.DataFrame(results, columns=["track_id", "waiting_time"]).to_csv(output_csv, index=False)
    print(f"[waiting] {len(results)} waiting pedestrians. Saved to {output_csv}")
