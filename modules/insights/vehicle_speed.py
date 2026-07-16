"""Vehicle metric speed profiles (module [V8], insight Rank 2).

Per-vehicle metric speed time-series statistics from the dense vehicle track dump
[V7]vehicle_tracks.csv: median / 85th-percentile / max speed, split into the
crosswalk band vs mid-block. Mirrors the [S1] pedestrian pipeline:

  ground point -> ego-motion compensation -> smoothing -> metric scale -> stats

* Ground point ((x1+x2)/2, y2) = the vehicle's road-contact point, where the
  ground-plane scale is valid (the bbox centre used by the counter is NOT).
* Ego motion: when [B3] says the camera is moving (median step_px >= 1.0) the
  interpolated camera displacement is subtracted before differencing.
* Metric scale priority (per spec):
    1. [S2] ground-plane scale(y) = a*y + b, only when quality == "good"
    2. global lane-width scale from [V5] (median lane width / 3.5 m)
    3. car-length prior (median bbox width / 4.5 m), ONLY for near-lateral
       tracks (|net dx| > 3 |net dy|) of cars/taxis, never flagged reliable
* Crosswalk band = union of [E7] boxes padded by 0.15x per side; step speeds are
  attributed by the (image-space) midpoint of each step.
* Honesty gate: reliable = enough steps AND a real scale AND the camera did not
  pan so far (> 200 px cumulative) that background-registration errors dominate.

Videos are deleted after analysis: this module is CSV-only and never opens the
video. Missing/empty inputs always yield a valid header-only output CSV.
"""

import ast
import math
import os

import numpy as np
import pandas as pd

from modules.speed.speed_estimation import _rolling_median, _lane_scale_px_per_m

CAR_LENGTH_M = 4.5
LENGTH_PRIOR_TYPES = {"car", "taxi"}
LATERAL_RATIO = 3.0                 # |net dx| > LATERAL_RATIO * |net dy| for length prior
CROSSWALK_PAD_FRAC = 0.15
MAX_STEP_SPEED_MPS = 50.0
MIN_RELIABLE_STEPS = 15
PAN_LIMIT_PX = 200.0
MIN_SCALE_PX_PER_M = 0.1            # below this a scale(y) row is degenerate -> skip step

OUTPUT_COLUMNS = [
    "track_id", "veh_type", "n_valid_steps", "median_speed_mps", "p85_speed_mps",
    "max_speed_mps", "speed_at_crosswalk_mps", "midblock_speed_mps",
    "scale_source", "camera_moving", "reliable",
]

REQUIRED_V7_COLUMNS = {"frame_id", "timestamp", "track_id", "x1", "y1", "x2", "y2"}


def _load_crosswalk_boxes(e7_csv, pad_frac=CROSSWALK_PAD_FRAC):
    """Union of [E7] crosswalk boxes across all frames, padded by pad_frac per side.

    Returns a list of [x1, y1, x2, y2]; empty list when the file is
    missing/empty/malformed (-> crosswalk split reported as NaN downstream).
    """
    if not (e7_csv and os.path.exists(e7_csv) and os.path.getsize(e7_csv) > 0):
        return []
    try:
        d = pd.read_csv(e7_csv)
    except Exception:
        return []
    if d.empty or "crosswalk_boxes" not in d.columns:
        return []
    seen = set()
    boxes = []
    for raw in d["crosswalk_boxes"].dropna():
        try:
            parsed = ast.literal_eval(raw) if isinstance(raw, str) else raw
        except (ValueError, SyntaxError):
            continue
        if not isinstance(parsed, (list, tuple)):
            continue
        for b in parsed:
            try:
                x1, y1, x2, y2 = (float(v) for v in b)
            except (TypeError, ValueError):
                continue
            key = (round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1))
            if key in seen:
                continue
            seen.add(key)
            px = (x2 - x1) * pad_frac
            py = (y2 - y1) * pad_frac
            boxes.append([x1 - px, y1 - py, x2 + px, y2 + py])
    return boxes


def _point_in_boxes(x, y, boxes):
    for bx1, by1, bx2, by2 in boxes:
        if bx1 <= x <= bx2 and by1 <= y <= by2:
            return True
    return False


def _majority_vtype(g):
    if "vtype" not in g.columns:
        return "unknown"
    vals = g["vtype"].dropna().astype(str)
    if vals.empty:
        return "unknown"
    return vals.value_counts().idxmax()


def compute_vehicle_speeds(veh_df, ego_df=None, stripe_ab=None, lane_scale_px_per_m=None,
                           crosswalk_boxes=None, max_step_speed_mps=MAX_STEP_SPEED_MPS,
                           smooth_window=3, min_reliable_steps=MIN_RELIABLE_STEPS,
                           pan_limit_px=PAN_LIMIT_PX, car_length_m=CAR_LENGTH_M):
    """Pure core: [V7]-shaped dataframe in -> list of per-track result dicts out.

    veh_df columns: frame_id, timestamp, track_id, x1, y1, x2, y2 (+ optional vtype).
    ego_df: [B3]-shaped (frame_id, cam_x, cam_y, step_px) or None.
    stripe_ab: (a, b) of the [S2] scale(y) = a*y + b fit; pass ONLY when quality=="good".
    lane_scale_px_per_m: global px/m from [V5], or None.
    crosswalk_boxes: pre-padded [x1, y1, x2, y2] boxes (see _load_crosswalk_boxes).
    """
    crosswalk_boxes = crosswalk_boxes or []

    # Ego-motion gate + interpolators (moving iff median step_px >= 1.0).
    camera_moving = False
    pan_ok = True
    ego_fr = ego_x = ego_y = None
    if ego_df is not None and len(ego_df) > 0:
        e = ego_df.sort_values("frame_id")
        camera_moving = bool(e["step_px"].median() >= 1.0)
        ego_fr = e["frame_id"].to_numpy(dtype=float)
        ego_x = e["cam_x"].to_numpy(dtype=float)
        ego_y = e["cam_y"].to_numpy(dtype=float)
        disp = np.hypot(ego_x - ego_x[0], ego_y - ego_y[0])
        pan_ok = bool(np.max(disp) <= pan_limit_px)

    rows = []
    for track_id, g in veh_df.groupby("track_id"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        if len(g) < 2:
            continue
        vtype = _majority_vtype(g)

        fr = g["frame_id"].to_numpy(dtype=float)
        t = g["timestamp"].to_numpy(dtype=float)
        gx_raw = (g["x1"].to_numpy(dtype=float) + g["x2"].to_numpy(dtype=float)) / 2.0
        gy_raw = g["y2"].to_numpy(dtype=float)          # ground contact row
        w_px = (g["x2"].to_numpy(dtype=float) - g["x1"].to_numpy(dtype=float))

        # Ego-compensate BEFORE smoothing (same rationale as [S1]): the rolling
        # median then attenuates camera jitter and box jitter together.
        if camera_moving and ego_fr is not None:
            cx = np.interp(fr, ego_fr, ego_x)
            cy = np.interp(fr, ego_fr, ego_y)
        else:
            cx = cy = np.zeros_like(fr)
        foot_x = _rolling_median(gx_raw - cx, smooth_window)
        foot_y = _rolling_median(gy_raw - cy, smooth_window)
        # Image-space series: scale(y) and the crosswalk band live in IMAGE
        # coordinates, so they are evaluated on the UNcompensated point.
        x_img = _rolling_median(gx_raw, smooth_window)
        y_img = _rolling_median(gy_raw, smooth_window)

        # --- scale-source resolution (track level) ---
        const_scale = None
        if stripe_ab is not None:
            scale_source = "stripe_ground_plane"
        elif lane_scale_px_per_m is not None and lane_scale_px_per_m > MIN_SCALE_PX_PER_M:
            scale_source = "lane_width"
            const_scale = float(lane_scale_px_per_m)
        else:
            net_dx = abs(gx_raw[-1] - gx_raw[0])
            net_dy = abs(gy_raw[-1] - gy_raw[0])
            med_w = float(np.median(w_px))
            if (vtype in LENGTH_PRIOR_TYPES and net_dx > LATERAL_RATIO * net_dy
                    and med_w > 1.0):
                scale_source = "length_prior"
                const_scale = med_w / car_length_m
            else:
                scale_source = "none"

        speeds, cw_speeds, mb_speeds = [], [], []
        if scale_source != "none":
            for i in range(len(g) - 1):
                dt = t[i + 1] - t[i]
                if dt <= 0:
                    continue
                if stripe_ab is not None:
                    a, b = stripe_ab
                    scale = 0.5 * ((a * y_img[i] + b) + (a * y_img[i + 1] + b))
                else:
                    scale = const_scale
                if scale is None or scale <= MIN_SCALE_PX_PER_M:
                    continue
                dxp = foot_x[i + 1] - foot_x[i]
                dyp = foot_y[i + 1] - foot_y[i]
                v = (math.hypot(dxp, dyp) / scale) / dt
                if v > max_step_speed_mps:
                    continue
                speeds.append(v)
                mx = 0.5 * (x_img[i] + x_img[i + 1])
                my = 0.5 * (y_img[i] + y_img[i + 1])
                if crosswalk_boxes and _point_in_boxes(mx, my, crosswalk_boxes):
                    cw_speeds.append(v)
                else:
                    mb_speeds.append(v)

        n_steps = len(speeds)
        reliable = bool(n_steps >= min_reliable_steps
                        and scale_source in ("stripe_ground_plane", "lane_width")
                        and pan_ok)

        def _r(vals, fn):
            return round(float(fn(vals)), 3) if vals else None

        rows.append({
            "track_id": track_id,
            "veh_type": vtype,
            "n_valid_steps": n_steps,
            "median_speed_mps": _r(speeds, np.median),
            "p85_speed_mps": _r(speeds, lambda v: np.percentile(v, 85)),
            "max_speed_mps": _r(speeds, np.max),
            "speed_at_crosswalk_mps": _r(cw_speeds, np.median),
            "midblock_speed_mps": _r(mb_speeds, np.median),
            "scale_source": scale_source,
            "camera_moving": camera_moving,
            "reliable": reliable,
        })
    return rows


def run_vehicle_speed(video_path, vehicle_tracks_csv=None, output_csv=None,
                      max_step_speed_mps=MAX_STEP_SPEED_MPS, smooth_window=3,
                      min_reliable_steps=MIN_RELIABLE_STEPS, pan_limit_px=PAN_LIMIT_PX):
    """Entry point (CSV-only; the video file itself is never opened)."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join("analysis_results", video_name)
    os.makedirs(output_dir, exist_ok=True)
    if output_csv is None:
        output_csv = os.path.join(output_dir, "[V8]vehicle_speed.csv")
    if vehicle_tracks_csv is None:
        vehicle_tracks_csv = os.path.join(output_dir, "[V7]vehicle_tracks.csv")

    def _write_empty(msg):
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(output_csv, index=False)
        print(f"[vehicle_speed] {msg} Empty results saved to {output_csv}")
        return output_csv

    if not (os.path.exists(vehicle_tracks_csv) and os.path.getsize(vehicle_tracks_csv) > 0):
        return _write_empty("[V7]vehicle_tracks.csv missing/empty.")
    try:
        veh_df = pd.read_csv(vehicle_tracks_csv)
    except Exception as e:
        return _write_empty(f"[V7] unreadable ({e}).")
    if veh_df.empty or not REQUIRED_V7_COLUMNS.issubset(veh_df.columns):
        return _write_empty("[V7] empty or malformed.")

    # [B3] ego motion (optional)
    ego_df = None
    ego_path = os.path.join(output_dir, "[B3]ego_motion.csv")
    if os.path.exists(ego_path) and os.path.getsize(ego_path) > 0:
        try:
            e = pd.read_csv(ego_path)
            if not e.empty and {"frame_id", "cam_x", "cam_y", "step_px"}.issubset(e.columns):
                ego_df = e
        except Exception as e:
            print(f"[vehicle_speed][warn] ego-motion read failed: {e}")

    # [S2] ground-plane scale, usable only when quality == "good"
    stripe_ab = None
    s2_path = os.path.join(output_dir, "[S2]scale_calibration.csv")
    if os.path.exists(s2_path) and os.path.getsize(s2_path) > 0:
        try:
            sc = pd.read_csv(s2_path)
            if not sc.empty and str(sc.iloc[0].get("quality")) == "good":
                stripe_ab = (float(sc.iloc[0]["a"]), float(sc.iloc[0]["b"]))
        except Exception as e:
            print(f"[vehicle_speed][warn] stripe calibration read failed: {e}")

    lane_scale = _lane_scale_px_per_m(os.path.join(output_dir, "[V5]lane_detection.csv"))
    crosswalk_boxes = _load_crosswalk_boxes(os.path.join(output_dir, "[E7]crosswalk_detection.csv"))

    rows = compute_vehicle_speeds(
        veh_df, ego_df=ego_df, stripe_ab=stripe_ab, lane_scale_px_per_m=lane_scale,
        crosswalk_boxes=crosswalk_boxes, max_step_speed_mps=max_step_speed_mps,
        smooth_window=smooth_window, min_reliable_steps=min_reliable_steps,
        pan_limit_px=pan_limit_px)

    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out.to_csv(output_csv, index=False)
    n_rel = int(out["reliable"].sum()) if not out.empty else 0
    srcs = out["scale_source"].value_counts().to_dict() if not out.empty else {}
    print(f"[vehicle_speed] {len(out)} vehicle tracks ({n_rel} reliable), "
          f"scale sources={srcs}, crosswalk_boxes={len(crosswalk_boxes)}. "
          f"Saved to {output_csv}")
    return output_csv


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Vehicle metric speed profiles ([V8]).")
    ap.add_argument("--source_video_path", required=True)
    args = ap.parse_args()
    run_vehicle_speed(args.source_video_path)
