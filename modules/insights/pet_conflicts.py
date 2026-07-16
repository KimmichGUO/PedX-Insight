"""Pedestrian-vehicle Post-Encroachment Time (PET) conflicts (module [I1]).

For every [C3] crossing, compute the minimum time gap between the pedestrian and
any vehicle occupying the same ground-plane cell — the standard video-based
surrogate-safety measure (PET < 1.5 s = serious conflict). PET is a pure time
difference, so monocular-scale errors only perturb the cell size, not the gap.

Pipeline:
  1. Ego-compensate both agents with [B3] (subtract interpolated camera position
     when the median step_px >= 1.0); skip the video entirely (camera_pan_ok=False)
     when the cumulative camera displacement exceeds ``max_pan_px`` (default 200 px).
  2. Grid the ground plane into ~1 m cells whose pixel side is scale(y):
     [S2] a*y+b when quality=='good' (with an implied-stature sanity guard),
     else the pedestrian's own [S1] scale_px_per_m_median, else a bbox-height prior.
  3. Stamp per-cell occupancy intervals: pedestrian foot point ((x1+x2)/2, y2)
     from [B2] restricted to [started_frame, ended_frame] +/- pad_s; vehicle ground
     point from [V7] (all frames, lightly densified so fast vehicles cannot skip cells).
  4. Per cell visited by both within max_pet_s: PET = t(second arriver enters) -
     t(first leaver exits); keep the per-crossing minimum and who passed first.

CSV-only: never opens the video. Missing/empty inputs (including the NEW producer
[V7]vehicle_tracks.csv) -> header-only output, never a crash. Pedestrians with no
vehicle co-occupancy get one row with NaN PET so every crossing is accounted for.
"""

import math
import os

import numpy as np
import pandas as pd

from modules.speed.speed_estimation import _resolve_assumed_height_m, _rolling_median

OUTPUT_COLUMNS = [
    "track_id", "veh_track_id", "veh_type", "min_pet_s", "first_agent",
    "cell_y_px", "n_shared_cells", "severity", "scale_source",
    "camera_pan_ok", "reliable",
]

MIN_IMPLIED_HEIGHT_M = 0.9    # same stripe-scale sanity window as [S1]
MAX_IMPLIED_HEIGHT_M = 2.8


def _truthy(v):
    return v is True or str(v).strip().lower() == "true"


def _read_csv_nonempty(path):
    """DataFrame or None for a missing / zero-byte / header-only CSV."""
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    return df if not df.empty else None


def _resolve_fps(fps, video_meta_csv, tracks_df):
    """fps priority: explicit arg -> [B0]video_meta.csv -> [B2] frame/timestamp ratio -> 30."""
    if fps is not None and fps > 0:
        return float(fps)
    meta = _read_csv_nonempty(video_meta_csv)
    if meta is not None and "fps" in meta.columns:
        try:
            v = float(meta.iloc[0]["fps"])
            if v > 0:
                return v
        except Exception:
            pass
    if tracks_df is not None and {"frame_id", "timestamp"}.issubset(tracks_df.columns):
        d = tracks_df[tracks_df["timestamp"] > 0.5]
        if not d.empty:
            ratio = (d["frame_id"].astype(float) / d["timestamp"].astype(float)).median()
            if np.isfinite(ratio) and ratio > 0:
                return float(ratio)
    return 30.0


def _densify(t, x, y, scale_fn, gap_break_s):
    """Insert linear sub-samples so consecutive spatial steps stay <= 0.5 cell.

    Fast vehicles can jump a whole ground cell between frames; occupancy stamping
    on raw samples would then miss the co-occupied cell. Original samples are kept
    verbatim (endpoints exact); gaps longer than gap_break_s are never bridged.
    """
    if len(t) < 2:
        return t, x, y
    ts, xs, ys = [t[0]], [x[0]], [y[0]]
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        if 0 < dt <= gap_break_s:
            s_mid = scale_fn(0.5 * (y[i] + y[i + 1]))
            max_step = max(0.5 * s_mid, 0.5) if (s_mid == s_mid and s_mid > 0) else 1e9
            n_sub = int(math.ceil(max(abs(x[i + 1] - x[i]), abs(y[i + 1] - y[i])) / max_step))
            for k in range(1, n_sub):
                f = k / n_sub
                ts.append(t[i] + f * dt)
                xs.append(x[i] + f * (x[i + 1] - x[i]))
                ys.append(y[i] + f * (y[i + 1] - y[i]))
        ts.append(t[i + 1]); xs.append(x[i + 1]); ys.append(y[i + 1])
    return np.asarray(ts), np.asarray(xs), np.asarray(ys)


def _cell_intervals(t, x, y, scale_fn, gap_break_s):
    """{(ix, iy): [(t_enter, t_exit, mean_y_px), ...]} contiguous occupancy runs.

    A cell is the ~1 m ground square containing the point: side = scale(y) px,
    index = floor(coord / side). Runs split when the point leaves for > gap_break_s.
    """
    occ = {}
    for ti, xi, yi in zip(t, x, y):
        s = scale_fn(yi)
        if not (s == s and s > 1e-6):
            continue
        key = (int(math.floor(xi / s)), int(math.floor(yi / s)))
        occ.setdefault(key, []).append((ti, yi))
    out = {}
    for key, pts in occ.items():
        pts.sort()
        ivs = []
        start_t, last_t, ys = pts[0][0], pts[0][0], [pts[0][1]]
        for ti, yi in pts[1:]:
            if ti - last_t > gap_break_s:
                ivs.append((start_t, last_t, float(np.mean(ys))))
                start_t, ys = ti, []
            last_t = ti
            ys.append(yi)
        ivs.append((start_t, last_t, float(np.mean(ys))))
        out[key] = ivs
    return out


def pet_from_tracks(ped_df, veh_df, scale_fn, max_pet_s=10.0, gap_break_s=1.0):
    """Pure core: minimum PET between ONE pedestrian and each vehicle track.

    ped_df: columns t, x, y — ego-compensated ground (foot) points of one pedestrian.
    veh_df: columns t, x, y, veh_track_id, veh_type — ego-compensated vehicle
            ground points (any number of tracks).
    scale_fn: y_px -> local pixels-per-metre (cell side).

    Returns a list of dicts (one per vehicle sharing >= 1 cell within max_pet_s):
    {veh_track_id, veh_type, min_pet_s, first_agent, cell_y_px, n_shared_cells}.
    Overlapping occupancy (a true joint presence) counts as PET = 0.
    """
    rows = []
    if ped_df is None or len(ped_df) < 2 or veh_df is None or len(veh_df) == 0:
        return rows
    pt, px, py = _densify(ped_df["t"].to_numpy(dtype=float),
                          ped_df["x"].to_numpy(dtype=float),
                          ped_df["y"].to_numpy(dtype=float), scale_fn, gap_break_s)
    ped_occ = _cell_intervals(pt, px, py, scale_fn, gap_break_s)
    if not ped_occ:
        return rows

    for veh_id, vg in veh_df.groupby("veh_track_id"):
        vg = vg.sort_values("t")
        if len(vg) < 2:
            continue
        vt, vx, vy = _densify(vg["t"].to_numpy(dtype=float),
                              vg["x"].to_numpy(dtype=float),
                              vg["y"].to_numpy(dtype=float), scale_fn, gap_break_s)
        veh_occ = _cell_intervals(vt, vx, vy, scale_fn, gap_break_s)
        best, n_shared = None, 0
        for key in set(ped_occ) & set(veh_occ):
            cell_best = None
            for (ps, pe, pyy) in ped_occ[key]:
                for (vs, ve, _) in veh_occ[key]:
                    if vs > pe:                       # pedestrian left first
                        pet, first = vs - pe, "ped"
                    elif ps > ve:                     # vehicle left first
                        pet, first = ps - ve, "vehicle"
                    else:                             # overlapping presence
                        pet, first = 0.0, ("ped" if ps <= vs else "vehicle")
                    if pet <= max_pet_s and (cell_best is None or pet < cell_best[0]):
                        cell_best = (pet, first, pyy)
            if cell_best is not None:
                n_shared += 1
                if best is None or cell_best[0] < best[0]:
                    best = cell_best
        if best is not None:
            mode = vg["veh_type"].mode()
            rows.append({
                "veh_track_id": veh_id,
                "veh_type": mode.iloc[0] if not mode.empty else None,
                "min_pet_s": best[0],
                "first_agent": best[1],
                "cell_y_px": best[2],
                "n_shared_cells": n_shared,
            })
    return rows


def _severity(pet, severe_pet_s, moderate_pet_s):
    if pet != pet:
        return "none"
    if pet < severe_pet_s:
        return "severe"
    if pet < moderate_pet_s:
        return "moderate"
    return "none"


def run_pet_conflicts(video_path, tracks_csv=None, vehicle_csv=None, crossing_csv=None,
                      ego_csv=None, scale_csv=None, speed_csv=None, video_meta_csv=None,
                      mapping_csv="mapping.csv", output_csv=None, fps=None,
                      pad_s=2.0, max_pet_s=10.0, gap_break_s=1.0, max_pan_px=200.0,
                      severe_pet_s=1.5, moderate_pet_s=3.0, smooth_window=3):
    """Compute [I1]pet_conflicts.csv for one video. CSV-only (video may be deleted)."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join("analysis_results", video_name)
    if output_csv is None:
        os.makedirs(output_dir, exist_ok=True)
        output_csv = os.path.join(output_dir, "[I1]pet_conflicts.csv")
    else:
        parent = os.path.dirname(output_csv)
        if parent:
            os.makedirs(parent, exist_ok=True)
    if tracks_csv is None:
        tracks_csv = os.path.join(output_dir, "[B2]dense_tracks.csv")
    if vehicle_csv is None:
        vehicle_csv = os.path.join(output_dir, "[V7]vehicle_tracks.csv")
    if crossing_csv is None:
        crossing_csv = os.path.join(output_dir, "[C3]crossing_judge.csv")
    if ego_csv is None:
        ego_csv = os.path.join(output_dir, "[B3]ego_motion.csv")
    if scale_csv is None:
        scale_csv = os.path.join(output_dir, "[S2]scale_calibration.csv")
    if speed_csv is None:
        speed_csv = os.path.join(output_dir, "[S1]pedestrian_speed.csv")
    if video_meta_csv is None:
        video_meta_csv = os.path.join(output_dir, "[B0]video_meta.csv")

    def _write_empty(msg):
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(output_csv, index=False)
        print(f"[pet] {msg} Empty results saved to {output_csv}")
        return output_csv

    ped = _read_csv_nonempty(tracks_csv)
    if ped is None or not {"frame_id", "timestamp", "track_id", "x1", "y1", "x2", "y2"}.issubset(ped.columns):
        return _write_empty("[B2] dense tracks missing/empty/malformed.")
    veh = _read_csv_nonempty(vehicle_csv)
    if veh is None or not {"frame_id", "timestamp", "track_id", "x1", "y1", "x2", "y2"}.issubset(veh.columns):
        return _write_empty("[V7] vehicle tracks missing/empty/malformed.")
    cj = _read_csv_nonempty(crossing_csv)
    if cj is None or not {"track_id", "crossed", "started_frame"}.issubset(cj.columns):
        return _write_empty("[C3] crossing judge missing/empty/malformed.")
    crossings = cj[[_truthy(v) and pd.notna(f) for v, f in zip(cj["crossed"], cj["started_frame"])]]
    if crossings.empty:
        return _write_empty("No completed crossings in [C3].")

    fps = _resolve_fps(fps, video_meta_csv, ped)

    # --- ego motion: interpolate camera position over TIME; gate on median step_px ---
    ego = _read_csv_nonempty(ego_csv)
    camera_moving, cam_t = False, None
    camera_pan_ok = True
    if ego is not None and {"timestamp", "cam_x", "cam_y", "step_px"}.issubset(ego.columns):
        ego = ego.sort_values("timestamp")
        camera_moving = bool(ego["step_px"].median() >= 1.0)
        et = ego["timestamp"].to_numpy(dtype=float)
        ex = ego["cam_x"].to_numpy(dtype=float)
        ey = ego["cam_y"].to_numpy(dtype=float)
        cam_t = (et, ex, ey)
        # cumulative camera displacement from its first position
        pan = float(np.max(np.hypot(ex - ex[0], ey - ey[0])))
        camera_pan_ok = pan <= max_pan_px

    def cam_at(ts):
        if cam_t is None or not camera_moving:
            return np.zeros_like(ts, dtype=float), np.zeros_like(ts, dtype=float)
        return (np.interp(ts, cam_t[0], cam_t[1]), np.interp(ts, cam_t[0], cam_t[2]))

    # --- scale sources ---
    stripe_a = stripe_b = None
    s2 = _read_csv_nonempty(scale_csv)
    if s2 is not None and str(s2.iloc[0].get("quality")) == "good":
        try:
            stripe_a, stripe_b = float(s2.iloc[0]["a"]), float(s2.iloc[0]["b"])
        except Exception:
            stripe_a = stripe_b = None
    s1_scale = {}
    s1 = _read_csv_nonempty(speed_csv)
    if s1 is not None and {"track_id", "scale_px_per_m_median"}.issubset(s1.columns):
        for _, r in s1.iterrows():
            v = r["scale_px_per_m_median"]
            if pd.notna(v) and float(v) > 0:
                s1_scale[r["track_id"]] = float(v)
    assumed_height_m, _hsrc = _resolve_assumed_height_m(video_name, mapping_csv)

    # --- vehicles: ego-compensated, smoothed ground points (computed once) ---
    veh = veh.sort_values(["track_id", "timestamp"])
    vparts = []
    for vid, vg in veh.groupby("track_id"):
        ts = vg["timestamp"].to_numpy(dtype=float)
        cx, cy = cam_at(ts)
        gx = _rolling_median((vg["x1"].to_numpy(dtype=float) + vg["x2"].to_numpy(dtype=float)) / 2.0 - cx,
                             smooth_window)
        gy = _rolling_median(vg["y2"].to_numpy(dtype=float) - cy, smooth_window)
        vtype = vg["vtype"] if "vtype" in vg.columns else pd.Series(["unknown"] * len(vg))
        vparts.append(pd.DataFrame({"t": ts, "x": gx, "y": gy,
                                    "veh_track_id": vid, "veh_type": vtype.to_numpy()}))
    veh_pts = pd.concat(vparts, ignore_index=True) if vparts else pd.DataFrame(
        columns=["t", "x", "y", "veh_track_id", "veh_type"])

    rows = []

    def _nan_row(tid, scale_source):
        rows.append({
            "track_id": tid, "veh_track_id": None, "veh_type": None,
            "min_pet_s": None, "first_agent": None, "cell_y_px": None,
            "n_shared_cells": 0, "severity": "none", "scale_source": scale_source,
            "camera_pan_ok": camera_pan_ok, "reliable": False,
        })

    for _, cr in crossings.iterrows():
        tid = cr["track_id"]
        g = ped[ped["track_id"] == tid].sort_values("timestamp")
        if len(g) < 2:
            _nan_row(tid, "none")
            continue
        h_px = (g["y2"].to_numpy(dtype=float) - g["y1"].to_numpy(dtype=float))
        median_h = float(np.median(h_px))
        med_row = float(np.median(g["y2"].to_numpy(dtype=float)))

        # scale priority: [S2] good (stature-sane) -> [S1] track median -> bbox prior
        scale_fn, scale_source = None, None
        if stripe_a is not None:
            s_med = stripe_a * med_row + stripe_b
            implied_h = (median_h / s_med) if s_med > 1 else float("inf")
            if MIN_IMPLIED_HEIGHT_M <= implied_h <= MAX_IMPLIED_HEIGHT_M:
                a, b = stripe_a, stripe_b
                scale_fn = lambda yy, a=a, b=b: a * yy + b
                scale_source = "stripe_ground_plane"
        if scale_fn is None and tid in s1_scale:
            const = s1_scale[tid]
            scale_fn = lambda yy, c=const: c
            scale_source = "s1_track_median"
        if scale_fn is None:
            if median_h <= 1:
                _nan_row(tid, "none")
                continue
            const = median_h / assumed_height_m
            scale_fn = lambda yy, c=const: c
            scale_source = "bbox_height_prior"

        if not camera_pan_ok:
            _nan_row(tid, scale_source)
            continue

        # pedestrian window: [started, ended] +/- pad_s (frames are [B2]-native units)
        start_f = float(cr["started_frame"]) - pad_s * fps
        ended = cr.get("ended_frame")
        end_f = (float(ended) if pd.notna(ended) else float(g["frame_id"].max())) + pad_s * fps
        w = g[(g["frame_id"] >= start_f) & (g["frame_id"] <= end_f)]
        if len(w) < 2:
            _nan_row(tid, scale_source)
            continue
        ts = w["timestamp"].to_numpy(dtype=float)
        cx, cy = cam_at(ts)
        ped_df = pd.DataFrame({
            "t": ts,
            "x": _rolling_median((w["x1"].to_numpy(dtype=float) + w["x2"].to_numpy(dtype=float)) / 2.0 - cx,
                                 smooth_window),
            "y": _rolling_median(w["y2"].to_numpy(dtype=float) - cy, smooth_window),
        })
        vsub = veh_pts[(veh_pts["t"] >= ts[0] - max_pet_s) & (veh_pts["t"] <= ts[-1] + max_pet_s)]
        conflicts = pet_from_tracks(ped_df, vsub, scale_fn,
                                    max_pet_s=max_pet_s, gap_break_s=gap_break_s)
        med_dt = float(np.median(np.diff(ts))) if len(ts) > 1 else float("inf")
        if not conflicts:
            _nan_row(tid, scale_source)
            continue
        for c in conflicts:
            pet = float(c["min_pet_s"])
            rows.append({
                "track_id": tid,
                "veh_track_id": c["veh_track_id"],
                "veh_type": c["veh_type"],
                "min_pet_s": round(pet, 3),
                "first_agent": c["first_agent"],
                "cell_y_px": round(float(c["cell_y_px"]), 1),
                "n_shared_cells": c["n_shared_cells"],
                "severity": _severity(pet, severe_pet_s, moderate_pet_s),
                "scale_source": scale_source,
                "camera_pan_ok": camera_pan_ok,
                "reliable": bool(camera_pan_ok and med_dt <= 0.25
                                 and scale_source != "bbox_height_prior"),
            })

    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out.to_csv(output_csv, index=False)
    n_conf = int(out["min_pet_s"].notna().sum()) if not out.empty else 0
    n_sev = int((out["severity"] == "severe").sum()) if not out.empty else 0
    print(f"[pet] {len(out)} rows over {crossings['track_id'].nunique()} crossings: "
          f"{n_conf} conflicts ({n_sev} severe), fps={fps:.2f}, "
          f"camera={'moving' if camera_moving else 'static'}, pan_ok={camera_pan_ok}. "
          f"Saved to {output_csv}")
    return output_csv


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Pedestrian-vehicle PET conflicts ([I1]).")
    ap.add_argument("--source_video_path", required=True)
    ap.add_argument("--mapping_csv", default="mapping.csv")
    ap.add_argument("--fps", type=float, default=None)
    args = ap.parse_args()
    run_pet_conflicts(args.source_video_path, mapping_csv=args.mapping_csv, fps=args.fps)
