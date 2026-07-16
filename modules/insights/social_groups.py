"""Social group detection and crossing platooning (insight module, Rank 6).

Links pedestrian tracks into persistent social groups (couples, families,
walking parties) from same-frame metric gap + velocity alignment, then
measures platooning at the crossing: leader/follower launch lags.

All existing modules treat track_ids as independent; group membership is a
known moderator of crossing risk, and the long-format output retroactively
enriches [C1]/[C5]/[P3]/[C4] rows with a solo/group condition via a plain
track_id join.

Algorithm
---------
For each pair of [B2] tracks with >= `min_overlap_s` seconds of shared
frames:

* same-frame foot-point gap in metres via ``scale(y)`` evaluated at the
  pair's mean foot row ([S2] stripe calibration when ``quality == 'good'``
  and per-track implied heights are plausible; otherwise the mean of the two
  per-track scales from [S1], falling back to a bbox-height / height-prior
  scale computed directly from [B2]).  Ego motion cancels within a frame, so
  the gap needs no ego compensation.
* velocity cosine of ego-compensated, median-smoothed step vectors on the
  shared frame grid ([B3] camera path subtracted when available).
* an edge is created when a contiguous run of shared samples with
  ``gap < gap_max_m`` spans >= `sustain_s` seconds AND over that run
  ``mean_gap < gap_max_m``, ``gap_std < gap_std_max_m`` and
  ``mean_vel_cos > cos_min``.  The sustain requirement rejects momentary
  passers-by.

Connected components over these edges are the groups.  Platooning: within
groups whose members have [C3] ``crossed == True``, the leader is the
earliest ``started_frame`` and each follower gets
``follower_lag_s = (started_frame_follower - started_frame_leader) / fps``
(note ~1 s quantization inherited from [C3]'s 1 Hz source).

Outputs
-------
[I2]pedestrian_groups.csv (long format, one row per group member):
    group_id, track_id, n_members, mean_gap_m, gap_std_m, mean_vel_cos,
    co_duration_s
[I3]group_crossings.csv (one row per follower; a group with a single
crosser yields one row with empty follower fields):
    group_id, n_members, n_crossers, leader_track_id, follower_track_id,
    follower_lag_s, movement_type_leader

Missing/empty inputs never crash: both outputs are written header-only.
The core (`detect_group_edges`, `connected_components`,
`build_group_rows`, `build_crossing_rows`) is pure dataframes-in /
rows-out so tests need no video.
"""

import math
import os

import numpy as np
import pandas as pd

from modules.speed.speed_estimation import _rolling_median, _resolve_assumed_height_m

GROUPS_COLUMNS = [
    "group_id", "track_id", "n_members", "mean_gap_m", "gap_std_m",
    "mean_vel_cos", "co_duration_s",
]
CROSSINGS_COLUMNS = [
    "group_id", "n_members", "n_crossers", "leader_track_id",
    "follower_track_id", "follower_lag_s", "movement_type_leader",
]

# Edge thresholds (metres / cosine / seconds) -- see module docstring.
GAP_MAX_M = 1.5
GAP_STD_MAX_M = 0.6
COS_MIN = 0.7
SUSTAIN_S = 3.0
MIN_OVERLAP_S = 3.0
# A hole in the shared grid longer than this breaks a run (detector dropout).
MAX_HOLE_S = 1.0
# Steps where either pedestrian moved less than this many pixels have an
# undefined direction and are excluded from the velocity-cosine mean.
SPEED_EPS_PX = 0.5
DEFAULT_FPS = 30.0


# --------------------------------------------------------------------------
# pure core
# --------------------------------------------------------------------------

def _pair_edge(ga, gb, cam_at, scale_fn, gap_max_m, gap_std_max_m, cos_min,
               sustain_s, min_overlap_s, max_hole_s, smooth_window):
    """Evaluate one track pair; return an edge dict or None.

    `ga`/`gb` are per-track dataframes with frame_id, timestamp, x1..y2,
    already de-duplicated on frame_id. `cam_at(frames) -> (cx, cy)` gives the
    cumulative camera position; `scale_fn(y_mean, tid_a, tid_b)` returns a
    px-per-metre array evaluated at the pair's mean foot rows.
    """
    m = pd.merge(ga, gb, on="frame_id", suffixes=("_a", "_b"))
    if len(m) < 2:
        return None
    m = m.sort_values("frame_id").reset_index(drop=True)
    t = m["timestamp_a"].to_numpy(dtype=float)
    if t[-1] - t[0] < min_overlap_s:
        return None

    fxa = (m["x1_a"].to_numpy(dtype=float) + m["x2_a"].to_numpy(dtype=float)) / 2.0
    fya = m["y2_a"].to_numpy(dtype=float)
    fxb = (m["x1_b"].to_numpy(dtype=float) + m["x2_b"].to_numpy(dtype=float)) / 2.0
    fyb = m["y2_b"].to_numpy(dtype=float)

    # Same-frame gap: ego motion cancels within a frame -> raw pixels are fine.
    gap_px = np.hypot(fxa - fxb, fya - fyb)
    y_mean = (fya + fyb) / 2.0
    tid_a = ga["track_id"].iloc[0]
    tid_b = gb["track_id"].iloc[0]
    scale = np.asarray(scale_fn(y_mean, tid_a, tid_b), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        gap_m = np.where(scale > 0, gap_px / scale, np.inf)

    # Ego-compensated, smoothed positions for the step-vector cosine.
    fr = m["frame_id"].to_numpy(dtype=float)
    cx, cy = cam_at(fr)
    ax = _rolling_median(fxa - cx, smooth_window)
    ay = _rolling_median(fya - cy, smooth_window)
    bx = _rolling_median(fxb - cx, smooth_window)
    by = _rolling_median(fyb - cy, smooth_window)
    dax, day = np.diff(ax), np.diff(ay)
    dbx, dby = np.diff(bx), np.diff(by)
    na = np.hypot(dax, day)
    nb = np.hypot(dbx, dby)
    dt = np.diff(t)
    cos = np.full(len(m) - 1, np.nan)
    valid = (na >= SPEED_EPS_PX) & (nb >= SPEED_EPS_PX) & (dt > 0) & (dt <= max_hole_s)
    if valid.any():
        cos[valid] = (dax[valid] * dbx[valid] + day[valid] * dby[valid]) / (na[valid] * nb[valid])

    # Sustained-proximity runs: consecutive shared samples with gap < gap_max
    # and no hole longer than max_hole_s.
    ok = gap_m < gap_max_m
    best = None
    i = 0
    n = len(m)
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and ok[j + 1] and (t[j + 1] - t[j]) <= max_hole_s:
            j += 1
        dur = t[j] - t[i]
        if dur >= sustain_s:
            seg_gap = gap_m[i:j + 1]
            mean_gap = float(np.mean(seg_gap))
            gap_std = float(np.std(seg_gap))
            seg_cos = cos[i:j]
            vel_cos = float(np.nanmean(seg_cos)) if np.isfinite(seg_cos).any() else float("nan")
            if (mean_gap < gap_max_m and gap_std < gap_std_max_m
                    and vel_cos == vel_cos and vel_cos > cos_min):
                cand = {
                    "a": tid_a, "b": tid_b,
                    "mean_gap_m": round(mean_gap, 3),
                    "gap_std_m": round(gap_std, 3),
                    "mean_vel_cos": round(vel_cos, 3),
                    "co_duration_s": round(dur, 2),
                }
                if best is None or cand["co_duration_s"] > best["co_duration_s"]:
                    best = cand
        i = j + 1
    return best


def detect_group_edges(tracks_df, cam_at=None, scale_fn=None,
                       gap_max_m=GAP_MAX_M, gap_std_max_m=GAP_STD_MAX_M,
                       cos_min=COS_MIN, sustain_s=SUSTAIN_S,
                       min_overlap_s=MIN_OVERLAP_S, max_hole_s=MAX_HOLE_S,
                       smooth_window=3):
    """All pairwise social edges in a [B2]-shaped dataframe.

    Returns a list of edge dicts {a, b, mean_gap_m, gap_std_m, mean_vel_cos,
    co_duration_s}.  `cam_at` defaults to a static camera; `scale_fn` defaults
    to 1 px per metre (tests pass their own constant).
    """
    if cam_at is None:
        cam_at = lambda f: (np.zeros_like(f, dtype=float), np.zeros_like(f, dtype=float))
    if scale_fn is None:
        scale_fn = lambda y, ta, tb: np.ones_like(np.asarray(y, dtype=float))

    per_track = {}
    for tid, g in tracks_df.groupby("track_id"):
        g = g.drop_duplicates(subset="frame_id").sort_values("frame_id").reset_index(drop=True)
        if len(g) < 2:
            continue
        per_track[tid] = g

    tids = sorted(per_track.keys())
    edges = []
    for ii in range(len(tids)):
        ga = per_track[tids[ii]]
        a_lo, a_hi = ga["timestamp"].iloc[0], ga["timestamp"].iloc[-1]
        for jj in range(ii + 1, len(tids)):
            gb = per_track[tids[jj]]
            # cheap time-overlap prefilter before the frame merge
            lo = max(a_lo, gb["timestamp"].iloc[0])
            hi = min(a_hi, gb["timestamp"].iloc[-1])
            if hi - lo < min_overlap_s:
                continue
            e = _pair_edge(ga, gb, cam_at, scale_fn, gap_max_m, gap_std_max_m,
                           cos_min, sustain_s, min_overlap_s, max_hole_s,
                           smooth_window)
            if e is not None:
                edges.append(e)
    return edges


def connected_components(edges):
    """Union-find over edge dicts -> list of sorted member lists (size >= 2)."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for e in edges:
        union(e["a"], e["b"])
    comps = {}
    for node in parent:
        comps.setdefault(find(node), []).append(node)
    return sorted((sorted(v) for v in comps.values()), key=lambda c: c[0])


def build_group_rows(edges):
    """(groups_rows, components) for [I2] from the edge list."""
    comps = connected_components(edges)
    incident = {}
    for e in edges:
        incident.setdefault(e["a"], []).append(e)
        incident.setdefault(e["b"], []).append(e)
    rows = []
    for gid, members in enumerate(comps, start=1):
        for tid in members:
            es = incident.get(tid, [])
            rows.append({
                "group_id": gid,
                "track_id": tid,
                "n_members": len(members),
                "mean_gap_m": round(float(np.mean([e["mean_gap_m"] for e in es])), 3) if es else None,
                "gap_std_m": round(float(np.mean([e["gap_std_m"] for e in es])), 3) if es else None,
                "mean_vel_cos": round(float(np.mean([e["mean_vel_cos"] for e in es])), 3) if es else None,
                "co_duration_s": round(float(max(e["co_duration_s"] for e in es)), 2) if es else None,
            })
    return rows, comps


def build_crossing_rows(components, crossing_df, fps):
    """[I3] rows: leader/follower launch lags per group from a [C3] dataframe."""
    info = {}
    if crossing_df is not None and not crossing_df.empty:
        for _, r in crossing_df.iterrows():
            crossed = str(r.get("crossed")).strip().lower() == "true"
            if crossed and pd.notna(r.get("started_frame")):
                info[r["track_id"]] = (float(r["started_frame"]),
                                       r.get("movement_type"))
    rows = []
    if not fps or fps <= 0:
        fps = DEFAULT_FPS
    for gid, members in enumerate(components, start=1):
        crossers = sorted(((info[t][0], t) for t in members if t in info))
        if not crossers:
            continue
        leader_start, leader_tid = crossers[0]
        mv = info[leader_tid][1]
        mv = mv if (mv == mv and mv is not None) else None
        base = {
            "group_id": gid,
            "n_members": len(members),
            "n_crossers": len(crossers),
            "leader_track_id": leader_tid,
            "movement_type_leader": mv,
        }
        if len(crossers) == 1:
            rows.append({**base, "follower_track_id": None, "follower_lag_s": None})
        else:
            for start, tid in crossers[1:]:
                rows.append({**base,
                             "follower_track_id": tid,
                             "follower_lag_s": round((start - leader_start) / fps, 3)})
    return rows


# --------------------------------------------------------------------------
# input resolution helpers
# --------------------------------------------------------------------------

def _readable(path):
    return path is not None and os.path.exists(path) and os.path.getsize(path) > 0


def _resolve_fps(meta_csv, video_name, tracks_df):
    """fps from [B0]video_meta.csv, else the [B2] frame_id/timestamp ratio, else 30."""
    if _readable(meta_csv):
        try:
            meta = pd.read_csv(meta_csv)
            if not meta.empty and "fps" in meta.columns:
                sel = meta
                if "video_name" in meta.columns:
                    hit = meta[meta["video_name"] == video_name]
                    if not hit.empty:
                        sel = hit
                fps = float(sel["fps"].iloc[0])
                if 1.0 <= fps <= 240.0:
                    return fps
        except Exception as e:
            print(f"[social_groups][warn] video_meta read failed: {e}")
    if tracks_df is not None and not tracks_df.empty:
        tt = tracks_df[tracks_df["timestamp"] > 1e-3]
        if not tt.empty:
            fps = float(np.median(tt["frame_id"].to_numpy(dtype=float)
                                  / tt["timestamp"].to_numpy(dtype=float)))
            if 1.0 <= fps <= 240.0:
                return fps
    return DEFAULT_FPS


def _make_cam_at(ego_csv):
    """Interpolated cumulative camera position from [B3]; zeros when absent."""
    ego = None
    if _readable(ego_csv):
        try:
            e = pd.read_csv(ego_csv)
            if not e.empty:
                ef = e.sort_values("frame_id")
                ego = (ef["frame_id"].to_numpy(dtype=float),
                       ef["cam_x"].to_numpy(dtype=float),
                       ef["cam_y"].to_numpy(dtype=float))
        except Exception as e:
            print(f"[social_groups][warn] ego-motion read failed: {e}")

    def cam_at(frames):
        frames = np.asarray(frames, dtype=float)
        if ego is None:
            return np.zeros_like(frames), np.zeros_like(frames)
        return np.interp(frames, ego[0], ego[1]), np.interp(frames, ego[0], ego[2])

    return cam_at


def _make_scale_fn(tracks_df, scale_csv, speed_csv, video_name, mapping_csv):
    """scale(y) resolver: [S2] good stripe fit (with per-track implied-height
    plausibility guard, mirroring [S1]) -> per-frame a*y+b; otherwise the mean
    of the two tracks' constant scales ([S1] scale_px_per_m_median when
    present, else bbox-height / height-prior straight from [B2])."""
    stripe = None
    if _readable(scale_csv):
        try:
            sc = pd.read_csv(scale_csv)
            if not sc.empty and str(sc.iloc[0].get("quality")) == "good":
                stripe = (float(sc.iloc[0]["a"]), float(sc.iloc[0]["b"]))
        except Exception as e:
            print(f"[social_groups][warn] stripe calibration read failed: {e}")

    s1_scale = {}
    if _readable(speed_csv):
        try:
            sp = pd.read_csv(speed_csv)
            if not sp.empty and "scale_px_per_m_median" in sp.columns:
                for _, r in sp.iterrows():
                    v = r["scale_px_per_m_median"]
                    if pd.notna(v) and float(v) > 0:
                        s1_scale[r["track_id"]] = float(v)
        except Exception as e:
            print(f"[social_groups][warn] [S1] speed read failed: {e}")

    assumed_h, _src = _resolve_assumed_height_m(video_name, mapping_csv)

    # per-track median bbox height and median foot row from [B2]
    tstats = {}
    for tid, g in tracks_df.groupby("track_id"):
        h = (g["y2"] - g["y1"]).to_numpy(dtype=float)
        tstats[tid] = (float(np.median(h)), float(np.median(g["y2"].to_numpy(dtype=float))))

    def const_scale(tid):
        v = s1_scale.get(tid)
        if v is not None:
            return v
        mh = tstats.get(tid, (0.0, 0.0))[0]
        return mh / assumed_h if mh > 1 else None

    def fn(y_mean, tid_a, tid_b):
        y_mean = np.asarray(y_mean, dtype=float)
        vals = [v for v in (const_scale(tid_a), const_scale(tid_b)) if v]
        const = float(np.mean(vals)) if vals else 1.0
        if stripe is None:
            return np.full_like(y_mean, const)
        a, b = stripe
        # plausibility guard: the stripe scale must imply a sane stature for
        # BOTH tracks at their own depth, else fall back to the constant.
        for tid in (tid_a, tid_b):
            mh, my = tstats.get(tid, (None, None))
            if mh is None:
                continue
            s = a * my + b
            implied_h = (mh / s) if s > 1 else float("inf")
            if not (0.9 <= implied_h <= 2.8):
                return np.full_like(y_mean, const)
        s = a * y_mean + b
        return np.where(s > 1, s, const)

    return fn


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def run_social_groups(video_path, tracks_csv=None, ego_csv=None, scale_csv=None,
                      speed_csv=None, crossing_csv=None, meta_csv=None,
                      mapping_csv="mapping.csv",
                      output_groups_csv=None, output_crossings_csv=None,
                      gap_max_m=GAP_MAX_M, gap_std_max_m=GAP_STD_MAX_M,
                      cos_min=COS_MIN, sustain_s=SUSTAIN_S,
                      min_overlap_s=MIN_OVERLAP_S, max_hole_s=MAX_HOLE_S,
                      smooth_window=3):
    """Detect social groups and crossing platooning for one video.

    Works CSV-only (the video itself is never opened, so it may already be
    deleted).  Writes [I2]pedestrian_groups.csv and [I3]group_crossings.csv
    into analysis_results/<video_name>/; missing or empty inputs yield valid
    header-only outputs.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join("analysis_results", video_name)
    os.makedirs(output_dir, exist_ok=True)
    if output_groups_csv is None:
        output_groups_csv = os.path.join(output_dir, "[I2]pedestrian_groups.csv")
    if output_crossings_csv is None:
        output_crossings_csv = os.path.join(output_dir, "[I3]group_crossings.csv")
    if tracks_csv is None:
        b2 = os.path.join(output_dir, "[B2]dense_tracks.csv")
        b1 = os.path.join(output_dir, "[B1]tracked_pedestrians.csv")
        tracks_csv = b2 if _readable(b2) else b1
    if ego_csv is None:
        ego_csv = os.path.join(output_dir, "[B3]ego_motion.csv")
    if scale_csv is None:
        scale_csv = os.path.join(output_dir, "[S2]scale_calibration.csv")
    if speed_csv is None:
        speed_csv = os.path.join(output_dir, "[S1]pedestrian_speed.csv")
    if crossing_csv is None:
        crossing_csv = os.path.join(output_dir, "[C3]crossing_judge.csv")
    if meta_csv is None:
        meta_csv = os.path.join(output_dir, "[B0]video_meta.csv")

    def _write_empty(msg):
        pd.DataFrame(columns=GROUPS_COLUMNS).to_csv(output_groups_csv, index=False)
        pd.DataFrame(columns=CROSSINGS_COLUMNS).to_csv(output_crossings_csv, index=False)
        print(f"[social_groups] {msg} Header-only results saved to "
              f"{output_groups_csv} and {output_crossings_csv}")
        return output_groups_csv, output_crossings_csv

    if not _readable(tracks_csv):
        return _write_empty("Trajectory CSV missing/empty.")
    try:
        tracks = pd.read_csv(tracks_csv)
    except Exception as e:
        return _write_empty(f"Trajectory CSV unreadable ({e}).")
    required = {"frame_id", "timestamp", "track_id", "x1", "y1", "x2", "y2"}
    if tracks.empty or not required.issubset(tracks.columns):
        return _write_empty("Trajectory CSV empty or malformed.")

    cam_at = _make_cam_at(ego_csv)
    scale_fn = _make_scale_fn(tracks, scale_csv, speed_csv, video_name, mapping_csv)

    edges = detect_group_edges(tracks, cam_at=cam_at, scale_fn=scale_fn,
                               gap_max_m=gap_max_m, gap_std_max_m=gap_std_max_m,
                               cos_min=cos_min, sustain_s=sustain_s,
                               min_overlap_s=min_overlap_s, max_hole_s=max_hole_s,
                               smooth_window=smooth_window)
    group_rows, comps = build_group_rows(edges)
    pd.DataFrame(group_rows, columns=GROUPS_COLUMNS).to_csv(output_groups_csv, index=False)

    crossing_df = None
    if _readable(crossing_csv):
        try:
            crossing_df = pd.read_csv(crossing_csv)
        except Exception as e:
            print(f"[social_groups][warn] crossing_judge read failed: {e}")
    fps = _resolve_fps(meta_csv, video_name, tracks)
    crossing_rows = build_crossing_rows(comps, crossing_df, fps)
    pd.DataFrame(crossing_rows, columns=CROSSINGS_COLUMNS).to_csv(output_crossings_csv, index=False)

    print(f"[social_groups] {len(comps)} groups over {len(group_rows)} member tracks, "
          f"{len(crossing_rows)} platooning rows (fps={fps:.2f}). "
          f"Saved to {output_groups_csv} and {output_crossings_csv}")
    return output_groups_csv, output_crossings_csv


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Social group detection & platooning ([I2]/[I3]).")
    ap.add_argument("--source_video_path", required=True)
    ap.add_argument("--mapping_csv", default="mapping.csv")
    args = ap.parse_args()
    run_social_groups(args.source_video_path, mapping_csv=args.mapping_csv)
