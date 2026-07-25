"""Per-pedestrian speed estimation from a foot-point trajectory (module [S1]).

This is the first module that actually MEASURES a pedestrian's speed from the
footage, rather than importing a city-level constant. Pipeline (Priority #1):

  foot-point -> smoothing -> ego-motion compensation -> metric scale -> region split

* Foot-point ((x1+x2)/2, y2) = ground contact; the ground plane is valid there.
* Smoothing: a short rolling median on the foot series kills box jitter.
* Ego-motion: if [B3]ego_motion.csv says the camera is moving, subtract the camera's
  displacement so we measure the pedestrian's motion, not the camera's.
* Metric scale: a walking adult's stature is ~constant, so bbox height in pixels
  encodes the LOCAL pixels-per-metre scale at their image depth -> per-step local
  scale handles perspective without calibration (rung B). A global lane-width scale
  from [V5] is computed as an independent cross-check when available (rung A-lite).
* Region split: [C3]crossing_judge started/ended frames separate free-flow WALKING
  speed from curb-to-curb CROSSING speed and a decision/start delay.

Prefers the dense [B2]dense_tracks.csv; falls back to 1 Hz [B1] (coarse, never
flagged reliable). Honesty: within-video/relative speed is trustworthy; absolute
cross-city m/s is approximate on uncalibrated monocular video -> gate on `reliable`.
"""

import os
import re
import math
import numpy as np
import pandas as pd

DEFAULT_HEIGHT_M = 1.70
MIN_PLAUSIBLE_HEIGHT_M = 1.2
MAX_PLAUSIBLE_HEIGHT_M = 2.2
ASSUMED_LANE_WIDTH_M = 3.5
WAIT_SPEED_MPS = 0.3          # below this a pedestrian is "not walking" (waiting)
RUN_SPEED_MPS = 2.2          # above this they are running
# Occlusion/truncation guard for the height-prior scale: a standing/walking adult's
# bbox aspect (h/w) sits around ~1.6-4.5. A lower-body-occluded (truncated) or merged
# box falls outside this band; its SHORT height inflates px/m and thus the speed, so
# such tracks must never be flagged reliable.
MIN_BBOX_ASPECT = 1.4
MAX_BBOX_ASPECT = 5.0
# Forward-camera gate: |median relative bbox-height growth| above this (per second) means
# the camera is translating toward/away from the scene, which [B3]'s median-translation ego
# model cannot remove (radial flow cancels in the median). Static reference: a stopped-at-
# lights dashcam clip measures 0.000/s; driving clips measure 0.10-0.18/s.
EGO_STATIC_MAX_EXPANSION = 0.05
EGO_SMOOTH_SAMPLES = 15          # ~1 s at the 15 Hz dense rate
# Second half of the gate: residual background TRANSLATION from [B3]. Expansion catches a
# forward-driving camera; step_px catches pan/lateral motion, which inflates speed just as
# badly (binning per-step speed by step_px showed 0.68 -> 3.29 m/s as step_px grew).
EGO_STATIC_MAX_STEP_PX = 0.5
# A crossing chord may span a few contaminated intervals without being ruined; require the
# window to be overwhelmingly - not perfectly - camera-static.
CROSSING_STATIC_MIN_FRAC = 0.9

OUTPUT_COLUMNS = [
    "track_id", "n_valid_steps", "walking_speed_mps", "crossing_speed_mps",
    "net_speed_mps", "decision_delay_s", "is_running", "mean_speed_mps",
    "median_bbox_h_px", "height_cv", "assumed_height_m", "scale_px_per_m_median",
    "scale_source", "lane_scale_px_per_m", "camera_moving", "median_bbox_aspect",
    "ego_regime", "ego_static_frac", "traj_source", "reliable",
]


def _city_from_video_name(video_name):
    if "_" not in video_name:
        return video_name
    return re.sub(r"\d+$", "", video_name[:video_name.index("_")])


def _resolve_assumed_height_m(video_name, mapping_csv):
    city = _city_from_video_name(video_name)
    if mapping_csv and os.path.exists(mapping_csv):
        try:
            mp = pd.read_csv(mapping_csv)
            match = mp.loc[mp["city"] == city, "avg_height"].dropna()
            if not match.empty:
                h_m = float(match.iloc[0]) / 100.0
                if MIN_PLAUSIBLE_HEIGHT_M <= h_m <= MAX_PLAUSIBLE_HEIGHT_M:
                    return h_m, f"height_prior:{city}"
        except Exception as e:
            print(f"[speed][warn] could not read avg_height for {city}: {e}")
    return DEFAULT_HEIGHT_M, "height_prior:default"


def _rolling_median(a, w=3):
    a = np.asarray(a, dtype=float)
    if w < 3 or len(a) < w:
        return a
    half = w // 2
    out = a.copy()
    for i in range(half, len(a) - half):
        out[i] = np.median(a[i - half:i + half + 1])
    return out


def estimate_ego_expansion(df, min_tracks=3, lag_s=0.5):
    """Per-frame FORWARD-camera indicator, computed from the dense tracks alone.

    [B3]'s ego model is a median background TRANSLATION, which is structurally blind to a
    forward-driving camera: that motion is radial (left of frame flows left, right flows
    right), so the median translation cancels to ~0 and compensates nothing. The residual
    expansion then enters every pedestrian's displacement and inflates measured speed
    (observed: dashcam cities reported 1.4-2.6 m/s median "walking", p90 > 5 m/s).

    Forward motion has an unambiguous signature that survives in the CSVs: static objects
    approach, so ALL bounding boxes grow together. We therefore take, for each pair of
    consecutive dense frames, the median RELATIVE height growth rate across the tracks
    visible in both:

        expansion(f) = median_over_tracks( (h_next/h_prev - 1) / dt )      [1/s]

    A pedestrian's own approach/recede motion is idiosyncratic, so with several tracks the
    median reflects the camera.

    The comparison spans `lag_s` seconds rather than one frame-step: box coordinates are
    written as INTEGERS, so over a single 1/15 s step even a fast approach moves the box
    edge by ~1 px and the signal is quantised away to zero. Over ~0.5 s it is ~10 px and
    measurable. Returns {frame_id: expansion_rate_per_s} anchored at the interval start,
    for anchors where at least `min_tracks` tracks were matched across the lag.
    Pure function of the dataframe -> unit-testable without a video.
    """
    need = {"frame_id", "track_id", "y1", "y2", "timestamp"}
    if df is None or df.empty or not need.issubset(df.columns):
        return {}
    d = df[["frame_id", "track_id", "y1", "y2", "timestamp"]].copy()
    d["h"] = (d["y2"] - d["y1"]).astype(float)
    d = d[d["h"] > 1]
    if d.empty:
        return {}

    # per-anchor-frame accumulation of each track's relative growth rate across the lag
    acc = {}
    for _, g in d.groupby("track_id"):
        g = g.sort_values("timestamp")
        t = g["timestamp"].to_numpy(dtype=float)
        h = g["h"].to_numpy(dtype=float)
        f = g["frame_id"].to_numpy(dtype=float)
        if t.size < 2:
            continue
        # j[i] = first sample at least lag_s after sample i
        j = np.searchsorted(t, t + lag_s, side="left")
        for i in range(t.size):
            k = j[i]
            if k >= t.size:
                break
            dt = t[k] - t[i]
            if dt <= 0 or h[i] <= 1:
                continue
            acc.setdefault(int(f[i]), []).append((h[k] / h[i] - 1.0) / dt)

    return {fid: float(np.median(v)) for fid, v in acc.items() if len(v) >= min_tracks}


def _lane_scale_px_per_m(lane_csv):
    """Global px/m from [V5] lane geometry: median lane width at the frame bottom / 3.5 m."""
    if not (os.path.exists(lane_csv) and os.path.getsize(lane_csv) > 0):
        return None
    try:
        d = pd.read_csv(lane_csv)
    except Exception:
        return None
    widths = []
    for _, r in d.iterrows():
        # bottom of frame = the endpoint with the larger y on each lane line
        lx = r["left_x1"] if r["left_y1"] >= r["left_y2"] else r["left_x2"]
        rx = r["right_x1"] if r["right_y1"] >= r["right_y2"] else r["right_x2"]
        # lane_detection writes all-zero coords for an undetected side; a one-sided row
        # would contribute |rx - 0| as if it were a real lane width. Require BOTH sides.
        if lx == 0 or rx == 0:
            continue
        wpx = abs(float(rx) - float(lx))
        if wpx > 5:
            widths.append(wpx)
    if not widths:
        return None
    return float(np.median(widths)) / ASSUMED_LANE_WIDTH_M


def run_speed_estimation(video_path, trajectory_csv=None, mapping_csv="mapping.csv",
                         output_csv=None, assumed_height_m=None, min_samples=3,
                         max_step_speed_mps=8.0, smooth_window=3):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join("analysis_results", video_name)
    os.makedirs(output_dir, exist_ok=True)
    if output_csv is None:
        output_csv = os.path.join(output_dir, "[S1]pedestrian_speed.csv")

    dense_path = os.path.join(output_dir, "[B2]dense_tracks.csv")
    b1_path = os.path.join(output_dir, "[B1]tracked_pedestrians.csv")
    if trajectory_csv is None:
        if os.path.exists(dense_path) and os.path.getsize(dense_path) > 0:
            trajectory_csv, traj_source = dense_path, "dense"
        else:
            trajectory_csv, traj_source = b1_path, "sparse_1hz"
    else:
        traj_source = "custom"

    def _write_empty(msg):
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(output_csv, index=False)
        print(f"[speed] {msg} Empty results saved to {output_csv}")

    if not os.path.exists(trajectory_csv) or os.path.getsize(trajectory_csv) == 0:
        return _write_empty("Trajectory CSV missing/empty.")
    df = pd.read_csv(trajectory_csv)
    required = {"frame_id", "timestamp", "track_id", "x1", "y1", "x2", "y2"}
    # A header-only [B2] (pedestrian-free or aborted tracking run) must not defeat the
    # documented fallback to [B1]: decide on ROW COUNT, not file size.
    if df.empty and trajectory_csv == dense_path and os.path.exists(b1_path) and os.path.getsize(b1_path) > 0:
        b1_df = pd.read_csv(b1_path)
        if not b1_df.empty:
            df, trajectory_csv, traj_source = b1_df, b1_path, "sparse_1hz"
    if df.empty or not required.issubset(df.columns):
        return _write_empty("Trajectory CSV empty or malformed.")

    if assumed_height_m is None:
        assumed_height_m, height_source = _resolve_assumed_height_m(video_name, mapping_csv)
    else:
        height_source = "height_prior:override"

    # Optional ego-motion ([B3]). ALWAYS subtract the interpolated camera position when
    # available: the old all-or-nothing gate (global median step >= 1 px) left partial
    # pans fully uncompensated, and subtracting a near-zero series on static clips is
    # harmless. camera_moving remains as a reported flag only.
    ego_fx = ego_fy = None
    ego_step_frames = ego_step_px = None
    camera_moving = False
    ego_path = os.path.join(output_dir, "[B3]ego_motion.csv")
    if os.path.exists(ego_path) and os.path.getsize(ego_path) > 0:
        try:
            e = pd.read_csv(ego_path)
            if not e.empty:
                camera_moving = bool(e["step_px"].median() >= 1.0)
                ef = e.sort_values("frame_id")
                fr = ef["frame_id"].to_numpy(dtype=float)
                ego_fx = (fr, ef["cam_x"].to_numpy(dtype=float))
                ego_fy = (fr, ef["cam_y"].to_numpy(dtype=float))
                # residual translation magnitude per frame, for the second gate
                ego_step_frames = fr
                ego_step_px = _rolling_median(ef["step_px"].to_numpy(dtype=float), 5)
        except Exception as e:
            print(f"[speed][warn] ego-motion read failed: {e}")

    def cam_at(frame_ids):
        if ego_fx is None:
            return np.zeros_like(frame_ids, dtype=float), np.zeros_like(frame_ids, dtype=float)
        return (np.interp(frame_ids, ego_fx[0], ego_fx[1]),
                np.interp(frame_ids, ego_fy[0], ego_fy[1]))

    lane_scale = _lane_scale_px_per_m(os.path.join(output_dir, "[V5]lane_detection.csv"))

    # Rung A: ground-plane scale(y) = a*y + b from crosswalk stripe periodicity ([S2]).
    # Preferred over the height prior when its fit is good; falls back automatically.
    stripe_a = stripe_b = None
    s2_path = os.path.join(output_dir, "[S2]scale_calibration.csv")
    if os.path.exists(s2_path) and os.path.getsize(s2_path) > 0:
        try:
            sc = pd.read_csv(s2_path)
            if not sc.empty and str(sc.iloc[0].get("quality")) == "good":
                stripe_a = float(sc.iloc[0]["a"]); stripe_b = float(sc.iloc[0]["b"])
        except Exception as e:
            print(f"[speed][warn] stripe calibration read failed: {e}")
    scale_source = "stripe_ground_plane" if stripe_a is not None else "height_prior"

    # Forward-camera gate. Build a smoothed per-frame expansion series; a step is usable
    # only while the camera is NOT translating forward/backward. On a driving dashcam this
    # keeps the stopped-at-the-lights intervals — exactly when pedestrians cross — and
    # discards the driving intervals whose radial flow would inflate every speed.
    exp_map = estimate_ego_expansion(df)
    ego_frames = ego_static_flags = None
    ego_expansion_median = float("nan")
    if exp_map:
        ego_frames = np.array(sorted(exp_map), dtype=float)
        ev = np.array([exp_map[int(f)] for f in ego_frames], dtype=float)
        win = EGO_SMOOTH_SAMPLES if EGO_SMOOTH_SAMPLES % 2 else EGO_SMOOTH_SAMPLES + 1
        ev_s = _rolling_median(ev, win)
        ego_expansion_median = float(np.median(np.abs(ev_s)))
        ego_static_flags = np.abs(ev_s) <= EGO_STATIC_MAX_EXPANSION
        ego_regime = ("static" if ego_expansion_median <= EGO_STATIC_MAX_EXPANSION
                      else "forward_motion")
        print(f"[speed] ego regime: {ego_regime} (median |expansion| "
              f"{ego_expansion_median:.3f}/s); {int(ego_static_flags.sum())}/"
              f"{ego_static_flags.size} intervals camera-static")
    else:
        ego_regime = "unknown"

    def step_is_ego_static(frame_id):
        """True when the camera is still over the interval containing this step.

        TWO conditions, because the two camera motions corrupt speed by different routes:
          * expansion  -> forward/backward drive (radial flow; invisible to [B3]'s median)
          * step_px    -> pan / lateral translation (visible to [B3])
        Anchors are looked up by nearest PRECEDING interval start; frames before the first
        anchor (or with too few co-visible tracks) are trusted only on a globally static
        video."""
        if ego_step_px is not None:
            j = int(np.searchsorted(ego_step_frames, float(frame_id), side="right")) - 1
            if j >= 0 and ego_step_px[j] > EGO_STATIC_MAX_STEP_PX:
                return False
        if ego_static_flags is None:
            return True                      # no expansion info -> translation gate only
        i = int(np.searchsorted(ego_frames, float(frame_id), side="right")) - 1
        if i < 0:
            return ego_regime == "static"
        return bool(ego_static_flags[i])

    # Region context: [C3] crossing windows.
    cross_win = {}
    c3 = os.path.join(output_dir, "[C3]crossing_judge.csv")
    if os.path.exists(c3) and os.path.getsize(c3) > 0:
        try:
            cj = pd.read_csv(c3)
            for _, r in cj.iterrows():
                if bool(r.get("crossed")) and pd.notna(r.get("started_frame")):
                    cross_win[r["track_id"]] = (float(r["started_frame"]),
                                                float(r["ended_frame"]) if pd.notna(r.get("ended_frame")) else None)
        except Exception:
            pass

    rows = []
    for track_id, g in df.groupby("track_id"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        if len(g) < 2:
            continue

        fr = g["frame_id"].to_numpy(dtype=float)
        t = g["timestamp"].to_numpy(dtype=float)
        h_px = (g["y2"].to_numpy() - g["y1"].to_numpy()).astype(float)
        w_px = (g["x2"].to_numpy() - g["x1"].to_numpy()).astype(float)
        # Occlusion/truncation guard input: per-row bbox aspect (h/w). Guard w<=0
        # (degenerate boxes) out of the median rather than dividing by zero.
        aspect_valid = w_px > 0
        median_aspect = (float(np.median(h_px[aspect_valid] / w_px[aspect_valid]))
                         if aspect_valid.any() else float("nan"))
        cx, cy = cam_at(fr)                      # cumulative camera position at each row
        raw_fx = (g["x1"].to_numpy() + g["x2"].to_numpy()) / 2.0
        raw_fy = g["y2"].to_numpy(dtype=float)
        # Displacement series: ego-compensate BEFORE smoothing, so the rolling median
        # attenuates camera jitter and box jitter together (smoothing the foot but
        # subtracting RAW camera diffs re-injected shake as fake pedestrian motion).
        foot_x = _rolling_median(raw_fx - cx, smooth_window)
        foot_y = _rolling_median(raw_fy - cy, smooth_window)
        # Image-row series for scale(y) lookups: the ground-plane scale lives in IMAGE
        # space, so it must be evaluated at the uncompensated row.
        foot_y_img = _rolling_median(raw_fy, smooth_window)

        # Physical plausibility guard (per track): a scale that implies an absurd
        # pedestrian stature means the calibration is wrong for this depth/video —
        # refuse it and fall back to the height prior (live run caught a degenerate
        # 4 px/m calibration implying 10.6 m tall pedestrians rated 7 m/s "walking").
        use_stripe = stripe_a is not None
        if use_stripe:
            s_med = stripe_a * float(np.median(foot_y_img)) + stripe_b
            implied_h = (float(np.median(h_px)) / s_med) if s_med > 1 else float("inf")
            if not (0.9 <= implied_h <= 2.8):
                use_stripe = False

        step_speeds, step_scales, step_dts, per_step = [], [], [], []
        n_moving_steps = 0
        for i in range(len(g) - 1):
            dt = t[i + 1] - t[i]
            h_avg = 0.5 * (h_px[i] + h_px[i + 1])
            if dt <= 0 or h_avg <= 1:
                per_step.append(None); continue
            # Discard steps taken while the camera translates: the uncompensated radial
            # flow would be counted as pedestrian motion.
            if not step_is_ego_static(fr[i + 1]):
                n_moving_steps += 1
                per_step.append(None); continue
            if use_stripe:
                # ground-plane scale at the IMAGE foot row (rung A); degenerate near/above
                # the horizon -> fall back to this pedestrian's own height prior (rung B).
                scale = 0.5 * ((stripe_a * foot_y_img[i] + stripe_b) +
                               (stripe_a * foot_y_img[i + 1] + stripe_b))
                if scale <= 1:
                    scale = h_avg / assumed_height_m
            else:
                scale = h_avg / assumed_height_m
            # foot series is already ego-compensated
            dxp = foot_x[i + 1] - foot_x[i]
            dyp = foot_y[i + 1] - foot_y[i]
            v = (math.hypot(dxp, dyp) / scale) / dt
            if v > max_step_speed_mps:
                per_step.append(None); continue
            step_speeds.append(v); step_scales.append(scale); step_dts.append(dt)
            per_step.append(v)

        if not step_speeds:
            continue
        median_scale = float(np.median(step_scales))
        # Fraction of this track's usable steps taken with a non-translating camera.
        total_considered = len(step_speeds) + n_moving_steps
        ego_static_frac = (len(step_speeds) / total_considered) if total_considered else 0.0

        # net (chord) speed; foot series is already ego-compensated. A chord that spans
        # camera-translation intervals is contaminated end-to-end, so refuse it there.
        net_dx = foot_x[-1] - foot_x[0]
        net_dy = foot_y[-1] - foot_y[0]
        net_dt = t[-1] - t[0]
        net_speed = (math.hypot(net_dx, net_dy) / median_scale) / net_dt if net_dt > 0 else float("nan")
        if ego_static_frac < 0.8:
            net_speed = float("nan")

        # region-aware crossing speed + decision delay from the [C3] window
        crossing_speed = float("nan"); decision_delay = float("nan")
        win = cross_win.get(track_id)
        if win is not None:
            s_fr, e_fr = win
            e_fr = e_fr if e_fr is not None else fr[-1]
            in_win = (fr >= s_fr) & (fr <= e_fr)
            if in_win.sum() >= 2:
                idx = np.where(in_win)[0]
                a, b = idx[0], idx[-1]
                # Only trust the curb-to-curb chord if the camera held still across it.
                # A few contaminated intervals are tolerable; a driving stretch is not.
                win_flags = [step_is_ego_static(fr[k]) for k in range(a + 1, b + 1)]
                win_static = (sum(win_flags) / len(win_flags)) >= CROSSING_STATIC_MIN_FRAC if win_flags else False
                cdx = foot_x[b] - foot_x[a]
                cdy = foot_y[b] - foot_y[a]
                cdt = t[b] - t[a]
                if cdt > 0 and win_static:
                    crossing_speed = (math.hypot(cdx, cdy) / median_scale) / cdt
            # decision delay: contiguous near-stationary time just before crossing start.
            # No observations before the start (track begins mid-crossing) -> unknown (NaN),
            # not a false 0.0.
            start_idx = int(np.argmax(fr >= s_fr))
            if start_idx > 0:
                delay = 0.0
                j = start_idx - 1
                while j >= 0 and per_step[j] is not None and per_step[j] < WAIT_SPEED_MPS:
                    delay += (t[j + 1] - t[j]); j -= 1
                decision_delay = round(delay, 2)

        median_h = float(np.median(h_px))
        height_cv = float(np.std(h_px) / np.mean(h_px)) if np.mean(h_px) > 0 else float("nan")
        median_dt = float(np.median(step_dts))
        n_steps = len(step_speeds)
        walking = float(np.median(step_speeds))
        # A truncated (lower-body-occluded) or merged box breaks the height-prior
        # scale, so an out-of-band aspect (or none computable) also fails the gate.
        aspect_ok = (median_aspect == median_aspect
                     and MIN_BBOX_ASPECT <= median_aspect <= MAX_BBOX_ASPECT)
        # An uncompensated translating camera is the single largest speed-inflation source,
        # so a track measured mostly while driving is never reliable.
        reliable = bool(median_dt <= 0.2 and n_steps >= min_samples
                        and median_h >= 40 and height_cv < 0.35 and aspect_ok
                        and ego_static_frac >= 0.8)

        rows.append({
            "track_id": track_id,
            "n_valid_steps": n_steps,
            "walking_speed_mps": round(walking, 3),
            "crossing_speed_mps": round(crossing_speed, 3) if crossing_speed == crossing_speed else None,
            "net_speed_mps": round(net_speed, 3) if net_speed == net_speed else None,
            "decision_delay_s": decision_delay if decision_delay == decision_delay else None,
            "is_running": bool(walking >= RUN_SPEED_MPS),
            "mean_speed_mps": round(float(np.mean(step_speeds)), 3),
            "median_bbox_h_px": round(median_h, 1),
            "height_cv": round(height_cv, 3) if height_cv == height_cv else None,
            "assumed_height_m": round(assumed_height_m, 3),
            "scale_px_per_m_median": round(median_scale, 3),
            "scale_source": "stripe_ground_plane" if use_stripe else "height_prior",
            "lane_scale_px_per_m": round(lane_scale, 3) if lane_scale else None,
            "camera_moving": camera_moving,
            "median_bbox_aspect": round(median_aspect, 3) if median_aspect == median_aspect else None,
            "ego_regime": ego_regime,
            "ego_static_frac": round(ego_static_frac, 3),
            "traj_source": traj_source,
            "reliable": reliable,
        })

    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out.to_csv(output_csv, index=False)
    n_rel = int(out["reliable"].sum()) if not out.empty else 0
    n_stripe = int((out["scale_source"] == "stripe_ground_plane").sum()) if not out.empty else 0
    print(f"[speed] {len(out)} tracks ({n_rel} reliable, {n_stripe} stripe-scaled / "
          f"{len(out) - n_stripe} height-prior), source={traj_source}, "
          f"height={assumed_height_m:.2f} m ({height_source}), "
          f"ego={ego_regime}, camera={'moving' if camera_moving else 'static'}, "
          f"lane_scale={'%.1f' % lane_scale if lane_scale else 'n/a'}. Saved to {output_csv}")
    return output_csv


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Per-pedestrian speed estimation ([S1]).")
    ap.add_argument("--source_video_path", required=True)
    ap.add_argument("--mapping_csv", default="mapping.csv")
    ap.add_argument("--assumed_height_m", type=float, default=None)
    args = ap.parse_args()
    run_speed_estimation(args.source_video_path, mapping_csv=args.mapping_csv,
                         assumed_height_m=args.assumed_height_m)
