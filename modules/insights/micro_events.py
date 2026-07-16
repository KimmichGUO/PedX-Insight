"""Curb-dance hesitation + evasive-burst micro-event profiler (module [P11]).

Segments each pedestrian's dense metric speed series into STOP/SLOW/WALK states
(hysteresis bands + minimum state dwell) and counts behavioral micro-events
anchored to the [C3] crossing window:

* ``aborted_start``  - a short WALK burst (< 1.5 s) that begins within 10 s
  BEFORE the crossing start and collapses straight back to STOP: the classic
  "curb dance" false start. [S1]'s ``decision_delay_s`` only sees the single
  contiguous stationary interval immediately before the start, so repeated
  false starts are invisible to it.
* ``midcross_stop``  - a STOP state that begins inside the crossing window
  (mid-road freeze).
* ``evasive``        - an acceleration burst >= 1.5 m/s^2 sustained >= 0.4 s
  whose terminal speed reaches max(2.0 m/s, 1.6x the pre-event median speed),
  inside the crossing window: the sprint-out-of-the-way signature.  Counted
  only for tracks whose speed measurement is ``reliable`` (from [S1] when
  available, else recomputed with [S1]'s own criteria), because acceleration
  doubles the scale-noise sensitivity.

Speed recomputation is intentionally IDENTICAL to [S1]pedestrian_speed
(same rolling-median smoothing, ego-motion compensation from [B3], and scale
priority: [S2] stripe ground-plane when quality=='good' and per-track height
plausibility holds, else the per-track bbox-height prior).

Inputs (all per-video CSVs, all optional -> graceful degradation):
  [B2]dense_tracks.csv (fallback [B1]tracked_pedestrians.csv), [B3]ego_motion.csv,
  [C3]crossing_judge.csv, [S1]pedestrian_speed.csv (reliable flag),
  [S2]scale_calibration.csv, [C1]risky_crossing.csv + [C6]crossing_ve_count.csv
  (validation join columns only).
No video access is needed: the module is CSV-only and safe to run after the
source video has been deleted (timestamps come from the trajectory CSV itself,
so [B0]video_meta.csv is not required).

Output: analysis_results/<video>/[P11]micro_events.csv, one row per pedestrian
track that has at least one valid speed step.
"""

import os
import math

import numpy as np
import pandas as pd

from modules.speed.speed_estimation import (
    _rolling_median,
    _resolve_assumed_height_m,
    WAIT_SPEED_MPS,
)

# --- segmentation thresholds -------------------------------------------------
STOP_SPEED_MPS = WAIT_SPEED_MPS      # v <  0.3 m/s -> STOP   (strict <)
WALK_SPEED_MPS = 0.8                 # v >  0.8 m/s -> WALK   (strict >)
                                     # in between   -> SLOW
MIN_STATE_DWELL_S = 0.4              # shorter runs are merged into neighbours

# --- event thresholds ---------------------------------------------------------
ABORT_LOOKBACK_S = 10.0              # aborted start must begin <= 10 s before crossing start
ABORT_MAX_DUR_S = 1.5                # ... and last < 1.5 s
EVASIVE_ACCEL_MPS2 = 1.5             # sustained acceleration threshold
EVASIVE_MIN_DUR_S = 0.4              # ... sustained at least this long
EVASIVE_MIN_SPEED_MPS = 2.0          # absolute terminal-speed floor
EVASIVE_SPEED_RATIO = 1.6            # terminal speed vs pre-event median speed

MAX_STEP_SPEED_MPS = 8.0             # same per-step outlier guard as [S1]
STOP_TIME_CAP_S = 30.0               # cap of the stop-time term in hesitation_score

OUTPUT_COLUMNS = [
    "track_id", "n_aborted_starts", "n_midcross_stops", "total_stop_time_s",
    "n_evasive_events", "max_burst_speed_mps", "first_event_progress_frac",
    "hesitation_score", "n_states", "reliable",
    # validation-join extras (not part of the core spec schema; adding is allowed)
    "c1_risk", "c6_total_vehicle_count",
]


# =============================================================================
# Pure core (arrays in -> rows out); unit-testable without any file I/O.
# =============================================================================

def segment_speed_states(t, per_step, stop_thr=STOP_SPEED_MPS,
                         walk_thr=WALK_SPEED_MPS, min_dwell_s=MIN_STATE_DWELL_S):
    """Segment a per-step speed series into STOP/SLOW/WALK states.

    ``t``        : row timestamps, length n (seconds).
    ``per_step`` : length n-1; speed of step i covering [t[i], t[i+1]] in m/s,
                   or None for an invalid step (invalid steps break contiguity).

    Banding is strict (STOP: v < stop_thr; WALK: v > walk_thr; SLOW between),
    then a temporal-hysteresis pass merges every run shorter than
    ``min_dwell_s`` into its longer neighbour, so jitter-length excursions
    never register as states.  Returns a list of dicts
    {label, t0, t1, dur} ordered in time.
    """
    t = np.asarray(t, dtype=float)
    n_steps = len(per_step)
    if len(t) != n_steps + 1:
        raise ValueError("t must have exactly len(per_step)+1 entries")

    # 1) raw labelled runs; None speed breaks the sequence into segments
    segments, current = [], []
    for i in range(n_steps):
        v = per_step[i]
        if v is None or (isinstance(v, float) and math.isnan(v)):
            if current:
                segments.append(current)
                current = []
            continue
        if v < stop_thr:
            label = "STOP"
        elif v > walk_thr:
            label = "WALK"
        else:
            label = "SLOW"
        if current and current[-1][0] == label:
            current[-1][2] = t[i + 1]
        else:
            current.append([label, t[i], t[i + 1]])
    if current:
        segments.append(current)

    # 2) dwell merging within each contiguous segment
    # (epsilon so a run of exactly min_dwell_s survives despite float rounding)
    dwell_eps = 1e-9
    states = []
    for runs in segments:
        runs = [list(r) for r in runs]
        while len(runs) > 1:
            durs = [r[2] - r[1] for r in runs]
            short = [(d, i) for i, d in enumerate(durs) if d < min_dwell_s - dwell_eps]
            if not short:
                break
            _, idx = min(short)             # merge the shortest offender first
            left = idx - 1 if idx > 0 else None
            right = idx + 1 if idx < len(runs) - 1 else None
            if left is None:
                tgt = right
            elif right is None:
                tgt = left
            else:
                tgt = left if durs[left] >= durs[right] else right
            lo, hi = (tgt, idx) if tgt < idx else (idx, tgt)
            runs[lo] = [runs[tgt][0], runs[lo][1], runs[hi][2]]
            del runs[hi]
            # coalesce adjacent same-label runs created by the merge
            merged = []
            for r in runs:
                if merged and merged[-1][0] == r[0]:
                    merged[-1][2] = r[2]
                else:
                    merged.append(r)
            runs = merged
        for label, t0, t1 in runs:
            states.append({"label": label, "t0": float(t0), "t1": float(t1),
                           "dur": float(t1 - t0)})
    return states


def detect_events(states, t, per_step, window_s=None, reliable=False,
                  abort_lookback_s=ABORT_LOOKBACK_S, abort_max_dur_s=ABORT_MAX_DUR_S,
                  evasive_accel=EVASIVE_ACCEL_MPS2, evasive_min_dur_s=EVASIVE_MIN_DUR_S,
                  evasive_min_speed=EVASIVE_MIN_SPEED_MPS, evasive_ratio=EVASIVE_SPEED_RATIO):
    """Count micro-events for one track.

    ``window_s`` = (t_start, t_end) of the [C3] crossing window in seconds, or
    None when the track never crossed (all window-anchored counts are then 0).
    Returns the per-track stat dict (without track_id / join columns).
    """
    t = np.asarray(t, dtype=float)
    eps = 1e-6

    total_stop = sum(s["dur"] for s in states if s["label"] == "STOP")

    n_aborted = 0
    midcross_onsets = []
    evasive_events = []           # (onset_s, terminal_speed, burst_max_speed)

    if window_s is not None:
        ts, te = float(window_s[0]), float(window_s[1])

        # aborted starts: short WALK burst just before the crossing start that
        # collapses straight back to STOP (next temporally-adjacent state).
        for k, s in enumerate(states):
            if s["label"] != "WALK" or s["dur"] >= abort_max_dur_s:
                continue
            if not (ts - abort_lookback_s <= s["t0"] < ts):
                continue
            nxt = states[k + 1] if k + 1 < len(states) else None
            if nxt is not None and nxt["label"] == "STOP" and nxt["t0"] <= s["t1"] + eps:
                n_aborted += 1

        # mid-crossing stops: STOP state beginning inside the window
        for s in states:
            if s["label"] == "STOP" and ts <= s["t0"] < te:
                midcross_onsets.append(s["t0"])

        # evasive bursts: sustained acceleration inside the window (reliable only)
        if reliable:
            tm, vv = [], []
            for i, v in enumerate(per_step):
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    tm.append(None)
                else:
                    tm.append(0.5 * (t[i] + t[i + 1]))
                    vv.append(float(v))
            # contiguous valid stretches of step midtimes/speeds
            stretches, cur_t, cur_v = [], [], []
            vi = 0
            for x in tm:
                if x is None:
                    if len(cur_t) >= 2:
                        stretches.append((np.array(cur_t), np.array(cur_v)))
                    cur_t, cur_v = [], []
                else:
                    cur_t.append(x)
                    cur_v.append(vv[vi])
                    vi += 1
            if len(cur_t) >= 2:
                stretches.append((np.array(cur_t), np.array(cur_v)))

            for st_t, st_v in stretches:
                dts = np.diff(st_t)
                acc = np.diff(st_v) / np.where(dts > 0, dts, np.nan)
                i = 0
                while i < len(acc):
                    if not (acc[i] == acc[i] and acc[i] >= evasive_accel):
                        i += 1
                        continue
                    j = i
                    while j + 1 < len(acc) and acc[j + 1] == acc[j + 1] and acc[j + 1] >= evasive_accel:
                        j += 1
                    onset, end = st_t[i], st_t[j + 1]
                    duration = end - onset
                    terminal = float(st_v[j + 1])
                    burst_max = float(np.max(st_v[i:j + 2]))
                    prior = st_v[st_t < onset]
                    floor = evasive_min_speed
                    if prior.size:
                        floor = max(floor, evasive_ratio * float(np.median(prior)))
                    if (duration >= evasive_min_dur_s and terminal >= floor
                            and ts <= onset <= te):
                        evasive_events.append((float(onset), terminal, burst_max))
                    i = j + 1

    # first event progress fraction (events inside the window only)
    first_frac = None
    if window_s is not None:
        ts, te = float(window_s[0]), float(window_s[1])
        onsets = sorted(midcross_onsets + [e[0] for e in evasive_events])
        if onsets and te > ts:
            first_frac = float(np.clip((onsets[0] - ts) / (te - ts), 0.0, 1.0))

    max_burst = max((e[2] for e in evasive_events), default=None)
    hesitation = (n_aborted + len(midcross_onsets)
                  + min(total_stop, STOP_TIME_CAP_S) / 10.0)

    return {
        "n_aborted_starts": n_aborted,
        "n_midcross_stops": len(midcross_onsets),
        "total_stop_time_s": round(total_stop, 2),
        "n_evasive_events": len(evasive_events),
        "max_burst_speed_mps": round(max_burst, 3) if max_burst is not None else None,
        "first_event_progress_frac": round(first_frac, 3) if first_frac is not None else None,
        "hesitation_score": round(hesitation, 3),
        "n_states": len(states),
    }


def compute_step_speeds(g, cam_at, stripe_a, stripe_b, assumed_height_m,
                        smooth_window=3, max_step_speed_mps=MAX_STEP_SPEED_MPS):
    """Per-step metric speeds for one sorted track dataframe.

    Replicates [S1]speed_estimation's loop exactly: ego-compensate before
    smoothing, stripe ground-plane scale when plausible for this track,
    per-track bbox-height prior otherwise, and the same outlier guard.
    Returns (t_rows, per_step, diagnostics) where per_step[i] covers
    [t[i], t[i+1]] and is None for an invalid/outlier step.
    """
    fr = g["frame_id"].to_numpy(dtype=float)
    t = g["timestamp"].to_numpy(dtype=float)
    h_px = (g["y2"].to_numpy() - g["y1"].to_numpy()).astype(float)
    cx, cy = cam_at(fr)
    raw_fx = (g["x1"].to_numpy() + g["x2"].to_numpy()) / 2.0
    raw_fy = g["y2"].to_numpy(dtype=float)
    foot_x = _rolling_median(raw_fx - cx, smooth_window)
    foot_y = _rolling_median(raw_fy - cy, smooth_window)
    foot_y_img = _rolling_median(raw_fy, smooth_window)

    use_stripe = stripe_a is not None
    if use_stripe:
        s_med = stripe_a * float(np.median(foot_y_img)) + stripe_b
        implied_h = (float(np.median(h_px)) / s_med) if s_med > 1 else float("inf")
        if not (0.9 <= implied_h <= 2.8):
            use_stripe = False

    per_step, step_dts = [], []
    for i in range(len(g) - 1):
        dt = t[i + 1] - t[i]
        h_avg = 0.5 * (h_px[i] + h_px[i + 1])
        if dt <= 0 or h_avg <= 1:
            per_step.append(None)
            continue
        if use_stripe:
            scale = 0.5 * ((stripe_a * foot_y_img[i] + stripe_b) +
                           (stripe_a * foot_y_img[i + 1] + stripe_b))
            if scale <= 1:
                scale = h_avg / assumed_height_m
        else:
            scale = h_avg / assumed_height_m
        dxp = foot_x[i + 1] - foot_x[i]
        dyp = foot_y[i + 1] - foot_y[i]
        v = (math.hypot(dxp, dyp) / scale) / dt
        if v > max_step_speed_mps:
            per_step.append(None)
            continue
        per_step.append(v)
        step_dts.append(dt)

    n_valid = sum(1 for v in per_step if v is not None)
    diag = {
        "n_valid_steps": n_valid,
        "median_dt": float(np.median(step_dts)) if step_dts else float("nan"),
        "median_h_px": float(np.median(h_px)),
        "height_cv": float(np.std(h_px) / np.mean(h_px)) if np.mean(h_px) > 0 else float("nan"),
    }
    return t, per_step, diag


def _internal_reliable(diag, min_samples=3):
    """[S1]'s reliability criteria, recomputed when [S1] has no row for a track."""
    return bool(diag["median_dt"] == diag["median_dt"] and diag["median_dt"] <= 0.2
                and diag["n_valid_steps"] >= min_samples
                and diag["median_h_px"] >= 40
                and diag["height_cv"] == diag["height_cv"] and diag["height_cv"] < 0.35)


# =============================================================================
# Entry point
# =============================================================================

def _safe_read_csv(path):
    if path and os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"[micro_events][warn] could not read {path}: {e}")
    return None


def _as_bool(v):
    return str(v).strip().lower() == "true"


def run_micro_events(video_path, trajectory_csv=None, mapping_csv="mapping.csv",
                     output_csv=None, assumed_height_m=None, smooth_window=3,
                     max_step_speed_mps=MAX_STEP_SPEED_MPS):
    """Profile curb-dance / freeze / evasive micro-events for every pedestrian.

    All inputs are read from analysis_results/<video_name>/; every one of them
    may be missing (a missing trajectory yields a header-only output CSV).
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join("analysis_results", video_name)
    os.makedirs(output_dir, exist_ok=True)
    if output_csv is None:
        output_csv = os.path.join(output_dir, "[P11]micro_events.csv")

    def _write_empty(msg):
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(output_csv, index=False)
        print(f"[micro_events] {msg} Empty results saved to {output_csv}")
        return output_csv

    dense_path = os.path.join(output_dir, "[B2]dense_tracks.csv")
    b1_path = os.path.join(output_dir, "[B1]tracked_pedestrians.csv")
    if trajectory_csv is None:
        if os.path.exists(dense_path) and os.path.getsize(dense_path) > 0:
            trajectory_csv, traj_source = dense_path, "dense"
        else:
            trajectory_csv, traj_source = b1_path, "sparse_1hz"
    else:
        traj_source = "custom"

    if not os.path.exists(trajectory_csv) or os.path.getsize(trajectory_csv) == 0:
        return _write_empty("Trajectory CSV missing/empty.")
    try:
        df = pd.read_csv(trajectory_csv)
    except Exception as e:
        return _write_empty(f"Trajectory CSV unreadable ({e}).")
    required = {"frame_id", "timestamp", "track_id", "x1", "y1", "x2", "y2"}
    # header-only [B2] must not defeat the [B1] fallback (row count, not file size)
    if df.empty and trajectory_csv == dense_path and os.path.exists(b1_path) \
            and os.path.getsize(b1_path) > 0:
        try:
            b1_df = pd.read_csv(b1_path)
            if not b1_df.empty:
                df, trajectory_csv, traj_source = b1_df, b1_path, "sparse_1hz"
        except Exception:
            pass
    if df.empty or not required.issubset(df.columns):
        return _write_empty("Trajectory CSV empty or malformed.")

    if assumed_height_m is None:
        assumed_height_m, _hsrc = _resolve_assumed_height_m(video_name, mapping_csv)

    # ego-motion ([B3]) -> interpolated cumulative camera position
    ego_fx = ego_fy = None
    e = _safe_read_csv(os.path.join(output_dir, "[B3]ego_motion.csv"))
    if e is not None and not e.empty:
        try:
            ef = e.sort_values("frame_id")
            frv = ef["frame_id"].to_numpy(dtype=float)
            ego_fx = (frv, ef["cam_x"].to_numpy(dtype=float))
            ego_fy = (frv, ef["cam_y"].to_numpy(dtype=float))
        except Exception as ex:
            print(f"[micro_events][warn] ego-motion read failed: {ex}")

    def cam_at(frame_ids):
        if ego_fx is None:
            return (np.zeros_like(frame_ids, dtype=float),
                    np.zeros_like(frame_ids, dtype=float))
        return (np.interp(frame_ids, ego_fx[0], ego_fx[1]),
                np.interp(frame_ids, ego_fy[0], ego_fy[1]))

    # stripe ground-plane scale ([S2]) - usable only when quality == 'good'
    stripe_a = stripe_b = None
    sc = _safe_read_csv(os.path.join(output_dir, "[S2]scale_calibration.csv"))
    if sc is not None and not sc.empty:
        try:
            if str(sc.iloc[0].get("quality")) == "good":
                stripe_a = float(sc.iloc[0]["a"])
                stripe_b = float(sc.iloc[0]["b"])
        except Exception as ex:
            print(f"[micro_events][warn] stripe calibration read failed: {ex}")

    # [C3] crossing windows (frames; ~1 s quantized because computed from 1 Hz [B1])
    cross_win = {}
    cj = _safe_read_csv(os.path.join(output_dir, "[C3]crossing_judge.csv"))
    if cj is not None:
        for _, r in cj.iterrows():
            try:
                if _as_bool(r.get("crossed")) and pd.notna(r.get("started_frame")):
                    ef_ = float(r["ended_frame"]) if pd.notna(r.get("ended_frame")) else None
                    cross_win[r["track_id"]] = (float(r["started_frame"]), ef_)
            except Exception:
                continue

    # [S1] reliable flags
    s1_reliable = {}
    s1 = _safe_read_csv(os.path.join(output_dir, "[S1]pedestrian_speed.csv"))
    if s1 is not None and "reliable" in s1.columns:
        for _, r in s1.iterrows():
            s1_reliable[r["track_id"]] = _as_bool(r["reliable"])

    # validation-join extras
    c1_risk = {}
    c1 = _safe_read_csv(os.path.join(output_dir, "[C1]risky_crossing.csv"))
    if c1 is not None and "risk" in c1.columns:
        for _, r in c1.iterrows():
            c1_risk[r["track_id"]] = r["risk"]
    c6_count = {}
    c6 = _safe_read_csv(os.path.join(output_dir, "[C6]crossing_ve_count.csv"))
    if c6 is not None and "total_vehicle_count" in c6.columns:
        for _, r in c6.iterrows():
            try:
                c6_count[r["track_id"]] = int(r["total_vehicle_count"])
            except Exception:
                continue

    rows = []
    for track_id, g in df.groupby("track_id"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        if len(g) < 2:
            continue
        t, per_step, diag = compute_step_speeds(
            g, cam_at, stripe_a, stripe_b, assumed_height_m,
            smooth_window=smooth_window, max_step_speed_mps=max_step_speed_mps)
        if diag["n_valid_steps"] == 0:
            continue

        # crossing window frames -> track-local seconds
        window_s = None
        win = cross_win.get(track_id)
        if win is not None:
            fr = g["frame_id"].to_numpy(dtype=float)
            s_fr, e_fr = win
            ts = float(np.interp(s_fr, fr, t))
            te = float(np.interp(e_fr, fr, t)) if e_fr is not None else float(t[-1])
            if te > ts:
                window_s = (ts, te)

        reliable = s1_reliable.get(track_id, _internal_reliable(diag))
        states = segment_speed_states(t, per_step)
        stats = detect_events(states, t, per_step, window_s=window_s,
                              reliable=reliable)
        stats.update({
            "track_id": track_id,
            "reliable": reliable,
            "c1_risk": c1_risk.get(track_id),
            "c6_total_vehicle_count": c6_count.get(track_id),
        })
        rows.append(stats)

    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out.to_csv(output_csv, index=False)
    if out.empty:
        print(f"[micro_events] no usable tracks (source={traj_source}). "
              f"Saved to {output_csv}")
    else:
        n_ev = int(out["n_evasive_events"].sum())
        n_ab = int(out["n_aborted_starts"].sum())
        n_ms = int(out["n_midcross_stops"].sum())
        print(f"[micro_events] {len(out)} tracks (source={traj_source}): "
              f"{n_ab} aborted starts, {n_ms} mid-crossing stops, {n_ev} evasive "
              f"bursts. Saved to {output_csv}")
    return output_csv


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Curb-dance / freeze / evasive micro-event profiler ([P11]).")
    ap.add_argument("--source_video_path", required=True)
    ap.add_argument("--mapping_csv", default="mapping.csv")
    ap.add_argument("--assumed_height_m", type=float, default=None)
    args = ap.parse_args()
    run_micro_events(args.source_video_path, mapping_csv=args.mapping_csv,
                     assumed_height_m=args.assumed_height_m)
