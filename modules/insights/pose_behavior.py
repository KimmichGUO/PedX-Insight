"""Pose-based head-scanning + gait profiler (module [P12], research tier).

For every pedestrian track that [C3]crossing_judge marks as ``crossed``, runs
ultralytics ``yolo11x-pose`` (COCO 17-keypoint) on video frames sampled at
~15 Hz inside the crossing window +/- 5 s, matches each detected pose person to
the crossing track via per-frame IoU against the interpolated [B2] bbox, and
computes:

* head-scanning proxy (pre-crossing 5 s window):
  per-frame head yaw sign from keypoint geometry -- nose x relative to the
  mid-shoulder x, normalized by shoulder pixel width; when the nose/shoulder
  geometry is missing or inconclusive (|yaw| below threshold), a strong
  ear/eye visibility asymmetry acts as tiebreak (left-side facial keypoints
  clearly more confident => head turned toward image-left, and vice versa;
  this assumes a roughly camera-facing pedestrian, which is the typical
  street-cam crossing geometry).  Sustained (>= 0.3 s) looks toward
  image-left / image-right are counted -> ``looked_left``, ``looked_right``,
  ``n_head_turns`` (total sustained looks, left + right).  "Left"/"right"
  are IMAGE-space directions, not egocentric.
* gait cadence (crossing window): FFT of the ankle-y series (mean of both
  ankles when both are confident, else the confident one), linearly
  detrended and Hann-windowed, evaluated on the ~15 Hz pose sampling grid.
  The dominant peak in the 0.5-3.5 Hz band (typical walking cadence
  1.5-2.5 Hz) is accepted only when it concentrates >= 30 % of the band
  energy (rejects noise-only spectra); the fine frequency comes from a
  zero-padded FFT (grid << 0.1 Hz) -> ``cadence_hz``, and
  ``step_count`` = round(cadence_hz * window duration).
* ``reliable`` honesty gate: >= 20 matched pose frames AND median keypoint
  confidence >= 0.3 AND median matched-person bbox height >= 80 px.

Inputs (per-video, under analysis_results/<video>/):
  the source VIDEO itself (this module must run before deletion),
  [C3]crossing_judge.csv  (crossing windows; required),
  [B2]dense_tracks.csv    (bbox interpolation; falls back to [B1]),
  [B0]video_meta.csv      (fps; falls back to cv2 metadata).

When the video (or any required CSV) is absent -- which is the case for every
archived folder, since videos are deleted after analysis -- a header-only
output CSV is written.  The module activates on future batch runs.

Output: analysis_results/<video>/[P12]pose_behavior.csv, one row per crossed
track (tracks with zero matched pose frames still get a row, flagged
unreliable).

The metric core is PURE (keypoint/conf arrays in -> metrics out): see
``head_yaw_series``, ``count_sustained_looks``, ``gait_cadence`` and
``compute_track_metrics``.  Unit tests need no video, no GPU and no
ultralytics import (heavy imports are lazy, inside the video branch).
"""

import os
import math

import numpy as np
import pandas as pd

# --- COCO-17 keypoint indices --------------------------------------------------
KP_NOSE = 0
KP_L_EYE, KP_R_EYE = 1, 2
KP_L_EAR, KP_R_EAR = 3, 4
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_ANKLE, KP_R_ANKLE = 15, 16

# --- head-scanning parameters ---------------------------------------------------
YAW_THR = 0.25            # |nose offset| / shoulder width that counts as a look
MIN_LOOK_S = 0.3          # a look must be sustained at least this long
MAX_GAP_S = 0.21          # bridge up to ~2 dropped 15 Hz samples inside a look
TIEBREAK_DELTA = 0.35     # ear+eye conf-sum asymmetry that resolves an ambiguous yaw
MIN_SHOULDER_PX = 5.0     # narrower shoulder spans make the normalization degenerate

# --- gait parameters ------------------------------------------------------------
CADENCE_BAND_HZ = (0.5, 3.5)   # search band (typical walking 1.5-2.5 Hz)
MIN_PEAK_FRAC = 0.3            # peak+/-1 coarse bin must hold >= this band-energy share
MIN_CADENCE_FRAMES = 16        # minimum valid ankle samples for an FFT
MIN_CADENCE_SPAN_S = 2.0       # minimum time span (frequency resolution floor)

# --- shared / reliability -------------------------------------------------------
KP_CONF_MIN = 0.3              # per-keypoint confidence floor for geometry use
RELIABLE_MIN_FRAMES = 20
RELIABLE_MIN_CONF = 0.3
RELIABLE_MIN_BBOX_H = 80.0

# --- video sampling / matching --------------------------------------------------
TARGET_HZ = 15.0
PRE_WINDOW_S = 5.0             # head-scanning lookback before the crossing start
POST_WINDOW_S = 5.0            # extra tail after the crossing end (context)
IOU_MATCH_THR = 0.3
DET_CONF = 0.25
POSE_MODEL = "yolo11x-pose.pt"     # ultralytics auto-downloads on first use

OUTPUT_COLUMNS = [
    "track_id", "n_pose_frames", "looked_left", "looked_right", "n_head_turns",
    "cadence_hz", "step_count", "median_kp_conf", "reliable",
]


# =============================================================================
# Pure core (keypoint arrays in -> metrics out); no I/O, no video, no GPU.
# =============================================================================

def head_yaw_series(kps, confs, kp_conf_min=KP_CONF_MIN, yaw_thr=YAW_THR,
                    tiebreak_delta=TIEBREAK_DELTA, min_shoulder_px=MIN_SHOULDER_PX):
    """Per-frame head-yaw proxy for one track.

    ``kps``   : (n, 17, 2) keypoint xy pixels.
    ``confs`` : (n, 17) keypoint confidences.

    Primary signal: (nose_x - mid_shoulder_x) / shoulder_width -- negative
    means the nose points toward image-LEFT, positive toward image-RIGHT.
    Tiebreak (only when the primary is unavailable or |yaw| < yaw_thr): a
    strong ear+eye confidence asymmetry snaps yaw to -/+ yaw_thr (left-side
    facial keypoints clearly more visible => looking image-left under the
    frontal-view assumption).  Frames with neither signal are NaN.
    """
    kps = np.asarray(kps, dtype=float)
    confs = np.asarray(confs, dtype=float)
    n = kps.shape[0]
    yaw = np.full(n, np.nan)
    for i in range(n):
        c = confs[i]
        val = float("nan")
        if c[KP_L_SHOULDER] >= kp_conf_min and c[KP_R_SHOULDER] >= kp_conf_min:
            xl = kps[i, KP_L_SHOULDER, 0]
            xr = kps[i, KP_R_SHOULDER, 0]
            width = abs(xl - xr)
            if width >= min_shoulder_px and c[KP_NOSE] >= kp_conf_min:
                val = (kps[i, KP_NOSE, 0] - 0.5 * (xl + xr)) / width
        if math.isnan(val) or abs(val) < yaw_thr:
            vis_l = c[KP_L_EYE] + c[KP_L_EAR]
            vis_r = c[KP_R_EYE] + c[KP_R_EAR]
            if vis_l - vis_r >= tiebreak_delta:
                val = -yaw_thr
            elif vis_r - vis_l >= tiebreak_delta:
                val = yaw_thr
        yaw[i] = val
    return yaw


def count_sustained_looks(t, yaw, yaw_thr=YAW_THR, min_look_s=MIN_LOOK_S,
                          max_gap_s=MAX_GAP_S):
    """Count sustained image-left / image-right looks in a yaw series.

    A look is a run of frames with yaw <= -yaw_thr (left) or >= +yaw_thr
    (right) whose covered duration (last - first sample time + one median
    sample period) is >= min_look_s.  NaN or neutral frames end a run; gaps
    between same-label samples larger than ``max_gap_s`` also end it.
    Returns (n_left, n_right).
    """
    t = np.asarray(t, dtype=float)
    yaw = np.asarray(yaw, dtype=float)
    if len(t) != len(yaw):
        raise ValueError("t and yaw must have the same length")
    dt_med = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    eps = 1e-9

    n_left = n_right = 0
    cur = 0            # current run label: -1 left, +1 right, 0 none
    run_t0 = None
    last_t = None

    def _close(label, t0, t1):
        nonlocal n_left, n_right
        if label == 0 or t0 is None:
            return
        if (t1 - t0) + dt_med >= min_look_s - eps:
            if label < 0:
                n_left += 1
            else:
                n_right += 1

    for ti, yi in zip(t, yaw):
        if math.isnan(yi):
            lab = 0
        elif yi >= yaw_thr:
            lab = 1
        elif yi <= -yaw_thr:
            lab = -1
        else:
            lab = 0
        if (lab == cur and cur != 0 and last_t is not None
                and ti - last_t <= max_gap_s + eps):
            last_t = ti
            continue
        _close(cur, run_t0, last_t)
        cur = lab
        run_t0 = ti if lab != 0 else None
        last_t = ti
    _close(cur, run_t0, last_t)
    return n_left, n_right


def ankle_y_series(kps, confs, kp_conf_min=KP_CONF_MIN):
    """Per-frame ankle-y: mean of the confident ankles, NaN when neither is."""
    kps = np.asarray(kps, dtype=float)
    confs = np.asarray(confs, dtype=float)
    n = kps.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        ys = []
        for k in (KP_L_ANKLE, KP_R_ANKLE):
            if confs[i, k] >= kp_conf_min:
                ys.append(kps[i, k, 1])
        if ys:
            out[i] = float(np.mean(ys))
    return out


def gait_cadence(t, ankle_y, min_frames=MIN_CADENCE_FRAMES,
                 min_span_s=MIN_CADENCE_SPAN_S, band_hz=CADENCE_BAND_HZ,
                 min_peak_frac=MIN_PEAK_FRAC):
    """Dominant gait frequency of an ankle-y series.

    Valid samples are resampled onto a uniform grid (median dt), linearly
    detrended and Hann-windowed.  The coarse (unpadded) spectrum gates on
    peak prominence: the peak +/- 1 bin must hold >= ``min_peak_frac`` of the
    band energy, so noise-only spectra return NaN.  The fine frequency comes
    from a zero-padded FFT (grid well below 0.1 Hz).

    Returns (cadence_hz, step_count); (nan, None) when no credible peak.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(ankle_y, dtype=float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    if len(t) < min_frames:
        return float("nan"), None
    order = np.argsort(t)
    t, y = t[order], y[order]
    span = float(t[-1] - t[0])
    if span < min_span_s:
        return float("nan"), None
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return float("nan"), None

    # uniform grid (tolerates occasional dropped frames)
    tu = np.arange(t[0], t[-1] + 0.5 * dt, dt)
    yu = np.interp(tu, t, y)
    n = len(tu)
    if n < min_frames:
        return float("nan"), None

    # linear detrend + Hann window
    A = np.vstack([tu, np.ones(n)]).T
    coef, _res, _rk, _sv = np.linalg.lstsq(A, yu, rcond=None)
    yw = (yu - A @ coef) * np.hanning(n)

    # coarse spectrum: prominence gate
    P = np.abs(np.fft.rfft(yw)) ** 2
    F = np.fft.rfftfreq(n, dt)
    band_idx = np.where((F >= band_hz[0]) & (F <= band_hz[1]))[0]
    if len(band_idx) < 3:
        return float("nan"), None
    band_energy = float(P[band_idx].sum())
    if band_energy <= 0:
        return float("nan"), None
    k = band_idx[int(np.argmax(P[band_idx]))]
    lo = max(k - 1, band_idx[0])
    hi = min(k + 1, band_idx[-1])
    if float(P[lo:hi + 1].sum()) / band_energy < min_peak_frac:
        return float("nan"), None

    # fine frequency via zero-padded FFT
    nfft = 4096
    while nfft < 8 * n:
        nfft *= 2
    Pf = np.abs(np.fft.rfft(yw, n=nfft)) ** 2
    Ff = np.fft.rfftfreq(nfft, dt)
    bf = np.where((Ff >= band_hz[0]) & (Ff <= band_hz[1]))[0]
    kf = bf[int(np.argmax(Pf[bf]))]
    cadence = float(Ff[kf])
    step_count = int(round(cadence * (span + dt)))
    return cadence, step_count


def compute_track_metrics(t, kps, confs, bbox_heights, window_s,
                          pre_window_s=PRE_WINDOW_S, kp_conf_min=KP_CONF_MIN):
    """All [P12] metrics for one track from its matched pose frames.

    ``t``            : (n,) sample times in seconds (any order).
    ``kps``          : (n, 17, 2) keypoint xy pixels.
    ``confs``        : (n, 17) keypoint confidences.
    ``bbox_heights`` : (n,) matched person-bbox heights in pixels.
    ``window_s``     : (t_start, t_end) of the [C3] crossing window (seconds).

    Head scanning is evaluated on frames in [t_start - pre_window_s, t_start);
    cadence on frames in [t_start, t_end].  Returns the output-row dict
    (without ``track_id``).
    """
    t = np.asarray(t, dtype=float)
    kps = np.asarray(kps, dtype=float)
    confs = np.asarray(confs, dtype=float)
    bh = np.asarray(bbox_heights, dtype=float)
    order = np.argsort(t)
    t, kps, confs, bh = t[order], kps[order], confs[order], bh[order]
    n = len(t)
    ts, te = float(window_s[0]), float(window_s[1])

    med_conf = float(np.median(confs)) if n else float("nan")
    reliable = bool(
        n >= RELIABLE_MIN_FRAMES
        and med_conf >= RELIABLE_MIN_CONF
        and n > 0 and float(np.median(bh)) >= RELIABLE_MIN_BBOX_H
    )

    n_left = n_right = 0
    pre = (t >= ts - pre_window_s) & (t < ts)
    if pre.any():
        yaw = head_yaw_series(kps[pre], confs[pre], kp_conf_min=kp_conf_min)
        n_left, n_right = count_sustained_looks(t[pre], yaw)

    cadence, step_count = float("nan"), None
    cross = (t >= ts) & (t <= te)
    if cross.any():
        ankle = ankle_y_series(kps[cross], confs[cross], kp_conf_min=kp_conf_min)
        cadence, step_count = gait_cadence(t[cross], ankle)

    return {
        "n_pose_frames": int(n),
        "looked_left": bool(n_left > 0),
        "looked_right": bool(n_right > 0),
        "n_head_turns": int(n_left + n_right),
        "cadence_hz": round(cadence, 3) if cadence == cadence else None,
        "step_count": step_count,
        "median_kp_conf": round(med_conf, 3) if med_conf == med_conf else None,
        "reliable": reliable,
    }


def _iou(a, b):
    """IoU of two [x1, y1, x2, y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def match_tracks_to_detections(track_boxes, det_boxes, iou_thr=IOU_MATCH_THR):
    """Greedy 1:1 IoU matching. ``track_boxes``: {track_id: box}; ``det_boxes``:
    list of boxes.  Returns {track_id: det_index}."""
    pairs = []
    for tid, tb in track_boxes.items():
        for j, db in enumerate(det_boxes):
            v = _iou(tb, db)
            if v >= iou_thr:
                pairs.append((v, tid, j))
    pairs.sort(key=lambda p: -p[0])
    used_t, used_d, out = set(), set(), {}
    for v, tid, j in pairs:
        if tid in used_t or j in used_d:
            continue
        used_t.add(tid)
        used_d.add(j)
        out[tid] = j
    return out


# =============================================================================
# Entry point (video + CSV I/O)
# =============================================================================

def _safe_read_csv(path):
    if path and os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"[pose_behavior][warn] could not read {path}: {e}")
    return None


def _as_bool(v):
    return str(v).strip().lower() == "true"


class _TrackBoxInterp:
    """Per-frame bbox interpolation from a track's [B2] rows (valid only
    inside the track's observed frame range)."""

    def __init__(self, g):
        g = g.sort_values("frame_id")
        self.fr = g["frame_id"].to_numpy(dtype=float)
        self.coords = [g[c].to_numpy(dtype=float) for c in ("x1", "y1", "x2", "y2")]
        self.f_min = float(self.fr[0])
        self.f_max = float(self.fr[-1])

    def box_at(self, frame_id):
        if not (self.f_min <= frame_id <= self.f_max):
            return None
        return [float(np.interp(frame_id, self.fr, c)) for c in self.coords]


def run_pose_behavior(video_path, output_csv=None, target_hz=TARGET_HZ,
                      pre_window_s=PRE_WINDOW_S, post_window_s=POST_WINDOW_S,
                      model_name=POSE_MODEL, det_conf=DET_CONF,
                      iou_match_thr=IOU_MATCH_THR, device=None):
    """Head-scanning + gait metrics for every crossed pedestrian track.

    NEEDS the source video (must run before deletion).  When the video or a
    required CSV is missing, a header-only [P12] CSV is written -- expected
    for every archived folder; the module activates on future batch runs.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join("analysis_results", video_name)
    os.makedirs(output_dir, exist_ok=True)
    if output_csv is None:
        output_csv = os.path.join(output_dir, "[P12]pose_behavior.csv")

    def _write_empty(msg):
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(output_csv, index=False)
        print(f"[pose_behavior] {msg} Empty results saved to {output_csv}")
        return output_csv

    if not (video_path and os.path.exists(video_path)):
        return _write_empty("Video not found (videos are deleted after analysis).")

    # --- [C3] crossing windows (original-video frame ids) -----------------------
    cj = _safe_read_csv(os.path.join(output_dir, "[C3]crossing_judge.csv"))
    if cj is None or cj.empty or "crossed" not in cj.columns:
        return _write_empty("[C3]crossing_judge.csv missing/empty.")
    windows = {}
    for _, r in cj.iterrows():
        try:
            if _as_bool(r.get("crossed")) and pd.notna(r.get("started_frame")):
                ef = float(r["ended_frame"]) if pd.notna(r.get("ended_frame")) else None
                windows[r["track_id"]] = (float(r["started_frame"]), ef)
        except Exception:
            continue
    if not windows:
        return _write_empty("No crossed tracks in [C3].")

    # --- trajectory ([B2] dense, fallback [B1] 1 Hz) -----------------------------
    traj = _safe_read_csv(os.path.join(output_dir, "[B2]dense_tracks.csv"))
    if traj is None or traj.empty:
        traj = _safe_read_csv(os.path.join(output_dir, "[B1]tracked_pedestrians.csv"))
    required = {"frame_id", "timestamp", "track_id", "x1", "y1", "x2", "y2"}
    if traj is None or traj.empty or not required.issubset(traj.columns):
        return _write_empty("No usable trajectory CSV ([B2]/[B1]).")

    interps = {}
    for tid, g in traj.groupby("track_id"):
        if tid in windows and len(g) >= 2:
            interps[tid] = _TrackBoxInterp(g)

    # --- video + fps -------------------------------------------------------------
    import cv2  # lazy: header-only paths never need it

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return _write_empty(f"Could not open video {video_path}.")
    fps = None
    meta = _safe_read_csv(os.path.join(output_dir, "[B0]video_meta.csv"))
    if meta is not None and not meta.empty and "fps" in meta.columns:
        try:
            fps = float(meta.iloc[0]["fps"])
        except Exception:
            fps = None
    if not fps or fps <= 0:
        fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        cap.release()
        return _write_empty("Could not determine video fps.")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, int(round(fps / target_hz)))

    # --- sampled frames per track window (union, aligned to the stride grid) ----
    frame_tracks = {}           # frame_id -> [track_id, ...]
    for tid, (f_start, f_end) in windows.items():
        itp = interps.get(tid)
        if itp is None:
            continue
        if f_end is None:
            f_end = itp.f_max
        f0 = max(0, int(f_start - pre_window_s * fps))
        f1 = int(f_end + post_window_s * fps)
        if total_frames > 0:
            f1 = min(f1, total_frames - 1)
        f0 = (f0 // stride) * stride
        for f in range(f0, f1 + 1, stride):
            if itp.box_at(float(f)) is not None:
                frame_tracks.setdefault(f, []).append(tid)
    if not frame_tracks:
        return _write_empty("No frames overlap [B2] coverage for crossed tracks.")

    # --- pose model (lazy; ultralytics auto-downloads the weights) --------------
    try:
        from ultralytics import YOLO
        model = YOLO(model_name)
    except Exception as e:
        cap.release()
        return _write_empty(f"Pose model unavailable ({e}).")

    # --- sequential decode, pose on sampled frames, IoU matching -----------------
    acc = {tid: {"t": [], "kps": [], "confs": [], "bh": []} for tid in windows}
    current = 0
    for f in sorted(frame_tracks):
        while current < f:
            if not cap.grab():
                break
            current += 1
        if current != f:
            break                      # ran off the end of the stream
        ok, img = cap.read()
        current += 1
        if not ok:
            break
        try:
            res = model.predict(img, conf=det_conf, device=device, verbose=False)[0]
        except Exception as e:
            print(f"[pose_behavior][warn] pose inference failed at frame {f}: {e}")
            continue
        if res.boxes is None or res.keypoints is None or len(res.boxes) == 0:
            continue
        det_boxes = res.boxes.xyxy.cpu().numpy()
        det_kps = res.keypoints.xy.cpu().numpy()          # (m, 17, 2)
        kc = res.keypoints.conf
        det_confs = (kc.cpu().numpy() if kc is not None
                     else np.ones(det_kps.shape[:2], dtype=float))

        track_boxes = {}
        for tid in frame_tracks[f]:
            b = interps[tid].box_at(float(f))
            if b is not None:
                track_boxes[tid] = b
        assign = match_tracks_to_detections(track_boxes, list(det_boxes),
                                            iou_thr=iou_match_thr)
        t_frame = f / fps
        for tid, j in assign.items():
            a = acc[tid]
            a["t"].append(t_frame)
            a["kps"].append(det_kps[j])
            a["confs"].append(det_confs[j])
            a["bh"].append(float(det_boxes[j][3] - det_boxes[j][1]))
    cap.release()

    # --- per-track metrics --------------------------------------------------------
    rows = []
    for tid in sorted(windows, key=lambda x: str(x)):
        f_start, f_end = windows[tid]
        itp = interps.get(tid)
        if f_end is None:
            f_end = itp.f_max if itp is not None else f_start
        a = acc[tid]
        if not a["t"]:
            rows.append({"track_id": tid, "n_pose_frames": 0,
                         "looked_left": False, "looked_right": False,
                         "n_head_turns": 0, "cadence_hz": None,
                         "step_count": None, "median_kp_conf": None,
                         "reliable": False})
            continue
        stats = compute_track_metrics(
            a["t"], a["kps"], a["confs"], a["bh"],
            window_s=(f_start / fps, f_end / fps), pre_window_s=pre_window_s)
        stats["track_id"] = tid
        rows.append(stats)

    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out.to_csv(output_csv, index=False)
    n_rel = int(out["reliable"].fillna(False).astype(bool).sum()) if not out.empty else 0
    print(f"[pose_behavior] {len(out)} crossed tracks ({n_rel} reliable). "
          f"Saved to {output_csv}")
    return output_csv


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Pose-based head-scanning + gait profiler ([P12]).")
    ap.add_argument("--source_video_path", required=True)
    ap.add_argument("--model", default=POSE_MODEL)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    run_pose_behavior(args.source_video_path, model_name=args.model,
                      device=args.device)
