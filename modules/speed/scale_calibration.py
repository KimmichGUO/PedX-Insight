"""Ground-plane metric scale from crosswalk stripe periodicity (module [S2], rung A).

The height prior ([S1] rung B) is depth-adaptive but rests on an assumed stature
(~10-20% per person). Road markings are a far stronger reference: a zebra crossing
is a PERIODIC ground pattern whose real period is standardised (~1.0 m in much of
Europe: 0.5 m stripe + 0.5 m gap). Measuring that period in pixels at a known image
row gives pixels-per-metre on the ground plane at that depth.

Under a pinhole camera looking at a flat ground plane, pixels-per-metre varies
linearly with image row y (it is 0 at the horizon and grows toward the camera), so
we fit

    scale(y) = a * y + b            [pixels per metre]

from stripe-period samples measured at several depths/frames. [S1] then converts a
pedestrian's foot-point displacement using scale(foot_y) — a real ground-plane
calibration rather than a stature guess.

Robustness: the period is recovered by AUTOCORRELATION of the ROI intensity profile
(projected onto each axis; the more periodic axis wins), which tolerates blur/noise
far better than Hough line fitting. Samples are filtered by autocorrelation
confidence, and the linear fit reports its residual so [S1] can refuse a bad
calibration and fall back to the height prior.

Honesty: `stripe_period_m` is a country-dependent ASSUMPTION (default 1.0 m). A wrong
period scales every speed proportionally, so the quality flag and the assumed period
are recorded in the output for auditability.
"""

import ast
import os
import numpy as np
import pandas as pd

try:
    import cv2
except Exception:                      # pragma: no cover
    cv2 = None

DEFAULT_STRIPE_PERIOD_M = 1.0          # 0.5 m stripe + 0.5 m gap (typical EU zebra)
MIN_AUTOCORR_CONF = 0.30
MIN_SAMPLES = 4
OUTPUT_COLUMNS = ["a", "b", "n_samples", "fit_residual_px", "stripe_period_m",
                  "median_scale_px_per_m", "quality"]


def dominant_period(profile, min_lag=4, min_period=8):
    """Dominant period (in samples) of a 1-D periodic signal via autocorrelation.
    Returns (period, confidence in [0,1]); (None, 0.0) when not periodic.

    A valid period must be a genuine INTERIOR local maximum of the autocorrelation
    that is preceded by a real dip. Smooth, non-periodic profiles (shading gradients,
    shadows, plain asphalt) have autocorrelation ~1 at small lags, so a bare argmax
    returned period==min_lag with conf~0.9 — the degenerate 4 px/m calibration seen
    on live dashcam footage. Boundary peaks and dipless "peaks" are now rejected, and
    the profile must contain at least 3 full periods. Pure function -> unit-testable."""
    p = np.asarray(profile, dtype=float)
    if p.size < 3 * max(min_lag, min_period):
        return None, 0.0
    p = p - p.mean()
    denom = float(np.dot(p, p))
    if denom < 1e-9:
        return None, 0.0
    ac = np.correlate(p, p, mode="full")[p.size - 1:] / denom
    hi = p.size // 2
    if hi <= min_lag + 1:
        return None, 0.0
    seg = ac[min_lag:hi]
    k = int(np.argmax(seg)) + min_lag
    conf = float(max(0.0, ac[k]))
    if k <= min_lag or k >= hi - 1:          # peak at the search boundary != a period
        return None, 0.0
    if k < min_period or k * 3 > p.size:     # too fine to be a stripe / <3 periods visible
        return None, 0.0
    if float(np.min(ac[1:k])) > 0.5 * conf:  # no real dip before the peak -> not periodic
        return None, 0.0
    if not (ac[k] >= ac[k - 1] and ac[k] >= ac[k + 1]):
        return None, 0.0
    return k, conf


def stripe_scale_from_roi(gray_roi, stripe_period_m=DEFAULT_STRIPE_PERIOD_M):
    """pixels-per-metre from the stripe period inside a crosswalk ROI.
    Projects onto both axes and keeps the more periodic one."""
    if gray_roi is None or gray_roi.size == 0 or min(gray_roi.shape[:2]) < 12:
        return None, 0.0
    cand = []
    for axis in (0, 1):                      # axis0 -> column profile (period along x)
        prof = gray_roi.mean(axis=axis)
        per, conf = dominant_period(prof)
        if per:
            cand.append((conf, per))
    if not cand:
        return None, 0.0
    conf, period_px = max(cand)
    if conf < MIN_AUTOCORR_CONF:
        return None, conf
    return period_px / float(stripe_period_m), conf


def _parse_boxes(cell):
    try:
        v = ast.literal_eval(cell) if isinstance(cell, str) else cell
        return v if isinstance(v, (list, tuple)) else []
    except Exception:
        return []


def run_scale_calibration(video_path, crosswalk_csv=None, output_csv=None,
                          stripe_period_m=DEFAULT_STRIPE_PERIOD_M, max_frames=60):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join("analysis_results", video_name)
    os.makedirs(output_dir, exist_ok=True)
    if crosswalk_csv is None:
        crosswalk_csv = os.path.join(output_dir, "[E7]crosswalk_detection.csv")
    if output_csv is None:
        output_csv = os.path.join(output_dir, "[S2]scale_calibration.csv")

    def _write(row):
        pd.DataFrame([row] if row else [], columns=OUTPUT_COLUMNS).to_csv(output_csv, index=False)

    if cv2 is None:
        _write(None); print("[scale] OpenCV unavailable."); return output_csv
    if not (os.path.exists(crosswalk_csv) and os.path.getsize(crosswalk_csv) > 0):
        _write(None); print("[scale] No [E7] crosswalk data; no stripe calibration."); return output_csv

    try:
        cw = pd.read_csv(crosswalk_csv)
    except Exception as e:
        _write(None); print(f"[scale] could not read {crosswalk_csv}: {e}"); return output_csv

    if "crosswalk_detected" not in cw.columns or "crosswalk_boxes" not in cw.columns:
        _write(None); print("[scale] [E7] malformed."); return output_csv

    hits = cw[cw["crosswalk_detected"].astype(str).str.lower() == "yes"]
    if hits.empty:
        _write(None); print("[scale] No crosswalk detected in this video."); return output_csv

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release(); _write(None)
        print(f"[scale] Video unavailable ({video_path})."); return output_csv

    boxes_by_frame = {}
    for _, r in hits.iterrows():
        b = _parse_boxes(r["crosswalk_boxes"])
        if b:
            boxes_by_frame[int(r["frame_id"])] = b
    # Spread samples across the WHOLE video. [E7] is forward-filled per native frame, so
    # taking the first max_frames keys grabbed ~2 s of near-identical clones whose zero
    # variance then self-certified the fit as 'good'.
    all_f = sorted(boxes_by_frame)
    step = max(1, len(all_f) // max_frames)
    wanted = all_f[::step][:max_frames]

    samples = []          # (roi_center_y, scale_px_per_m)
    for fid in wanted:
        # [E7] frame_id is 0-based (crosswalk.py counts from 0), so seek to fid directly;
        # fid-1 (correct for the tracker's 1-based ids) returned the frame BEFORE the boxes.
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, fid))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape[:2]
        for box in boxes_by_frame[fid]:
            try:
                x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
            except Exception:
                continue
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            # A real zebra ROI must be able to hold >=3 stripe periods at >=8 px each;
            # the live run showed 27x42 px pedestrian-sized false detections poisoning
            # the calibration, so tiny boxes are rejected outright.
            if x2 - x1 < 32 or y2 - y1 < 16:
                continue
            scale, conf = stripe_scale_from_roi(gray[y1:y2, x1:x2], stripe_period_m)
            if scale and scale > 1:
                samples.append(((y1 + y2) / 2.0, scale))
    cap.release()

    # Clones (the same box re-detected / forward-filled) are not independent evidence:
    # count unique (y, scale) pairs, not raw rows.
    unique_samples = sorted(set((round(y, 1), round(s, 2)) for y, s in samples))
    if len(unique_samples) < MIN_SAMPLES:
        _write(None)
        print(f"[scale] Only {len(unique_samples)} distinct stripe samples (<{MIN_SAMPLES}); "
              f"no calibration -> [S1] keeps the height prior.")
        return output_csv

    ys = np.array([s[0] for s in unique_samples], dtype=float)
    ss = np.array([s[1] for s in unique_samples], dtype=float)
    # scale(y) = a*y + b ; needs spread in y to be identifiable, else use a flat model.
    if ys.max() - ys.min() >= 20 and len(ys) >= MIN_SAMPLES:
        a, b = np.polyfit(ys, ss, 1)
    else:
        a, b = 0.0, float(np.median(ss))
    resid = float(np.sqrt(np.mean((ss - (a * ys + b)) ** 2)))
    med = float(np.median(ss))
    # Quality: fit must explain the samples to within ~25% of the typical scale.
    quality = "good" if (med > 0 and resid / med <= 0.25) else "poor"

    _write({
        "a": round(float(a), 6), "b": round(float(b), 4), "n_samples": len(unique_samples),
        "fit_residual_px": round(resid, 3), "stripe_period_m": stripe_period_m,
        "median_scale_px_per_m": round(med, 3), "quality": quality,
    })
    print(f"[scale] {len(samples)} stripe samples -> scale(y)={a:.4f}*y+{b:.2f} px/m "
          f"(median {med:.1f}, residual {resid:.1f}, {quality}). Saved to {output_csv}")
    return output_csv


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Crosswalk-stripe ground-plane scale ([S2]).")
    ap.add_argument("--source_video_path", required=True)
    ap.add_argument("--stripe_period_m", type=float, default=DEFAULT_STRIPE_PERIOD_M)
    args = ap.parse_args()
    run_scale_calibration(args.source_video_path, stripe_period_m=args.stripe_period_m)
