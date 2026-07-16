"""Camera ego-motion estimation (module [B3]).

Many YouTube pedestrian clips are handheld / walking / dashcam / panning. When the
camera moves, a pedestrian's pixel displacement mixes real motion with camera
motion, which corrupts speed ([S1]) and waiting-time detection. This module
estimates the per-frame global background translation (in pixels) so the kinematic
modules can subtract it.

Method: on each tracked frame, track background feature points (Lucas-Kanade) from
the previous tracked frame, masking out the pedestrian boxes from [B2] so people
don't drag the estimate. The median background displacement is the camera step; we
accumulate it into a cumulative camera position so consumers can take the exact
displacement between any two frames as a difference.

Output [B3]ego_motion.csv: frame_id, timestamp, cam_x, cam_y (cumulative px),
step_px (per-step magnitude), n_bg_points. A `moving` summary is printed. If the
video is unavailable, an empty file is written and consumers assume a static camera.
"""

import os
import numpy as np
import pandas as pd

try:
    import cv2
except Exception:                      # pragma: no cover
    cv2 = None

OUTPUT_COLUMNS = ["frame_id", "timestamp", "cam_x", "cam_y", "step_px", "n_bg_points"]


def estimate_step(prev_gray, cur_gray, boxes, min_points=8, pad_frac=0.12):
    """Median background translation (dx, dy) from prev->cur, masking `boxes`
    (list of (x1,y1,x2,y2)). Returns (dx, dy, n_points); (0,0,0) if unreliable.
    Pure array function -> unit-testable without a video."""
    if cv2 is None:
        return 0.0, 0.0, 0
    h, w = prev_gray.shape[:2]
    mask = np.full((h, w), 255, dtype=np.uint8)
    for (x1, y1, x2, y2) in boxes:
        pw, ph = (x2 - x1), (y2 - y1)
        mx1 = max(0, int(x1 - pw * pad_frac)); my1 = max(0, int(y1 - ph * pad_frac))
        mx2 = min(w, int(x2 + pw * pad_frac)); my2 = min(h, int(y2 + ph * pad_frac))
        mask[my1:my2, mx1:mx2] = 0        # exclude pedestrian regions from the background

    p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01,
                                 minDistance=8, mask=mask)
    if p0 is None or len(p0) < min_points:
        return 0.0, 0.0, 0
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, p0.astype(np.float32), None)
    if p1 is None:
        return 0.0, 0.0, 0
    good = st.flatten() == 1
    if good.sum() < min_points:
        return 0.0, 0.0, 0
    disp = (p1[good] - p0[good]).reshape(-1, 2)
    return float(np.median(disp[:, 0])), float(np.median(disp[:, 1])), int(good.sum())


def run_ego_motion(video_path, dense_csv=None, output_csv=None, moving_threshold_px=1.0,
                   sample_fps=15.0):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join("analysis_results", video_name)
    os.makedirs(output_dir, exist_ok=True)
    if dense_csv is None:
        dense_csv = os.path.join(output_dir, "[B2]dense_tracks.csv")
    if output_csv is None:
        output_csv = os.path.join(output_dir, "[B3]ego_motion.csv")

    def _write(rows):
        pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(output_csv, index=False)

    if cv2 is None:
        _write([]); print("[ego] OpenCV unavailable; wrote empty ego-motion."); return output_csv

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release(); _write([])
        print(f"[ego] Video unavailable ({video_path}); assuming static camera."); return output_csv

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        fps = 30.0

    # Pedestrian boxes are used only to MASK people out of the background estimate.
    boxes_by_frame = {}
    if os.path.exists(dense_csv) and os.path.getsize(dense_csv) > 0:
        try:
            d = pd.read_csv(dense_csv)
            if not d.empty:
                for fid, g in d.groupby("frame_id"):
                    boxes_by_frame[int(fid)] = list(zip(g["x1"], g["y1"], g["x2"], g["y2"]))
        except Exception:
            pass
    # Camera motion is a property of the CAMERA, not of pedestrian presence: sample a regular
    # grid (~sample_fps). Keying the grid off [B2] frames made a pedestrian-free dashcam clip
    # report "static". [S1] interpolates cam_x/cam_y over frame_id, so a regular grid is ideal.
    stride = max(1, round(fps / sample_fps)) if sample_fps and sample_fps > 0 else 1

    rows = []
    cam_x = cam_y = 0.0
    prev_gray = None
    frame_id = 0
    steps = []
    while cap.isOpened():
        # grab() skips the decode for frames outside the sampling grid.
        if not cap.grab():
            break
        frame_id += 1
        if frame_id % stride != 0:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            dx, dy, n = estimate_step(prev_gray, gray, boxes_by_frame.get(frame_id, []))
            cam_x += dx; cam_y += dy
            step = float(np.hypot(dx, dy)); steps.append(step)
            rows.append({"frame_id": frame_id, "timestamp": round(frame_id / fps, 3),
                         "cam_x": round(cam_x, 2), "cam_y": round(cam_y, 2),
                         "step_px": round(step, 3), "n_bg_points": n})
        prev_gray = gray
    cap.release()

    _write(rows)
    med = float(np.median(steps)) if steps else float("nan")
    moving = bool(steps) and med >= moving_threshold_px
    print(f"[ego] {len(rows)} steps @ ~{fps/stride:.1f} Hz, median step="
          f"{med:.2f}px -> camera {'MOVING' if moving else 'static'}. Saved to {output_csv}")
    return output_csv


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Camera ego-motion estimation ([B3]).")
    ap.add_argument("--source_video_path", required=True)
    args = ap.parse_args()
    run_ego_motion(args.source_video_path)
