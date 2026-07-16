import cv2
import numpy as np
from collections import deque
import csv
import os
import math

ASSUMED_LANE_WIDTH_M = 3.5


def _lane_scale_px_per_m(output_dir):
    """Rough px/m from [V5] lane geometry (median lane width at frame bottom / 3.5 m).
    Returns None when no lane data is available -> road width in metres stays NaN
    rather than being fabricated from a magic constant."""
    lane_csv = os.path.join(output_dir, "[V5]lane_detection.csv")
    if not (os.path.exists(lane_csv) and os.path.getsize(lane_csv) > 0):
        return None
    try:
        import pandas as pd
        d = pd.read_csv(lane_csv)
    except Exception:
        return None
    widths = []
    for _, r in d.iterrows():
        try:
            lx = r["left_x1"] if r["left_y1"] >= r["left_y2"] else r["left_x2"]
            rx = r["right_x1"] if r["right_y1"] >= r["right_y2"] else r["right_x2"]
        except KeyError:
            return None
        # lane_detection.py writes all-zero coords for an UNDETECTED side, and one-sided
        # detections are common. Requiring only "both zero" let a one-sided row contribute
        # |rx - 0| (e.g. 1500 px) as if it were a real 3.5 m lane, inflating the median
        # scale several-fold. Skip the row unless BOTH sides are present.
        if lx == 0 or rx == 0:
            continue
        wpx = abs(float(rx) - float(lx))
        if wpx > 5:
            widths.append(wpx)
    return (float(np.median(widths)) / ASSUMED_LANE_WIDTH_M) if widths else None


def run_road_width_analysis(video_path, analyze_interval_sec=1.0, output_csv_path=None):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if output_csv_path is None:
        output_dir = os.path.join(".", "analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "[E5]road_width.csv")
    else:
        output_dir = os.path.dirname(output_csv_path) or "."
        os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        # Still write a valid header so downstream pd.read_csv does not crash.
        with open(output_csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(['Frame Index', 'Time (s)', 'Road Width (px)', 'Road Width (m)'])
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    analyze_every_n_frames = max(1, math.ceil(fps * analyze_interval_sec))

    # px -> m scale from lane geometry (rough); None => metres reported as NaN, not fabricated.
    # [V5] lives next to the output CSV (output_dir == ./analysis_results/<video_name> by default).
    lane_scale = _lane_scale_px_per_m(output_dir)

    src_points = np.float32([
        [frame_width * 0.45, frame_height * 0.6],
        [frame_width * 0.55, frame_height * 0.6],
        [frame_width * 0.1, frame_height * 0.95],
        [frame_width * 0.9, frame_height * 0.95]
    ])
    dest_points = np.float32([
        [frame_width * 0.25, 0],
        [frame_width * 0.75, 0],
        [frame_width * 0.25, frame_height],
        [frame_width * 0.75, frame_height]
    ])

    # Forward warp (native -> birdseye) for edge/line detection, inverse (birdseye -> native)
    # to bring the detected boundaries back into NATIVE pixels. The lane scale from [V5] is in
    # native px/m, so widths must be measured in native space too: the warp stretches the
    # measurement ROI horizontally by anywhere between ~0.6x and ~5x, which previously made
    # 'Road Width (m)' wrong by that same factor.
    warp_matrix = cv2.getPerspectiveTransform(src_points, dest_points)
    inv_warp_matrix = cv2.getPerspectiveTransform(dest_points, src_points)

    road_width_history = deque(maxlen=10)  # widths in NATIVE pixels

    def detect_road_width(birdseye_frame):
        # ROI as fractions of the (birdseye) frame so it scales with resolution instead of
        # the old hardcoded 50/300/550/400 pixels that only made sense at one size.
        h, w = birdseye_frame.shape[:2]
        rx1, rx2 = int(w * 0.08), int(w * 0.92)
        ry1, ry2 = int(h * 0.55), int(h * 0.75)
        mask = np.zeros_like(birdseye_frame)
        roi_vertices = np.array([[(rx1, ry1), (rx2, ry1), (rx2, ry2), (rx1, ry2)]], dtype=np.int32)
        cv2.fillPoly(mask, roi_vertices, 255)
        masked_frame = cv2.bitwise_and(birdseye_frame, mask)
        edges = cv2.Canny(masked_frame, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=20)

        left_boundary = right_boundary = None
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1 + 1e-6)
                if slope < 0:
                    if left_boundary is None or x1 < left_boundary[0]:
                        left_boundary = (x1, y1, x2, y2)
                elif slope > 0:
                    if right_boundary is None or x1 > right_boundary[0]:
                        right_boundary = (x1, y1, x2, y2)

        road_width_pixels = None
        if left_boundary and right_boundary:
            # Midpoints of each boundary segment in birdseye space...
            left_mid = ((left_boundary[0] + left_boundary[2]) / 2.0,
                        (left_boundary[1] + left_boundary[3]) / 2.0)
            right_mid = ((right_boundary[0] + right_boundary[2]) / 2.0,
                         (right_boundary[1] + right_boundary[3]) / 2.0)
            # ...inverse-projected back to NATIVE image coordinates before measuring,
            # so the width is in the same pixel space as the [V5]-derived lane scale.
            pts = np.array([left_mid, right_mid], dtype=np.float32).reshape(-1, 1, 2)
            native = cv2.perspectiveTransform(pts, inv_warp_matrix).reshape(-1, 2)
            if np.all(np.isfinite(native)):
                road_width_pixels = float(abs(native[1][0] - native[0][0]))
                road_width_history.append(road_width_pixels)
        if road_width_pixels is None and road_width_history:
            road_width_pixels = float(np.mean(road_width_history))
        return road_width_pixels

    with open(output_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Frame Index', 'Time (s)', 'Road Width (px)', 'Road Width (m)'])

        frame_idx = 0
        while True:
            # grab() advances without decoding; retrieve() decodes only the ~1/interval
            # frames we actually analyze instead of paying full decode on every frame.
            if not cap.grab():
                break

            frame_idx += 1
            if frame_idx % analyze_every_n_frames != 0:
                continue

            ret, frame = cap.retrieve()
            if not ret:
                continue

            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            birdseye = cv2.warpPerspective(enhanced, warp_matrix,
                                           (enhanced.shape[1], enhanced.shape[0]))

            road_width_px = detect_road_width(birdseye)
            time_sec = frame_idx / fps

            if road_width_px is not None:
                width_m = (road_width_px / lane_scale) if lane_scale else float('nan')
                writer.writerow([frame_idx, f"{time_sec:.2f}", f"{road_width_px:.1f}",
                                 f"{width_m:.2f}" if width_m == width_m else "NaN"])
            else:
                writer.writerow([frame_idx, f"{time_sec:.2f}", "NaN", "NaN"])

    cap.release()
    print(f"Road width analysis saved to: {output_csv_path}")
    print(f"Total frames processed: {frame_idx} | lane_scale="
          f"{'%.1f px/m' % lane_scale if lane_scale else 'n/a -> metres = NaN'}")
