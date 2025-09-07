import cv2
import numpy as np
from collections import deque
import csv
import os
import math

def run_road_width_analysis(video_path, analyze_interval_sec=1.0, output_csv_path=None):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if output_csv_path is None:
        output_dir = os.path.join(".", "analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "[E5]road_width.csv")
    else:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    # Calculate frame interval based on analyze_interval_sec
    analyze_every_n_frames = max(1, math.ceil(fps * analyze_interval_sec))

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

    road_width_history = deque(maxlen=10)

    def perspective_transform(frame, src_pts, dest_pts):
        matrix = cv2.getPerspectiveTransform(src_pts, dest_pts)
        return cv2.warpPerspective(frame, matrix, (frame.shape[1], frame.shape[0]))

    def detect_road_width(birdseye_frame):
        mask = np.zeros_like(birdseye_frame)
        roi_vertices = np.array([[(50, 300), (550, 300), (550, 400), (50, 400)]], dtype=np.int32)
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
            left_x = (left_boundary[0] + left_boundary[2]) // 2
            right_x = (right_boundary[0] + right_boundary[2]) // 2
            road_width_pixels = abs(right_x - left_x)
            road_width_history.append(road_width_pixels)
        elif road_width_history:
            road_width_pixels = np.mean(road_width_history)

        calibration_factor = 0.05
        return road_width_pixels * calibration_factor if road_width_pixels else None

    with open(output_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Frame Index', 'Time (s)', 'Road Width (m)'])

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Skip frames based on analysis interval
            if frame_idx % analyze_every_n_frames != 0:
                continue

            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            birdseye = perspective_transform(enhanced, src_points, dest_points)

            road_width_m = detect_road_width(birdseye)
            time_sec = frame_idx / fps

            if road_width_m is not None:
                writer.writerow([frame_idx, f"{time_sec:.2f}", f"{road_width_m:.2f}"])
            else:
                writer.writerow([frame_idx, f"{time_sec:.2f}", "NaN"])

    cap.release()
    print(f"Road width analysis saved to: {output_csv_path}")
    print(f"Total frames processed: {frame_idx}")
    print(f"Analysis interval: {analyze_interval_sec} seconds ({analyze_every_n_frames} frames)")