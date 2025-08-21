import os
import cv2
import numpy as np
import math
import csv
from ultralytics import YOLO

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    return cv2.bitwise_and(img, mask)

def draw_lane_lines(img, left_line, right_line, color=[0, 255, 0], thickness=10):
    line_img = np.zeros_like(img)
    poly_pts = np.array([[left_line[:2], left_line[2:], right_line[2:], right_line[:2]]], dtype=np.int32)
    cv2.fillPoly(line_img, poly_pts, color)
    return cv2.addWeighted(img, 0.8, line_img, 0.5, 0.0)

def pipeline(image):
    height, width = image.shape[:2]
    roi_vertices = [(0, height), (width / 2, height / 2), (width, height)]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    cropped = region_of_interest(edges, np.array([roi_vertices], np.int32))

    lines = cv2.HoughLinesP(cropped, 6, np.pi / 60, 160, np.array([]), minLineLength=40, maxLineGap=25)
    left_x, left_y, right_x, right_y = [], [], [], []

    if lines is None:
        return image, None, None

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1 + 1e-5)
            if abs(slope) < 0.5:
                continue
            if slope <= 0:
                left_x.extend([x1, x2])
                left_y.extend([y1, y2])
            else:
                right_x.extend([x1, x2])
                right_y.extend([y1, y2])

    min_y = int(image.shape[0] * (3 / 5))
    max_y = image.shape[0]

    left_line, right_line = None, None
    if left_x and left_y:
        poly_left = np.poly1d(np.polyfit(left_y, left_x, 1))
        left_line = [int(poly_left(max_y)), max_y, int(poly_left(min_y)), min_y]
    if right_x and right_y:
        poly_right = np.poly1d(np.polyfit(right_y, right_x, 1))
        right_line = [int(poly_right(max_y)), max_y, int(poly_right(min_y)), min_y]

    if left_line and right_line:
        image = draw_lane_lines(image, left_line, right_line)
    return image, left_line, right_line

def estimate_distance(bbox_width, bbox_height):
    focal_length = 500
    known_width = 2.0
    return (known_width * focal_length) / bbox_width if bbox_width > 0 else 0

def run_lane_detection(video_path, weights="yolo11n.pt", interval_sec=1, output_csv_path=None):
    model = YOLO(weights)
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if output_csv_path is None:
        output_dir = os.path.join(".", "analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "[V5]lane_detection.csv")

    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    fps = math.ceil(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval_frames = interval_sec * fps

    results = []
    last_left_line, last_right_line, last_nearest_distance = None, None, None
    frame_idx = 0

    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            left_line, right_line, nearest_distance = last_left_line, last_right_line, last_nearest_distance
        else:
            frame = cv2.resize(frame, (1280, 720))
            lane_frame, left_line, right_line = pipeline(frame.copy())
            nearest_distance = float('inf')

            results_yolo = model(frame)
            for result in results_yolo:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    conf = box.conf[0]
                    if model.names[cls] == 'car' and conf >= 0.5:
                        bbox_width = x2 - x1
                        distance = estimate_distance(bbox_width, y2 - y1)
                        if distance < nearest_distance:
                            nearest_distance = distance

            last_left_line, last_right_line, last_nearest_distance = left_line, right_line, nearest_distance

        for f in range(frame_idx, min(frame_idx + interval_frames, total_frames)):
            results.append({
                'frame': f,
                'left_x1': left_line[0] if left_line else 0,
                'left_y1': left_line[1] if left_line else 0,
                'left_x2': left_line[2] if left_line else 0,
                'left_y2': left_line[3] if left_line else 0,
                'right_x1': right_line[0] if right_line else 0,
                'right_y1': right_line[1] if right_line else 0,
                'right_x2': right_line[2] if right_line else 0,
                'right_y2': right_line[3] if right_line else 0,
                'nearest_car_distance_m': f"{nearest_distance:.2f}" if nearest_distance != float('inf') else ''
            })
        frame_idx += interval_frames

    cap.release()

    fieldnames = ['frame', 'left_x1', 'left_y1', 'left_x2', 'left_y2',
                  'right_x1', 'right_y1', 'right_x2', 'right_y2',
                  'nearest_car_distance_m']
    with open(output_csv_path, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        if results:
            for row in results:
                writer.writerow(row)

    print(f"\nLane detection results saved to {output_csv_path}")
