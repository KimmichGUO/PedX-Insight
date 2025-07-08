import argparse
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import os
import csv
import json
from collections import defaultdict

def detect_pedestrian_speed(
        source_video_path: str,
        weights: str = "yolov8m-pose.pt",
) -> None:
    model = YOLO(weights)
    edge_annotator = sv.EdgeAnnotator()
    vertex_annotator = sv.VertexAnnotator()
    box_annotator = sv.BoxAnnotator()
    tracker = sv.ByteTrack() 

    cap = cv2.VideoCapture(source_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    video_basename = os.path.splitext(os.path.basename(source_video_path))[0]
    csv_filename = f"{video_basename}_pedestrian_speed.csv"

    with open(csv_filename, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["frame_index", "id_speed_map"])

    prev_centers = defaultdict(tuple)  # {id: (cx, cy)}
    id_bbox_heights = defaultdict(list)  # {id: [height1, height2, ...]}

    def callback(frame: np.ndarray, frame_index: int) -> np.ndarray:
        results = model(frame)[0]
        key_points = sv.KeyPoints.from_ultralytics(results)
        detections = key_points.as_detections()

        detections = tracker.update_with_detections(detections)

        annotated_frame = edge_annotator.annotate(frame.copy(), key_points=key_points)
        annotated_frame = vertex_annotator.annotate(annotated_frame, key_points=key_points)
        annotated_frame = box_annotator.annotate(annotated_frame, detections=detections)

        id_speed_map = {}

        for idx, tracker_id in enumerate(detections.tracker_id):
            if tracker_id is None:
                continue

            x1, y1, x2, y2 = detections.xyxy[idx]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            height = y2 - y1

            id_bbox_heights[tracker_id].append(height)
            if len(id_bbox_heights[tracker_id]) > 30:
                id_bbox_heights[tracker_id].pop(0)

            if tracker_id in prev_centers:
                prev_cx, prev_cy = prev_centers[tracker_id]
                dist_px = np.linalg.norm([cx - prev_cx, cy - prev_cy])
                speed_px_per_sec = dist_px * fps

                avg_height = np.mean(id_bbox_heights[tracker_id])
                pixel_to_meter = 1.5 / avg_height if avg_height > 0 else 0.02
                speed_mps = speed_px_per_sec * pixel_to_meter
                speed_kph = min(speed_mps * 3.6, 10)

                label = f"ID: {tracker_id} | {speed_kph:.1f} km/h"
                cv2.putText(annotated_frame, label, (cx + 10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                id_speed_map[int(tracker_id)] = round(speed_kph, 1)

            prev_centers[tracker_id] = (cx, cy)

        if frame_index % 30 == 0:
            with open(csv_filename, mode="a", newline="") as csvfile:
                csv_writer = csv.writer(csvfile)
                json_str = json.dumps(id_speed_map)
                csv_writer.writerow([frame_index, json_str])

        cv2.imshow("Pedestrian Speed Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            os._exit(0)

        return annotated_frame

    sv.process_video(
        source_path=source_video_path,
        target_path=None,
        callback=callback
    )
