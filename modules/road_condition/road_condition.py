import os
import cv2
import pandas as pd
from ultralytics import YOLO
import math

def run_road_defect_detection(video_path, output_csv_path=None, analyze_interval_sec=1.0):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if output_csv_path is None:
        output_dir = os.path.join("analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "[E4]road_condition.csv")

    model = YOLO("modules/road_condition/YOLOv8road.pt")
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = math.ceil(fps) if fps > 0 else 30
    analyze_every_n_frames = int(fps * analyze_interval_sec)

    class_map = {
        0: 'Longitudinal Crack',
        1: 'Transverse Crack',
        2: 'Alligator Crack',
        3: 'Potholes'
    }

    frame_cache = {}
    results_list = []
    frame_id = -1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        if frame_id % analyze_every_n_frames == 0:
            frame_result = {
                "Longitudinal Crack": 0,
                "Transverse Crack": 0,
                "Alligator Crack": 0,
                "Potholes": 0
            }

            result = model(frame, verbose=False)[0]
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = class_map.get(cls_id)
                if cls_name:
                    frame_result[cls_name] = 1

            frame_cache[frame_id] = frame_result

        nearest_analysis_frame = (frame_id // analyze_every_n_frames) * analyze_every_n_frames
        detections = frame_cache.get(nearest_analysis_frame, {
            "Longitudinal Crack": 0,
            "Transverse Crack": 0,
            "Alligator Crack": 0,
            "Potholes": 0
        })

        results_list.append({
            "frame_id": frame_id,
            **detections
        })

    cap.release()

    df = pd.DataFrame(results_list)
    df.to_csv(output_csv_path, index=False)
    print(f"Road defect detection completed. Results saved to {output_csv_path}")
