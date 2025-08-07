import os
import cv2
import pandas as pd
from ultralytics import YOLO

def run_road_defect_detection(video_path, output_csv_path=None):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if output_csv_path is None:
        output_dir = os.path.join("analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "road_defect_detection.csv")

    model = YOLO("modules/road_condition/YOLOv8road.pt")
    cap = cv2.VideoCapture(video_path)

    class_map = {
        0: 'Longitudinal Crack',
        1: 'Transverse Crack',
        2: 'Alligator Crack',
        3: 'Potholes'
    }

    results_list = []
    frame_id = -1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        frame_result = {
            "frame_id": frame_id,
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

        results_list.append(frame_result)

    cap.release()

    df = pd.DataFrame(results_list)
    df.to_csv(output_csv_path, index=False)
    print(f"Road defect detection completed. Results saved to {output_csv_path}")
