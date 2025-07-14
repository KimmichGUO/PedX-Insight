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

    results_list = []
    frame_id = -1

    class_map = {
        0: 'Longitudinal Crack',
        1: 'Transverse Crack',
        2: 'Alligator Crack',
        3: 'Potholes'
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        result = model(frame, verbose=False)[0]

        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = class_map.get(cls_id, "unknown")
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            results_list.append({
                "frame_id": frame_id,
                "defect_type": cls_name,
                "confidence": round(conf, 4),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })

    cap.release()
    pd.DataFrame(results_list).to_csv(output_csv_path, index=False)
    print(f"Road defect detection completed. Results saved to {output_csv_path}")
