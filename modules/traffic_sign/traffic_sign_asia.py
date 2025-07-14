import os
import cv2
import pandas as pd
from ultralytics import YOLO

def run_traffic_sign_asia(video_path, output_csv_path=None, conf=0.25):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if output_csv_path is None:
        output_dir = os.path.join("analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "traffic_sign_detection_asia.csv")

    model = YOLO("modules/traffic_sign/best_asia.pt")
    cap = cv2.VideoCapture(video_path)

    results_list = []
    frame_id = -1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        result = model(frame, imgsz=640, conf=conf, verbose=False)[0]

        if result.boxes is not None and result.boxes.data.size(0) > 0:
            detected = True
            class_names = [model.names[int(cls)] for cls in result.boxes.cls]
            class_str = ";".join(class_names)
        else:
            detected = False
            class_str = ""

        results_list.append({
            "frame_id": frame_id,
            "traffic_sign_detected": int(detected),
            "traffic_sign_classes": class_str
        })

    cap.release()
    pd.DataFrame(results_list).to_csv(output_csv_path, index=False)
    print(f"Traffic sign detection completed. Results saved to {output_csv_path}")
