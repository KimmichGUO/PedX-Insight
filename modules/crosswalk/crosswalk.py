import os
import cv2
import pandas as pd
from ultralytics import YOLO

def run_crosswalk_detection(video_path, output_csv_path=None, conf=0.25):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if output_csv_path is None:
        output_dir = os.path.join("analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "crosswalk_detection.csv")

    model = YOLO("modules/crosswalk/best.pt")
    cap = cv2.VideoCapture(video_path)

    results_list = []
    frame_id = -1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        result = model(frame, imgsz=640, conf=conf, verbose=False)[0]

        detected = False
        if result.boxes is not None and result.boxes.data.size(0) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id == 0:  # Crosswalk class
                    detected = True
                    break

        results_list.append({
            "frame_id": frame_id,
            "crosswalk_detected": int(detected)
        })

    cap.release()
    pd.DataFrame(results_list).to_csv(output_csv_path, index=False)
    print(f"Crosswalk detection completed. Results saved to {output_csv_path}")
