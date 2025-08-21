import os
import cv2
import pandas as pd
from ultralytics import YOLO
import math

def run_traffic_sign(video_path, output_csv_path=None, conf=0.25, analyze_interval_sec=1.0):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if output_csv_path is None:
        output_dir = os.path.join("analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "[E3]traffic_sign.csv")

    model_asia = YOLO("modules/traffic_sign/best_asia.pt")
    model_new = YOLO("modules/traffic_sign/best_new.pt")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = math.ceil(fps) if fps > 0 else 30
    analyze_every_n_frames = max(1, int(fps * analyze_interval_sec))

    results_list = []
    frame_id = -1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        if frame_id % analyze_every_n_frames != 0:
            continue

        # Asia traffic sign detection
        result_asia = model_asia(frame, imgsz=640, conf=conf, verbose=False)[0]
        detected_asia = 0
        class_str_asia = ""
        if result_asia.boxes is not None and result_asia.boxes.data.size(0) > 0:
            detected_asia = 1
            for box in result_asia.boxes:
                cls_id = int(box.cls)
                label = model_asia.names[cls_id]
                class_str_asia += label + ";"

        # New traffic sign detection
        result_new = model_new(frame, imgsz=640, conf=conf, verbose=False)[0]
        detected_new = 0
        class_str_new = ""
        if result_new.boxes is not None and result_new.boxes.data.size(0) > 0:
            for box in result_new.boxes:
                cls_id = int(box.cls)
                label = model_new.names[cls_id]

                if cls_id == 24 or label.lower() == "traffic_signal":
                    continue
                detected_new = 1
                class_str_new += label + ";"

        results_list.append({
            "frame_id": frame_id,
            "sign_detected_1": detected_asia,
            "sign_classes_1": class_str_asia.strip(";"),
            "sign_detected_2": detected_new,
            "sign_classes_2": class_str_new.strip(";")
        })

    cap.release()
    cv2.destroyAllWindows()
    pd.DataFrame(results_list).to_csv(output_csv_path, index=False)
    print(f"Traffic Sign detection completed. Results saved to {output_csv_path}")
