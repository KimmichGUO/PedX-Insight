import os
import cv2
import pandas as pd
from collections import Counter, defaultdict
from ultralytics import YOLO
import supervision as sv
import math

def run_vehicle_frame_analysis(video_path, weights="modules/type_vehicle/best.pt",
                               output_csv_path=None, analyze_interval_sec=1.0):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if output_csv_path is None:
        output_dir = os.path.join("analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "[V1]vehicle_type.csv")

    model = YOLO(weights)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = math.ceil(fps) if fps > 0 else 30
    analyze_every_n_frames = max(1, int(fps * analyze_interval_sec))

    class_name_list = list(model.model.names.values())
    box_annotator = sv.BoxAnnotator()

    results_list = []
    frame_id = -1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        # 按 analyze_interval_sec 跳过帧
        if frame_id % analyze_every_n_frames != 0:
            continue

        result = model.predict(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        frame_counter = Counter(detections.class_id)
        stat = defaultdict(int)

        for class_id, count in frame_counter.items():
            class_name = model.model.names[class_id]
            stat[class_name] = count

        for class_name in class_name_list:
            stat[class_name] = stat.get(class_name, 0)

        stat["frame_id"] = frame_id
        stat["total"] = sum(stat[class_name] for class_name in class_name_list)

        results_list.append(stat)

    cap.release()

    df = pd.DataFrame(results_list)
    columns = ["frame_id"] + sorted(class_name_list) + ["total"]
    df = df[columns]
    df.to_csv(output_csv_path, index=False)
    print(f"Vehicle frame-level statistics saved to {output_csv_path}")
