import os
import cv2
import pandas as pd
import math
from ultralytics import YOLO

CLASS_NAMES = {0: 'police_car', 1: 'Arrow Board', 2: 'cones', 3: 'accident'}

def run_accident_scene_detection(
    video_path,
    output_csv_path=None,
    conf=0.25,
    analyze_interval_sec=1.0
):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if output_csv_path is None:
        output_dir = os.path.join("analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "[E8]accident_detection.csv")

    model = YOLO("modules/accident/best.pt")
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    analyze_every_n_frames = max(1, math.ceil(fps * analyze_interval_sec))

    print(f"Video FPS: {fps:.2f}, analyzing every {analyze_every_n_frames} frames (~{analyze_interval_sec}s)")

    results_list = []
    frame_id = -1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        if frame_id % analyze_every_n_frames != 0:
            continue

        result = model(frame, imgsz=640, conf=conf, verbose=False)[0]

        detected_classes = {name: 0 for name in CLASS_NAMES.values()}

        if result.boxes is not None and result.boxes.data.size(0) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id in CLASS_NAMES:
                    detected_classes[CLASS_NAMES[cls_id]] = 1

        frame_result = {"frame_id": frame_id}
        frame_result.update(detected_classes)

        results_list.append(frame_result)

    cap.release()

    df = pd.DataFrame(results_list, columns=["frame_id"] + list(CLASS_NAMES.values()))
    df.to_csv(output_csv_path, index=False)
    print(f"Accident scene detection completed. Results saved to {output_csv_path}")
