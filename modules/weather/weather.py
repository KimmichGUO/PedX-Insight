import os
import cv2
import pandas as pd
from ultralytics import YOLO
import math

def run_weather_detection(video_path, output_csv_path=None, analyze_interval_sec=1.0):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if output_csv_path is None:
        output_dir = os.path.join("analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "[E1]weather.csv")

    model = YOLO('modules/weather/best.pt')
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = math.ceil(fps) if fps > 0 else 30
    analyze_every_n_frames = max(1, int(fps * analyze_interval_sec))

    results_list = []
    frame_id = -1
    last_label = "unknown"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        if frame_id % analyze_every_n_frames == 0:
            result = model(frame, verbose=False)[0]

            if result.probs is not None:
                pred_index = int(result.probs.top1)
                last_label = model.names[pred_index]
            elif result.boxes.data.size(0) > 0:
                best_det = result.boxes.conf.argmax().item()
                pred_index = int(result.boxes.cls[best_det].item())
                last_label = model.names[pred_index]
            else:
                last_label = "unknown"

        results_list.append({
            "frame_id": frame_id,
            "weather_label": last_label
        })

    cap.release()
    pd.DataFrame(results_list).to_csv(output_csv_path, index=False)
    print(f"Weather detection completed. Results saved to {output_csv_path}")
