import os
import cv2
import numpy as np
import pandas as pd
import math

def run_daytime_detection(video_path, brightness_threshold=100, analyze_interval_sec=1, output_csv_path=None):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if output_csv_path is None:
        output_dir = os.path.join("analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "[E6]daytime.csv")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        pd.DataFrame(columns=["frame_id", "avg_brightness", "daytime_label"]).to_csv(output_csv_path, index=False)
        return

    fps = math.ceil(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval_frames = analyze_interval_sec * fps

    results = []
    last_avg_brightness = None
    last_label = None
    frame_id = 0

    while frame_id < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            avg_brightness = last_avg_brightness
            label = last_label
        else:
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            avg_brightness = float(np.mean(hsv_frame[:, :, 2]))
            label = "Day" if avg_brightness > brightness_threshold else "Evening"
            last_avg_brightness = avg_brightness
            last_label = label

        for f in range(int(frame_id), min(int(frame_id) + int(interval_frames), int(total_frames))):
            results.append({
                "frame_id": f,
                "avg_brightness": avg_brightness,
                "daytime_label": label
            })

        frame_id += interval_frames

    cap.release()
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"Daytime detection completed. Results saved to {output_csv_path}")
