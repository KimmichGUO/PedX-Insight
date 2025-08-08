import os
import cv2
import numpy as np
import pandas as pd


def run_daytime_detection(video_path, brightness_threshold=100, output_csv_path=None):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if output_csv_path is None:
        output_dir = os.path.join("analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "[E6]daytime.csv")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    results = []
    frame_id = -1
    brightness_values = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        avg_brightness = float(np.mean(hsv_frame[:, :, 2]))
        brightness_values.append(avg_brightness)

    cap.release()

    overall_avg_brightness = np.mean(brightness_values)
    overall_label = "Day" if overall_avg_brightness > brightness_threshold else "Evening"
    print(f"Overall condition: {overall_label} (Average brightness: {overall_avg_brightness:.2f})")

    results = [{
        "frame_id": i,
        "avg_brightness": brightness_values[i],
        "daytime_label": overall_label
    } for i in range(len(brightness_values))]

    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"Daytime detection completed. Results saved to {output_csv_path}")
