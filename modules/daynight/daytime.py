import cv2
import numpy as np

def run_daytime_detection(source_video_path, brightness_threshold=100):
    cap = cv2.VideoCapture(source_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    brightness_values = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        avg_brightness = np.mean(hsv_frame[:, :, 2])
        brightness_values.append(avg_brightness)

    cap.release()

    overall_avg_brightness = np.mean(brightness_values)

    if overall_avg_brightness > brightness_threshold:
        print("Day")
    else:
        print("Evening")
