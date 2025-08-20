# import os
# import cv2
# import pandas as pd
# from ultralytics import YOLO
#
# def run_weather_detection(video_path, output_csv_path=None):
#     video_name = os.path.splitext(os.path.basename(video_path))[0]
#     if output_csv_path is None:
#         output_dir = os.path.join("analysis_results", video_name)
#         os.makedirs(output_dir, exist_ok=True)
#         output_csv_path = os.path.join(output_dir, "[E1]weather.csv")
#
#     model = YOLO('modules/weather/best.pt')
#     cap = cv2.VideoCapture(video_path)
#
#     results_list = []
#     frame_id = -1
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_id += 1
#
#         result = model(frame, verbose=False)[0]
#
#         if result.probs is not None:
#             pred_index = int(result.probs.top1)
#             pred_label = model.names[pred_index]
#         elif result.boxes.data.size(0) > 0:
#             best_det = result.boxes.conf.argmax().item()
#             pred_index = int(result.boxes.cls[best_det].item())
#             pred_label = model.names[pred_index]
#         else:
#             pred_label = "unknown"
#
#         results_list.append({
#             "frame_id": frame_id,
#             "weather_label": pred_label
#         })
#
#     cap.release()
#     pd.DataFrame(results_list).to_csv(output_csv_path, index=False)
#     print(f"Weather detection completed. Results saved to {output_csv_path}")
import os
import cv2
import pandas as pd
from ultralytics import YOLO

def run_weather_detection(video_path, output_csv_path=None, detect_interval=3000):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if output_csv_path is None:
        output_dir = os.path.join("analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "[E1]weather.csv")

    model = YOLO('modules/weather/best.pt')
    cap = cv2.VideoCapture(video_path)

    results_list = []
    frame_id = -1
    current_label = "unknown"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        if frame_id % detect_interval == 0:
            result = model(frame, verbose=False)[0]

            if result.probs is not None:
                pred_index = int(result.probs.top1)
                current_label = model.names[pred_index]
            elif result.boxes.data.size(0) > 0:
                best_det = result.boxes.conf.argmax().item()
                pred_index = int(result.boxes.cls[best_det].item())
                current_label = model.names[pred_index]
            else:
                current_label = "unknown"

        results_list.append({
            "frame_id": frame_id,
            "weather_label": current_label
        })

    cap.release()
    pd.DataFrame(results_list).to_csv(output_csv_path, index=False)
    print(f"Weather detection completed. Results saved to {output_csv_path}")
