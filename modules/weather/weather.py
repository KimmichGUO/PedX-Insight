import os
import cv2
import pandas as pd
from ultralytics import YOLO

def run_weather_detection(video_path, output_csv_path=None):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if output_csv_path is None:
        output_dir = os.path.join("analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "[E1]weather.csv")

    model = YOLO('modules/weather/best.pt')
    cap = cv2.VideoCapture(video_path)

    results_list = []
    frame_id = -1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        result = model(frame, verbose=False)[0]

        if result.probs is not None:  
            pred_index = int(result.probs.top1)
            pred_label = model.names[pred_index]
        elif result.boxes.data.size(0) > 0:  
            best_det = result.boxes.conf.argmax().item()
            pred_index = int(result.boxes.cls[best_det].item())
            pred_label = model.names[pred_index]
        else:
            pred_label = "unknown"

        results_list.append({
            "frame_id": frame_id,
            "weather_label": pred_label
        })

    cap.release()
    pd.DataFrame(results_list).to_csv(output_csv_path, index=False)
    print(f"Weather detection completed. Results saved to {output_csv_path}")
# from tensorflow.keras.models import load_model
# from tensorflow.keras.utils import img_to_array
# import cv2
# import numpy as np

# classes = ["Cloudy","Sunny","Rainy","Snowy","Foggy"]

# def preprocess_frame(frame):
#     frame = cv2.resize(frame, (100, 100))
#     frame = img_to_array(frame)
#     frame = np.expand_dims(frame, axis=0)
#     frame = frame / 255.0
#     return frame

# def run_weather_detection(source_video_path):
#     model = load_model("modules/weather/trainedModelE40.h5")
#     cap = cv2.VideoCapture(source_video_path)

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         input_img = preprocess_frame(frame)
#         pred = model.predict(input_img)
#         label = classes[np.argmax(pred)]

#         cv2.putText(frame, f"Weather: {label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
#                     1.2, (0, 255, 0), 2, cv2.LINE_AA)
#         cv2.imshow("Weather Detection", frame)

#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
