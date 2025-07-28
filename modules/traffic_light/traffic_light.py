import os
import cv2
import pandas as pd
from ultralytics import YOLO

def classify_light_shape(x1, y1, x2, y2):
    width = x2 - x1
    height = y2 - y1
    return "vertical" if height > width else "horizontal"

def run_traffic_light_detection(video_path, output_csv_path=None):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if output_csv_path is None:
        output_dir = os.path.join("analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "traffic_light_detection.csv")

    model = YOLO("modules/traffic_light/v9 - 48 epochs.pt")
    cap = cv2.VideoCapture(video_path)

    results_list = []
    frame_id = -1

    light_id_map = {
        0: 'green',
        1: 'red',
        2: 'yellow'
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        result = model(frame, verbose=False)[0]

        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in light_id_map:
                continue

            cls_name = light_id_map[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            shape = classify_light_shape(x1, y1, x2, y2)

            color = (0, 255, 0) if cls_name == "green" else (0, 0, 255) if cls_name == "red" else (0, 255, 255)
            label = f"{cls_name} | {shape}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            results_list.append({
                "frame_id": frame_id,
                "light_color": cls_name,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "shape": shape
            })

        cv2.imshow("Traffic Light Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pd.DataFrame(results_list).to_csv(output_csv_path, index=False)
    print(f"Traffic light detection completed. Results saved to {output_csv_path}")
