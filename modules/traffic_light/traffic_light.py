import os
import cv2
import pandas as pd
from ultralytics import YOLO

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

    color_map = {
        "green": (0, 255, 0),
        "red": (0, 0, 255),
        "yellow": (0, 255, 255)
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        result = model(frame, verbose=False)[0]
        lights = []

        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in light_id_map:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            cls_name = light_id_map[cls_id]

            lights.append({
                "color": cls_name,
                "area": area,
                "box": (x1, y1, x2, y2)
            })

            cv2.rectangle(frame, (x1, y1), (x2, y2), color_map[cls_name], 2)
            cv2.putText(frame, cls_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_map[cls_name], 2)

        if not lights:
            results_list.append({
                "frame_id": frame_id,
                "main_light_color": "None",
                "other_lights": "None"
            })
        else:
            lights.sort(key=lambda x: x["area"], reverse=True)
            main_color = lights[0]["color"]
            other_colors = [light["color"] for light in lights[1:]]
            other_str = str(other_colors) if other_colors else "None"

            results_list.append({
                "frame_id": frame_id,
                "main_light_color": main_color,
                "other_lights": other_str
            })

        cv2.imshow("Traffic Light Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pd.DataFrame(results_list).to_csv(output_csv_path, index=False)
    print(f"Traffic light detection completed. Results saved to {output_csv_path}")
