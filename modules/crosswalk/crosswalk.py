import os
import cv2
import pandas as pd
from ultralytics import YOLO
import math
import torch


def run_crosswalk_detection(video_path, analyze_interval_sec=1.0, output_csv_path=None, conf=0.1, show=False):

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if output_csv_path is None:
        output_dir = os.path.join("analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "[E7]crosswalk_detection.csv")

    model = YOLO("modules/crosswalk/best.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = math.ceil(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, int(fps * analyze_interval_sec))

    results_list = []
    frame_id = -1

    last_detected = "No"
    last_boxes = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        if frame_id % frame_interval == 0:

            result = model(frame, imgsz=640, conf=conf, verbose=False)[0]
            detected = False
            crosswalk_boxes = []

            if result.boxes is not None and result.boxes.data.size(0) > 0:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id == 0:
                        detected = True
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        width = (x2 - x1) * 2
                        extended_x1 = x1 - width
                        extended_x2 = x2 + width
                        coords = [round(extended_x1, 2), round(y1, 2), round(extended_x2, 2), round(y2, 2)]
                        crosswalk_boxes.append(coords)

                        if show:
                            cv2.rectangle(
                                frame,
                                (int(x1), int(y1)),
                                (int(x2), int(y2)),
                                (0, 0, 255), 2
                            )
                            cv2.putText(frame, "Crosswalk", (int(x1), int(y1) - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            last_detected = "Yes" if detected else "No"
            last_boxes = crosswalk_boxes if detected else []

        results_list.append({
            "frame_id": frame_id,
            "crosswalk_detected": last_detected,
            "crosswalk_boxes": last_boxes
        })

        if show:
            cv2.imshow("Crosswalk Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Detection stopped by user.")
                break

    cap.release()
    cv2.destroyAllWindows()

    if not results_list:
        results_df = pd.DataFrame(columns=["frame_id", "crosswalk_detected", "crosswalk_boxes"])
    else:
        results_df = pd.DataFrame(results_list)

    results_df.to_csv(output_csv_path, index=False)
    print(f"Crosswalk detection completed. Results saved to {output_csv_path}")
