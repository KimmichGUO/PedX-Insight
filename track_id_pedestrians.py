import os
import cv2
import pandas as pd
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
from types import SimpleNamespace
import numpy as np
np.float = float

def run_pedestrian_tracking(video_path, weights="yolov8n.pt", output_csv_path=None):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if output_csv_path is None:
        output_dir = os.path.join("analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "tracked_pedestrians.csv")
    else:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    model = YOLO(weights)
    tracker_args = SimpleNamespace(
        track_thresh=0.3,
        match_thresh=0.8,
        track_buffer=30,
        frame_rate=30,
        mot20=False
    )
    tracker = BYTETracker(tracker_args)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_id = 0
    results = []
    target_cls = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_id += 1
        timestamp = round(frame_id / fps, 2)

        dets = model(frame)[0]
        if dets.boxes.shape[0] == 0:
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        boxes = []
        for box, cls, conf in zip(dets.boxes.xyxy, dets.boxes.cls, dets.boxes.conf):
            if int(cls.item()) == target_cls:
                x1, y1, x2, y2 = box.tolist()
                boxes.append([x1, y1, x2, y2, conf.item()])

        if not boxes:
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        dets_array = np.array(boxes, dtype=np.float32)
        online_targets = tracker.update(
            dets_array,
            img_info=(frame.shape[0], frame.shape[1], 1.0),
            img_size=(frame.shape[0], frame.shape[1])
        )

        for target in online_targets:
            tlwh = target.tlwh
            track_id = target.track_id
            x1, y1 = int(tlwh[0]), int(tlwh[1])
            x2, y2 = int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            results.append({
                "frame_id": frame_id,
                "timestamp": timestamp,
                "track_id": track_id,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"Tracking results saved to: {output_csv_path}")
