import cv2
import pandas as pd
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
from types import SimpleNamespace
import numpy as np

model = YOLO("yolov8n.pt")

args = SimpleNamespace(track_thresh=0.3, match_thresh=0.8, track_buffer=30, frame_rate=30, mot20=False)
tracker = BYTETracker(args)

video_path = "pedestrian.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_id = 0
results = []

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame_id += 1
    timestamp = frame_id / fps

    dets = model(frame)[0]
    if dets.boxes.shape[0] == 0:
        continue

    boxes = []
    for box, cls, conf in zip(dets.boxes.xyxy, dets.boxes.cls, dets.boxes.conf):
        if int(cls.item()) == 0: 
            x1, y1, x2, y2 = box.tolist()
            boxes.append([x1, y1, x2, y2, conf.item()])

    if len(boxes) == 0:
        continue

    dets_array = np.array(boxes, dtype=np.float32)

    online_targets = tracker.update(dets_array, img_info=(frame.shape[0], frame.shape[1], 1.0), img_size=(frame.shape[0], frame.shape[1]))

    for target in online_targets:
        tlwh = target.tlwh
        tid = target.track_id
        results.append({
            "frame_id": frame_id,
            "timestamp": round(timestamp, 2),
            "track_id": tid,
            "x1": round(tlwh[0], 2),
            "y1": round(tlwh[1], 2),
            "x2": round(tlwh[0] + tlwh[2], 2),
            "y2": round(tlwh[1] + tlwh[3], 2)
        })

cap.release()

df = pd.DataFrame(results)
df.to_csv("tracked_pedestrians.csv", index=False)
