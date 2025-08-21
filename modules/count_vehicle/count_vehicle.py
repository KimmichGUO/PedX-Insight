import os
import cv2
import pandas as pd
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
import numpy as np
np.float = float
from types import SimpleNamespace
import math

id2name = {
    0: 'ambulance',
    1: 'army vehicle',
    2: 'auto rickshaw',
    3: 'bicycle',
    4: 'bus',
    5: 'car',
    6: 'garbagevan',
    7: 'human hauler',
    8: 'minibus',
    9: 'minivan',
    10: 'motorbike',
    11: 'pickup',
    12: 'policecar',
    13: 'rickshaw',
    14: 'scooter',
    15: 'suv',
    16: 'taxi',
    17: 'three wheelers -CNG-',
    18: 'truck',
    19: 'van',
    20: 'wheelbarrow'
}

def vehicle_count(video_path, output_csv_path=None, analyze_interval_sec=1.0):
    model = YOLO("modules/count_vehicle/best.pt")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if output_csv_path is None:
        output_dir = os.path.join("analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "[V6]vehicle_count.csv")

    tracker_args = SimpleNamespace(
        track_thresh=0.3,
        match_thresh=0.8,
        track_buffer=120,
        frame_rate=60,
        mot20=False
    )
    tracker = BYTETracker(tracker_args)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    analyze_every_n_frames = max(1, math.ceil(fps * analyze_interval_sec))
    print(f"Video FPS: {fps:.2f}, analyzing every {analyze_every_n_frames} frames (~{analyze_interval_sec}s)")

    tracked_vehicles = {}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % analyze_every_n_frames != 0:
            continue

        results = model(frame)[0]

        bboxes = []
        scores = []
        cls_ids = []

        for box, score, cls_id in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            cls_id = int(cls_id)
            if cls_id in id2name and score > 0.3:
                bboxes.append(box.cpu().numpy())
                scores.append(score.cpu().numpy())
                cls_ids.append(cls_id)

        if len(bboxes) == 0:
            continue

        dets = np.array([np.append(bbox, score) for bbox, score in zip(bboxes, scores)])

        online_targets = tracker.update(dets, [frame.shape[0], frame.shape[1]], [frame.shape[0], frame.shape[1]])

        for t in online_targets:
            track_id = t.track_id
            tlwh = t.tlwh
            bbox = [tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]]

            dists = np.linalg.norm(dets[:, :4] - bbox, axis=1)
            min_idx = np.argmin(dists)
            cls_id = cls_ids[min_idx]

            if track_id not in tracked_vehicles:
                tracked_vehicles[track_id] = id2name[cls_id]

    cap.release()

    count_dict = {name: 0 for name in id2name.values()}
    for vtype in tracked_vehicles.values():
        count_dict[vtype] += 1

    df = pd.DataFrame(list(count_dict.items()), columns=['Vehicle_Type', 'Count'])
    total_count = df['Count'].sum()
    total_df = pd.DataFrame([{'Vehicle_Type': 'Total', 'Count': total_count}])
    df = pd.concat([df, total_df], ignore_index=True)

    df.to_csv(output_csv_path, index=False)
    print(f"Vehicle count completed. Results saved to {output_csv_path}")
