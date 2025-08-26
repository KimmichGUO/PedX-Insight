import os
import cv2
import pandas as pd
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
from types import SimpleNamespace
import numpy as np
import math

np.float = float


def run_pedestrian_tracking_with_imgsave(video_path, weights="yolo11n.pt", output_csv_path=None,
                                         analyze_interval_sec=1.0):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if output_csv_path is None:
        output_dir = os.path.join("analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "[B1]tracked_pedestrians.csv")
    else:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    pedestrian_img_dir = os.path.join(os.path.dirname(output_csv_path), "pedestrian_img")
    os.makedirs(pedestrian_img_dir, exist_ok=True)

    model = YOLO(weights)
    model.to("cuda")
    tracker_args = SimpleNamespace(
        track_thresh=0.3,
        match_thresh=0.8,
        track_buffer=300,
        frame_rate=60,
        mot20=False
    )
    tracker = BYTETracker(tracker_args)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    analyze_every_n_frames = max(1, math.ceil(fps * analyze_interval_sec))

    frame_id = 0
    results = []
    target_cls = 0

    saved_frames = {}

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_id += 1

        if frame_id % analyze_every_n_frames != 0:
            continue

        timestamp = round(frame_id / fps, 2)

        dets = model(frame)[0]
        if dets.boxes.shape[0] == 0:
            continue

        boxes = []
        for box, cls, conf in zip(dets.boxes.xyxy, dets.boxes.cls, dets.boxes.conf):
            if int(cls.item()) == target_cls:
                x1, y1, x2, y2 = box.tolist()
                boxes.append([x1, y1, x2, y2, conf.item()])

        if not boxes:
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

            results.append({
                "frame_id": frame_id,
                "timestamp": timestamp,
                "track_id": track_id,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })

            if track_id not in saved_frames:
                saved_frames[track_id] = []

            if len(saved_frames[track_id]) < 2:
                min_frame_interval = max(analyze_every_n_frames * 2, 60)
                if not saved_frames[track_id] or (frame_id - saved_frames[track_id][-1]) >= min_frame_interval:
                    expand_left = int((x2 - x1) * 0.2)
                    expand_right = int((x2 - x1) * 0.2)
                    expand_top = int((y2 - y1) * 0.5)

                    x1_exp = max(0, x1 - expand_left)
                    x2_exp = min(frame.shape[1], x2 + expand_right)
                    y1_exp = max(0, y1 - expand_top)
                    y2_exp = y2 + expand_top

                    crop = frame[y1_exp:y2_exp, x1_exp:x2_exp]
                    if crop.size == 0:
                        continue

                    person_dir = os.path.join(pedestrian_img_dir, f"id_{track_id}")
                    os.makedirs(person_dir, exist_ok=True)
                    # img_path = os.path.join(person_dir, f"frame_{frame_id}.jpg")
                    # cv2.imwrite(img_path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    img_path = os.path.join(person_dir, f"frame_{frame_id}.png")
                    cv2.imwrite(img_path, crop)
                    saved_frames[track_id].append(frame_id)

    cap.release()

    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"Tracking results saved to: {output_csv_path}")
    print(f"Pedestrian images saved in: {pedestrian_img_dir}")
    print(f"YOLO is running on: {model.device}")
    print(f"Total frames processed: {len(df)}")
    print(f"Total pedestrians tracked: {len(df['track_id'].unique()) if not df.empty else 0}")