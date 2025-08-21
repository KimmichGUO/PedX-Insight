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

def analyze_vehicle_during_crossing(
    video_path,
    crossing_csv_path=None,
    output_csv_path=None,
    analyze_interval_sec=1.0
):
    model = YOLO("modules/count_vehicle/best.pt")
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if output_csv_path is None:
        output_dir = os.path.join("analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "[C6]crossing_ve_count.csv")
    if crossing_csv_path is None:
        crossing_csv_path = os.path.join(output_dir, "[C3]crossing_judge.csv")

    crossing_df = pd.read_csv(crossing_csv_path)
    crossing_df = crossing_df[crossing_df['crossed'] == True]

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

    frame_idx = 0
    frame_tracks = {}

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
            frame_tracks[frame_idx] = []
            continue

        dets = np.array([np.append(bbox, score) for bbox, score in zip(bboxes, scores)])
        online_targets = tracker.update(dets, [frame.shape[0], frame.shape[1]], [frame.shape[0], frame.shape[1]])

        tracks_this_frame = []
        for t in online_targets:
            tlwh = t.tlwh
            bbox = [tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]]

            dists = np.linalg.norm(dets[:, :4] - bbox, axis=1)
            min_idx = np.argmin(dists)
            cls_id = cls_ids[min_idx]

            tracks_this_frame.append((t.track_id, id2name[cls_id]))

        frame_tracks[frame_idx] = tracks_this_frame

    cap.release()

    output_data = []
    for _, row in crossing_df.iterrows():
        person_id = row['track_id']
        start_frame = int(row['started_frame'])
        end_frame = int(row['ended_frame'])

        unique_tracks = {}
        for f in range(start_frame, end_frame + 1):
            if f in frame_tracks:
                for track_id, vt in frame_tracks[f]:
                    unique_tracks[track_id] = vt

        cumulative_counts = {name: 0 for name in id2name.values()}
        for vt in unique_tracks.values():
            cumulative_counts[vt] += 1

        total_vehicles = sum(cumulative_counts.values())
        row_data = [person_id, True, total_vehicles] + [cumulative_counts[vt] for vt in id2name.values()]
        output_data.append(row_data)

    columns = ['track_id', 'crossed', 'total_vehicle_count'] + list(id2name.values())
    output_df = pd.DataFrame(output_data, columns=columns)

    if output_df.empty:
        output_df = pd.DataFrame(columns=columns)

    output_df.to_csv(output_csv_path, index=False)
    print(f"Vehicle count when crossing completed. Results saved to {output_csv_path}")
