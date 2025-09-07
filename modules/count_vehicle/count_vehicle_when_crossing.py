import os
import cv2
import pandas as pd
from ultralytics import YOLO
import math
import torch

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if output_csv_path is None:
        output_dir = os.path.join("analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "[C6]crossing_ve_count.csv")
    else:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    if crossing_csv_path is None:
        output_dir = os.path.join("analysis_results", video_name)
        crossing_csv_path = os.path.join(output_dir, "[C3]crossing_judge.csv")

    # Read crossing data
    crossing_df = pd.read_csv(crossing_csv_path)
    crossing_df = crossing_df[crossing_df['crossed'] == True]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    analyze_every_n_frames = max(1, math.ceil(fps * analyze_interval_sec))

    frame_idx = 0
    frame_tracks = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Skip frames based on analysis interval
        if frame_idx % analyze_every_n_frames != 0:
            continue

        # Choose appropriate tracker config based on FPS
        if fps < 30:
            track_results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=0.3,
                verbose=False
            )
        elif fps > 45:
            track_results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=0.3,
                verbose=False
            )
        else:
            track_results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=0.3,
                verbose=False
            )

        # Check if any detections exist
        if track_results[0].boxes is None or len(track_results[0].boxes) == 0:
            frame_tracks[frame_idx] = []
            continue

        boxes = track_results[0].boxes

        # Check if tracking IDs exist
        if boxes.id is None:
            frame_tracks[frame_idx] = []
            continue

        # Process each tracked vehicle
        tracks_this_frame = []
        for i, (box, track_id, cls_id, conf) in enumerate(zip(
                boxes.xyxy, boxes.id, boxes.cls, boxes.conf
        )):
            track_id = int(track_id.cpu().numpy())
            cls_id = int(cls_id.cpu().numpy())
            conf = float(conf.cpu().numpy())

            # Check if the class is in our vehicle categories and confidence is above threshold
            if cls_id in id2name and conf > 0.3:
                tracks_this_frame.append((track_id, id2name[cls_id]))

        frame_tracks[frame_idx] = tracks_this_frame

    cap.release()

    # Process crossing data and count vehicles
    output_data = []
    for _, row in crossing_df.iterrows():
        person_id = row['track_id']
        start_frame = int(row['started_frame'])
        end_frame = int(row['ended_frame'])

        # Collect unique vehicle tracks during crossing period
        unique_tracks = {}
        for f in range(start_frame, end_frame + 1):
            if f in frame_tracks:
                for track_id, vt in frame_tracks[f]:
                    unique_tracks[track_id] = vt

        # Count vehicles by type
        cumulative_counts = {name: 0 for name in id2name.values()}
        for vt in unique_tracks.values():
            cumulative_counts[vt] += 1

        total_vehicles = sum(cumulative_counts.values())
        row_data = [person_id, True, total_vehicles] + [cumulative_counts[vt] for vt in id2name.values()]
        output_data.append(row_data)

    # Create output DataFrame
    columns = ['track_id', 'crossed', 'total_vehicle_count'] + list(id2name.values())
    output_df = pd.DataFrame(output_data, columns=columns)

    if output_df.empty:
        output_df = pd.DataFrame(columns=columns)

    output_df.to_csv(output_csv_path, index=False)
    print(f"Vehicle count when crossing completed. Results saved to {output_csv_path}")
    print(f"YOLO is running on: {model.device}")
    print(f"Total crossing events analyzed: {len(crossing_df)}")

    return output_df