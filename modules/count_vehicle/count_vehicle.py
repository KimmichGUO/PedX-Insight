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


def vehicle_count(video_path, output_csv_path=None, analyze_interval_sec=1.0):
    model = YOLO("modules/count_vehicle/best.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if output_csv_path is None:
        output_dir = os.path.join("analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "[V6]vehicle_count.csv")
    else:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    analyze_every_n_frames = max(1, math.ceil(fps * analyze_interval_sec))

    tracked_vehicles = {}
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Skip frames based on analysis interval
        if frame_idx % analyze_every_n_frames != 0:
            continue

        # Choose appropriate tracker config based on FPS (similar to pedestrian code)
        if fps < 30:
            track_results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",  # or use custom config if needed
                conf=0.3,
                verbose=False
            )
        elif fps > 45:
            track_results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",  # or use custom config if needed
                conf=0.3,
                verbose=False
            )
        else:
            track_results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",  # or use custom config if needed
                conf=0.3,
                verbose=False
            )

        # Check if any detections exist
        if track_results[0].boxes is None or len(track_results[0].boxes) == 0:
            continue

        boxes = track_results[0].boxes

        # Check if tracking IDs exist
        if boxes.id is None:
            continue

        # Process each tracked vehicle
        for i, (box, track_id, cls_id, conf) in enumerate(zip(
                boxes.xyxy, boxes.id, boxes.cls, boxes.conf
        )):
            track_id = int(track_id.cpu().numpy())
            cls_id = int(cls_id.cpu().numpy())
            conf = float(conf.cpu().numpy())

            # Check if the class is in our vehicle categories and confidence is above threshold
            if cls_id in id2name and conf > 0.3:
                # Store the vehicle type for this track_id
                if track_id not in tracked_vehicles:
                    tracked_vehicles[track_id] = id2name[cls_id]

    cap.release()

    # Count vehicles by type
    count_dict = {name: 0 for name in id2name.values()}
    for vtype in tracked_vehicles.values():
        count_dict[vtype] += 1

    # Create DataFrame with results
    df = pd.DataFrame(list(count_dict.items()), columns=['Vehicle_Type', 'Count'])
    total_count = df['Count'].sum()
    total_df = pd.DataFrame([{'Vehicle_Type': 'Total', 'Count': total_count}])
    df = pd.concat([df, total_df], ignore_index=True)

    # Save results
    df.to_csv(output_csv_path, index=False)
    print(f"Vehicle count completed. Results saved to {output_csv_path}")
    print(f"YOLO is running on: {model.device}")
    print(f"Total vehicles tracked: {len(tracked_vehicles)}")

    return df