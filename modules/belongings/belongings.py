import os
import cv2
import pandas as pd
from ultralytics import YOLO
import math

def run_belongings_detection(
    video_path,
    weights="yolo11n.pt",
    tracking_csv_path=None,
    output_csv_path=None,
    analyze_interval_sec=1.0
):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if tracking_csv_path is None:
        tracking_csv_path = os.path.join("analysis_results", video_name, "[B1]tracked_pedestrians.csv")
    if output_csv_path is None:
        output_dir = os.path.join("analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "[P9]pedestrian_belongings.csv")

    TARGET_CLASSES = {
        24: "backpack",
        25: "umbrella",
        26: "handbag",
        28: "suitcase"
    }

    if not os.path.exists(tracking_csv_path) or os.path.getsize(tracking_csv_path) == 0:
        empty_df = pd.DataFrame(columns=["frame_id", "track_id"] + list(TARGET_CLASSES.values()))
        empty_df.to_csv(output_csv_path, index=False)
        print(f"No tracking data found. Empty results saved to {output_csv_path}")
        return

    df = pd.read_csv(tracking_csv_path)
    if df.empty:
        empty_df = pd.DataFrame(columns=["frame_id", "track_id"] + list(TARGET_CLASSES.values()))
        empty_df.to_csv(output_csv_path, index=False)
        print(f"Tracking CSV is empty. Empty results saved to {output_csv_path}")
        return

    df.sort_values(by=["frame_id", "track_id"], inplace=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    analyze_every_n_frames = max(1, math.ceil(fps * analyze_interval_sec))
    print(f"Video FPS: {fps:.2f}, analyzing every {analyze_every_n_frames} frames (~{analyze_interval_sec}s)")

    model = YOLO(weights)

    results_list = []
    frame_cache = dict()
    frame_id = -1
    success, frame = cap.read()

    while success:
        frame_id += 1
        if frame_id % analyze_every_n_frames == 0:
            frame_cache[frame_id] = frame.copy()
        success, frame = cap.read()

    for fid in sorted(df["frame_id"].unique()):
        if fid % analyze_every_n_frames != 0:
            continue

        frame_data = df[df["frame_id"] == fid]
        frame = frame_cache.get(fid)
        if frame is None:
            continue

        for _, row in frame_data.iterrows():
            track_id = int(row["track_id"])
            x1, y1, x2, y2 = map(int, [row["x1"], row["y1"], row["x2"], row["y2"]])
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            result = model(crop, verbose=False)[0]

            item_flags = {name: 0 for name in TARGET_CLASSES.values()}
            for det in result.boxes.data.cpu().numpy():
                cls_id = int(det[5])
                if cls_id in TARGET_CLASSES:
                    name = TARGET_CLASSES[cls_id]
                    item_flags[name] = 1

            row_data = {
                "frame_id": fid,
                "track_id": track_id,
                **item_flags
            }
            results_list.append(row_data)

    df_out = pd.DataFrame(results_list, columns=["frame_id", "track_id"] + list(TARGET_CLASSES.values()))
    df_out.to_csv(output_csv_path, index=False)
    print(f"Belongings detection completed. Results saved to {output_csv_path}")
