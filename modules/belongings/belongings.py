import os
import cv2
import pandas as pd
from ultralytics import YOLO
import math
import torch
from collections import defaultdict

def run_belongings_detection(
    video_path,
    analyze_interval_sec=1.0,
    weights="yolo11n.pt",
    tracking_csv_path=None,
    output_csv_path=None,
    vote_fraction=0.2,
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # FIX #24: aggregate per pedestrian with a VOTE instead of an OR-latch, and read frames
    # on demand instead of caching the whole video to RAM. For each track we count how many of
    # its analyzed frames each item was detected in; the item is reported present only if that
    # fraction reaches `vote_fraction` (so one lucky single-frame detection no longer latches an
    # item on for the whole pedestrian).
    detect_counts = defaultdict(lambda: {name: 0 for name in TARGET_CLASSES.values()})
    frame_counts = defaultdict(int)  # analyzed frames actually processed per track (vote denominator)
    first_frame = {}                 # first analyzed frame per track, kept for the frame_id column

    # Only the sampled frames that actually carry tracking rows need to be decoded.
    analyzed_fids = sorted(
        fid for fid in df["frame_id"].unique() if fid % analyze_every_n_frames == 0
    )

    for fid in analyzed_fids:
        frame_data = df[df["frame_id"] == fid]
        # Read this frame on demand. tracking frame_id is 1-indexed while OpenCV's
        # CAP_PROP_POS_FRAMES is 0-indexed, so seek to fid - 1. If the video is missing/deleted
        # the read fails and we simply skip (empty output stays a valid header-only CSV).
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid - 1)
        success, frame = cap.read()
        if not success or frame is None:
            continue

        for _, row in frame_data.iterrows():
            track_id = int(row["track_id"])
            x1, y1, x2, y2 = map(int, [row["x1"], row["y1"], row["x2"], row["y2"]])
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            result = model(crop, verbose=False)[0]

            detected = set()
            for det in result.boxes.data.cpu().numpy():
                cls_id = int(det[5])
                if cls_id in TARGET_CLASSES:
                    detected.add(TARGET_CLASSES[cls_id])

            frame_counts[track_id] += 1
            first_frame.setdefault(track_id, fid)
            for name in detected:
                detect_counts[track_id][name] += 1

    cap.release()

    results_list = []
    for track_id in sorted(frame_counts.keys()):
        total = frame_counts[track_id]
        item_flags = {
            name: (1 if detect_counts[track_id][name] / total >= vote_fraction else 0)
            for name in TARGET_CLASSES.values()
        }
        results_list.append({
            "frame_id": first_frame[track_id],
            "track_id": track_id,
            **item_flags,
        })

    df_out = pd.DataFrame(results_list, columns=["frame_id", "track_id"] + list(TARGET_CLASSES.values()))
    df_out.to_csv(output_csv_path, index=False)
    print(f"Belongings detection completed. Results saved to {output_csv_path}")
