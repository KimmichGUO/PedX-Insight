import cv2
import pandas as pd
import os
from ultralytics import YOLO
import math
import torch

def run_phone_detection(
    video_path,
    analyze_interval_sec=1.0,
    weights="yolo11n.pt",
    tracking_csv_path=None
):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if tracking_csv_path is None:
        tracking_csv_path = os.path.join(".", "analysis_results", video_name, "[B1]tracked_pedestrians.csv")

    output_dir = os.path.join(".", "analysis_results", video_name)
    os.makedirs(output_dir, exist_ok=True)
    phone_csv_path = os.path.join(output_dir, "[P5]phone_usage.csv")

    if not os.path.exists(tracking_csv_path) or os.path.getsize(tracking_csv_path) == 0:
        empty_phone_df = pd.DataFrame(columns=["frame_id", "track_id", "phone_using"])
        empty_phone_df.to_csv(phone_csv_path, index=False)
        print(f"Tracking CSV not found or empty. Empty results saved to {phone_csv_path}")
        return

    df = pd.read_csv(tracking_csv_path)
    if df.empty:
        empty_phone_df = pd.DataFrame(columns=["frame_id", "track_id", "phone_using"])
        empty_phone_df.to_csv(phone_csv_path, index=False)
        print(f"Tracking CSV is empty. Empty results saved to {phone_csv_path}")
        return

    df.sort_values(by=["frame_id", "track_id"], inplace=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    analyze_every_n_frames = max(1, math.ceil(fps * analyze_interval_sec))
    print(f"Video FPS: {fps:.2f}, analyzing every {analyze_every_n_frames} frames (~{analyze_interval_sec}s)")

    model = YOLO(weights)
    names = model.names
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    phone_results = []

    grouped = df.groupby("frame_id")

    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

    frame_cache = {}
    for frame_id in sorted(df["frame_id"].unique()):
        if frame_id % analyze_every_n_frames != 0:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
        ret, frame = cap.read()
        if not ret:
            continue
        yolo_results = model.predict(frame, conf=0.25, show=False)[0]
        boxes = yolo_results.boxes.xyxy.cpu().numpy()
        clss = yolo_results.boxes.cls.cpu().numpy().astype(int)

        phone_boxes = [box for box, cls in zip(boxes, clss) if names[cls] == "cell phone"]

        frame_cache[frame_id] = phone_boxes

    for frame_id, group in grouped:
        analysis_frame_id = (frame_id // analyze_every_n_frames) * analyze_every_n_frames
        phone_boxes = frame_cache.get(analysis_frame_id, [])

        for _, row in group.iterrows():
            track_id = row["track_id"]
            x1, y1, x2, y2 = map(int, [row["x1"], row["y1"], row["x2"], row["y2"]])
            person_box = [x1, y1, x2, y2]

            phone_using = any(iou(person_box, pb) > 0.1 for pb in phone_boxes)
            phone_results.append({
                "frame_id": frame_id,
                "track_id": track_id,
                "phone_using": phone_using
            })

    cap.release()

    pd.DataFrame(phone_results, columns=["frame_id", "track_id", "phone_using"]).to_csv(phone_csv_path, index=False)

    print(f"Phone usage results saved to {phone_csv_path}")
