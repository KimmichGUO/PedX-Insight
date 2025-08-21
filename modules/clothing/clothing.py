import cv2
import pandas as pd
import os
from ultralytics import YOLO
import math

def run_clothing_detection(
    video_path,
    tracking_csv_path=None,
    output_csv_path=None,
    analyze_interval_sec=1.0
):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if tracking_csv_path is None:
        tracking_csv_path = os.path.join(".", "analysis_results", video_name, "[B1]tracked_pedestrians.csv")
    if output_csv_path is None:
        output_dir = os.path.join(".", "analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "[P8]clothing.csv")

    if not os.path.exists(tracking_csv_path) or os.path.getsize(tracking_csv_path) == 0:
        class_names = [
            'short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_outwear',
            'long_sleeved_outwear', 'vest', 'sling', 'shorts', 'trousers',
            'skirt', 'short_sleeved_dress', 'long_sleeved_dress', 'vest_dress', 'sling_dress'
        ]
        empty_df = pd.DataFrame(columns=["frame_id", "track_id"] + class_names)
        empty_df.to_csv(output_csv_path, index=False)
        print(f"Tracking CSV not found or empty. Empty results saved to {output_csv_path}")
        return

    df = pd.read_csv(tracking_csv_path)
    if df.empty:
        class_names = [
            'short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_outwear',
            'long_sleeved_outwear', 'vest', 'sling', 'shorts', 'trousers',
            'skirt', 'short_sleeved_dress', 'long_sleeved_dress', 'vest_dress', 'sling_dress'
        ]
        empty_df = pd.DataFrame(columns=["frame_id", "track_id"] + class_names)
        empty_df.to_csv(output_csv_path, index=False)
        print(f"Tracking CSV is empty. Empty results saved to {output_csv_path}")
        return

    df.sort_values(by=["frame_id", "track_id"], inplace=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    analyze_every_n_frames = max(1, math.ceil(fps * analyze_interval_sec))
    print(f"Video FPS: {fps:.2f}, analyzing every {analyze_every_n_frames} frames (~{analyze_interval_sec}s)")

    model = YOLO("modules/clothing/deepfashion2_yolov8s-seg.pt")
    class_names = {
        0: 'short_sleeved_shirt', 1: 'long_sleeved_shirt', 2: 'short_sleeved_outwear',
        3: 'long_sleeved_outwear', 4: 'vest', 5: 'sling', 6: 'shorts',
        7: 'trousers', 8: 'skirt', 9: 'short_sleeved_dress',
        10: 'long_sleeved_dress', 11: 'vest_dress', 12: 'sling_dress'
    }

    grouped = df.groupby("frame_id")
    results = []

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
        detections = [{"box": boxes[i], "cls": clss[i], "name": class_names[clss[i]]} for i in range(len(boxes))]
        frame_cache[frame_id] = detections

    for frame_id, group in grouped:
        analysis_frame_id = (frame_id // analyze_every_n_frames) * analyze_every_n_frames
        detections = frame_cache.get(analysis_frame_id, [])
        for _, row in group.iterrows():
            track_id = row["track_id"]
            x1, y1, x2, y2 = map(int, [row["x1"], row["y1"], row["x2"], row["y2"]])
            person_box = [x1, y1, x2, y2]
            clothing_flags = {name: 0 for name in class_names.values()}
            for det in detections:
                if iou(person_box, det["box"]) > 0.1:
                    clothing_flags[det["name"]] = 1
            result = {
                "frame_id": frame_id,
                "track_id": track_id,
                **clothing_flags
            }
            results.append(result)

    cap.release()
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, index=False)
    print(f"\nClothing detection results saved to {output_csv_path}")
