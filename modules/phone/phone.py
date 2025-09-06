import cv2
import pandas as pd
import os
from ultralytics import YOLO
import math

def run_phone_detection(video_path, weights = "yolov8n.pt", tracking_csv_path=None, output_csv_path=None):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if tracking_csv_path is None:
        tracking_csv_path = os.path.join(".", "analysis_results", video_name, "[B1]tracked_pedestrians.csv")
    if output_csv_path is None:
        output_dir = os.path.join(".", "analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "[P5]phone_usage.csv")

    df = pd.read_csv(tracking_csv_path)
    cap = cv2.VideoCapture(video_path)

    model = YOLO(weights)
    names = model.names

    results = []

    print("Start analyzing phone usage frame by frame...")

    grouped = df.groupby("frame_id")

    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
        return iou

    for frame_id, group in grouped:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
        ret, frame = cap.read()
        if not ret:
            print(f"Frame {frame_id} not readable.")
            continue

        yolo_results = model.predict(frame, conf=0.25, show=False)
        boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
        clss = yolo_results[0].boxes.cls.cpu().numpy().astype(int)

        phone_boxes = [box for box, cls in zip(boxes, clss) if names[cls] == "cell phone"]

        for _, row in group.iterrows():
            track_id = row["track_id"]
            x1, y1, x2, y2 = map(int, [row["x1"], row["y1"], row["x2"], row["y2"]])
            person_box = [x1, y1, x2, y2]

            phone_using = False
            for pb in phone_boxes:
                if iou(person_box, pb) > 0.1:
                    phone_using = True
                    break

            results.append({
                "frame_id": frame_id,
                "track_id": track_id,
                "phone_using": phone_using
            })

        print(f"Frame {frame_id} processed.")

    cap.release()

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, index=False)
    print(f"\nPhone usage results saved to {output_csv_path}")

# def run_phone_and_belongings_detection(
#     video_path,
#     weights="yolo11n.pt",
#     tracking_csv_path=None,
#     analyze_interval_sec=1.0
# ):
#     video_name = os.path.splitext(os.path.basename(video_path))[0]
#
#     if tracking_csv_path is None:
#         tracking_csv_path = os.path.join(".", "analysis_results", video_name, "[B1]tracked_pedestrians.csv")
#
#     output_dir = os.path.join(".", "analysis_results", video_name)
#     os.makedirs(output_dir, exist_ok=True)
#     phone_csv_path = os.path.join(output_dir, "[P5]phone_usage.csv")
#     belongings_csv_path = os.path.join(output_dir, "[P9]pedestrian_belongings.csv")
#
#     belongings_classes = ["backpack", "umbrella", "handbag", "suitcase"]
#     if not os.path.exists(tracking_csv_path) or os.path.getsize(tracking_csv_path) == 0:
#         empty_phone_df = pd.DataFrame(columns=["frame_id", "track_id", "phone_using"])
#         empty_belongings_df = pd.DataFrame(columns=["frame_id", "track_id"] + belongings_classes)
#         empty_phone_df.to_csv(phone_csv_path, index=False)
#         empty_belongings_df.to_csv(belongings_csv_path, index=False)
#         print(f"Tracking CSV not found or empty. Empty results saved to {phone_csv_path} and {belongings_csv_path}")
#         return
#
#     df = pd.read_csv(tracking_csv_path)
#     if df.empty:
#         empty_phone_df = pd.DataFrame(columns=["frame_id", "track_id", "phone_using"])
#         empty_belongings_df = pd.DataFrame(columns=["frame_id", "track_id"] + belongings_classes)
#         empty_phone_df.to_csv(phone_csv_path, index=False)
#         empty_belongings_df.to_csv(belongings_csv_path, index=False)
#         print(f"Tracking CSV is empty. Empty results saved to {phone_csv_path} and {belongings_csv_path}")
#         return
#
#     df.sort_values(by=["frame_id", "track_id"], inplace=True)
#
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     if fps <= 0:
#         fps = 30.0
#     analyze_every_n_frames = max(1, math.ceil(fps * analyze_interval_sec))
#     print(f"Video FPS: {fps:.2f}, analyzing every {analyze_every_n_frames} frames (~{analyze_interval_sec}s)")
#
#     model = YOLO(weights)
#     names = model.names
#
#     phone_results = []
#     belongings_results = []
#
#     grouped = df.groupby("frame_id")
#
#     def iou(boxA, boxB):
#         xA = max(boxA[0], boxB[0])
#         yA = max(boxA[1], boxB[1])
#         xB = min(boxA[2], boxB[2])
#         yB = min(boxA[3], boxB[3])
#         interArea = max(0, xB - xA) * max(0, yB - yA)
#         boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
#         boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
#         return interArea / float(boxAArea + boxBArea - interArea + 1e-5)
#
#     frame_cache = {}
#     for frame_id in sorted(df["frame_id"].unique()):
#         if frame_id % analyze_every_n_frames != 0:
#             continue
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
#         ret, frame = cap.read()
#         if not ret:
#             continue
#         yolo_results = model.predict(frame, conf=0.25, show=False)[0]
#         boxes = yolo_results.boxes.xyxy.cpu().numpy()
#         clss = yolo_results.boxes.cls.cpu().numpy().astype(int)
#
#         phone_boxes = [box for box, cls in zip(boxes, clss) if names[cls] == "cell phone"]
#         items_boxes = {cls_name: [] for cls_name in belongings_classes}
#         for box, cls in zip(boxes, clss):
#             name = names[cls]
#             if name in belongings_classes:
#                 items_boxes[name].append(box)
#
#         frame_cache[frame_id] = (phone_boxes, items_boxes)
#
#     for frame_id, group in grouped:
#         analysis_frame_id = (frame_id // analyze_every_n_frames) * analyze_every_n_frames
#         phone_boxes, items_boxes = frame_cache.get(analysis_frame_id, ([], {cls_name: [] for cls_name in belongings_classes}))
#
#         for _, row in group.iterrows():
#             track_id = row["track_id"]
#             x1, y1, x2, y2 = map(int, [row["x1"], row["y1"], row["x2"], row["y2"]])
#             person_box = [x1, y1, x2, y2]
#
#             phone_using = any(iou(person_box, pb) > 0.1 for pb in phone_boxes)
#             phone_results.append({
#                 "frame_id": frame_id,
#                 "track_id": track_id,
#                 "phone_using": phone_using
#             })
#
#             belongings = {item_name: any(iou(person_box, pb) > 0.1 for pb in items_boxes[item_name])
#                           for item_name in belongings_classes}
#             belongings_row = {"frame_id": frame_id, "track_id": track_id}
#             belongings_row.update(belongings)
#             belongings_results.append(belongings_row)
#
#     cap.release()
#
#     pd.DataFrame(phone_results, columns=["frame_id", "track_id", "phone_using"]).to_csv(phone_csv_path, index=False)
#     pd.DataFrame(belongings_results, columns=["frame_id", "track_id"] + belongings_classes).to_csv(belongings_csv_path, index=False)
#
#     print(f"\nPhone usage results saved to {phone_csv_path}")
#     print(f"Pedestrian belongings results saved to {belongings_csv_path}")
