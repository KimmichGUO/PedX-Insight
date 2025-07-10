import cv2
import pandas as pd
import os
from ultralytics import YOLO

def run_clothing_detection(video_path, tracking_csv_path=None, output_csv_path=None):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if tracking_csv_path is None:
        tracking_csv_path = os.path.join(".", "analysis_results", video_name, "tracked_pedestrians.csv")
    if output_csv_path is None:
        output_dir = os.path.join(".", "analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "clothing_analysis.csv")

    df = pd.read_csv(tracking_csv_path)
    cap = cv2.VideoCapture(video_path)

    model = YOLO("modules/clothing/deepfashion2_yolov8s-seg.pt")
    class_names = {
        0: 'short_sleeved_shirt', 1: 'long_sleeved_shirt', 2: 'short_sleeved_outwear',
        3: 'long_sleeved_outwear', 4: 'vest', 5: 'sling', 6: 'shorts',
        7: 'trousers', 8: 'skirt', 9: 'short_sleeved_dress',
        10: 'long_sleeved_dress', 11: 'vest_dress', 12: 'sling_dress'
    }

    print("Start analyzing clothing per frame...")

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

    results = []

    for frame_id, group in grouped:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
        ret, frame = cap.read()
        if not ret:
            print(f"Frame {frame_id} not readable.")
            continue

        yolo_results = model.predict(frame, conf=0.25, show=False)[0]
        boxes = yolo_results.boxes.xyxy.cpu().numpy()
        clss = yolo_results.boxes.cls.cpu().numpy().astype(int)

        detections = [
            {"box": boxes[i], "cls": clss[i], "name": class_names[clss[i]]}
            for i in range(len(boxes))
        ]

        for _, row in group.iterrows():
            track_id = row["track_id"]
            x1, y1, x2, y2 = map(int, [row["x1"], row["y1"], row["x2"], row["y2"]])
            person_box = [x1, y1, x2, y2]

            matched_classes = set()
            for det in detections:
                if iou(person_box, det["box"]) > 0.1:
                    matched_classes.add(det["name"])

            results.append({
                "frame_id": frame_id,
                "track_id": track_id,
                "clothing_items": ", ".join(sorted(matched_classes)) if matched_classes else "None"
            })

        print(f"Frame {frame_id} processed.")

    cap.release()
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, index=False)
    print(f"\nClothing detection results saved to {output_csv_path}")
