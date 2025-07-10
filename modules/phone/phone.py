import cv2
import pandas as pd
import os
from ultralytics import YOLO

def run_phone_detection(video_path, weights = "yolov8n.pt", tracking_csv_path=None, output_csv_path=None):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if tracking_csv_path is None:
        tracking_csv_path = os.path.join(".", "analysis_results", video_name, "tracked_pedestrians.csv")
    if output_csv_path is None:
        output_dir = os.path.join(".", "analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "phone_usage_analysis.csv")

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
