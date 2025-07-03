import cv2
from ultralytics import YOLO
import numpy as np
import os
from pathlib import Path
import time

def run_traffic_total_detection(source_video_path, weights="modules/traffic_total/best_model.pt", target_video_path=None):
    # Load YOLO model
    model = YOLO(weights)

    if target_video_path is None:
        video_name = Path(source_video_path).stem
        output_folder = "./output"
        os.makedirs(output_folder, exist_ok=True)
        target_video_path = os.path.join(output_folder, f"{video_name}_traffic.mp4")


    cap = cv2.VideoCapture(source_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {source_video_path}!")
        return


    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(
        target_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )

    print(f"\nProcessing: {source_video_path}")
    print(f"Total frames: {total_frames}")

    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        results = model(frame)
        frame = draw_boxes(frame, results, model)

        out.write(frame)
        cv2.imshow("Traffic Light Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            print(f"Progress: {frame_count}/{total_frames} - FPS: {frame_count / elapsed:.2f}", end='\r')

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nFinished processing {source_video_path}")
    print(f"Output saved to: {target_video_path}")

def draw_boxes(frame, results, model):
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            if conf > 0.25:
                x1, y1, x2, y2 = map(int, box)
                label = f"{model.names[cls_id][:10]}-{conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame
