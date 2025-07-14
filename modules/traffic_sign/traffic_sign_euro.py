import cv2
import torch
import csv
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def run_traffic_sign_euro(video_path, output_csv=None):

    model = torch.hub.load('ultralytics/yolov5', 'custom', path='modules/traffic_sign/best_93.pt', force_reload=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if output_csv is None:
        output_dir = os.path.join(".", "analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv = os.path.join(output_dir, "traffic_sign_detections_euro.csv")

    with open(output_csv, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["frame_id", "detections"])

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        results = model(frame)
        labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

        detections = []
        for label, cord in zip(labels, cords):
            x1 = int(cord[0] * frame.shape[1])
            y1 = int(cord[1] * frame.shape[0])
            x2 = int(cord[2] * frame.shape[1])
            y2 = int(cord[3] * frame.shape[0])
            conf = float(cord[4])
            cls_name = model.names[int(label)]

            detections.append(f"{cls_name}:{conf:.2f}({x1},{y1},{x2},{y2})")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        detection_str = "; ".join(detections)

        with open(output_csv, mode="a", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([frame_id, detection_str])

        cv2.imshow("Traffic Sign Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Detection and CSV saving completed, saved to {output_csv}")
