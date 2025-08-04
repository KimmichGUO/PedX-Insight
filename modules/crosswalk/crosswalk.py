import os
import cv2
import pandas as pd
from ultralytics import YOLO

def run_crosswalk_detection(video_path, output_csv_path=None, conf=0.25, show=True):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if output_csv_path is None:
        output_dir = os.path.join("analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "crosswalk_detection.csv")

    model = YOLO("modules/crosswalk/best_cw.pt")
    cap = cv2.VideoCapture(video_path)

    results_list = []
    frame_id = -1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        result = model(frame, imgsz=640, conf=conf, verbose=False)[0]

        detected = False
        crosswalk_boxes = []

        if result.boxes is not None and result.boxes.data.size(0) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id == 0:
                    detected = True
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    coords = [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
                    crosswalk_boxes.append(coords)

                    if show:
                        cv2.rectangle(
                            frame,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (0, 0, 255), 2
                        )
                        cv2.putText(frame, "Crosswalk", (int(x1), int(y1) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        if show:
            cv2.imshow("Crosswalk Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Detection stopped by user.")
                break

        results_list.append({
            "frame_id": frame_id,
            "crosswalk_detected": "Yes" if detected else "No",
            "crosswalk_boxes": crosswalk_boxes if detected else []
        })

    cap.release()
    cv2.destroyAllWindows()
    pd.DataFrame(results_list).to_csv(output_csv_path, index=False)
    print(f"Crosswalk detection completed. Results saved to {output_csv_path}")
