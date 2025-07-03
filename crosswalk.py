# modules/crosswalk/crosswalk.py

import cv2
from ultralytics import YOLO

def run_crosswalk_detection(source_video_path, conf=0.25):
    model = YOLO("modules/crosswalk/best_cw.pt")

    cap = cv2.VideoCapture(source_video_path)

    if not cap.isOpened():
        print("Error opening video file.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, imgsz=640, conf=conf)
        detected = False

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                if cls_id == 0:
                    detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"Crosswalk {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if detected:
            print("Detected crosswalk")
        else:
            print("No crosswalk")

        cv2.imshow("Crosswalk Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
