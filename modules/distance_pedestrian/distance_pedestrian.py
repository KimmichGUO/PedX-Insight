import os
import cv2
import time
import pandas as pd


def estimate_distance(bbox_width, focal_length=1000, known_width=0.5):
    if bbox_width == 0:
        return 0.0
    return (known_width * focal_length) / bbox_width


def visualize_and_estimate_distance(video_path, tracked_csv_path=None, output_csv_path=None):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if output_csv_path is None:
        output_dir = os.path.join("analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "[V4]distance_ve_pe.csv")
    if tracked_csv_path is None:
        tracked_csv_path = os.path.join(".", "analysis_results", video_name, "[B1]tracked_pedestrians.csv")

    tracked_df = pd.read_csv(tracked_csv_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    frame_id = -1
    target_fps = 3
    frame_time = 1.0 / target_fps
    results_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        frame_detections = tracked_df[tracked_df['frame_id'] == frame_id]

        for _, row in frame_detections.iterrows():
            tid = int(row['track_id'])
            x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
            bbox_width = x2 - x1

            distance = estimate_distance(bbox_width)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"ID: {tid}", (x1, y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"Dist: {distance:.2f}m", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            results_list.append({
                "frame_id": frame_id,
                "track_id": tid,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "distance(m)": round(distance, 2)
            })

        cv2.imshow("Distance Estimation from Tracking", frame)

        time.sleep(frame_time)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    pd.DataFrame(results_list).to_csv(output_csv_path, index=False)
    print(f"Distance estimation completed. Results saved to {output_csv_path}")
