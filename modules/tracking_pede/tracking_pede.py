import cv2
import pandas as pd
import numpy as np
import os
from collections import defaultdict, deque

def run_pede_direction_analysis(video_path, tracking_csv_path=None, output_csv_path=None):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if tracking_csv_path is None:
        tracking_csv_path = os.path.join(".", "analysis_results", video_name, "[B1]tracked_pedestrians.csv")
    if output_csv_path is None:
        output_dir = os.path.join(".", "analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "[P4]pedestrian_tracking.csv")

    df = pd.read_csv(tracking_csv_path)
    df.sort_values(by=["frame_id", "track_id"], inplace=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    position_history = defaultdict(lambda: deque(maxlen=10))

    with open(output_csv_path, "w") as f:
        f.write("frame_id,track_id,direction\n")

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        current_frame = df[df["frame_id"] == frame_id]
        for _, row in current_frame.iterrows():
            track_id = int(row["track_id"])
            x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            position_history[track_id].append((cx, cy))

            direction = "unknown"
            if len(position_history[track_id]) >= 2:
                old_cx, old_cy = position_history[track_id][0]
                delta_x = cx - old_cx
                delta_y = cy - old_cy

                threshold = 5
                if abs(delta_x) < threshold and abs(delta_y) < threshold:
                    direction = "static"
                elif abs(delta_y) >= abs(delta_x):
                    direction = "backward" if delta_y > 0 else "forward"
                else:
                    direction = "right" if delta_x > 0 else "left"

                with open(output_csv_path, "a") as f:
                    f.write(f"{frame_id},{track_id},{direction}\n")

    cap.release()
    print(f"Direction analysis completed. Results saved to {output_csv_path}")
# import argparse
# import cv2
# import numpy as np
# import supervision as sv
# from ultralytics import YOLO
# import os
# import csv
# from collections import defaultdict, deque

# def tracking_pede(
#         source_video_path: str,
#         weights: str = "yolov8m-pose.pt",
# ) -> None:
#     model = YOLO(weights)
#     edge_annotator = sv.EdgeAnnotator()
#     vertex_annotator = sv.VertexAnnotator()
#     box_annotator = sv.BoxAnnotator()
#     tracker = sv.ByteTrack()
#     smoother = sv.DetectionsSmoother()
#     trace_annotator = sv.TraceAnnotator()

#     cap = cv2.VideoCapture(source_video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     cap.release()

#     video_basename = os.path.splitext(os.path.basename(source_video_path))[0]
#     csv_filename = f"{video_basename}_pedestrian_tracking.csv"

#     with open(csv_filename, mode="w", newline="") as csvfile:
#         csv_writer = csv.writer(csvfile)
#         csv_writer.writerow(["frame_index", "id_direction_map"])

#     id_centers = defaultdict(list)
#     id_distance = defaultdict(float)
#     # id_bbox_widths = defaultdict(lambda: deque(maxlen=30))  

#     def callback(frame: np.ndarray, frame_index: int) -> np.ndarray:
#         results = model(frame)[0]
#         key_points = sv.KeyPoints.from_ultralytics(results)
#         detections = key_points.as_detections()
#         detections = tracker.update_with_detections(detections)
#         detections = smoother.update_with_detections(detections)

#         annotated_frame = edge_annotator.annotate(frame.copy(), key_points=key_points)
#         annotated_frame = vertex_annotator.annotate(annotated_frame, key_points=key_points)
#         annotated_frame = box_annotator.annotate(annotated_frame, detections=detections)
#         annotated_frame = trace_annotator.annotate(annotated_frame, detections=detections)

#         id_speed_map = {}

#         # for idx, tracker_id in enumerate(detections.tracker_id):
#         #     if tracker_id is None:
#         #         continue
#         #
#         #     x1, y1, x2, y2 = detections.xyxy[idx]
#         #     cx = int((x1 + x2) / 2)
#         #     cy = int((y1 + y2) / 2)
#         #
#         #     if len(id_centers[tracker_id]) > 0:
#         #         prev_cx, prev_cy = id_centers[tracker_id][-1]
#         #         dist = np.linalg.norm([cx - prev_cx, cy - prev_cy])
#         #         id_distance[tracker_id] += dist
#         #     id_centers[tracker_id].append((cx, cy))

#             # bbox_width = x2 - x1
#             # id_bbox_widths[tracker_id].append(bbox_width)
#             #
#             # duration = len(id_centers[tracker_id]) / fps
#             # speed_px_per_sec = id_distance[tracker_id] / duration if duration > 0 else 0
#             #
#             # if len(id_bbox_widths[tracker_id]) == 30:
#             #     avg_bbox_width = np.mean(id_bbox_widths[tracker_id])
#             # else:
#             #     avg_bbox_width = bbox_width
#             #
#             # pixel_to_meter = 0.8 / avg_bbox_width if avg_bbox_width > 0 else 0.02
#             # speed_mps = speed_px_per_sec * pixel_to_meter
#             # speed_kph = speed_mps * 3.6
#             #
#             # if speed_kph > 10:
#             #     speed_kph = 10
#             # label = f"ID: {tracker_id} | {speed_kph:.1f} km/h"
#         #     cv2.putText(annotated_frame, label, (cx + 10, cy),
#         #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#         #
#         #     id_speed_map[tracker_id] = round(speed_kph, 1)
#         #
#         # if frame_index > 0 and frame_index % 30 == 0:
#         #     with open(csv_filename, mode="a", newline="") as csvfile:
#         #         csv_writer = csv.writer(csvfile)
#         #         for id_, speed in id_speed_map.items():
#         #             csv_writer.writerow([frame_index, id_, speed])
#         direction_threshold = 1  # 设置方向判断阈值
#         id_direction_map = {}  # 存储本帧每个ID的方向

#         for idx, tracker_id in enumerate(detections.tracker_id):
#             if tracker_id is None:
#                 continue

#             x1, y1, x2, y2 = detections.xyxy[idx]
#             cx = int((x1 + x2) / 2)
#             cy = int((y1 + y2) / 2)

#             id_centers[tracker_id].append((cx, cy))

#             direction = "static"
#             if len(id_centers[tracker_id]) >= 10:
#                 old_cx = id_centers[tracker_id][-10][0]
#                 delta_x = cx - old_cx
#                 if abs(delta_x) >= direction_threshold:
#                     direction = "right" if delta_x > 0 else "left"

#             id_direction_map[tracker_id] = direction

#             # 可视化标签
#             label = f"ID: {tracker_id} | {direction}"
#             cv2.putText(annotated_frame, label, (cx + 10, cy),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


#         if frame_index > 0 and frame_index % 30 == 0:
#             with open(csv_filename, mode="a", newline="") as csvfile:
#                 csv_writer = csv.writer(csvfile)
#                 row = {id_: dir_ for id_, dir_ in id_direction_map.items()}
#                 csv_writer.writerow([frame_index, row])

#         cv2.imshow("Processed Video", annotated_frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             os._exit(0)

#         return annotated_frame

#     sv.process_video(
#         source_path=source_video_path,
#         target_path=None,
#         callback=callback
#     )

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Pose tracking on a video using YOLO and Supervision.")
#     parser.add_argument(
#         "--source_video_path", type=str, required=True, help="Path to input video."
#     )
#     parser.add_argument(
#         "--weights", type=str, default="yolov8m-pose.pt", help="YOLO pose weights file."
#     )
#     args = parser.parse_args()

#     tracking_pede(
#         source_video_path=args.source_video_path,
#         weights=args.weights,
#     )
