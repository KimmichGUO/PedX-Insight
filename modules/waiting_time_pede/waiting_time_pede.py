import cv2
import numpy as np
import pandas as pd
import os

def run_waiting_time_analysis(video_path, csv_path=None, output_csv=None,
                             move_thresh=2.0, frame_thresh=30, min_good_points=5):

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if csv_path is None:
        csv_path = os.path.join("analysis_results", video_name, "[B1]tracked_pedestrians.csv")
    if output_csv is None:
        output_dir = os.path.join("analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv = os.path.join(output_dir, "[P3]waiting_time.csv")
    df = pd.read_csv(csv_path)
    cap = cv2.VideoCapture(video_path)
    frame_cache = {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    for i in range(1, total_frames + 1):
        ret, frame = cap.read()
        if not ret:
            break
        frame_cache[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cap.release()

    results = []

    for track_id, group in df.groupby("track_id"):
        group = group.sort_values("frame_id").reset_index(drop=True)
        waiting_counter = 0
        total_waiting_frames = 0

        for i in range(len(group) - 1):
            f0, f1 = group.loc[i, "frame_id"], group.loc[i + 1, "frame_id"]
            if f0 not in frame_cache or f1 not in frame_cache:
                continue

            x1, y1, x2, y2 = group.loc[i, ["x1", "y1", "x2", "y2"]].astype(int)
            roi0 = frame_cache[f0][y1:y2, x1:x2]
            roi1 = frame_cache[f1][y1:y2, x1:x2]

            p0 = cv2.goodFeaturesToTrack(roi0, maxCorners=20, qualityLevel=0.01, minDistance=3)
            if p0 is None or len(p0) < min_good_points:
                continue
            p0 += np.array([[x1, y1]])

            p1, st, _ = cv2.calcOpticalFlowPyrLK(frame_cache[f0], frame_cache[f1], p0.astype(np.float32), None)
            if p1 is None:
                continue

            good = st.flatten() == 1
            if good.sum() < min_good_points:
                continue

            dists = np.linalg.norm(p1[good] - p0[good], axis=1)
            median_dist = np.median(dists)

            if median_dist < move_thresh:
                waiting_counter += 1
            else:
                if waiting_counter >= frame_thresh:
                    total_waiting_frames += waiting_counter
                waiting_counter = 0  # reset

        if waiting_counter >= frame_thresh:
            total_waiting_frames += waiting_counter

        waiting_time = round(total_waiting_frames / fps, 2)


        if waiting_time >= 1.0:
            results.append({"track_id": track_id, "waiting_time": waiting_time})

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"\nWaiting time detection results saved to {output_csv}")

# import os
# import cv2
# import pandas as pd
# from collections import defaultdict
# import math
#
# def run_waiting_time_analysis(video_path, tracking_csv_path=None, output_csv_path=None, distance_threshold=15):
#     video_name = os.path.splitext(os.path.basename(video_path))[0]
#
#     if tracking_csv_path is None:
#         tracking_csv_path = os.path.join("analysis_results", video_name, "tracked_pedestrians.csv")
#     if output_csv_path is None:
#         output_dir = os.path.join("analysis_results", video_name)
#         os.makedirs(output_dir, exist_ok=True)
#         output_csv_path = os.path.join(output_dir, "waiting_time_pedestrian.csv")
#
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     cap.release()
#
#     df = pd.read_csv(tracking_csv_path)
#
#     track_points = defaultdict(list)
#     for _, row in df.iterrows():
#         frame_id = int(row["frame_id"])
#         track_id = int(row["track_id"])
#         x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
#         cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
#         track_points[track_id].append((frame_id, cx, cy))
#
#     results = []
#
#     for track_id, points in track_points.items():
#         points.sort()
#         segment_frames = []
#         base_cx, base_cy = None, None
#
#         for frame_id, cx, cy in points:
#             if not segment_frames:
#                 base_cx, base_cy = cx, cy
#                 segment_frames = [frame_id]
#                 continue
#
#             dist = math.hypot(cx - base_cx, cy - base_cy)
#
#             if dist <= distance_threshold:
#                 segment_frames.append(frame_id)
#             else:
#                 if len(segment_frames) > 1:
#                     duration_sec = len(segment_frames) / fps
#                     if duration_sec >= 1.0:
#                         results.append({
#                             "track_id": track_id,
#                             "start_frame": segment_frames[0],
#                             "end_frame": segment_frames[-1],
#                             "frame_count": len(segment_frames),
#                             "waiting_time_sec": round(duration_sec, 2)
#                         })
#                 base_cx, base_cy = cx, cy
#                 segment_frames = [frame_id]
#
#         if len(segment_frames) > 1:
#             duration_sec = len(segment_frames) / fps
#             if duration_sec >= 1.0:
#                 results.append({
#                     "track_id": track_id,
#                     "start_frame": segment_frames[0],
#                     "end_frame": segment_frames[-1],
#                     "frame_count": len(segment_frames),
#                     "waiting_time_sec": round(duration_sec, 2)
#                 })
#
#     results_df = pd.DataFrame(results)
#     results_df.to_csv(output_csv_path, index=False)
#     print(f"\nwaiting detection results saved to {output_csv_path}")
#
