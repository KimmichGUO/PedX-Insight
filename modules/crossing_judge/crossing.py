import os
import cv2
import pandas as pd

def get_video_mid_x(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap.release()
    return width / 2

def detect_crossing(video_path, tracked_csv_path=None, output_csv_path=None):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if tracked_csv_path is None:
        tracked_csv_path = os.path.join(".", "analysis_results", video_name, "[B1]tracked_pedestrians.csv")

    if os.path.exists(tracked_csv_path):
        df_tracks = pd.read_csv(tracked_csv_path)
    else:
        df_tracks = pd.DataFrame(columns=['track_id', 'frame_id', 'x1', 'y1', 'x2', 'y2'])

    video_mid_x = get_video_mid_x(video_path)

    grouped = df_tracks.groupby('track_id')
    results = []

    for track_id, group in grouped:
        group = group.sort_values('frame_id')
        frames = group['frame_id'].values
        xs1 = group['x1'].values
        xs2 = group['x2'].values

        x_center_history = [(xs1[i] + xs2[i]) / 2 for i in range(len(frames))]

        min_x = min(x_center_history)
        max_x = max(x_center_history)
        crossed_mid = (min_x < video_mid_x) and (max_x > video_mid_x)

        start_cross_frame = None
        end_cross_frame = None

        if crossed_mid:
            for i, x_center in enumerate(x_center_history):
                if (x_center < video_mid_x and max_x > video_mid_x) or (x_center > video_mid_x and min_x < video_mid_x):
                    start_cross_frame = frames[i]
                    break
            for i in range(len(x_center_history)-1, -1, -1):
                x_center = x_center_history[i]
                if (x_center < video_mid_x and max_x > video_mid_x) or (x_center > video_mid_x and min_x < video_mid_x):
                    end_cross_frame = frames[i]
                    break

        results.append({
            'track_id': track_id,
            'crossed': crossed_mid,
            'started_frame': start_cross_frame if crossed_mid else None,
            'ended_frame': end_cross_frame if crossed_mid else None
        })

    if output_csv_path is None:
        output_csv_path = os.path.join(os.path.dirname(tracked_csv_path), "[C3]crossing_judge.csv")

    if not results:
        results_df = pd.DataFrame(columns=['track_id', 'crossed', 'started_frame', 'ended_frame'])
    else:
        results_df = pd.DataFrame(results)

    results_df.to_csv(output_csv_path, index=False)
    print(f"Crossing detection results saved to: {output_csv_path}")
