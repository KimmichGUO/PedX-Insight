import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os


def analyze_acceleration(video_path, input_csv=None, output_csv=None, window_size=30):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if input_csv is None:
        input_csv = os.path.join(".", "analysis_results", video_name, "[B1]tracked_pedestrians.csv")
    df = pd.read_csv(input_csv)
    df['x_center'] = (df['x1'] + df['x2']) / 2

    results = []

    for track_id, group in df.groupby('track_id'):
        group = group.sort_values('frame_id').reset_index(drop=True)
        x_centers = group['x_center'].values
        frames = group['frame_id'].values

        total_windows = len(group) // window_size

        for w in range(total_windows):
            start = w * window_size
            end = start + window_size

            window_frames = frames[start:end].reshape(-1, 1)
            window_centers = x_centers[start:end]

            model = LinearRegression()
            model.fit(window_frames, window_centers)
            slope = model.coef_[0]

            trend = "accelerating" if slope > 0.5 else "decelerating" if slope < -0.5 else "constant"

            results.append({
                "track_id": track_id,
                "start_frame": frames[start],
                "end_frame": frames[end - 1],
                "slope": round(slope, 3),
                "trend": trend
            })

    result_df = pd.DataFrame(results, columns=["track_id", "start_frame", "end_frame", "slope", "trend"])

    if output_csv is None:
        output_csv = os.path.join(os.path.dirname(input_csv), "[C2]pedestrian_acc.csv")
    result_df.to_csv(output_csv, index=False)
    print(f"Acceleration trend analysis saved to: {output_csv}")
