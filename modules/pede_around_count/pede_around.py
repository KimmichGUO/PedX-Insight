import pandas as pd
import numpy as np
import os
import math

def calculate_nearby_count(video_path, crossing_csv=None, tracked_pede_csv=None, output_csv_path=None):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if output_csv_path is None:
        output_dir = os.path.join("analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "[C10]nearby_count.csv")
    if tracked_pede_csv is None:
        tracked_pede_csv = os.path.join(output_dir, "[B1]tracked_pedestrians.csv")
    if crossing_csv is None:
        crossing_csv = os.path.join(output_dir, "[C3]crossing_judge.csv")

    if not os.path.exists(tracked_pede_csv) or os.path.getsize(tracked_pede_csv) == 0 or \
       not os.path.exists(crossing_csv) or os.path.getsize(crossing_csv) == 0:
        empty_df = pd.DataFrame(columns=['track_id', 'nearby_first_1_5', 'nearby_all'])
        empty_df.to_csv(output_csv_path, index=False)
        print(f"No input data. Empty results saved to {output_csv_path}")
        return

    crossing_df = pd.read_csv(crossing_csv)
    crossing_df = crossing_df[(crossing_df['crossed'] == True)]
    tracked_df = pd.read_csv(tracked_pede_csv)

    if crossing_df.empty or tracked_df.empty:
        empty_df = pd.DataFrame(columns=['track_id', 'nearby_first_1_5', 'nearby_all'])
        empty_df.to_csv(output_csv_path, index=False)
        print(f"No data after filtering. Empty results saved to {output_csv_path}")
        return

    results = []

    for _, row in crossing_df.iterrows():
        track_id = row['track_id']
        start_frame = int(row['started_frame'])
        end_frame = int(row['ended_frame'])
        total_frames = end_frame - start_frame + 1
        first_5th_frames = start_frame + total_frames // 5

        person_traj = tracked_df[(tracked_df['track_id'] == track_id) &
                                 (tracked_df['frame_id'] >= start_frame) &
                                 (tracked_df['frame_id'] <= end_frame)]

        nearby_counts_all = []
        nearby_counts_first_5th = []

        for _, p in person_traj.iterrows():
            frame_id = p['frame_id']
            px = (p['x1'] + p['x2']) / 2
            py = p['y2']
            radius = (p['x2'] - p['x1'])

            frame_peds = tracked_df[tracked_df['frame_id'] == frame_id]

            dx = (frame_peds['x1'] + frame_peds['x2']) / 2 - px
            dy = frame_peds['y2'] - py
            distances = np.sqrt(dx**2 + dy**2)

            nearby_count = ((distances <= radius) & (frame_peds['track_id'] != track_id)).sum()
            nearby_counts_all.append(nearby_count)

            if frame_id <= first_5th_frames:
                nearby_counts_first_5th.append(nearby_count)

        avg_first_5th = math.ceil(np.mean(nearby_counts_first_5th)) if nearby_counts_first_5th else 0
        avg_all = math.ceil(np.mean(nearby_counts_all)) if nearby_counts_all else 0

        results.append([track_id, avg_first_5th, avg_all])

    result_df = pd.DataFrame(results, columns=['track_id', 'nearby_first_1_5', 'nearby_all'])
    result_df.to_csv(output_csv_path, index=False)
    print(f"People around detection Done! Results saved to {output_csv_path}")
