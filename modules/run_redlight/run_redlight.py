import os
import pandas as pd
import numpy as np


def load_traffic_light_status(traffic_light_csv_path):
    df = pd.read_csv(traffic_light_csv_path)
    light_dict = {}
    for _, row in df.iterrows():
        frame_id = int(row["frame_id"])
        color = str(row["main_light_color"]).strip().lower()
        if color in ["red", "green", "yellow"]:
            light_dict[frame_id] = color
    return light_dict


def determine_red_light_violation(
        video_path,
        crossing_csv_path=None,
        traffic_light_csv_path=None,
        output_csv_path=None
):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if crossing_csv_path is None:
        crossing_csv_path = os.path.join("analysis_results", video_name, "[C3]crossing_judge.csv")
    if traffic_light_csv_path is None:
        traffic_light_csv_path = os.path.join("analysis_results", video_name, "[E2]traffic_light.csv")
    if output_csv_path is None:
        output_csv_path = os.path.join("analysis_results", video_name, "[C5]red_light_runner.csv")

    crossing_df = pd.read_csv(crossing_csv_path)
    traffic_light_dict = load_traffic_light_status(traffic_light_csv_path)

    result_list = []

    for _, row in crossing_df.iterrows():
        tid = int(row["track_id"])
        crossed = bool(row["crossed"])

        if not crossed:
            continue

        start_frame = int(row["started_frame"])
        end_frame = int(row["ended_frame"])

        ran_red_light = False
        red_start = None
        red_end = None

        green_frames = [
            frame_id
            for frame_id in range(start_frame, end_frame + 1)
            if traffic_light_dict.get(frame_id) == "green"
        ]

        if green_frames:
            ran_red_light = True
            red_start = min(green_frames)
            red_end = max(green_frames)

        result_list.append({
            "track_id": tid,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "ran_red_light": ran_red_light,
            "red_start_frame": red_start if red_start is not None else np.nan,
            "red_end_frame": red_end if red_end is not None else np.nan
        })

    columns = [
        "track_id",
        "start_frame",
        "end_frame",
        "ran_red_light",
        "red_start_frame",
        "red_end_frame"
    ]

    result_df = pd.DataFrame(result_list, columns=columns)
    result_df.to_csv(output_csv_path, index=False)
    print(f"Red light violation detection results saved to {output_csv_path}")
