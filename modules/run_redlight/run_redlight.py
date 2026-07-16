import os
import bisect
import pandas as pd
import numpy as np


def load_traffic_light_status(traffic_light_csv_path):
    df = pd.read_csv(traffic_light_csv_path)
    light_dict = {}
    for _, row in df.iterrows():
        try:
            frame_id = int(row["frame_id"])
        except (ValueError, TypeError):
            continue
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

    columns = [
        "track_id",
        "start_frame",
        "end_frame",
        "ran_red_light",
        "red_start_frame",
        "red_end_frame"
    ]

    # Missing-input guard: no crossing_judge input -> valid header-only output.
    if not os.path.exists(crossing_csv_path):
        pd.DataFrame(columns=columns).to_csv(output_csv_path, index=False)
        print(f"Red light violation detection results saved to {output_csv_path}")
        return

    crossing_df = pd.read_csv(crossing_csv_path)

    # Missing-input guard: no traffic-light readings -> treat the light as
    # unknown (flag nothing) instead of raising.
    if traffic_light_csv_path and os.path.exists(traffic_light_csv_path):
        traffic_light_dict = load_traffic_light_status(traffic_light_csv_path)
    else:
        traffic_light_dict = {}

    # FIX (#15): the raw [E2] samples are sparse and riddled with "None" gaps, so
    # the old code over-triggered ran_red_light on a single lucky green sample and
    # disagreed with risky_crossing. Instead, build a time-ordered, FORWARD-FILLED
    # light state: sort the frames that carry a real color and bisect to the last
    # known color at/just before any pedestrian frame. Then flag ran_red_light only
    # when green is SUSTAINED across the pedestrian's on-road window (green covers a
    # majority of the [C3] started_frame..ended_frame span, backed by >= 2 real
    # readings), never on a single isolated sample.
    sorted_light_frames = sorted(traffic_light_dict.keys())

    def light_color_at(frame):
        # Forward-fill: last known real color at or before `frame` (None if none exists yet).
        idx = bisect.bisect_right(sorted_light_frames, frame) - 1
        if idx < 0:
            return None
        return traffic_light_dict[sorted_light_frames[idx]]

    result_list = []

    for _, row in crossing_df.iterrows():
        tid = int(row["track_id"])
        crossed = bool(row["crossed"])

        if not crossed:
            continue

        start_frame = int(row["started_frame"])
        end_frame = int(row["ended_frame"])

        # Piecewise-constant boundaries of the forward-filled state across the
        # on-road window: window start, every light sample strictly inside it, then
        # window end. The forward-filled color is constant on each [a, b) segment.
        boundaries = [start_frame]
        lo = bisect.bisect_right(sorted_light_frames, start_frame)
        hi = bisect.bisect_right(sorted_light_frames, end_frame)
        boundaries.extend(sorted_light_frames[lo:hi])
        boundaries.append(end_frame)

        window_span = max(end_frame - start_frame, 1)
        green_span = 0
        green_eval_frames = []
        known_points = 0
        for i in range(len(boundaries) - 1):
            a = boundaries[i]
            b = boundaries[i + 1]
            color = light_color_at(a)
            if color is not None:
                known_points += 1
            if color == "green" and b > a:
                green_span += (b - a)
                green_eval_frames.append(a)

        # Sustained green over at least half the on-road window, backed by more than
        # one real reading -> ran red light. Kills the old single-sample false positive.
        ran_red_light = (known_points >= 2) and (green_span / window_span >= 0.5)

        if ran_red_light and green_eval_frames:
            red_start = min(green_eval_frames)
            red_end = max(green_eval_frames)
        else:
            red_start = None
            red_end = None

        result_list.append({
            "track_id": tid,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "ran_red_light": ran_red_light,
            "red_start_frame": red_start if red_start is not None else np.nan,
            "red_end_frame": red_end if red_end is not None else np.nan
        })

    result_df = pd.DataFrame(result_list, columns=columns)
    result_df.to_csv(output_csv_path, index=False)
    print(f"Red light violation detection results saved to {output_csv_path}")
