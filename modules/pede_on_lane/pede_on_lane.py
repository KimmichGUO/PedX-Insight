import pandas as pd
import numpy as np
from shapely.geometry import Polygon, Point
import os

def build_lane_polygon(lane_row):
    polygon_coords = [
        (lane_row['left_x1'], lane_row['left_y1']),
        (lane_row['left_x2'], lane_row['left_y2']),
        (lane_row['right_x2'], lane_row['right_y2']),
        (lane_row['right_x1'], lane_row['right_y1'])
    ]
    return Polygon(polygon_coords)

def pedestrian_on_lane(video_path, tracking_csv_path=None, lane_csv_path=None, output_csv_path=None):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if output_csv_path is None:
        output_dir = os.path.join("analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "pedestrian_on_lane.csv")
    if tracking_csv_path is None:
        tracking_csv_path = os.path.join(output_dir, "tracked_pedestrians.csv")
    if lane_csv_path is None:
        lane_csv_path = os.path.join(output_dir, "lane_detection.csv")

    tracking_df = pd.read_csv(tracking_csv_path)
    lane_df = pd.read_csv(lane_csv_path)

    lane_polygons = {}
    for _, row in lane_df.iterrows():
        frame_id = int(row['frame'])
        poly = build_lane_polygon(row)
        lane_polygons[frame_id] = poly

    on_lane_flags = {}

    for _, row in tracking_df.iterrows():
        frame_id = int(row['frame_id'])
        track_id = int(row['track_id'])

        if frame_id not in lane_polygons:
            continue

        x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
        center_x = (x1 + x2) / 2
        center_y = y1
        radius = (x2 - x1) / 2

        person_circle = Point(center_x, center_y).buffer(radius)

        polygon = lane_polygons[frame_id]
        is_on_lane = person_circle.intersects(polygon)

        if track_id not in on_lane_flags:
            on_lane_flags[track_id] = []
        on_lane_flags[track_id].append((frame_id, is_on_lane))

    results = []

    for track_id, flags in on_lane_flags.items():
        in_lane = False
        start_frame = None
        has_entry = False

        for frame, is_on in flags:
            if is_on and not in_lane:
                in_lane = True
                start_frame = frame
            elif not is_on and in_lane:
                in_lane = False
                has_entry = True
                results.append({
                    "track_id": track_id,
                    "entered_lane": True,
                    "start_frame": start_frame,
                    "end_frame": frame - 1
                })

        if in_lane and start_frame is not None:
            has_entry = True
            results.append({
                "track_id": track_id,
                "entered_lane": True,
                "start_frame": start_frame,
                "end_frame": flags[-1][0]
            })

        if not has_entry:
            results.append({
                "track_id": track_id,
                "entered_lane": False,
                "start_frame": None,
                "end_frame": None
            })

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv_path, index=False)
    print(f"Pedestrian on lane detecion Done. Results saved to {output_csv_path}")
