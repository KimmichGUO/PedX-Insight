import os
import cv2
import pandas as pd
from shapely.geometry import Point, Polygon

def get_video_mid_x(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap.release()
    return width / 2

def point_in_sidewalk(circle_center, radius, polygons):
    point = Point(circle_center)
    for poly in polygons:
        if poly.distance(point) <= radius or poly.contains(point):
            return True
    return False

def parse_sidewalk_polygons(sidewalk_str):
    polygons = []
    for poly_str in sidewalk_str.split('|'):
        points = []
        for coord in poly_str.split(';'):
            x, y = coord.split(',')
            points.append((float(x), float(y)))
        polygons.append(Polygon(points))
    return polygons

def detect_crossing_strict(video_path, tracked_csv_path=None, sidewalk_csv_path=None, output_csv_path=None):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if tracked_csv_path is None:
        tracked_csv_path = os.path.join(".", "analysis_results", video_name, "[B1]tracked_pedestrians.csv")
    if sidewalk_csv_path is None:
        sidewalk_csv_path = os.path.join(".", "analysis_results", video_name, "[E9]sidewalk_detection_strict.csv")

    df_tracks = pd.read_csv(tracked_csv_path)
    df_polygons = pd.read_csv(sidewalk_csv_path)

    video_mid_x = get_video_mid_x(video_path)

    polygons_by_frame = {}
    for _, row in df_polygons.iterrows():
        polygons_by_frame[row['frame_id']] = parse_sidewalk_polygons(row['polygons'])

    grouped = df_tracks.groupby('track_id')

    results = []

    for track_id, group in grouped:
        group = group.sort_values('frame_id')
        frames = group['frame_id'].values
        xs1 = group['x1'].values
        ys1 = group['y1'].values
        xs2 = group['x2'].values
        ys2 = group['y2'].values

        start_x_center = (xs1[0] + xs2[0]) / 2
        on_left = start_x_center < video_mid_x

        left_below_idx = 0
        right_below_idx = 1

        in_sidewalk_status = []

        for i in range(len(frames)):
            frame_id = frames[i]
            x1, y1, x2, y2 = xs1[i], ys1[i], xs2[i], ys2[i]
            width = x2 - x1

            if (x1 + x2) / 2 < video_mid_x:
                center = (x1, y1)
            else:
                center = (x2, y1)

            radius = width

            polys = polygons_by_frame.get(frame_id, [])

            in_sidewalk = point_in_sidewalk(center, radius, polys)
            in_sidewalk_status.append(in_sidewalk)

        crossed = False
        start_cross_frame = None
        end_cross_frame = None

        left_count = 0
        left_start_idx = None
        returned = False

        for idx, status in enumerate(in_sidewalk_status):
            if status:
                if left_count >= 6 and left_start_idx is not None and not crossed:
                    crossed = True
                    start_cross_frame = frames[left_start_idx]
                    end_cross_frame = frames[idx]
                    break
                left_count = 0
                left_start_idx = None
            else:
                if left_start_idx is None:
                    left_start_idx = idx
                left_count += 1

        results.append({
            'track_id': track_id,
            'crossed': crossed,
            'started_frame': start_cross_frame if crossed else None,
            'ended_frame': end_cross_frame if crossed else None
        })

    if output_csv_path is None:
        output_csv_path = os.path.join(os.path.dirname(tracked_csv_path), "crossing_results.csv")

    pd.DataFrame(results).to_csv(output_csv_path, index=False)
    print(f"Crossing detection results saved to: {output_csv_path}")

