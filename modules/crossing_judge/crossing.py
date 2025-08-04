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

def detect_crossing(video_path, tracked_csv_path=None, sidewalk_csv_path=None, output_csv_path=None):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if tracked_csv_path is None:
        tracked_csv_path = os.path.join("analysis_results", video_name, "tracked_pedestrians.csv")
    if sidewalk_csv_path is None:
        sidewalk_csv_path = os.path.join("analysis_results", video_name, "sidewalk_polygons.csv")

    df_tracks = pd.read_csv(tracked_csv_path)
    df_polygons = pd.read_csv(sidewalk_csv_path)

    video_mid_x = get_video_mid_x(video_path)
    polygons_by_frame = {
        row['frame_id']: parse_sidewalk_polygons(row['polygons'])
        for _, row in df_polygons.iterrows()
    }

    results = []

    for track_id, group in df_tracks.groupby('track_id'):
        group = group.sort_values('frame_id')
        frames = group['frame_id'].values
        xs1, ys1, xs2, ys2 = group['x1'].values, group['y1'].values, group['x2'].values, group['y2'].values

        x_centers = []
        in_sidewalk_status = []

        for i in range(len(frames)):
            x1, y1, x2, y2 = xs1[i], ys1[i], xs2[i], ys2[i]
            width = x2 - x1
            frame_id = frames[i]

            # if (x1 + x2) / 2 < video_mid_x:
            #     center = (x1, y1)
            # else:
            #     center = (x2, y1)

            center = ((x1 + x2)/2 , y1)

            radius = width
            polygons = polygons_by_frame.get(frame_id, [])
            in_sidewalk = point_in_sidewalk(center, radius, polygons)

            in_sidewalk_status.append(in_sidewalk)
            x_centers.append((x1 + x2) / 2)

        crossed = False
        start_cross_frame = None
        end_cross_frame = None
        movement_type = None

        i = 0
        while i <= len(in_sidewalk_status) - 6:
            if all(not s for s in in_sidewalk_status[i:i + 6]):
                start_idx = i
                end_idx = i + 5
                while end_idx + 1 < len(in_sidewalk_status) and not in_sidewalk_status[end_idx + 1]:
                    end_idx += 1

                min_x = min(x_centers[start_idx:end_idx + 1])
                max_x = max(x_centers[start_idx:end_idx + 1])
                if min_x < video_mid_x and max_x > video_mid_x:
                    crossed = True
                    start_cross_frame = frames[start_idx]
                    end_cross_frame = frames[end_idx]
                    break
                i = end_idx + 1
            else:
                i += 1

        if crossed:
            start_status = in_sidewalk_status[0]
            end_status = in_sidewalk_status[-1]
            if start_status and end_status:
                movement_type = "sidewalk-road-sidewalk"
            elif start_status and not end_status:
                movement_type = "sidewalk-to-road"
            elif not start_status and end_status:
                movement_type = "road-to-sidewalk"
            else:
                movement_type = "road-to-road"

        results.append({
            'track_id': track_id,
            'crossed': crossed,
            'started_frame': start_cross_frame if crossed else None,
            'ended_frame': end_cross_frame if crossed else None,
            'movement_type': movement_type if crossed else None
        })

    if output_csv_path is None:
        output_csv_path = os.path.join(os.path.dirname(tracked_csv_path), "crossing_results.csv")
    pd.DataFrame(results).to_csv(output_csv_path, index=False)
    print(f"Crossing detection results saved to: {output_csv_path}")
