import os
import cv2
import pandas as pd
from shapely.geometry import Point, Polygon

OUTPUT_COLUMNS = ['track_id', 'crossed', 'started_frame', 'ended_frame', 'movement_type']


def get_video_width(video_path):
    """Frame width in pixels, or None if the video is unavailable (e.g. deleted after
    analysis). Callers fall back to inferring width from the tracked boxes."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap.release()
    return width if width and width > 0 else None


def point_in_sidewalk(circle_center, radius, polygons):
    point = Point(circle_center)
    for poly in polygons:
        if poly.distance(point) <= radius or poly.contains(point):
            return True
    return False


def parse_sidewalk_polygons(sidewalk_str):
    polygons = []
    # Frames with no detected sidewalk are written as an empty cell, which pandas reads back
    # as NaN (a float); calling .split on it would raise AttributeError.
    if not isinstance(sidewalk_str, str) or not sidewalk_str:
        return polygons
    for poly_str in sidewalk_str.split('|'):
        points = []
        for coord in poly_str.split(';'):
            x, y = coord.split(',')
            points.append((float(x), float(y)))
        if len(points) >= 3:
            polygons.append(Polygon(points))
    return polygons


def detect_crossing(video_path, tracked_csv_path=None, sidewalk_csv_path=None, output_csv_path=None):
    """Region-based crossing judge (replaces the raw frame-midline test).

    A track counts as crossed when its foot-point sweeps >= 2 of 5 vertical image
    regions AND it leaves the detected sidewalk at some point. This rejects
    pedestrians walking parallel to the road on the sidewalk (the midline test's
    main false positive) and is less sensitive to exactly where the crossing sits
    in frame. Output schema matches the old [C3]crossing_judge.csv (plus an extra
    movement_type column that downstream consumers ignore).
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(".", "analysis_results", video_name)

    if tracked_csv_path is None:
        tracked_csv_path = os.path.join(output_dir, "[B1]tracked_pedestrians.csv")
    if sidewalk_csv_path is None:
        sidewalk_csv_path = os.path.join(output_dir, "[E9]sidewalk_detection.csv")
    if output_csv_path is None:
        output_csv_path = os.path.join(os.path.dirname(tracked_csv_path), "[C3]crossing_judge.csv")

    out_dir = os.path.dirname(output_csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    def _write(results):
        df = pd.DataFrame(results, columns=OUTPUT_COLUMNS) if results else pd.DataFrame(columns=OUTPUT_COLUMNS)
        df.to_csv(output_csv_path, index=False)
        print(f"Crossing detection results saved to: {output_csv_path}")

    if not os.path.exists(tracked_csv_path) or os.path.getsize(tracked_csv_path) == 0:
        return _write([])
    df_tracks = pd.read_csv(tracked_csv_path)
    if df_tracks.empty:
        return _write([])

    # Sidewalk polygons are optional; without them, `crossed` reduces to the region sweep.
    if os.path.exists(sidewalk_csv_path) and os.path.getsize(sidewalk_csv_path) > 0:
        df_polygons = pd.read_csv(sidewalk_csv_path)
    else:
        df_polygons = pd.DataFrame(columns=['frame_id', 'polygons'])

    # Frame width: video -> [B0] sidecar (survives video deletion) -> tracked boxes.
    # The bare x2.max() fallback understates the true width and shrinks the regions,
    # which could flip crossed=True on archived reruns.
    video_width = get_video_width(video_path)
    if video_width is None:
        meta_path = os.path.join(output_dir, "[B0]video_meta.csv")
        if os.path.exists(meta_path) and os.path.getsize(meta_path) > 0:
            try:
                meta = pd.read_csv(meta_path)
                if not meta.empty and float(meta["width"].iloc[0]) > 0:
                    video_width = float(meta["width"].iloc[0])
            except Exception:
                pass
    if video_width is None:
        video_width = float(df_tracks['x2'].max()) if not df_tracks.empty else 1.0
    region_width = max(video_width / 5.0, 1.0)

    polygons_by_frame = {}
    if 'polygons' in df_polygons.columns:
        for _, row in df_polygons.iterrows():
            polygons_by_frame[row['frame_id']] = parse_sidewalk_polygons(row['polygons'])

    results = []
    for track_id, group in df_tracks.groupby('track_id'):
        group = group.sort_values('frame_id')
        frames = group['frame_id'].values
        xs1 = group['x1'].values
        xs2 = group['x2'].values
        ys2 = group['y2'].values

        in_sidewalk_status = []
        x_center_history = []
        for i in range(len(frames)):
            x1, x2, y2 = xs1[i], xs2[i], ys2[i]
            width = x2 - x1
            center = ((x1 + x2) / 2, y2)          # foot-point (ground contact), not the head
            radius = width
            polys = polygons_by_frame.get(frames[i], [])
            in_sidewalk_status.append(point_in_sidewalk(center, radius, polys))
            x_center_history.append((x1 + x2) / 2)

        min_region = int(min(x_center_history) // region_width)
        max_region = int(max(x_center_history) // region_width)
        crossed_region = (max_region - min_region) >= 2
        ever_left_sidewalk = not all(in_sidewalk_status)
        crossed = crossed_region and ever_left_sidewalk

        start_cross_frame = end_cross_frame = movement_type = None
        if crossed and polygons_by_frame:
            for i in range(len(in_sidewalk_status)):
                if not in_sidewalk_status[i]:
                    start_cross_frame = frames[i]
                    break
            for i in range(len(in_sidewalk_status)):
                if start_cross_frame is not None and frames[i] > start_cross_frame and in_sidewalk_status[i]:
                    end_cross_frame = frames[i]
                    break
            if end_cross_frame is None:
                end_cross_frame = frames[-1]

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
        elif crossed:
            # No sidewalk data ([E9] absent): in_sidewalk_status is all-False, which would
            # degrade started/ended to the whole track span (inflating every downstream
            # window). Approximate the crossing window from the region sweep instead: from
            # the last sample still in the starting region to the first sample reaching the
            # far extreme region.
            regions = [int(x // region_width) for x in x_center_history]
            r0 = regions[0]
            min_r, max_r = min(regions), max(regions)
            target = max_r if abs(max_r - r0) >= abs(min_r - r0) else min_r
            start_i = 0
            for i in range(len(regions)):
                if regions[i] != r0:
                    start_i = max(0, i - 1)
                    break
            end_i = len(regions) - 1
            for i in range(start_i, len(regions)):
                if regions[i] == target:
                    end_i = i
                    break
            start_cross_frame = frames[start_i]
            end_cross_frame = frames[end_i]
            movement_type = "unknown (no sidewalk data)"

        results.append({
            'track_id': track_id,
            'crossed': crossed,
            'started_frame': start_cross_frame if crossed else None,
            'ended_frame': end_cross_frame if crossed else None,
            'movement_type': movement_type if crossed else None,
        })

    _write(results)
