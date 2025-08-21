import os
import pandas as pd
import ast
from shapely.geometry import box, Point

def load_crosswalk_boxes(crosswalk_csv_path):
    df = pd.read_csv(crosswalk_csv_path)
    crosswalk_dict = {}
    for _, row in df.iterrows():
        frame_id = int(row["frame_id"])
        detected = row["crosswalk_detected"] == "Yes"
        if detected:
            try:
                boxes = ast.literal_eval(row["crosswalk_boxes"])
                crosswalk_dict[frame_id] = [tuple(b) for b in boxes]
            except:
                continue
    return crosswalk_dict

def circle_overlaps_crosswalk(circle_center, radius, crosswalk_boxes):
    circle = Point(circle_center).buffer(radius)
    for bx1, by1, bx2, by2 in crosswalk_boxes:
        rect = box(bx1, by1, bx2, by2)
        if rect.distance(circle) <= 0 or rect.contains(circle):
            return True
    return False

def determine_crosswalk_usage(video_path, crossing_csv_path=None, track_csv_path=None, crosswalk_csv_path=None, output_csv_path=None):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if crossing_csv_path is None:
        crossing_csv_path = os.path.join("analysis_results", video_name, "[C3]crossing_judge.csv")
    if track_csv_path is None:
        track_csv_path = os.path.join("analysis_results", video_name, "[B1]tracked_pedestrians.csv")
    if crosswalk_csv_path is None:
        crosswalk_csv_path = os.path.join("analysis_results", video_name, "[E7]crosswalk_detection.csv")
    if output_csv_path is None:
        output_csv_path = os.path.join("analysis_results", video_name, "[C4]crosswalk_usage.csv")

    crossing_df = pd.read_csv(crossing_csv_path)
    tracked_df = pd.read_csv(track_csv_path)
    crosswalk_dict = load_crosswalk_boxes(crosswalk_csv_path)

    result_list = []

    for _, row in crossing_df.iterrows():
        tid = int(row["track_id"])
        if not row["crossed"]:
            continue

        person_track = tracked_df[tracked_df["track_id"] == tid]
        used_crosswalk = False

        for _, trow in person_track.iterrows():
            frame_id = int(trow["frame_id"])
            x1, y1, x2, y2 = trow["x1"], trow["y1"], trow["x2"], trow["y2"]
            cx = (x1 + x2) / 2
            cy = y1
            radius = (x2 - x1) * 2
            circle_center = (cx, cy)

            for offset in range(-25, 26):
                check_frame = frame_id + offset
                if check_frame in crosswalk_dict:
                    if circle_overlaps_crosswalk(circle_center, radius, crosswalk_dict[check_frame]):
                        used_crosswalk = True
                        break
            if used_crosswalk:
                break

        result_list.append({
            "track_id": tid,
            "used_crosswalk": used_crosswalk
        })

    if not result_list:
        results_df = pd.DataFrame(columns=["track_id", "used_crosswalk"])
    else:
        results_df = pd.DataFrame(result_list)

    results_df.to_csv(output_csv_path, index=False)
    print(f"Crosswalk Usage detection results saved to {output_csv_path}")