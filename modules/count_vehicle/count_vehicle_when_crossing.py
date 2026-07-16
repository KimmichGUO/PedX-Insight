import os
import cv2
import pandas as pd
from collections import defaultdict, Counter

# Output-schema vehicle types (kept identical so [C6]crossing_ve_count.csv columns never change).
id2name = {
    0: 'ambulance',
    1: 'army vehicle',
    2: 'auto rickshaw',
    3: 'bicycle',
    4: 'bus',
    5: 'car',
    6: 'garbagevan',
    7: 'human hauler',
    8: 'minibus',
    9: 'minivan',
    10: 'motorbike',
    11: 'pickup',
    12: 'policecar',
    13: 'rickshaw',
    14: 'scooter',
    15: 'suv',
    16: 'taxi',
    17: 'three wheelers -CNG-',
    18: 'truck',
    19: 'van',
    20: 'wheelbarrow'
}

# COCO class id -> output vehicle type name (all 5 map onto names already present in
# id2name, so the CSV schema is unchanged). Used for GLOBAL (worldwide) counting.
COCO_TO_TYPE = {
    1: 'bicycle',
    2: 'car',
    3: 'motorbike',   # COCO 'motorcycle'
    5: 'bus',
    7: 'truck',
}


def _load_sidecar_events(output_dir):
    """Line-crossing events from count_vehicle's [V10] sidecar, if that pass already ran.
    Returns (events, model_note) or (None, None) when unavailable."""
    events_path = os.path.join(output_dir, "[V10]line_crossing_events.csv")
    if not (os.path.exists(events_path) and os.path.getsize(events_path) > 0):
        return None, None
    try:
        ev = pd.read_csv(events_path)
    except Exception:
        return None, None
    if not {"frame_id", "veh_type"}.issubset(ev.columns):
        return None, None
    events = [{"frame": int(r["frame_id"]), "type": str(r["veh_type"])} for _, r in ev.iterrows()]
    model_note = "from [V10] sidecar (count_vehicle pass)"
    v6_path = os.path.join(output_dir, "[V6]vehicle_count.csv")
    if os.path.exists(v6_path):
        try:
            v6 = pd.read_csv(v6_path)
            if "Model" in v6.columns and not v6.empty:
                model_note = str(v6["Model"].iloc[0])
        except Exception:
            pass
    return events, model_note


def analyze_vehicle_during_crossing(
        video_path,
        crossing_csv_path=None,
        output_csv_path=None,
        analyze_interval_sec=1.0,
        use_coco=True,
        coco_model_path="yolo11n.pt",
        counting_line_ratio=0.5,
        conf_thresh=0.3,
):
    """Vehicles counted during each pedestrian crossing window.

    Perf fix: this module used to duplicate count_vehicle's ENTIRE dense full-fps GPU
    tracking pass. It now (a) exits before loading any model when no pedestrian crossed,
    and (b) consumes the [V10]line_crossing_events.csv sidecar that count_vehicle already
    writes, re-running its own tracking pass only when the sidecar is missing.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join("analysis_results", video_name)
    os.makedirs(output_dir, exist_ok=True)
    if output_csv_path is None:
        output_csv_path = os.path.join(output_dir, "[C6]crossing_ve_count.csv")
    else:
        os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    if crossing_csv_path is None:
        crossing_csv_path = os.path.join(output_dir, "[C3]crossing_judge.csv")

    columns = ['track_id', 'crossed', 'total_vehicle_count'] + list(id2name.values()) + ['model']

    def _write(rows):
        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(output_csv_path, index=False)
        print(f"Vehicle count when crossing completed. Results saved to {output_csv_path}")
        return df

    # Missing/unreadable crossing CSV guard -> empty output with valid header.
    if not (os.path.exists(crossing_csv_path) and os.path.getsize(crossing_csv_path) > 0):
        print(f"Crossing CSV not found: {crossing_csv_path}")
        return _write([])

    crossing_df = pd.read_csv(crossing_csv_path)
    crossing_df = crossing_df[crossing_df['crossed'] == True] if 'crossed' in crossing_df.columns else crossing_df.iloc[0:0]

    # Early-exit BEFORE any model/GPU work: no crossings -> nothing to attribute.
    if crossing_df.empty:
        print("No crossed pedestrians; skipping vehicle attribution pass.")
        return _write([])

    # Preferred path: reuse the [V10] events count_vehicle already produced.
    vehicle_crossings, model_note = _load_sidecar_events(output_dir)

    if vehicle_crossings is None:
        # Fallback: run our own dense pass (sidecar unavailable, e.g. standalone mode).
        from ultralytics import YOLO
        import torch
        if use_coco:
            model = YOLO(coco_model_path)
            class_map = COCO_TO_TYPE
            model_note = "yolo11-COCO (car/bus/truck/motorcycle/bicycle)"
        else:
            model = YOLO("modules/count_vehicle/best.pt")
            class_map = id2name
            model_note = "best.pt (region-specific)"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        use_half = device == "cuda"

        vehicle_crossings = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error opening video")
        else:
            allowed_classes = list(class_map.keys())
            # Dual counting lines (a lone horizontal line misses cross traffic).
            line_y = line_x = None
            frame_idx = 0
            track_last_side_y = {}
            track_last_side_x = {}
            track_frames = defaultdict(int)
            track_class_votes = defaultdict(Counter)
            counted_ids = set()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if line_y is None:
                    line_y = frame.shape[0] * counting_line_ratio
                    line_x = frame.shape[1] * counting_line_ratio

                track_results = model.track(
                    frame, persist=True, tracker="bytetrack.yaml",
                    conf=conf_thresh, classes=allowed_classes,
                    half=use_half, verbose=False,
                )
                boxes = track_results[0].boxes
                if boxes is None or len(boxes) == 0 or boxes.id is None:
                    continue

                for box, track_id, cls_id, conf in zip(boxes.xyxy, boxes.id, boxes.cls, boxes.conf):
                    track_id = int(track_id.cpu().numpy())
                    cls_id = int(cls_id.cpu().numpy())
                    conf = float(conf.cpu().numpy())
                    if cls_id not in class_map or conf < conf_thresh:
                        continue
                    vtype = class_map[cls_id]
                    cx = float((box[0] + box[2]) / 2.0)
                    cy = float((box[1] + box[3]) / 2.0)
                    track_frames[track_id] += 1
                    track_class_votes[track_id][vtype] += 1
                    side_y = 1 if cy >= line_y else -1
                    side_x = 1 if cx >= line_x else -1
                    crossed_y = track_last_side_y.get(track_id) not in (None, side_y)
                    crossed_x = track_last_side_x.get(track_id) not in (None, side_x)
                    if ((crossed_y or crossed_x)
                            and track_frames[track_id] >= 2 and track_id not in counted_ids):
                        majority_type = track_class_votes[track_id].most_common(1)[0][0]
                        vehicle_crossings.append({'frame': frame_idx, 'type': majority_type})
                        counted_ids.add(track_id)
                    track_last_side_y[track_id] = side_y
                    track_last_side_x[track_id] = side_x
            cap.release()
    else:
        print(f"Reusing {len(vehicle_crossings)} line-crossing events from [V10] sidecar "
              f"(skipped the duplicate dense tracking pass).")

    # For each pedestrian crossing event, count vehicle line-crossings within its window.
    output_data = []
    for _, row in crossing_df.iterrows():
        person_id = row['track_id']
        try:
            start_frame = int(row['started_frame'])
            end_frame = int(row['ended_frame'])
        except (ValueError, TypeError):
            start_frame, end_frame = None, None

        cumulative_counts = {name: 0 for name in id2name.values()}
        if start_frame is not None and end_frame is not None:
            for vc in vehicle_crossings:
                if start_frame <= vc['frame'] <= end_frame:
                    cumulative_counts[vc['type']] += 1

        total_vehicles = sum(cumulative_counts.values())
        output_data.append([person_id, True, total_vehicles]
                           + [cumulative_counts[vt] for vt in id2name.values()]
                           + [model_note])

    df = _write(output_data)
    print(f"Total crossing events analyzed: {len(crossing_df)}")
    return df
