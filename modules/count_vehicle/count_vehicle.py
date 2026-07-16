import os
import cv2
import pandas as pd
from ultralytics import YOLO
import torch
from collections import defaultdict, Counter

# Output-schema vehicle types (kept identical so [V6]vehicle_count.csv rows never change).
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


def _write_counts(count_dict, model_note, output_csv_path):
    """Write the [V6]vehicle_count.csv (Vehicle_Type, Count, + Model) with the Total row."""
    df = pd.DataFrame(list(count_dict.items()), columns=['Vehicle_Type', 'Count'])
    total_count = int(df['Count'].sum())
    total_df = pd.DataFrame([{'Vehicle_Type': 'Total', 'Count': total_count}])
    df = pd.concat([df, total_df], ignore_index=True)
    # Record which detector produced the count (added column; existing columns untouched).
    df['Model'] = model_note
    df.to_csv(output_csv_path, index=False)
    return df


TRACKS_COLUMNS = ["frame_id", "timestamp", "track_id", "vtype", "conf", "x1", "y1", "x2", "y2"]
EVENTS_COLUMNS = ["track_id", "frame_id", "time_s", "cx", "cy", "direction", "axis", "veh_type"]


def _write_sidecars(track_rows, event_rows, output_dir):
    """Persist the per-frame vehicle trajectories ([V7]) and line-crossing events ([V10])
    that the dense pass already computes — they power the PET/vehicle-speed/headway
    insight modules instead of being discarded."""
    pd.DataFrame(track_rows, columns=TRACKS_COLUMNS).to_csv(
        os.path.join(output_dir, "[V7]vehicle_tracks.csv"), index=False)
    pd.DataFrame(event_rows, columns=EVENTS_COLUMNS).to_csv(
        os.path.join(output_dir, "[V10]line_crossing_events.csv"), index=False)


def vehicle_count(
        video_path,
        output_csv_path=None,
        analyze_interval_sec=1.0,
        use_coco=True,
        coco_model_path="yolo11n.pt",
        counting_line_ratio=0.5,
        conf_thresh=0.3,
):
    # FIX #12/#13: Count vehicles with DENSE tracking + a virtual line-crossing counter
    # instead of sub-sampled per-frame detections. Every frame is tracked (persist=True) so
    # IDs stay coherent and fast vehicles between old 1-fps samples are no longer missed; a
    # vehicle is counted exactly once, when its track first crosses a horizontal counting
    # line and only after >=2 frames of support (kills fragmented-ID double counting). For
    # GLOBAL data we detect with a COCO yolo11 model (car/bus/truck/motorcycle/bicycle)
    # instead of the Bangladesh-specific best.pt, and record the model in a new column.
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

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if output_csv_path is None:
        output_dir = os.path.join("analysis_results", video_name)
    else:
        output_dir = os.path.dirname(output_csv_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    if output_csv_path is None:
        output_csv_path = os.path.join(output_dir, "[V6]vehicle_count.csv")

    count_dict = {name: 0 for name in id2name.values()}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # Empty/unreadable video guard: still emit valid header CSVs for all outputs.
        print("Error opening video")
        _write_sidecars([], [], output_dir)
        return _write_counts(count_dict, model_note, output_csv_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    allowed_classes = list(class_map.keys())
    use_half = device == "cuda"          # FP16 roughly halves GPU inference time

    # Dual counting lines: a horizontal line only registers vertically-moving traffic,
    # which systematically missed cross traffic moving left-right through the frame
    # (confirmed finding). A track is counted once, when it first crosses EITHER line.
    line_y = None
    line_x = None
    frame_idx = 0
    track_last_side_y = {}               # track_id -> side of the horizontal line (-1/+1)
    track_last_side_x = {}               # track_id -> side of the vertical line (-1/+1)
    track_frames = defaultdict(int)      # track_id -> number of frames it was seen
    track_class_votes = defaultdict(Counter)  # track_id -> vote over type names
    counted_ids = set()
    track_rows = []                      # [V7] per-frame vehicle trajectory dump
    event_rows = []                      # [V10] line-crossing events

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if line_y is None:
            line_y = frame.shape[0] * counting_line_ratio
            line_x = frame.shape[1] * counting_line_ratio

        # Dense per-frame tracking (no frame skipping) restricted to vehicle classes.
        track_results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=conf_thresh,
            classes=allowed_classes,
            half=use_half,
            verbose=False,
        )

        boxes = track_results[0].boxes
        if boxes is None or len(boxes) == 0 or boxes.id is None:
            continue

        timestamp = round(frame_idx / fps, 3)
        for box, track_id, cls_id, conf in zip(
                boxes.xyxy, boxes.id, boxes.cls, boxes.conf
        ):
            track_id = int(track_id.cpu().numpy())
            cls_id = int(cls_id.cpu().numpy())
            conf = float(conf.cpu().numpy())

            if cls_id not in class_map or conf < conf_thresh:
                continue

            vtype = class_map[cls_id]
            x1, y1, x2, y2 = [float(v) for v in box.cpu().numpy()]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            track_frames[track_id] += 1
            track_class_votes[track_id][vtype] += 1
            track_rows.append({
                "frame_id": frame_idx, "timestamp": timestamp, "track_id": track_id,
                "vtype": vtype, "conf": round(conf, 3),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })

            side_y = 1 if cy >= line_y else -1
            side_x = 1 if cx >= line_x else -1
            prev_y = track_last_side_y.get(track_id)
            prev_x = track_last_side_x.get(track_id)
            crossed_y = prev_y is not None and prev_y != side_y
            crossed_x = prev_x is not None and prev_x != side_x

            # Count once when the track crosses either line, with >=2 frames of support.
            if ((crossed_y or crossed_x)
                    and track_frames[track_id] >= 2 and track_id not in counted_ids):
                majority_type = track_class_votes[track_id].most_common(1)[0][0]
                count_dict[majority_type] += 1
                counted_ids.add(track_id)
                axis = "y" if crossed_y else "x"
                direction = (side_y if crossed_y else side_x)
                event_rows.append({
                    "track_id": track_id, "frame_id": frame_idx, "time_s": timestamp,
                    "cx": round(cx, 1), "cy": round(cy, 1),
                    "direction": direction, "axis": axis, "veh_type": majority_type,
                })

            track_last_side_y[track_id] = side_y
            track_last_side_x[track_id] = side_x

    cap.release()

    _write_sidecars(track_rows, event_rows, output_dir)
    df = _write_counts(count_dict, model_note, output_csv_path)
    print(f"Vehicle count completed. Results saved to {output_csv_path}")
    print(f"[V7] {len(track_rows)} trajectory rows, [V10] {len(event_rows)} crossing events")
    print(f"YOLO is running on: {model.device}")
    print(f"Total vehicles counted (line crossings): {len(counted_ids)}")

    return df
