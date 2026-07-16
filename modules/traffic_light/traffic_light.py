import os
import cv2
import pandas as pd
from ultralytics import YOLO
import math
import torch
from collections import Counter

# Tie-break priority used by the temporal smoother. Yellow ranks highest so a
# genuinely brief amber phase (only ~3-4 samples at 1 s sampling) survives the
# majority vote instead of being erased by the surrounding red/green. On a tie
# this order decides the winner; a colour that is not present in the window is
# never injected, so smoothing can only pick a colour that was actually detected.
_COLOR_PRIORITY = {"yellow": 3, "red": 2, "green": 1, "None": 0}


def _smooth_main_light(colors, window=5):
    """FIX #19/#21: temporal hysteresis for main_light_color.

    The per-frame ``main_light_color`` is picked as the LARGEST detected box,
    which (a) makes a single misdetection flip the label for one sample and
    (b) can never be corrected across time. Here we majority-vote each sample
    over a small centred sliding window so an isolated one-frame flip reverts to
    its neighbours' value, while any real phase lasting >= 2 samples (including
    yellow) is preserved. Ties favour yellow (see ``_COLOR_PRIORITY``) so short
    amber phases stay representable.
    """
    if window <= 1 or len(colors) <= 2:
        return list(colors)
    half = window // 2
    n = len(colors)
    smoothed = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        counts = Counter(colors[lo:hi])
        # winner = most frequent colour in the window; break ties by priority.
        best_color = max(counts.items(), key=lambda kv: (kv[1], _COLOR_PRIORITY.get(kv[0], 0)))[0]
        smoothed.append(best_color)
    return smoothed


def run_traffic_light_detection(video_path, analyze_interval_sec=1.0, output_csv_path=None, smoothing_window=5):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if output_csv_path is None:
        output_dir = os.path.join("analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "[E2]traffic_light.csv")

    model = YOLO("modules/traffic_light/v9 - 48 epochs.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        pd.DataFrame(columns=["frame_id", "main_light_color", "other_lights"]).to_csv(output_csv_path, index=False)
        cap.release()
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = math.ceil(fps) if fps > 0 else 30
    analyze_every_n_frames = max(1, int(fps * analyze_interval_sec))

    results_list = []
    frame_id = -1

    light_id_map = {
        0: 'green',
        1: 'red',
        2: 'yellow'
    }

    color_map = {
        "green": (0, 255, 0),
        "red": (0, 0, 255),
        "yellow": (0, 255, 255)
    }

    # PERF: grab() advances the stream without decoding; only the sampled frames
    # are decoded via retrieve(). read() == grab()+retrieve(), so frame indexing
    # and sampling semantics are unchanged.
    while cap.isOpened():
        ok = cap.grab()
        if not ok:
            break
        frame_id += 1

        if frame_id % analyze_every_n_frames != 0:
            continue

        ok, frame = cap.retrieve()
        if not ok:
            # Decode failed for a grabbed frame: keep the per-sample row cadence
            # with an explicit no-detection row instead of dropping the sample.
            results_list.append({
                "frame_id": frame_id,
                "main_light_color": "None",
                "other_lights": "None"
            })
            continue

        result = model(frame, verbose=False)[0]
        lights = []

        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in light_id_map:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            cls_name = light_id_map[cls_id]

            lights.append({
                "color": cls_name,
                "area": area,
                "box": (x1, y1, x2, y2)
            })

        if not lights:
            results_list.append({
                "frame_id": frame_id,
                "main_light_color": "None",
                "other_lights": "None"
            })
        else:
            # LIMITATION: "main" light = the largest bbox, i.e. the nearest signal
            # on screen. This is often a vehicle signal rather than the one facing
            # the pedestrian's crossing direction. The frames carry no crossing-
            # geometry data, so we cannot reliably associate a light to the walk
            # direction; we keep the largest-box heuristic but stabilise it in time
            # (see _smooth_main_light below). other_lights stays as the raw per-frame
            # detection and is intentionally NOT smoothed.
            lights.sort(key=lambda x: x["area"], reverse=True)
            main_color = lights[0]["color"]
            other_colors = [light["color"] for light in lights[1:]]
            other_str = str(other_colors) if other_colors else "None"

            results_list.append({
                "frame_id": frame_id,
                "main_light_color": main_color,
                "other_lights": other_str
            })

    cap.release()

    # Temporal hysteresis: replace each frame's raw largest-box pick with the
    # majority vote over a small sliding window so one-frame flips don't flip the
    # label. Preserves >= 2-sample phases and keeps short yellow phases (tie -> yellow).
    smoothed_main = _smooth_main_light([r["main_light_color"] for r in results_list], window=smoothing_window)
    for row, main_color in zip(results_list, smoothed_main):
        row["main_light_color"] = main_color

    pd.DataFrame(results_list, columns=["frame_id", "main_light_color", "other_lights"]).to_csv(output_csv_path, index=False)
    print(f"Traffic light detection completed. Results saved to {output_csv_path}")
