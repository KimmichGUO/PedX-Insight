import os
import cv2
import pandas as pd
from ultralytics import YOLO
import math
import torch
from collections import Counter

def run_weather_detection(video_path, analyze_interval_sec=1.0, output_csv_path=None,
                          smoothing_window=5):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if output_csv_path is None:
        output_dir = os.path.join("analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "[E1]weather.csv")

    model = YOLO('modules/weather/best.pt')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        pd.DataFrame(columns=["frame_id", "weather_label"]).to_csv(output_csv_path, index=False)
        cap.release()
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = math.ceil(fps) if fps > 0 else 30
    analyze_every_n_frames = max(1, int(fps * analyze_interval_sec))

    frame_ids = []          # frame_id for every read frame (one output row each)
    sample_index = []       # index into sampled_labels that governs each frame
    sampled_labels = []     # one raw model prediction per analysis point
    frame_id = -1
    cur_sample = -1
    last_label = "unknown"

    # PERF: grab() advances the stream without decoding; only the sampled frames
    # are decoded via retrieve(). read() == grab()+retrieve(), so frame indexing,
    # sampling semantics and the one-row-per-native-frame expansion are unchanged.
    while cap.isOpened():
        ok = cap.grab()
        if not ok:
            break
        frame_id += 1

        if frame_id % analyze_every_n_frames == 0:
            ok, frame = cap.retrieve()
            if ok:
                result = model(frame, verbose=False)[0]

                if result.probs is not None:
                    pred_index = int(result.probs.top1)
                    last_label = model.names[pred_index]
                elif result.boxes.data.size(0) > 0:
                    best_det = result.boxes.conf.argmax().item()
                    pred_index = int(result.boxes.cls[best_det].item())
                    last_label = model.names[pred_index]
                else:
                    last_label = "unknown"
            else:
                # Decode failed for a grabbed frame: keep the sample cadence with
                # an explicit unknown so sample_index stays valid for every frame.
                last_label = "unknown"

            sampled_labels.append(last_label)
            cur_sample += 1

        frame_ids.append(frame_id)
        sample_index.append(cur_sample)

    cap.release()

    # FIX #21: the raw per-analysis-point predictions flicker frame-to-frame; smooth
    # each sampled label with a centered sliding-window majority vote so weather spans
    # stay stable, then expand the smoothed labels back to every frame.
    win = max(1, int(smoothing_window))
    half = win // 2
    smoothed_labels = []
    for i in range(len(sampled_labels)):
        lo = max(0, i - half)
        hi = min(len(sampled_labels), i + half + 1)
        smoothed_labels.append(Counter(sampled_labels[lo:hi]).most_common(1)[0][0])

    results_list = [
        {"frame_id": fid, "weather_label": smoothed_labels[s]}
        for fid, s in zip(frame_ids, sample_index)
    ]
    pd.DataFrame(results_list, columns=["frame_id", "weather_label"]).to_csv(output_csv_path, index=False)
    print(f"Weather detection completed. Results saved to {output_csv_path}")
