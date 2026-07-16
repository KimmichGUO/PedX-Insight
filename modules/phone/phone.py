import cv2
import pandas as pd
import os
from ultralytics import YOLO
import math
import torch

def run_phone_detection(
    video_path,
    analyze_interval_sec=1.0,
    weights="yolo11n.pt",
    tracking_csv_path=None,
    phone_conf=0.15,      # NEW: lower conf recovers small/far phones missed at 0.25 on the full frame
    min_crop_px=320       # NEW: upscale each upper-body crop so its longer side >= this many px
):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if tracking_csv_path is None:
        tracking_csv_path = os.path.join(".", "analysis_results", video_name, "[B1]tracked_pedestrians.csv")

    output_dir = os.path.join(".", "analysis_results", video_name)
    os.makedirs(output_dir, exist_ok=True)
    phone_csv_path = os.path.join(output_dir, "[P5]phone_usage.csv")

    if not os.path.exists(tracking_csv_path) or os.path.getsize(tracking_csv_path) == 0:
        empty_phone_df = pd.DataFrame(columns=["frame_id", "track_id", "phone_using"])
        empty_phone_df.to_csv(phone_csv_path, index=False)
        print(f"Tracking CSV not found or empty. Empty results saved to {phone_csv_path}")
        return

    df = pd.read_csv(tracking_csv_path)
    if df.empty:
        empty_phone_df = pd.DataFrame(columns=["frame_id", "track_id", "phone_using"])
        empty_phone_df.to_csv(phone_csv_path, index=False)
        print(f"Tracking CSV is empty. Empty results saved to {phone_csv_path}")
        return

    df.sort_values(by=["frame_id", "track_id"], inplace=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # Video missing/unreadable (e.g. deleted after analysis): phone usage is
        # UNKNOWN, not False. Write a header-only CSV so downstream consumers
        # (crossed_info) fall back to None instead of a confident 0.
        cap.release()
        pd.DataFrame(columns=["frame_id", "track_id", "phone_using"]).to_csv(phone_csv_path, index=False)
        print(f"Video not readable: {video_path}. Header-only (unknown) results saved to {phone_csv_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    analyze_every_n_frames = max(1, math.ceil(fps * analyze_interval_sec))
    print(f"Video FPS: {fps:.2f}, analyzing every {analyze_every_n_frames} frames (~{analyze_interval_sec}s)")

    model = YOLO(weights)
    names = model.names
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    phone_results = []

    grouped = df.groupby("frame_id")

    # FIX #23: A nano model run on the full frame almost never sees a phone (it is a
    # handful of pixels far away), and an IoU>0.1 gate between a tiny phone and a whole
    # person box is mathematically almost never satisfied -> near-zero recall. Instead,
    # for each pedestrian we crop just their upper body (head/torso/hands), UPSAMPLE it so
    # the phone becomes large enough for the detector, run detection at a lower confidence,
    # and count a phone only when it is detected inside that head/hands region.
    phone_class_ids = {cid for cid, cname in names.items() if cname == "cell phone"}

    def detect_phone_on_pedestrian(frame, box):
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        if w <= 1 or h <= 1:
            return False
        # Upper body only: head + torso + raised hands ~ top 60% of the person, plus
        # horizontal padding so a phone held out from the body still falls in the crop.
        pad = max(2, int(0.25 * w))
        cx1, cy1 = max(0, x1 - pad), max(0, y1)
        cx2, cy2 = min(fw, x2 + pad), min(fh, y1 + int(0.60 * h))
        if cx2 - cx1 < 2 or cy2 - cy1 < 2:
            return False
        crop = frame[cy1:cy2, cx1:cx2]
        ch, cw = crop.shape[:2]
        # Upsample small/far crops so a distant phone spans enough pixels to be detected
        # (never downsample near pedestrians; cap the factor to avoid huge images).
        scale = min(6.0, max(1.0, float(min_crop_px) / max(ch, cw)))
        if scale > 1.0:
            crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)),
                              interpolation=cv2.INTER_CUBIC)
        res = model.predict(crop, conf=phone_conf, show=False, verbose=False)[0]
        pboxes = res.boxes.xyxy.cpu().numpy()
        pcls = res.boxes.cls.cpu().numpy().astype(int)
        for pb, cid in zip(pboxes, pcls):
            if cid not in phone_class_ids:
                continue
            # Map the phone box back to original-frame pixels.
            px1, py1 = cx1 + pb[0] / scale, cy1 + pb[1] / scale
            px2, py2 = cx1 + pb[2] / scale, cy1 + pb[3] / scale
            pw, ph = px2 - px1, py2 - py1
            # A real phone is small relative to the body: drop box-filling mis-detections.
            if pw > 0.8 * w or ph > 0.6 * h:
                continue
            # Associate only if the phone's centre lies INSIDE this pedestrian's own
            # bbox (upper 60%). The padded crop overlaps neighbours, so gating on the
            # crop/pad bounds would be a tautology and attribute a neighbour's phone
            # to this track (side-by-side walkers within 0.25*w of each other).
            mx, my = 0.5 * (px1 + px2), 0.5 * (py1 + py2)
            if x1 <= mx <= x2 and y1 <= my <= (y1 + 0.60 * h):
                return True
        return False

    for frame_id, group in grouped:
        # Decode the exact sampled frame these boxes came from (CSV holds ~1 row/interval).
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
        ret, frame = cap.read()
        if not ret:
            # Undecodable frame: phone usage is UNKNOWN here, not False. Skip the
            # rows (matching belongings.py) so [P5] carries no confident negatives
            # for frames that were never actually inspected.
            continue

        for _, row in group.iterrows():
            track_id = row["track_id"]
            box = list(map(int, [row["x1"], row["y1"], row["x2"], row["y2"]]))
            phone_using = detect_phone_on_pedestrian(frame, box)
            phone_results.append({
                "frame_id": frame_id,
                "track_id": track_id,
                "phone_using": phone_using
            })

    cap.release()

    pd.DataFrame(phone_results, columns=["frame_id", "track_id", "phone_using"]).to_csv(phone_csv_path, index=False)

    print(f"Phone usage results saved to {phone_csv_path}")
