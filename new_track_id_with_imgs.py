import os
import cv2
import pandas as pd
from ultralytics import YOLO
import math
import torch


def ultralytics_pedestrian_tracking_with_imgsave(video_path, analyze_interval_sec=1.0,
                                                 weights="yolo11n.pt", output_csv_path=None,
                                                 tracking_fps=15.0, det_weights="yolo11x.pt",
                                                 tracker_cfg="mybotsort.yaml"):
    """Detect + track pedestrians with a TWO-RATE design.

    Blocker fixed: ByteTrack's Kalman/IoU association assumes near-consecutive
    frames. The previous code stepped the tracker once per ~1 s (analysis_interval),
    which fragmented every moving pedestrian into fresh IDs. Now the tracker runs
    DENSELY (up to `tracking_fps`, default 15 Hz) so association holds, and two CSVs
    are written:
      * [B1]tracked_pedestrians.csv - 1 row per pedestrian per ANALYSIS-interval frame
        (unchanged schema; feeds all the appearance/environment/crossing modules).
      * [B2]dense_tracks.csv       - 1 row per pedestrian per TRACKED frame (dense);
        feeds the kinematic modules (speed [S1], future gait / TTC).

    Set tracking_fps<=0 to disable dense tracking and step only at the analysis
    interval (old behaviour), e.g. to bound runtime on CPU.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if output_csv_path is None:
        output_dir = os.path.join("analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "[B1]tracked_pedestrians.csv")
    else:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    output_dir = os.path.dirname(output_csv_path)
    dense_csv_path = os.path.join(output_dir, "[B2]dense_tracks.csv")
    pedestrian_img_dir = os.path.join(output_dir, "pedestrian_img")
    os.makedirs(pedestrian_img_dir, exist_ok=True)

    # #7 detector upgrade: use a stronger detector for the tracking pass (better recall on
    # small/distant/occluded pedestrians -> fewer track gaps). Fall back to `weights`
    # (typically yolo11n) if the larger model can't be loaded/downloaded (e.g. offline).
    try:
        model = YOLO(det_weights)
        used_weights = det_weights
    except Exception as e:
        print(f"[track] could not load {det_weights} ({e}); falling back to {weights}")
        model = YOLO(weights)
        used_weights = weights
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    # [B0] sidecar: fps/width/height for consumers that run AFTER the video is deleted
    # (crossing judge frame-width fallback, insight modules needing fps, etc.).
    pd.DataFrame([{
        "video_name": video_name,
        "fps": round(fps, 3),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }]).to_csv(os.path.join(output_dir, "[B0]video_meta.csv"), index=False)

    # [B1] cadence (classification/environment modules): every N frames ~= analysis_interval.
    analyze_every_n_frames = max(1, math.ceil(fps * analyze_interval_sec))
    # Dense tracking cadence: aim for ~tracking_fps so a walking pedestrian moves well under
    # one box width between tracker updates and ByteTrack association stays valid.
    if tracking_fps and tracking_fps > 0:
        track_stride = max(1, round(fps / tracking_fps))
        # Snap the [B1] cadence onto the dense grid: at e.g. 25 fps analyze_every=25 with
        # stride 2 would insert off-grid tracker steps with irregular dt, degrading the
        # Kalman prediction. A ~4% shorter B1 interval is harmless; irregular steps are not.
        analyze_every_n_frames = max(track_stride,
                                     (analyze_every_n_frames // track_stride) * track_stride)
    else:
        track_stride = analyze_every_n_frames

    frame_id = 0
    b1_results = []
    dense_results = []
    target_cls = 0
    saved_frames = {}

    use_half = device == "cuda"          # FP16 roughly halves GPU inference time

    while cap.isOpened():
        # grab() parses the frame without decoding it; the full decode (retrieve) is paid
        # only on frames we actually track — a large saving at 60 fps with stride 4.
        if not cap.grab():
            break

        frame_id += 1

        record_b1 = (frame_id % analyze_every_n_frames == 0)
        do_track = (frame_id % track_stride == 0) or record_b1
        if not do_track:
            continue

        success, frame = cap.retrieve()
        if not success:
            continue

        timestamp = round(frame_id / fps, 3)
        # BoT-SORT (ReID + GMC) by default for occlusion/moving-camera robustness;
        # pass tracker_cfg="mybytetrack.yaml" to fall back to plain ByteTrack.
        track_results = model.track(
            frame,
            persist=True,
            tracker=tracker_cfg,
            classes=[target_cls],
            half=use_half,
            verbose=False,
        )

        if track_results[0].boxes is None or len(track_results[0].boxes) == 0:
            continue

        boxes = track_results[0].boxes
        if boxes.id is None:
            continue

        for box, track_id in zip(boxes.xyxy, boxes.id):
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            track_id = int(track_id.cpu().numpy())

            row = {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "track_id": track_id,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            }
            dense_results.append(row)
            if record_b1:
                b1_results.append(row)

            # Save up to 2 reference crops per id (appearance modules), at the [B1] cadence.
            if record_b1 and track_id not in saved_frames:
                saved_frames[track_id] = []
            if record_b1 and len(saved_frames.get(track_id, [])) < 2:
                min_frame_interval = max(analyze_every_n_frames * 2, 60)
                if not saved_frames[track_id] or (frame_id - saved_frames[track_id][-1]) >= min_frame_interval:
                    expand_left = int((x2 - x1) * 0.2)
                    expand_right = int((x2 - x1) * 0.2)
                    expand_top = int((y2 - y1) * 0.5)

                    x1_exp = max(0, x1 - expand_left)
                    x2_exp = min(frame.shape[1], x2 + expand_right)
                    y1_exp = max(0, y1 - expand_top)
                    y2_exp = y2 + expand_top

                    crop = frame[y1_exp:y2_exp, x1_exp:x2_exp]
                    if crop.size == 0:
                        continue

                    person_dir = os.path.join(pedestrian_img_dir, f"id_{track_id}")
                    os.makedirs(person_dir, exist_ok=True)
                    img_path = os.path.join(person_dir, f"frame_{frame_id}.png")
                    cv2.imwrite(img_path, crop)
                    saved_frames[track_id].append(frame_id)

    cap.release()

    # Build with explicit columns so a pedestrian-free video still writes a valid header
    # row instead of a 0-byte file that makes every downstream pd.read_csv raise EmptyDataError.
    cols = ["frame_id", "timestamp", "track_id", "x1", "y1", "x2", "y2"]
    df = pd.DataFrame(b1_results, columns=cols)
    df.to_csv(output_csv_path, index=False)
    dense_df = pd.DataFrame(dense_results, columns=cols)
    dense_df.to_csv(dense_csv_path, index=False)

    print(f"Tracking results saved to: {output_csv_path}")
    print(f"Dense trajectory saved to: {dense_csv_path}")
    print(f"Pedestrian images saved in: {pedestrian_img_dir}")
    print(f"YOLO ({used_weights}) on {model.device} | tracker={tracker_cfg}")
    print(f"[B1] rows: {len(df)} | [B2] dense rows: {len(dense_df)} | "
          f"track_stride={track_stride} (~{fps / track_stride:.1f} Hz), "
          f"analyze_every={analyze_every_n_frames}")
    print(f"Total pedestrians tracked: {df['track_id'].nunique() if not df.empty else 0}")
