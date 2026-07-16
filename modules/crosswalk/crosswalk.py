import os
import cv2
import pandas as pd
from ultralytics import YOLO
import math
import torch

# Minimum plausible on-screen size for a crosswalk detection. A live run showed
# tiny (e.g. 27x42 px) false positives being accepted and poisoning the downstream
# scale calibration; real crosswalks in dashcam/street footage are wide, low
# structures, so anything narrower than ~60 px or shorter than ~24 px is rejected.
MIN_CROSSWALK_BOX_W_PX = 60
MIN_CROSSWALK_BOX_H_PX = 24


# FIX #18: default confidence raised from 0.1 to 0.35 so weak/false crosswalk
# detections are filtered out (kept as a defaulted kwarg for backward compatibility).
def run_crosswalk_detection(video_path, analyze_interval_sec=1.0, output_csv_path=None, conf=0.35, show=False):

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if output_csv_path is None:
        output_dir = os.path.join("analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "[E7]crosswalk_detection.csv")

    model = YOLO("modules/crosswalk/best.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        # Match the sibling environment modules: emit a valid header-only CSV so
        # downstream consumers (e.g. crosswalk_usage) never hit FileNotFoundError.
        print(f"Error: Could not open video: {video_path}")
        pd.DataFrame(columns=["frame_id", "crosswalk_detected", "crosswalk_boxes"]).to_csv(output_csv_path, index=False)
        cap.release()
        return

    fps_raw = cap.get(cv2.CAP_PROP_FPS)
    fps = math.ceil(fps_raw) if fps_raw > 0 else 30
    frame_interval = max(1, int(fps * analyze_interval_sec))

    results_list = []
    frame_id = -1

    # Per-sample raw detections collected during the pass; smoothing is applied afterwards.
    frame_ids = []
    sample_detected = []
    sample_boxes = []

    # PERF: grab() advances the stream without decoding; only frames that are
    # actually analyzed are decoded via retrieve(). read() == grab()+retrieve(),
    # so frame indexing and outputs are unchanged.
    while True:
        ok = cap.grab()
        if not ok:
            break
        frame_id += 1
        frame_ids.append(frame_id)

        if frame_id % frame_interval == 0:
            ok, frame = cap.retrieve()
            if not ok:
                # Decode failed for a grabbed frame: record an empty sample so the
                # sample list stays aligned with the frame -> sample expansion below.
                sample_detected.append(False)
                sample_boxes.append([])
                continue

            result = model(frame, imgsz=640, conf=conf, verbose=False)[0]
            detected = False
            crosswalk_boxes = []

            if result.boxes is not None and result.boxes.data.size(0) > 0:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id == 0:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        # Minimum plausible size gate: tiny detections are not
                        # crosswalks (see MIN_CROSSWALK_BOX_* above).
                        if (x2 - x1) < MIN_CROSSWALK_BOX_W_PX or (y2 - y1) < MIN_CROSSWALK_BOX_H_PX:
                            continue
                        detected = True
                        # FIX #18: pad the box by only ~0.15x its width on each side instead
                        # of the previous 2x-per-side widening (which inflated boxes ~5x wide).
                        pad = (x2 - x1) * 0.15
                        extended_x1 = x1 - pad
                        extended_x2 = x2 + pad
                        coords = [round(extended_x1, 2), round(y1, 2), round(extended_x2, 2), round(y2, 2)]
                        crosswalk_boxes.append(coords)

                        if show:
                            cv2.rectangle(
                                frame,
                                (int(x1), int(y1)),
                                (int(x2), int(y2)),
                                (0, 0, 255), 2
                            )
                            cv2.putText(frame, "Crosswalk", (int(x1), int(y1) - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            sample_detected.append(detected)
            sample_boxes.append(crosswalk_boxes)

            # Skipped frames are no longer decoded, so the debug preview only
            # shows the analyzed (retrieved) frames.
            if show:
                cv2.imshow("Crosswalk Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Detection stopped by user.")
                    break

    cap.release()
    cv2.destroyAllWindows()

    # FIX #18: majority-vote temporal smoothing of the per-sample crosswalk detection over a
    # small centered sliding window to suppress frame-to-frame Yes/No label flicker.
    n = len(sample_detected)
    smooth_window = 5
    half = smooth_window // 2
    smoothed_detected = []
    smoothed_boxes = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        votes = sample_detected[lo:hi]
        is_detected = sum(votes) * 2 > (hi - lo)  # strict majority within the window
        if is_detected:
            boxes = sample_boxes[i]
            if not boxes:
                # smoothing turned a No into Yes: borrow the nearest detected boxes in the window
                for j in range(lo, hi):
                    if sample_boxes[j]:
                        boxes = sample_boxes[j]
                        break
        else:
            boxes = []
        smoothed_detected.append("Yes" if is_detected else "No")
        smoothed_boxes.append(boxes)

    # Expand smoothed per-sample results back to one row per native frame (frame f belongs to
    # sample f // frame_interval, matching the original "hold last sampled value" behaviour).
    for f in frame_ids:
        s = f // frame_interval
        if s >= n:
            s = n - 1
        results_list.append({
            "frame_id": f,
            "crosswalk_detected": smoothed_detected[s],
            "crosswalk_boxes": smoothed_boxes[s]
        })

    if not results_list:
        results_df = pd.DataFrame(columns=["frame_id", "crosswalk_detected", "crosswalk_boxes"])
    else:
        results_df = pd.DataFrame(results_list)

    results_df.to_csv(output_csv_path, index=False)
    print(f"Crosswalk detection completed. Results saved to {output_csv_path}")
