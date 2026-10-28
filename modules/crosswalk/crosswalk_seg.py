"""[E7] crosswalk detection from scene segmentation (Mask2Former / Mapillary Vistas).

Replaces the YOLO detector in modules/crosswalk/crosswalk.py. That detector keyed on class 0,
which its weights name 'Flexible_delineator_post', and scored at chance against JAAD's
per-frame crosswalk label (ROC-AUC 0.50 on 346 clips; its class 4, 'crosswalk', is also at
chance, 0.49). Segmentation is the validated replacement: against the same labels it reaches
F1 0.91 (threshold chosen out of fold) and ROC-AUC 0.95 -- see validation/results/JAAD_REPORT.md.

Output contract = the old [E7]crosswalk_detection.csv, so every consumer keeps working
(crosswalk_usage, risky_crossing, scale_calibration, vehicle_speed, video_info/env_info):

    one row per NATIVE frame: frame_id (0-indexed), crosswalk_detected ('Yes'/'No'),
    crosswalk_boxes (python-literal list of [x1, y1, x2, y2] in frame pixels)

How a frame becomes a detection and boxes:
  * detected  <=>  crosswalk pixel fraction >= scene_seg.PRESENCE_THRESHOLDS['crosswalk']
    (the JAAD-validated criterion), then the same 5-sample majority vote the old module used.
  * boxes: the crosswalk mask (plain crossing + zebra marking classes) is morphologically closed
    so the stripes of one zebra merge, connected components are taken, and each component that
    passes the old size gate (>= 60 x 24 px) becomes a box padded by 0.15x its width per side
    (the old module's convention). The boxes are derived deterministically from the validated
    mask; box GEOMETRY itself has no ground truth in JAAD and is not validated.

By-product: the per-analysed-frame segmentation fractions are written as
[E10]scene_segmentation.csv next to [E7]. run.py deletes the video after analysis, so this is
the only record from which the threshold could later be recalibrated.

Cost: ~0.15 s per analysed frame on an RTX 4090 (1 s sampling).
"""

import ast  # noqa: F401  (consumers parse crosswalk_boxes with ast.literal_eval)
import math
import os

import cv2
import numpy as np
import pandas as pd

from modules.paths import RESULTS_ROOT

# Minimum plausible on-screen size of a crosswalk region. A live run showed tiny (e.g. 27x42 px)
# false positives poisoning the downstream scale calibration; real crosswalks in dashcam/street
# footage are wide, low structures, so anything narrower than ~60 px or shorter than ~24 px is
# rejected. (Single source of truth: the legacy modules/crosswalk/crosswalk.py imports these.)
MIN_CROSSWALK_BOX_W_PX = 60
MIN_CROSSWALK_BOX_H_PX = 24

BOX_PAD_FRAC = 0.15          # per side, as the old module
CLOSE_KERNEL_FRAC = 0.02     # closing kernel = 2% of the frame width: merges zebra stripes
MIN_CLOSE_KERNEL_PX = 5
MAX_BOXES_PER_FRAME = 10
SMOOTH_WINDOW = 5            # majority vote over a centred window of samples, as the old module

COLUMNS = ["frame_id", "crosswalk_detected", "crosswalk_boxes"]


def crosswalk_boxes_from_mask(mask, min_w=MIN_CROSSWALK_BOX_W_PX, min_h=MIN_CROSSWALK_BOX_H_PX,
                              pad_frac=BOX_PAD_FRAC):
    """Boxes [x1, y1, x2, y2] (floats, padded in x) of the connected crosswalk regions in a
    boolean (h, w) mask, largest first. Stripes of one zebra are merged by a morphological
    closing before components are taken."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        return []
    h, w = mask.shape
    k = max(MIN_CLOSE_KERNEL_PX, int(round(CLOSE_KERNEL_FRAC * w)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    n, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    comps = []
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if bw < min_w or bh < min_h:
            continue
        pad = bw * pad_frac
        comps.append((area, [round(x - pad, 2), round(float(y), 2),
                             round(x + bw + pad, 2), round(float(y + bh), 2)]))
    comps.sort(key=lambda c: -c[0])
    return [box for _, box in comps[:MAX_BOXES_PER_FRAME]]


def _smooth_and_expand(frame_ids, sample_detected, sample_boxes, frame_interval):
    """The old module's post-processing: strict-majority vote over a centred window of
    samples (a No turned into Yes borrows the nearest boxes in the window), then each native
    frame takes the value of the sample it belongs to (frame f -> sample f // frame_interval)."""
    n = len(sample_detected)
    half = SMOOTH_WINDOW // 2
    smoothed_detected, smoothed_boxes = [], []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        is_detected = sum(sample_detected[lo:hi]) * 2 > (hi - lo)
        if is_detected:
            boxes = sample_boxes[i]
            if not boxes:
                for j in range(lo, hi):
                    if sample_boxes[j]:
                        boxes = sample_boxes[j]
                        break
        else:
            boxes = []
        smoothed_detected.append("Yes" if is_detected else "No")
        smoothed_boxes.append(boxes)
    rows = []
    for f in frame_ids:
        s = min(f // frame_interval, n - 1)
        rows.append({"frame_id": f, "crosswalk_detected": smoothed_detected[s],
                     "crosswalk_boxes": smoothed_boxes[s]})
    return rows


def run_crosswalk_detection(video_path, analyze_interval_sec=1.0, output_csv_path=None,
                            conf=None, show=False, model_id=None, device=None, short_side=None):
    """Write [E7]crosswalk_detection.csv (and the [E10] by-product) for `video_path`.

    `conf` and `show` are accepted only for call compatibility with the retired YOLO module and
    are ignored (the criterion is scene_seg.PRESENCE_THRESHOLDS['crosswalk']).
    """
    from modules.scene_seg import scene_seg as ss

    model_id = model_id or ss.DEFAULT_MODEL_ID
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if output_csv_path is None:
        output_dir = os.path.join(RESULTS_ROOT, video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "[E7]crosswalk_detection.csv")
    e10_path = os.path.join(os.path.dirname(output_csv_path), ss.OUTPUT_FILENAME)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # Valid header-only CSVs so downstream consumers never hit FileNotFoundError.
        print(f"Error: Could not open video: {video_path}")
        pd.DataFrame(columns=COLUMNS).to_csv(output_csv_path, index=False)
        ss._write([], e10_path)
        cap.release()
        return

    fps_raw = cap.get(cv2.CAP_PROP_FPS)
    fps = math.ceil(fps_raw) if fps_raw > 0 else 30
    frame_interval = max(1, int(fps * analyze_interval_sec))

    frame_ids, sample_detected, sample_boxes, e10_rows = [], [], [], []
    frame_id = -1
    model = processor = group_ids = None
    num_labels = 0
    torch = None
    used_device = device
    try:
        # grab() advances the stream without decoding; only analysed frames are decoded.
        while True:
            if not cap.grab():
                break
            frame_id += 1
            frame_ids.append(frame_id)
            if frame_id % frame_interval != 0:
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None or frame.size == 0:
                # keep the sample list aligned with the frame -> sample expansion below
                sample_detected.append(False)
                sample_boxes.append([])
                continue

            if model is None:                      # lazy: unreadable videos never load it
                import torch
                if used_device is None:
                    used_device = "cuda" if torch.cuda.is_available() else "cpu"
                model, processor = ss._load_model(model_id, used_device)
                ss._record_scene_assets(model, model_id, output_csv_path, used_device, short_side)
                id2label = model.config.id2label
                group_ids = ss._group_class_ids(id2label)
                num_labels = max(int(i) for i in id2label) + 1
                cw_ids = torch.tensor(group_ids["crosswalk"], device=used_device)

            with torch.inference_mode():
                seg = ss._segment_map(frame, model, processor, used_device, short_side)
                counts = torch.bincount(seg.reshape(-1), minlength=num_labels).cpu().numpy()
                row = ss._row_from_counts(frame_id, counts, frame.shape[0] * frame.shape[1],
                                          group_ids, model_id)
                detected = bool(row["crosswalk_present"])
                boxes = []
                if detected:
                    mask = torch.isin(seg.to(cw_ids.device), cw_ids).cpu().numpy()
                    boxes = crosswalk_boxes_from_mask(mask)
            e10_rows.append(row)
            sample_detected.append(detected)
            sample_boxes.append(boxes)
    finally:
        cap.release()
        if model is not None:
            del model, processor
            import gc
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    if sample_detected:
        rows = _smooth_and_expand(frame_ids, sample_detected, sample_boxes, frame_interval)
        results_df = pd.DataFrame(rows, columns=COLUMNS)
    else:
        results_df = pd.DataFrame(columns=COLUMNS)
    results_df.to_csv(output_csv_path, index=False)
    ss._write(e10_rows, e10_path)
    n_yes = int((results_df["crosswalk_detected"] == "Yes").sum()) if len(results_df) else 0
    print(f"Crosswalk detection (segmentation) completed: {n_yes}/{len(results_df)} frames "
          f"with a crosswalk. Results saved to {output_csv_path}")
