import os
import re
import cv2
import numpy as np
import pandas as pd
import math
from datetime import datetime
from collections import Counter


def _solar_noon_elevation(video_name, analyze_date=None):
    """Solar-elevation prior for the video's city (from mapping.csv lat) on the analysis date.

    Returns the sun's elevation angle (deg) at local solar noon, or None if the city
    cannot be resolved. This is a per-video location/date prior only (we have no per-frame
    wall-clock time), so it gently biases the absolute brightness anchors rather than
    deciding labels on its own.
    """
    try:
        # City = filename token before the first underscore with trailing digits stripped
        # (e.g. "Berlin1_ORPGr4m2-Sw" -> "Berlin").
        city = re.sub(r"\d+$", "", str(video_name).split("_")[0]).strip()
        if not city:
            return None
        # Resolve mapping.csv relative to the repo root (…/modules/daynight/daytime.py -> repo),
        # falling back to the current working directory used by main.py.
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(here, "..", "..", "mapping.csv"),
            os.path.join(os.getcwd(), "mapping.csv"),
            "mapping.csv",
        ]
        mapping_path = next((p for p in candidates if os.path.exists(p)), None)
        if mapping_path is None:
            return None
        m = pd.read_csv(mapping_path)
        row = m[m["city"].astype(str).str.lower() == city.lower()]
        if row.empty:
            return None
        lat = float(row.iloc[0]["lat"])
        # Deterministic fallback: a fixed equinox (Mar 21, solar declination ~0) instead of
        # datetime.now(), so with no date the prior is a stable function of latitude only and
        # re-running the same video always yields the same labels. (2001 is non-leap: day 80.)
        d = analyze_date or datetime(2001, 3, 21)
        n = d.timetuple().tm_yday
        # Cooper's approximation of solar declination for day-of-year n.
        decl = 23.45 * math.sin(math.radians(360.0 * (284 + n) / 365.0))
        # Elevation at local solar noon for this latitude/date.
        return round(90.0 - abs(lat - decl), 2)
    except Exception:
        return None


def run_daytime_detection(video_path, brightness_threshold=100, analyze_interval_sec=1, output_csv_path=None):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if output_csv_path is None:
        output_dir = os.path.join("analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "[E6]daytime.csv")

    # Explicit column list so an empty/failed video still yields a valid header CSV.
    out_cols = ["frame_id", "avg_brightness", "daytime_label", "brightness_norm", "solar_noon_elev_deg"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        pd.DataFrame(columns=out_cols).to_csv(output_csv_path, index=False)
        return

    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = math.ceil(raw_fps) if raw_fps > 0 else 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # max(1, ...) is essential: without it a 0 fps report makes interval_frames == 0 and the
    # `frame_id += interval_frames` loop below never advances, hanging the process forever.
    interval_frames = max(1, int(analyze_interval_sec * fps))

    # --- Pass 1: sample one brightness value per interval block ---------------------------------
    blocks = []  # (start_frame, end_frame, avg_brightness or NaN)
    last_avg_brightness = None
    frame_id = 0
    while frame_id < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            avg_brightness = last_avg_brightness if last_avg_brightness is not None else float("nan")
        else:
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            avg_brightness = float(np.mean(hsv_frame[:, :, 2]))
            last_avg_brightness = avg_brightness
        start_f = int(frame_id)
        end_f = min(int(frame_id) + int(interval_frames), int(total_frames))
        blocks.append((start_f, end_f, avg_brightness))
        frame_id += interval_frames
    cap.release()

    # FIX #20: replace the single global brightness cutoff (which mislabels a lit night as "Day"
    # and an overcast day as "Evening") with a 3-class {Day, Evening, Night} decision that combines
    # (a) per-video brightness normalization — each frame's position within THIS video's own
    # brightness distribution, so an overcast day still reads brighter than its own night; with
    # (b) absolute pixel-value anchors nudged by a solar-elevation prior from the city's lat/lon
    # (mapping.csv) and the analysis date, so an all-dark video isn't uniformly called "Day"; then
    # (c) majority-vote temporal smoothing to remove single-sample flicker.
    #
    # Date for the solar prior: the video file's mtime when the file exists (a stable
    # proxy for when the footage was fetched, identical across re-runs of the same file),
    # never the wall-clock analysis date — otherwise the Day/Evening/Night anchors would
    # drift with the season the pipeline happens to run in. When the file is gone
    # (videos are deleted after analysis), _solar_noon_elevation falls back to a fixed
    # equinox so the prior depends on latitude only.
    analyze_date = None
    try:
        if os.path.exists(video_path):
            analyze_date = datetime.fromtimestamp(os.path.getmtime(video_path))
    except (OSError, OverflowError, ValueError):
        analyze_date = None
    solar_elev = _solar_noon_elevation(video_name, analyze_date=analyze_date)

    bright = np.array([b for (_, _, b) in blocks], dtype=float)
    finite = bright[np.isfinite(bright)]

    if finite.size >= 2:
        p_low = float(np.percentile(finite, 10))
        p_high = float(np.percentile(finite, 90))
    elif finite.size == 1:
        p_low = p_high = float(finite[0])
    else:
        p_low = p_high = 0.0
    span = max(p_high - p_low, 1e-6)
    # (a) per-video normalized brightness (0..1) within this video's own range. When the video's
    # brightness barely varies (uniform day/night), stretching that tiny range to 0..1 would
    # amplify noise into spurious labels, so damp the relative term toward neutral (0.5) as the
    # span shrinks below ~25 levels and let the absolute anchor decide.
    rel_raw = np.clip((bright - p_low) / span, 0.0, 1.0)
    rel_conf = float(np.clip(span / 25.0, 0.0, 1.0))
    rel = 0.5 + (rel_raw - 0.5) * rel_conf

    # (b) absolute anchors derived from brightness_threshold (default 100 -> night 55, day 125),
    # gently shifted (<=15% of threshold) by the solar-noon-elevation prior: a high-sun location/
    # date leans brighter so dimmer pixels still count as day, a low-sun one leans darker.
    night_anchor = 0.55 * brightness_threshold
    day_anchor = 1.25 * brightness_threshold
    if solar_elev is not None:
        nudge = float(np.clip((solar_elev - 40.0) / 50.0, -1.0, 1.0)) * 0.15 * brightness_threshold
        day_anchor -= nudge
        night_anchor -= 0.5 * nudge
    abs_score = np.clip((bright - night_anchor) / max(day_anchor - night_anchor, 1e-6), 0.0, 1.0)

    # Absolute band dominates (keeps an all-dark video out of "Day"); relative position refines
    # within-video contrast. NaN brightness -> neutral "Evening".
    score = 0.65 * abs_score + 0.35 * rel

    def _label(s):
        if not np.isfinite(s):
            return "Evening"
        if s < 0.35:
            return "Night"
        if s < 0.60:
            return "Evening"
        return "Day"

    raw_labels = [_label(s) for s in score]

    # (c) majority-vote temporal smoothing over the ~1s-spaced samples (±2 -> ~5s window).
    smoothed = []
    n = len(raw_labels)
    for i in range(n):
        window = raw_labels[max(0, i - 2):min(n, i + 3)]
        smoothed.append(Counter(window).most_common(1)[0][0])

    # --- Pass 2: expand each block back to its native frames ------------------------------------
    results = []
    for (start_f, end_f, b), label, norm in zip(blocks, smoothed, rel):
        for f in range(start_f, end_f):
            results.append({
                "frame_id": f,
                "avg_brightness": b,
                "daytime_label": label,
                "brightness_norm": round(float(norm), 4),
                "solar_noon_elev_deg": solar_elev,
            })

    df = pd.DataFrame(results, columns=out_cols)
    df.to_csv(output_csv_path, index=False)
    print(f"Daytime detection completed. Results saved to {output_csv_path}")
