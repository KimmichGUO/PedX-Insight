"""Time-headway and platooning statistics at the vehicle counting line.

Consumes [V10]line_crossing_events.csv (written by the vehicle-count producer:
one row per vehicle at the instant its centroid crosses the virtual counting
line) and emits [V11]headway_stats.csv: per direction (and per cx lane-half
when enough events exist) the classic flow-regularity fingerprint --
shifted-exponential headway fit, platoon fraction, coefficient of variation
and flow rate.

[V10] schema: track_id, frame_id, time_s, cx, cy, direction, axis, veh_type
[V11] schema: direction, lane_half, n_events, n_gaps, tau_s, lambda_hz,
              mean_headway_s, cv_headway, platoon_frac, flow_veh_per_min

Design notes
------------
* Pure-function core (`headway_stats`) takes a dataframe and returns row
  dicts, so it is unit-testable without any video or filesystem.
* Gaps longer than `max_gap_s` (default 20 s) are treated as flow
  interruptions (red phase, platoon hole, detector dropout) and discarded
  from the distribution rather than poisoning the mean.
* Shifted-exponential MLE: tau = min(gap); lambda = 1 / (mean(gap) - tau).
  For a degenerate (deterministic) gap sample mean == tau, so lambda is
  reported as NaN instead of infinity.
* Directions are NEVER pooled: every output row belongs to exactly one
  direction value as it appears in [V10].
"""

import math
import os

import numpy as np
import pandas as pd

OUTPUT_COLUMNS = [
    "direction",
    "lane_half",
    "n_events",
    "n_gaps",
    "tau_s",
    "lambda_hz",
    "mean_headway_s",
    "cv_headway",
    "platoon_frac",
    "flow_veh_per_min",
]

# Gaps above this are flow interruptions (signal phase / dropout), not headways.
MAX_GAP_S = 20.0
# Classic car-following threshold: a follower closer than this is platooning.
PLATOON_THRESHOLD_S = 1.5
# Minimum events in a direction before it is additionally split into cx halves.
MIN_EVENTS_FOR_LANE_SPLIT = 30
# Numerical floor below which (mean - tau) is treated as zero (deterministic flow).
_EPS = 1e-9


def _stats_for_times(times, max_gap_s, platoon_threshold_s):
    """Headway statistics for one already-isolated stratum.

    `times` is a 1-D array-like of crossing timestamps in seconds (any order).
    Returns the dict of the stat columns (everything except direction /
    lane_half / n_events, which the caller owns).
    """
    times = np.sort(np.asarray(times, dtype=float))
    gaps = np.diff(times)
    # Discard flow interruptions and non-positive/duplicate-timestamp gaps.
    gaps = gaps[(gaps > 0) & (gaps <= max_gap_s)]

    n_gaps = int(gaps.size)
    if n_gaps == 0:
        return {
            "n_gaps": 0,
            "tau_s": np.nan,
            "lambda_hz": np.nan,
            "mean_headway_s": np.nan,
            "cv_headway": np.nan,
            "platoon_frac": np.nan,
            "flow_veh_per_min": np.nan,
        }

    mean_gap = float(np.mean(gaps))
    tau = float(np.min(gaps))
    excess = mean_gap - tau
    lambda_hz = (1.0 / excess) if excess > _EPS else np.nan
    std_gap = float(np.std(gaps))  # population std: deterministic flow -> exactly 0
    cv = std_gap / mean_gap if mean_gap > _EPS else np.nan
    platoon_frac = float(np.mean(gaps < platoon_threshold_s))
    flow = 60.0 / mean_gap if mean_gap > _EPS else np.nan

    return {
        "n_gaps": n_gaps,
        "tau_s": tau,
        "lambda_hz": lambda_hz,
        "mean_headway_s": mean_gap,
        "cv_headway": cv,
        "platoon_frac": platoon_frac,
        "flow_veh_per_min": flow,
    }


def headway_stats(
    events_df,
    max_gap_s=MAX_GAP_S,
    platoon_threshold_s=PLATOON_THRESHOLD_S,
    min_events_for_lane_split=MIN_EVENTS_FOR_LANE_SPLIT,
    frame_width=None,
):
    """Pure core: [V10] events dataframe in -> list of [V11] row dicts out.

    One "all" row is always emitted per direction; when a direction holds at
    least `min_events_for_lane_split` events with usable cx values it is
    additionally split into "left"/"right" cx halves. The split boundary is
    frame_width / 2 when the caller knows the frame width, otherwise the
    median cx of that direction's events.
    """
    rows = []
    if events_df is None or len(events_df) == 0:
        return rows
    if "time_s" not in events_df.columns:
        return rows

    df = events_df.copy()
    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    df = df[df["time_s"].notna()]
    if len(df) == 0:
        return rows

    if "direction" not in df.columns:
        df["direction"] = "unknown"
    direction_series = df["direction"].fillna("unknown")

    # Directions are never pooled: iterate strictly per direction value.
    for direction, dir_df in df.groupby(direction_series, sort=True):
        times = dir_df["time_s"].to_numpy(dtype=float)
        stats = _stats_for_times(times, max_gap_s, platoon_threshold_s)
        rows.append(
            {"direction": direction, "lane_half": "all", "n_events": int(len(dir_df)), **stats}
        )

        # Optional cx lane-half split within the same direction.
        if "cx" not in dir_df.columns:
            continue
        cx = pd.to_numeric(dir_df["cx"], errors="coerce")
        usable = dir_df[cx.notna()]
        if len(usable) < min_events_for_lane_split:
            continue
        cx_usable = cx[cx.notna()]
        if frame_width is not None and frame_width > 0:
            split_cx = float(frame_width) / 2.0
        else:
            split_cx = float(cx_usable.median())
        for half_name, half_df in (
            ("left", usable[cx_usable < split_cx]),
            ("right", usable[cx_usable >= split_cx]),
        ):
            if len(half_df) < 2:
                continue
            half_times = half_df["time_s"].to_numpy(dtype=float)
            half_stats = _stats_for_times(half_times, max_gap_s, platoon_threshold_s)
            rows.append(
                {
                    "direction": direction,
                    "lane_half": half_name,
                    "n_events": int(len(half_df)),
                    **half_stats,
                }
            )

    return rows


def _read_frame_width(video_meta_csv_path, video_name):
    """Frame width from the [B0]video_meta.csv sidecar; None when unavailable."""
    try:
        if not (video_meta_csv_path and os.path.exists(video_meta_csv_path)):
            return None
        meta = pd.read_csv(video_meta_csv_path)
        if "width" not in meta.columns or len(meta) == 0:
            return None
        if "video_name" in meta.columns and video_name is not None:
            match = meta[meta["video_name"].astype(str) == str(video_name)]
            if len(match) > 0:
                meta = match
        width = pd.to_numeric(meta["width"], errors="coerce").dropna()
        if len(width) == 0:
            return None
        width = float(width.iloc[0])
        return width if math.isfinite(width) and width > 0 else None
    except Exception:
        return None


def run_headway_stats(
    video_path,
    line_crossing_csv_path=None,
    video_meta_csv_path=None,
    output_csv_path=None,
    max_gap_s=MAX_GAP_S,
    platoon_threshold_s=PLATOON_THRESHOLD_S,
    min_events_for_lane_split=MIN_EVENTS_FOR_LANE_SPLIT,
):
    """Module entry point: compute [V11]headway_stats.csv for one video.

    CSV-only -- the video file itself is never opened, so this keeps working
    after the source videos are deleted. A missing or empty
    [V10]line_crossing_events.csv yields a valid header-only output CSV.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join("analysis_results", video_name)

    if line_crossing_csv_path is None:
        line_crossing_csv_path = os.path.join(output_dir, "[V10]line_crossing_events.csv")
    if video_meta_csv_path is None:
        video_meta_csv_path = os.path.join(output_dir, "[B0]video_meta.csv")
    if output_csv_path is None:
        output_csv_path = os.path.join(output_dir, "[V11]headway_stats.csv")

    out_dir = os.path.dirname(output_csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    events_df = None
    if os.path.exists(line_crossing_csv_path):
        try:
            events_df = pd.read_csv(line_crossing_csv_path)
        except Exception:
            events_df = None  # unreadable/corrupt producer file -> header-only output

    frame_width = _read_frame_width(video_meta_csv_path, video_name)

    rows = headway_stats(
        events_df,
        max_gap_s=max_gap_s,
        platoon_threshold_s=platoon_threshold_s,
        min_events_for_lane_split=min_events_for_lane_split,
        frame_width=frame_width,
    )

    result_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    result_df.to_csv(output_csv_path, index=False)
    print(f"Headway statistics saved to {output_csv_path}")
    return result_df
