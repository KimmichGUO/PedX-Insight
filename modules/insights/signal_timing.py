"""[P10] Signal-phase micro-timing of crossing starts.

Builds a continuous, phase-relative timing distribution of when pedestrians
step off the curb relative to traffic-light transitions:

* anticipatory starts   -> pedestrian starts BEFORE the pedestrian-favorable
                           transition (negative ``delta_start_s``);
* startup latency       -> seconds between the pedestrian-favorable
                           transition and the actual step-off;
* red-clearance exposure-> seconds the pedestrian is still on the roadway
                           while the vehicle signal is (sustained) green.

Inputs (all per-video CSVs, no video required):
  [E2]traffic_light.csv   frame_id (0-based), main_light_color, other_lights
  [C3]crossing_judge.csv  track_id, crossed, started_frame, ended_frame, movement_type
  [B0]video_meta.csv      video_name, fps, width, height, total_frames  (optional, for fps)
  [B2]dense_tracks.csv    frame_id (1-based), timestamp, ...            (fps fallback)

Output ``[P10]signal_timing.csv`` (one row per crossed track):
  track_id, t_start_s, nearest_transition, delta_start_s, anticipatory,
  startup_latency_s, red_exposure_s, n_transitions_in_video,
  phase_convention, light_available

Documented caveats:
* [C3] ``started_frame`` is built from 1 Hz [B1] samples, so start times are
  ~1 s quantized.
* [E2] ``main_light_color`` tracks the LARGEST detected signal head, which is
  usually the VEHICLE signal. The pedestrian-favorable phase is therefore
  vehicle-red; this is recorded per row in ``phase_convention``.
* 'None' / missing [E2] samples are forward-filled (last known real color
  holds until contradicted), matching the bisect pattern in run_redlight.py.
* Transitions are only accepted when BOTH adjacent states are sustained for
  at least ``MIN_SUSTAIN_S`` seconds; one-sample flickers are rejected.
"""

import math
import os

import numpy as np
import pandas as pd

OUTPUT_COLUMNS = [
    "track_id",
    "t_start_s",
    "nearest_transition",
    "delta_start_s",
    "anticipatory",
    "startup_latency_s",
    "red_exposure_s",
    "n_transitions_in_video",
    "phase_convention",
    "light_available",
]

PHASE_CONVENTION = "main_light=vehicle;ped_go=vehicle_red"
REAL_COLORS = ("red", "green", "yellow")

# A phase state must hold at least this long on both sides of a change for the
# change to count as a real transition (flicker rejection).
MIN_SUSTAIN_S = 3.0

# Accept a transition as "nearest" only when delta = t_start - t_transition
# falls inside this window (seconds).
DELTA_MIN_S = -10.0
DELTA_MAX_S = 15.0

DEFAULT_FPS = 30.0


# --------------------------------------------------------------------------
# Pure-function core (dataframes in -> rows out); no filesystem access here.
# --------------------------------------------------------------------------

def _to_bool(value):
    """Robust truthiness for the [C3] 'crossed' column (bool/str/num/NaN)."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    try:
        if isinstance(value, float) and math.isnan(value):
            return False
        return bool(int(value))
    except (TypeError, ValueError):
        return False


def _to_float(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return f


def extract_color_samples(e2_df, fps):
    """[E2] dataframe -> sorted list of (time_s, color) for real colors only.

    Rows whose main_light_color is not in {red, green, yellow} (e.g. 'None',
    NaN) are dropped, which forward-fills the last real color through gaps.
    """
    if e2_df is None or len(e2_df) == 0:
        return []
    if "frame_id" not in e2_df.columns or "main_light_color" not in e2_df.columns:
        return []
    samples = []
    for frame, color in zip(e2_df["frame_id"], e2_df["main_light_color"]):
        try:
            f = int(frame)
        except (TypeError, ValueError):
            continue
        c = str(color).strip().lower()
        if c in REAL_COLORS:
            samples.append((f / float(fps), c))
    samples.sort(key=lambda s: s[0])
    return samples


def build_sustained_segments(samples, min_sustain_s=MIN_SUSTAIN_S, end_time_s=None):
    """Turn color samples into sustained phase segments and valid transitions.

    samples : sorted list of (time_s, color) with real colors only.
    Returns (segments, transitions):
      segments    : list of [t_start_s, t_end_s, color] sustained states,
                    time-ordered, adjacent entries have different colors;
      transitions : list of (t_s, from_color, to_color); t_s is the first
                    sample time of the new sustained state.

    A raw run's duration is measured until the next run starts (state holds
    until contradicted); the last run extends to ``end_time_s`` (default: last
    sample time + median sampling interval). Runs shorter than
    ``min_sustain_s`` are rejected as flicker; adjacent surviving runs of the
    same color are merged, so a one-sample blip inside a long phase produces
    no transition at all.
    """
    if not samples:
        return [], []

    # Raw runs of consecutive identical colors.
    runs = []  # [start_time, color]
    for t, c in samples:
        if not runs or runs[-1][1] != c:
            runs.append([t, c])

    last_t = samples[-1][0]
    if end_time_s is None or end_time_s < last_t:
        if len(samples) > 1:
            intervals = [b[0] - a[0] for a, b in zip(samples, samples[1:]) if b[0] > a[0]]
            step = float(np.median(intervals)) if intervals else 1.0
        else:
            step = 1.0
        end_time_s = last_t + step

    # Bound each run: it holds until the next run starts.
    bounded = []
    for i, (t, c) in enumerate(runs):
        t_next = runs[i + 1][0] if i + 1 < len(runs) else end_time_s
        bounded.append((t, t_next, c))

    # Flicker rejection: keep only sustained runs.
    sustained = [seg for seg in bounded if (seg[1] - seg[0]) >= min_sustain_s]

    # Merge adjacent same-color survivors (e.g. green | flicker | green).
    merged = []
    for t0, t1, c in sustained:
        if merged and merged[-1][2] == c:
            merged[-1][1] = t1
        else:
            merged.append([t0, t1, c])

    segments = [list(m) for m in merged]
    transitions = []
    for a, b in zip(segments, segments[1:]):
        transitions.append((b[0], a[2], b[2]))
    return segments, transitions


def _nearest_transition(t_start_s, transitions,
                        delta_min_s=DELTA_MIN_S, delta_max_s=DELTA_MAX_S):
    """Smallest-|delta| transition with delta = t_start - t_trans in window."""
    best = None
    for t_trans, from_c, to_c in transitions:
        delta = t_start_s - t_trans
        if delta_min_s <= delta <= delta_max_s:
            if best is None or abs(delta) < abs(best[0]):
                best = (delta, from_c, to_c)
    return best


def _red_exposure_s(t_start_s, t_end_s, segments):
    """Seconds of [t_start, t_end] overlapping sustained vehicle-GREEN phases."""
    if math.isnan(t_start_s) or math.isnan(t_end_s) or t_end_s < t_start_s:
        return float("nan")
    total = 0.0
    for s0, s1, color in segments:
        if color == "green":
            total += max(0.0, min(t_end_s, s1) - max(t_start_s, s0))
    return total


def compute_signal_timing_rows(crossing_df, e2_df, fps,
                               min_sustain_s=MIN_SUSTAIN_S,
                               delta_min_s=DELTA_MIN_S,
                               delta_max_s=DELTA_MAX_S):
    """Pure core: [C3] + [E2] dataframes -> list of output-row dicts."""
    samples = extract_color_samples(e2_df, fps)
    light_available = len(samples) > 0
    segments, transitions = build_sustained_segments(samples, min_sustain_s)
    n_transitions = len(transitions)

    rows = []
    if crossing_df is None or len(crossing_df) == 0:
        return rows

    for _, row in crossing_df.iterrows():
        if not _to_bool(row.get("crossed")):
            continue
        try:
            track_id = int(row["track_id"])
        except (TypeError, ValueError, KeyError):
            continue

        started_frame = _to_float(row.get("started_frame"))
        ended_frame = _to_float(row.get("ended_frame"))
        t_start = started_frame / fps if not math.isnan(started_frame) else float("nan")
        t_end = ended_frame / fps if not math.isnan(ended_frame) else float("nan")

        nearest_label = "none"
        delta_start = float("nan")
        anticipatory = np.nan  # tri-state: True / False / NaN (no usable transition)
        startup_latency = float("nan")

        if not math.isnan(t_start) and transitions:
            best = _nearest_transition(t_start, transitions, delta_min_s, delta_max_s)
            if best is not None:
                delta_start, from_c, to_c = best
                nearest_label = "{}_to_{}".format(from_c, to_c)
                ped_favorable = (to_c == "red")  # vehicle red == pedestrian go
                anticipatory = bool(ped_favorable and delta_start < 0)
                if ped_favorable and delta_start >= 0:
                    startup_latency = delta_start

        red_exposure = (
            _red_exposure_s(t_start, t_end, segments)
            if (segments and not math.isnan(t_start)) else float("nan")
        )

        rows.append({
            "track_id": track_id,
            "t_start_s": t_start,
            "nearest_transition": nearest_label,
            "delta_start_s": delta_start,
            "anticipatory": anticipatory,
            "startup_latency_s": startup_latency,
            "red_exposure_s": red_exposure,
            "n_transitions_in_video": n_transitions,
            "phase_convention": PHASE_CONVENTION,
            "light_available": light_available,
        })
    return rows


# --------------------------------------------------------------------------
# fps resolution: [B0]video_meta.csv -> [B2] frame_id/timestamp ratio -> 30.
# --------------------------------------------------------------------------

def _resolve_fps(video_meta_csv_path=None, dense_tracks_csv_path=None,
                 video_name=None, default=DEFAULT_FPS):
    # Preferred: [B0]video_meta.csv sidecar.
    if video_meta_csv_path and os.path.exists(video_meta_csv_path):
        try:
            meta = pd.read_csv(video_meta_csv_path)
            if "fps" in meta.columns and len(meta) > 0:
                sel = meta
                if video_name is not None and "video_name" in meta.columns:
                    match = meta[meta["video_name"].astype(str) == str(video_name)]
                    if len(match) > 0:
                        sel = match
                fps = float(sel.iloc[0]["fps"])
                if math.isfinite(fps) and fps > 0:
                    return fps
        except Exception:
            pass
    # Fallback: median frame_id/timestamp ratio from [B2] dense tracks.
    if dense_tracks_csv_path and os.path.exists(dense_tracks_csv_path):
        try:
            b2 = pd.read_csv(dense_tracks_csv_path, usecols=["frame_id", "timestamp"])
            b2 = b2[pd.to_numeric(b2["timestamp"], errors="coerce") > 1e-9]
            if len(b2) > 0:
                ratio = float(np.median(
                    pd.to_numeric(b2["frame_id"], errors="coerce")
                    / pd.to_numeric(b2["timestamp"], errors="coerce")
                ))
                if math.isfinite(ratio) and 1.0 <= ratio <= 240.0:
                    return ratio
        except Exception:
            pass
    return float(default)


def _read_csv_or_none(path):
    if not path or not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


# --------------------------------------------------------------------------
# Module entry point.
# --------------------------------------------------------------------------

def run_signal_timing(video_path,
                      crossing_csv_path=None,
                      traffic_light_csv_path=None,
                      video_meta_csv_path=None,
                      dense_tracks_csv_path=None,
                      output_csv_path=None,
                      fps=None):
    """Compute [P10]signal_timing.csv for one video. CSV-only (video not read).

    All inputs default to the standard per-video paths under
    analysis_results/<video_name>/. Any missing input degrades gracefully:
    no [C3] -> header-only output; no/empty [E2] -> rows with
    light_available=False and NaN timing metrics.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join("analysis_results", video_name)

    if crossing_csv_path is None:
        crossing_csv_path = os.path.join(output_dir, "[C3]crossing_judge.csv")
    if traffic_light_csv_path is None:
        traffic_light_csv_path = os.path.join(output_dir, "[E2]traffic_light.csv")
    if video_meta_csv_path is None:
        video_meta_csv_path = os.path.join(output_dir, "[B0]video_meta.csv")
    if dense_tracks_csv_path is None:
        dense_tracks_csv_path = os.path.join(output_dir, "[B2]dense_tracks.csv")
    if output_csv_path is None:
        output_csv_path = os.path.join(output_dir, "[P10]signal_timing.csv")

    out_parent = os.path.dirname(output_csv_path)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)

    crossing_df = _read_csv_or_none(crossing_csv_path)
    if crossing_df is None or len(crossing_df) == 0:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(output_csv_path, index=False)
        print(f"Signal timing results saved to {output_csv_path}")
        return output_csv_path

    e2_df = _read_csv_or_none(traffic_light_csv_path)

    if fps is None or not (isinstance(fps, (int, float)) and math.isfinite(fps) and fps > 0):
        fps = _resolve_fps(video_meta_csv_path, dense_tracks_csv_path,
                           video_name=video_name)
    fps = float(fps)

    rows = compute_signal_timing_rows(crossing_df, e2_df, fps)
    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(output_csv_path, index=False)
    print(f"Signal timing results saved to {output_csv_path}")
    return output_csv_path
