import os
import bisect
import pandas as pd


def _read_csv_or_empty(csv_path, columns):
    """Read an input CSV defensively.

    Upstream env modules run in their own subprocesses, so any of them can fail
    without stopping the pipeline (e.g. missing model weights -> no [E2]/[E3]/
    [E7] written). A missing, empty, or unparseable input must degrade to an
    empty DataFrame with the expected columns -- never a crash -- so this module
    still writes a valid [C1] output.
    """
    if not csv_path or not os.path.exists(csv_path):
        return pd.DataFrame(columns=columns)
    try:
        df = pd.read_csv(csv_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=columns)
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def detect_crossing_risk(
        video_path,
        traffic_light_csv=None,
        crosswalk_csv=None,
        traffic_sign_csv=None,
        crossing_judge_csv=None,
        output_csv_path=None,
        tracked_csv=None
):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join("analysis_results", video_name)
    os.makedirs(output_dir, exist_ok=True)

    if output_csv_path is None:
        output_csv_path = os.path.join(output_dir, "[C1]risky_crossing.csv")
    if traffic_light_csv is None:
        traffic_light_csv = os.path.join(output_dir, "[E2]traffic_light.csv")
    if crosswalk_csv is None:
        crosswalk_csv = os.path.join(output_dir, "[E7]crosswalk_detection.csv")
    if traffic_sign_csv is None:
        traffic_sign_csv = os.path.join(output_dir, "[E3]traffic_sign.csv")
    if crossing_judge_csv is None:
        crossing_judge_csv = os.path.join(output_dir, "[C3]crossing_judge.csv")
    if tracked_csv is None:
        tracked_csv = os.path.join(output_dir, "[B1]tracked_pedestrians.csv")

    # Missing/empty [C3] -> no crossings -> header-only output below.
    df_crossing = _read_csv_or_empty(
        crossing_judge_csv,
        ['track_id', 'crossed', 'started_frame', 'ended_frame']
    )
    crossed_pedestrians = df_crossing[df_crossing['crossed'] == True].copy()

    if crossed_pedestrians.empty:
        columns = [
            'track_id', 'crossed', 'started_frame', 'ended_frame',
            'total_frames', 'risky_frames', 'risky_ratio', 'risk'
        ]
        pd.DataFrame(columns=columns).to_csv(output_csv_path, index=False)
        print(f"No crossed pedestrians found. Empty results saved to {output_csv_path}")
        return

    # [E2] traffic light. The producer writes the literal string 'None' as
    # main_light_color for every no-light sample, and 'None' is in pandas'
    # default na_values, so read_csv hands those cells back as NaN. Without
    # normalization the `light_state == "None"` rule below never fires (a
    # video with no lights at all would classify every frame 'not risky').
    # Normalize every value to 'red'/'green'/'yellow' or the literal 'None',
    # and KEEP the 'None' samples in the map so the forward-fill correctly
    # returns to the unknown state after the last real color sample instead
    # of propagating a stale green.
    df_light = _read_csv_or_empty(traffic_light_csv, ["frame_id", "main_light_color"])
    light_state_map = {}
    for _, row in df_light.iterrows():
        try:
            light_frame_id = int(row["frame_id"])
        except (ValueError, TypeError):
            continue
        color = str(row["main_light_color"]).strip().lower()
        light_state_map[light_frame_id] = color if color in ("red", "green", "yellow") else "None"

    # [E7] crosswalk. str() guards NaN cells (float) that would break .strip().
    df_crosswalk = _read_csv_or_empty(crosswalk_csv, ["frame_id", "crosswalk_detected"])
    crosswalk_map = {}
    for _, row in df_crosswalk.iterrows():
        try:
            crosswalk_frame_id = int(row["frame_id"])
        except (ValueError, TypeError):
            continue
        crosswalk_map[crosswalk_frame_id] = str(row["crosswalk_detected"]).strip().lower() == "yes"

    df_sign = _read_csv_or_empty(traffic_sign_csv, ["frame_id", "sign_classes_1", "sign_classes_2"])
    sign_map = {}
    for _, row in df_sign.iterrows():
        try:
            sign_frame_id = int(row["frame_id"])
        except (ValueError, TypeError):
            continue
        sign_map[sign_frame_id] = {
            "sign_classes_1": str(row["sign_classes_1"]).split(";") if pd.notna(row["sign_classes_1"]) else [],
            "sign_classes_2": str(row["sign_classes_2"]).split(";") if pd.notna(row["sign_classes_2"]) else []
        }

    # The traffic-light and traffic-sign CSVs are sampled only ~once per second, so directly
    # looking up every integer crossing frame would leave almost all frames as 'None'/no-sign
    # and dilute risky_ratio (flipping the risk classification). Forward-fill from the nearest
    # sampled frame at or before each frame, matching how the state persists between samples.
    light_frames_sorted = sorted(light_state_map.keys())
    sign_frames_sorted = sorted(sign_map.keys())

    def light_at(frame_id):
        idx = bisect.bisect_right(light_frames_sorted, frame_id) - 1
        return light_state_map[light_frames_sorted[idx]] if idx >= 0 else "None"

    def signs_at(frame_id):
        idx = bisect.bisect_right(sign_frames_sorted, frame_id) - 1
        return sign_map[sign_frames_sorted[idx]] if idx >= 0 else {"sign_classes_1": [], "sign_classes_2": []}

    # Per-track full-frame span from the tracking CSV, used only as a fallback
    # window when [C3] has no crossing window for a track (guarded: missing file
    # -> empty map -> such a track is skipped rather than crashing).
    df_tracks = _read_csv_or_empty(tracked_csv, ['frame_id', 'track_id'])
    full_track_span = {}
    if not df_tracks.empty:
        for tid, grp in df_tracks.groupby('track_id'):
            full_track_span[tid] = (int(grp['frame_id'].min()), int(grp['frame_id'].max()))

    results = []

    for _, pedestrian in crossed_pedestrians.iterrows():
        track_id = pedestrian['track_id']

        # FIX #17: evaluate risk ONLY over the on-carriageway crossing sub-window
        # (started_frame..ended_frame from [C3]crossing_judge.csv). Spanning the
        # whole track let post-crossing sidewalk walking inflate risky_ratio.
        # Fall back to the full track span only when [C3] has no valid window.
        started_raw = pedestrian['started_frame']
        ended_raw = pedestrian['ended_frame']
        if pd.notna(started_raw) and pd.notna(ended_raw):
            started_frame = int(started_raw)
            ended_frame = int(ended_raw)
        elif track_id in full_track_span:
            started_frame, ended_frame = full_track_span[track_id]
        else:
            # No crossing window and no track geometry available -> skip safely.
            continue

        risky_frame_count = 0
        total_frame_count = ended_frame - started_frame + 1

        for frame_id in range(started_frame, ended_frame + 1):
            light_state = light_at(frame_id)
            crosswalk_present = crosswalk_map.get(frame_id, False)

            signs = signs_at(frame_id)
            sign_classes_1 = signs["sign_classes_1"]
            sign_classes_2 = signs["sign_classes_2"]

            special_not_risky_signs = {"w57", "pg", "i1"}
            special_not_risky_flag = (
                    any(sign in special_not_risky_signs for sign in sign_classes_1) or
                    "Pedestrian Crossing" in sign_classes_2
            )
            special_risky_flag = "p9" in sign_classes_1

            frame_risk = "not risky"

            if special_risky_flag:
                frame_risk = "risky"
            elif light_state == "green":
                frame_risk = "risky"
            elif special_not_risky_flag:
                frame_risk = "not risky"
            else:
                if light_state == "yellow" and not crosswalk_present:
                    frame_risk = "risky"
                elif light_state == "None" and not crosswalk_present:
                    frame_risk = "risky"
                else:
                    frame_risk = "not risky"

            if frame_risk == "risky":
                risky_frame_count += 1

        risky_ratio = risky_frame_count / total_frame_count if total_frame_count > 0 else 0.0
        final_risk = "risky" if risky_ratio > 0.8 else "not risky"

        results.append({
            "track_id": track_id,
            "crossed": True,
            "started_frame": started_frame,
            "ended_frame": ended_frame,
            "total_frames": total_frame_count,
            "risky_frames": risky_frame_count,
            "risky_ratio": round(risky_ratio, 3),
            "risk": final_risk
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv_path, index=False)