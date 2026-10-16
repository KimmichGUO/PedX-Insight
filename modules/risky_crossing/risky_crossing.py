"""Per-crossing risk classification -> [C1]risky_crossing.csv.

CONSTRUCT CHANGE 2026-09-07. The previous rule classified a frame as risky when
`light_state == "None" and not crosswalk_present`, i.e. when NEITHER environment
detector fired. Because the light detector reports a colour in only 0-37% of
samples and the crosswalk detector fires on 0% of frames in 17 of 28 analysed
videos, that branch dominated: median `risky_crossing_ratio` was 0.915 with 9 of
24 videos at exactly 1.000, and the published value correlated with
`traffic_light_prob` at Spearman -0.785 (p = 0.0005, n = 15 dense_v2). It was an
inverse index of detector recall, not a measure of behaviour.

`risky_ratio` is now EVIDENCE-POSITIVE: a frame enters the denominator only when
the signal state was actually observed (a real light colour, or an unambiguous
sign), and enters the numerator only when that observation shows the pedestrian
crossing against traffic. Frames with no observation are `undetermined` and are
excluded, so a silent detector lowers `signal_coverage` instead of inflating risk.

MEASURED EFFECT (recomputed over the 28 analysed videos, 2026-09-07). On a FIXED
sample -- the 19 videos where both the old and new metric are defined -- the
correlation with `traffic_light_prob` falls from -0.522 (p = 0.022) to -0.152
(p = 0.53). The wider -0.656 -> -0.152 figure is not apples-to-apples: part of it
is the correct removal of 5 zero-coverage videos.

The confound is REDUCED, NOT ELIMINATED. Within the comparable `dense_v2` tier it
remains significant: -0.785 -> -0.650 (p = 0.022, n = 12). `signal_coverage` is
itself correlated with detector recall (+0.580, p = 0.003), so a low-coverage
city's ratio rests on a biased subsample. Two consequences:

  * consumers MUST gate on `signal_coverage` / the determined-crossing count;
  * this metric is not yet safe to read as cross-city behaviour. That needs the
    detector threshold recalibration against a labelled ground-truth set, which
    does not exist yet.

The pre-2026-09 construct is retained as `unsignalled_ratio`. It correlates -0.891
(p < 0.0001) with `traffic_light_prob`: that was the confounded part.
"""

import os
import bisect
import pandas as pd
from modules.paths import RESULTS_ROOT


# Signs that make a crossing "protected" (a marked crossing with priority): a protective sign
# overrides a green vehicle light as evidence of risk. Class names of the two [E3] models
# (best_asia.pt = TT100K codes, best_new.pt = long names). Module constants so the JAAD
# benchmark (validation/jaad/score.py) scores exactly what production uses.
#
# 'ip' (TT100K "pedestrian crossing") was missing from the original set {w57, pg, i1}. On the
# 346-clip JAAD benchmark [E3] flags 'ip' on frames JAAD labels with a pedestrian sign with
# 97% precision (453 flagged frames, 99 clips; base rate 29%), while the original set scored
# F1 0.02 (recall 0.01). w57/i1 fire too rarely there (<10 frames) to judge; 'pg' is at
# base rate (precision 0.19, n=42) and is kept only because nothing shows it is wrong.
PROTECTIVE_SIGNS_1 = {"w57", "pg", "i1", "ip"}
PROTECTIVE_SIGNS_2 = {"Pedestrian Crossing"}
RISKY_SIGN_1 = "p9"     # unvalidated: fires on too few JAAD frames to check


def _read_csv_or_empty(csv_path, columns):
    """Read an input CSV defensively, reporting whether it was actually available.

    Upstream env modules run in their own subprocesses, so any of them can fail
    without stopping the pipeline (e.g. missing model weights -> no [E2]/[E3]/
    [E7] written). A missing, empty, or unparseable input must degrade to an
    empty DataFrame with the expected columns -- never a crash -- so this module
    still writes a valid [C1] output.

    Returns (df, available). `available` is False when the producer did not run
    or produced nothing usable. Callers MUST NOT treat an unavailable input as
    evidence of absence: doing so is what made `risky_crossing_ratio` an inverse
    index of detector recall (a failed detector scored as maximum risk). When
    `available` is False the affected frames are classified `undetermined` and
    excluded from the denominator rather than counted as risky.
    """
    if not csv_path or not os.path.exists(csv_path):
        return pd.DataFrame(columns=columns), False
    try:
        df = pd.read_csv(csv_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=columns), False
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df, not df.empty


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
    output_dir = os.path.join(RESULTS_ROOT, video_name)
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

    # Output schema. `risky_ratio`/`risk` are EVIDENCE-POSITIVE (see the module note
    # below): they describe only the frames in which the environment was actually
    # observed. `signal_coverage` says how much of the crossing that was, and
    # `unsignalled_ratio` preserves the pre-2026-09 construct under an honest name.
    OUTPUT_COLUMNS = [
        'track_id', 'crossed', 'started_frame', 'ended_frame',
        'total_frames', 'signal_obs_frames', 'signal_coverage',
        'risky_frames', 'risky_ratio', 'risk',
        'unsignalled_frames', 'unsignalled_ratio', 'env_available',
    ]

    # Missing/empty [C3] -> no crossings -> header-only output below.
    df_crossing, _ = _read_csv_or_empty(
        crossing_judge_csv,
        ['track_id', 'crossed', 'started_frame', 'ended_frame']
    )
    crossed_pedestrians = df_crossing[df_crossing['crossed'] == True].copy()

    if crossed_pedestrians.empty:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(output_csv_path, index=False)
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
    df_light, light_available = _read_csv_or_empty(traffic_light_csv, ["frame_id", "main_light_color"])
    light_state_map = {}
    for _, row in df_light.iterrows():
        try:
            light_frame_id = int(row["frame_id"])
        except (ValueError, TypeError):
            continue
        color = str(row["main_light_color"]).strip().lower()
        light_state_map[light_frame_id] = color if color in ("red", "green", "yellow") else "None"

    # [E7] crosswalk. str() guards NaN cells (float) that would break .strip().
    df_crosswalk, crosswalk_available = _read_csv_or_empty(crosswalk_csv, ["frame_id", "crosswalk_detected"])
    crosswalk_map = {}
    for _, row in df_crosswalk.iterrows():
        try:
            crosswalk_frame_id = int(row["frame_id"])
        except (ValueError, TypeError):
            continue
        crosswalk_map[crosswalk_frame_id] = str(row["crosswalk_detected"]).strip().lower() == "yes"

    df_sign, _ = _read_csv_or_empty(traffic_sign_csv, ["frame_id", "sign_classes_1", "sign_classes_2"])
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
    df_tracks, _ = _read_csv_or_empty(tracked_csv, ['frame_id', 'track_id'])
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

        risky_frame_count = 0        # risky on POSITIVE evidence
        signal_obs_frame_count = 0   # frames where the signal state was actually observed
        unsignalled_frame_count = 0  # neither light nor crosswalk observed (old construct)
        total_frame_count = ended_frame - started_frame + 1

        for frame_id in range(started_frame, ended_frame + 1):
            light_state = light_at(frame_id)
            # A light colour is EVIDENCE. The literal 'None' is not: it means the
            # detector reported no light in this frame, which is indistinguishable
            # from a miss without a labelled ground-truth set (see validation/).
            light_observed = light_available and light_state in ("red", "green", "yellow")
            # [E7] carries one row per frame (carry-forward producer), so presence in
            # the map is a genuine observation; absence means the frame was not covered.
            crosswalk_observed = crosswalk_available and frame_id in crosswalk_map
            crosswalk_present = crosswalk_map.get(frame_id, False)

            signs = signs_at(frame_id)
            sign_classes_1 = signs["sign_classes_1"]
            sign_classes_2 = signs["sign_classes_2"]

            special_not_risky_flag = (
                    any(sign in PROTECTIVE_SIGNS_1 for sign in sign_classes_1) or
                    any(sign in PROTECTIVE_SIGNS_2 for sign in sign_classes_2)
            )
            special_risky_flag = RISKY_SIGN_1 in sign_classes_1

            # --- old construct, retained under an honest name -------------------
            # "No light seen AND no crosswalk seen" describes an environment the
            # detectors did not characterise. It is dominated by detector recall
            # (Spearman -0.785 against traffic_light_prob over the 15 dense videos),
            # so it is NOT published as behaviour any more -- but it is still a
            # useful descriptor, so it keeps its own column.
            if not light_observed and not crosswalk_present:
                unsignalled_frame_count += 1

            # --- evidence-positive risk ----------------------------------------
            # A frame counts toward the denominator only when the signal state was
            # observed, or a sign gives an unambiguous verdict. Everything else is
            # `undetermined` and is excluded rather than scored.
            if special_risky_flag:
                signal_obs_frame_count += 1
                risky_frame_count += 1
            elif light_observed:
                signal_obs_frame_count += 1
                # Vehicle light green -> the pedestrian is crossing against traffic.
                # A protective sign overrides (marked crossing with priority).
                if light_state == "green" and not special_not_risky_flag:
                    risky_frame_count += 1
            # else: no positive evidence either way -> undetermined, not counted.

        if signal_obs_frame_count > 0:
            risky_ratio = risky_frame_count / signal_obs_frame_count
            final_risk = "risky" if risky_ratio > 0.8 else "not risky"
            risky_ratio_out = round(risky_ratio, 3)
        else:
            # No observed signal anywhere in the crossing window. Emit NA, never 0.0
            # and never "not risky" -- a silent detector is not a measurement.
            risky_ratio_out = pd.NA
            final_risk = pd.NA

        results.append({
            "track_id": track_id,
            "crossed": True,
            "started_frame": started_frame,
            "ended_frame": ended_frame,
            "total_frames": total_frame_count,
            "signal_obs_frames": signal_obs_frame_count,
            "signal_coverage": round(signal_obs_frame_count / total_frame_count, 3)
            if total_frame_count > 0 else 0.0,
            "risky_frames": risky_frame_count,
            "risky_ratio": risky_ratio_out,
            "risk": final_risk,
            "unsignalled_frames": unsignalled_frame_count,
            "unsignalled_ratio": round(unsignalled_frame_count / total_frame_count, 3)
            if total_frame_count > 0 else pd.NA,
            "env_available": bool(light_available and crosswalk_available),
        })

    result_df = pd.DataFrame(results, columns=OUTPUT_COLUMNS)
    result_df.to_csv(output_csv_path, index=False)
    determined = int(result_df["risk"].notna().sum())
    print(f"[C1] {len(result_df)} crossings, {determined} with an observed signal state "
          f"({determined / len(result_df):.1%} coverage) -> {output_csv_path}"
          if len(result_df) else f"[C1] no crossings -> {output_csv_path}")