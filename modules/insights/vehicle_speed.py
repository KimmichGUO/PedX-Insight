"""Vehicle metric speed profiles (module [V8], insight Rank 2).

Per-vehicle metric speed time-series statistics from the dense vehicle track dump
[V7]vehicle_tracks.csv: median / 85th-percentile / max speed, split into the
crosswalk band vs mid-block. Mirrors the [S1] pedestrian pipeline:

  ground point -> ego-motion compensation -> smoothing -> metric scale -> stats

* Ground point ((x1+x2)/2, y2) = the vehicle's road-contact point, where the
  ground-plane scale is valid (the bbox centre used by the counter is NOT).

--------------------------------------------------------------------------------
METRIC SCALE  (audit fix: the old lane-width scale was ~5-7x too large)
--------------------------------------------------------------------------------
The scale must be a function of the image row, because px/m on a ground plane
grows linearly with the distance below the horizon:  scale(y) = a*y + b.

The previous implementation took the [V5] lane separation at the BOTTOM row of
the frame and applied it as a single GLOBAL CONSTANT to every vehicle.  That is
wrong twice over, and the two errors multiply:

  1. y-dependence: the frame-bottom row is the closest ground row in the image
     and therefore has the LARGEST px/m of anywhere in the frame.  Vehicles sit
     far higher up (measured median ground row y2 ~ 515-575 on 720p clips), where
     the true scale is ~2x smaller.
  2. absolute width: [V5]'s "lane" is a Hough left/right ROAD-EDGE pair, i.e. the
     whole carriageway, not one 3.5 m lane.  Dividing it by 3.5 m inflates px/m by
     a further ~2-3.7x.

Measured on 11 cities, the resulting constant disagreed with the pedestrian-height
scale at the vehicles' own image row by 4.3x - 7.4x (median 6.0x), which is
exactly the reported "median vehicle speed 0.245-0.441 m/s" defect: dividing a
pixel displacement by a px/m that is 6x too large makes the speed 6x too small.

The scale we can actually defend is the one [S1] already validates: a pedestrian
is ~1.7 m tall, so scale(y) = bbox_height / assumed_height at that pedestrian's
ground row.  Fitting those (ground row, px/m) samples with a Theil-Sen line gives
a full ground-plane map, from the same physics and the same clips as [S1].
Scale priority is therefore:

    1. [S2] ground-plane scale(y), only when quality == "good"      stripe_ground_plane
    2. pedestrian-height plane fit from [B2]/[B1]                   ped_height_plane
    3. [V5] lane geometry as a PLANE, width(y)/3.5 m                lane_width_plane
    4. [V5] lane geometry as the legacy global constant             lane_width
    5. car-length prior (median bbox width / 4.5 m), lateral        length_prior
       tracks of cars/taxis only, never flagged reliable
    6. none

`scale_cross_check_ratio` records lane-scale / pedestrian-scale at the track's own
rows whenever both exist, so the disagreement above stays visible in the output.
Sources 3 and 4 are never flagged reliable when a pedestrian cross-check exists
and rejects them.

--------------------------------------------------------------------------------
CAMERA MOTION  (audit fix: reliable was 0 for all 138,830 tracks)
--------------------------------------------------------------------------------
[B3]'s cam_x/cam_y are the CUMULATIVE integrated camera position - a random walk
that reaches 22,808-89,278 px on a full-length video.  The old gate compared that
cumulative displacement against a 200 px limit, so it fired on every video longer
than a few seconds and forced reliable = False everywhere.  It was a length test,
not a camera test.

Cumulative drift is in fact harmless here: speeds are built from DIFFERENCES of
consecutive ego-compensated samples, so any constant offset cancels.  What does
corrupt a step is local: per-interval registration error (tracked by [B3]'s
step_px) and forward camera translation (radial flow, structurally invisible to
[B3]'s median-translation model - see estimate_ego_expansion in [S1]).  Both are
length-independent.  Binning per-step vehicle speed by step_px on real data shows
the inflation directly (Chicago: 0.19 m/s at step_px <= 0.25 rising monotonically
to 2.10 m/s at step_px 4-8), which is why steps are now kept only while the camera
is still, exactly as [S1] does, and `local_pan_px` measures the ego displacement
over the TRACK'S OWN frame span rather than over the whole video.

--------------------------------------------------------------------------------
CROSSWALK BAND
--------------------------------------------------------------------------------
Crosswalk band = union of [E7] boxes padded by 0.15x per side; step speeds are
attributed by the (image-space) midpoint of each step.  The mechanism works: on
the clips where [E7] actually returns boxes it populates the column (Manila: 950
tracks, Cincinnati: 20).  It is all-NaN elsewhere because [E7] detected ZERO
crosswalks in those videos (0 boxes in 10 of the 11 audited cities), which is an
honest NaN, not a matching bug.  The appended `n_crosswalk_boxes` column makes
that distinction readable straight from the output.

Videos are deleted after analysis: this module is CSV-only and never opens the
video. Missing/empty inputs always yield a valid header-only output CSV.
"""

import ast
import math
import os

import numpy as np
import pandas as pd

from modules.speed.speed_estimation import (
    ASSUMED_LANE_WIDTH_M,
    EGO_SMOOTH_SAMPLES,
    EGO_STATIC_MAX_EXPANSION,
    EGO_STATIC_MAX_STEP_PX,
    _lane_scale_px_per_m,
    _resolve_assumed_height_m,
    _rolling_median,
    estimate_ego_expansion,
)

CAR_LENGTH_M = 4.5
LENGTH_PRIOR_TYPES = {"car", "taxi"}
LATERAL_RATIO = 3.0                 # |net dx| > LATERAL_RATIO * |net dy| for length prior
CROSSWALK_PAD_FRAC = 0.15
MAX_STEP_SPEED_MPS = 50.0
MIN_RELIABLE_STEPS = 15
PAN_LIMIT_PX = 200.0                # now applied to the LOCAL (per-track) ego displacement
MIN_SCALE_PX_PER_M = 0.1            # below this a scale(y) row is degenerate -> skip step
MIN_EGO_STATIC_FRAC = 0.8           # mirrors [S1]: a mostly-driving track is never reliable

# --- pedestrian-height ground-plane fit -------------------------------------------------
PED_MIN_BBOX_H_PX = 20.0
PED_MIN_BBOX_W_PX = 4.0
PED_MIN_ASPECT = 1.4                # upright-person bbox aspect band, as in [S1]
PED_MAX_ASPECT = 5.0
PED_MIN_SAMPLES_PER_TRACK = 5
PED_MAX_HEIGHT_CV = 0.25            # a track whose box height jitters is not a scale sample
PED_MIN_TRACKS = 10                 # below this the plane fit is not trustworthy
PED_PLANE_MAX_PAIRS = 40000
PED_PLANE_SEED = 0

# A lane-derived scale is accepted only if it agrees with the pedestrian-height scale
# within this factor; measured disagreement on the audited cities was 4.3x - 7.4x.
SCALE_AGREEMENT_TOL = 1.5

# Traffic-engineering split: below this a vehicle counts as stopped (queued at the
# signal), so `running_speed_mps` reports the speed while actually moving.
STOPPED_SPEED_MPS = 0.5

OUTPUT_COLUMNS = [
    "track_id", "veh_type", "n_valid_steps", "median_speed_mps", "p85_speed_mps",
    "max_speed_mps", "speed_at_crosswalk_mps", "midblock_speed_mps",
    "scale_source", "camera_moving", "reliable",
    # --- appended by the scale / camera-gate audit fix (never reorder the above) ---
    "running_speed_mps", "stopped_frac", "scale_px_per_m_median",
    "scale_cross_check_ratio", "ego_regime", "ego_static_frac", "local_pan_px",
    "n_crosswalk_boxes",
]

REQUIRED_V7_COLUMNS = {"frame_id", "timestamp", "track_id", "x1", "y1", "x2", "y2"}
PED_TRACK_SOURCES = ("[B2]dense_tracks.csv", "[B1]tracked_pedestrians.csv")


# ---------------------------------------------------------------------------------------
# ground-plane scale helpers
# ---------------------------------------------------------------------------------------

def _theil_sen(x, y, max_pairs=PED_PLANE_MAX_PAIRS, seed=PED_PLANE_SEED):
    """Robust line fit y = a*x + b (median of pairwise slopes, median intercept).

    Sampled rather than exhaustive so the cost stays bounded on large inputs; the
    seed is fixed so the module is deterministic. Returns None when undetermined.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 8:
        return None
    rng = np.random.default_rng(seed)
    i = rng.integers(0, x.size, max_pairs)
    j = rng.integers(0, x.size, max_pairs)
    m = x[i] != x[j]
    if int(m.sum()) < 20:
        return None
    a = float(np.median((y[i][m] - y[j][m]) / (x[i][m] - x[j][m])))
    if not np.isfinite(a):
        return None
    return a, float(np.median(y - a * x))


def fit_ped_height_plane(ped_df, assumed_height_m=1.7):
    """Ground-plane scale(y) = a*y + b in px/m, from pedestrian bbox heights.

    Each usable pedestrian track contributes ONE sample: (median ground row y2,
    median bbox height / assumed_height_m). Per-track medians rather than raw
    detections keep a single long track from dominating the fit.

    Returns (a, b, n_tracks) or (None, None, 0) when the fit is not defensible
    (too few tracks, non-positive slope, or a horizon at/below the samples).
    """
    need = {"track_id", "x1", "y1", "x2", "y2"}
    if ped_df is None or len(ped_df) == 0 or not need.issubset(ped_df.columns):
        return None, None, 0
    d = ped_df[["track_id", "x1", "y1", "x2", "y2"]].astype(
        {"x1": float, "y1": float, "x2": float, "y2": float}).copy()
    d["h"] = d["y2"] - d["y1"]
    d["w"] = d["x2"] - d["x1"]
    d = d[(d["h"] > PED_MIN_BBOX_H_PX) & (d["w"] > PED_MIN_BBOX_W_PX)]
    if d.empty:
        return None, None, 0
    aspect = d["h"] / d["w"]
    d = d[(aspect >= PED_MIN_ASPECT) & (aspect <= PED_MAX_ASPECT)]
    if d.empty:
        return None, None, 0

    g = d.groupby("track_id").agg(h=("h", "median"), y=("y2", "median"),
                                  n=("h", "size"), sd=("h", "std"), mu=("h", "mean"))
    g = g[g["n"] >= PED_MIN_SAMPLES_PER_TRACK]
    if g.empty:
        return None, None, 0
    cv = g["sd"].fillna(0.0) / g["mu"].clip(lower=1e-6)
    g = g[cv < PED_MAX_HEIGHT_CV]
    if len(g) < PED_MIN_TRACKS:
        return None, None, 0

    fit = _theil_sen(g["y"].to_numpy(), (g["h"] / float(assumed_height_m)).to_numpy())
    if fit is None:
        return None, None, 0
    a, b = fit
    if a <= 0:
        return None, None, 0                      # px/m must grow downward in the image
    # The horizon (where scale -> 0) must sit ABOVE the rows the fit is built from.
    # Tested at the 10th percentile, not the minimum: a handful of tracks always land
    # near or above the horizon (tiny far detections, people on steps), and letting
    # those veto the whole fit is what silently dropped whole cities back onto the
    # discredited lane scale. Individual steps on degenerate rows are still skipped
    # later by the MIN_SCALE_PX_PER_M check.
    y_lo = float(np.percentile(g["y"].to_numpy(), 10))
    if a * y_lo + b <= MIN_SCALE_PX_PER_M:
        return None, None, 0
    return a, b, int(len(g))


def fit_lane_width_plane(lane_csv, assumed_lane_width_m=ASSUMED_LANE_WIDTH_M):
    """Ground-plane scale(y) = a*y + b from the [V5] left/right lane lines.

    The two lane lines are straight segments, so their separation is linear in y;
    sampling it along each detected pair and fitting robustly gives the y-dependent
    scale the old frame-bottom constant was missing. Returns (a, b) or None.

    NOTE this only repairs the y-dependence. It cannot repair the assumption that
    the detected separation really is `assumed_lane_width_m` metres wide - see the
    module docstring and `scale_cross_check_ratio`.
    """
    if not (lane_csv and os.path.exists(lane_csv) and os.path.getsize(lane_csv) > 0):
        return None
    try:
        d = pd.read_csv(lane_csv)
    except Exception:
        return None
    cols = ["left_x1", "left_y1", "left_x2", "left_y2",
            "right_x1", "right_y1", "right_x2", "right_y2"]
    if d.empty or not set(cols).issubset(d.columns):
        return None
    u = d[cols].drop_duplicates().to_numpy(dtype=float)
    ys, ss = [], []
    for lx1, ly1, lx2, ly2, rx1, ry1, rx2, ry2 in u:
        # lane_detection writes all-zero coords for an undetected side
        if 0 in (lx1, lx2, rx1, rx2) or ly1 == ly2 or ry1 == ry2:
            continue
        y_lo = min(ly1, ly2, ry1, ry2)
        y_hi = max(ly1, ly2, ry1, ry2)
        if y_hi - y_lo < 1:
            continue
        for y in np.linspace(y_lo, y_hi, 11):
            lx = lx1 + (lx2 - lx1) * (y - ly1) / (ly2 - ly1)
            rx = rx1 + (rx2 - rx1) * (y - ry1) / (ry2 - ry1)
            w = abs(rx - lx)
            if w > 5:
                ys.append(y)
                ss.append(w / float(assumed_lane_width_m))
    # The separation is exactly linear in y, so a single detected pair already
    # determines the line; real clips contribute thousands of pairs and the
    # Theil-Sen median then absorbs the per-frame Hough noise.
    if len(ys) < 10:
        return None
    fit = _theil_sen(np.array(ys), np.array(ss))
    if fit is None or fit[0] <= 0:
        return None
    return fit


def _load_crosswalk_boxes(e7_csv, pad_frac=CROSSWALK_PAD_FRAC):
    """Union of [E7] crosswalk boxes across all frames, padded by pad_frac per side.

    Returns a list of [x1, y1, x2, y2]; empty list when the file is
    missing/empty/malformed (-> crosswalk split reported as NaN downstream).
    """
    if not (e7_csv and os.path.exists(e7_csv) and os.path.getsize(e7_csv) > 0):
        return []
    try:
        d = pd.read_csv(e7_csv)
    except Exception:
        return []
    if d.empty or "crosswalk_boxes" not in d.columns:
        return []
    seen = set()
    boxes = []
    for raw in d["crosswalk_boxes"].dropna():
        try:
            parsed = ast.literal_eval(raw) if isinstance(raw, str) else raw
        except (ValueError, SyntaxError):
            continue
        if not isinstance(parsed, (list, tuple)):
            continue
        for b in parsed:
            try:
                x1, y1, x2, y2 = (float(v) for v in b)
            except (TypeError, ValueError):
                continue
            key = (round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1))
            if key in seen:
                continue
            seen.add(key)
            px = (x2 - x1) * pad_frac
            py = (y2 - y1) * pad_frac
            boxes.append([x1 - px, y1 - py, x2 + px, y2 + py])
    return boxes


def _point_in_boxes(x, y, boxes):
    for bx1, by1, bx2, by2 in boxes:
        if bx1 <= x <= bx2 and by1 <= y <= by2:
            return True
    return False


def _majority_vtype(g):
    if "vtype" not in g.columns:
        return "unknown"
    vals = g["vtype"].dropna().astype(str)
    if vals.empty:
        return "unknown"
    return vals.value_counts().idxmax()


def _ego_static_series(ego_df, ped_df, max_step_px, max_expansion):
    """Build the length-independent camera-still test.

    Returns (flags_fn, ego_regime) where flags_fn(frame_ids) -> bool array, True
    where the camera was still over the interval starting at that frame.

    TWO conditions, because the two camera motions corrupt a step by different
    routes (identical reasoning to [S1].step_is_ego_static):
      * step_px   -> pan / lateral translation, visible to [B3]
      * expansion -> forward drive, radial flow, invisible to [B3]'s median
    Neither looks at the cumulative cam_x/cam_y, so neither grows with video length.
    """
    step_frames = step_px = None
    if ego_df is not None and len(ego_df) > 0 and "step_px" in ego_df.columns:
        e = ego_df.sort_values("frame_id")
        step_frames = e["frame_id"].to_numpy(dtype=float)
        step_px = _rolling_median(e["step_px"].to_numpy(dtype=float), 5)

    exp_frames = exp_static = None
    ego_regime = "unknown"
    exp_map = estimate_ego_expansion(ped_df) if ped_df is not None and len(ped_df) else {}
    if exp_map:
        exp_frames = np.array(sorted(exp_map), dtype=float)
        ev = np.array([exp_map[int(f)] for f in exp_frames], dtype=float)
        win = EGO_SMOOTH_SAMPLES if EGO_SMOOTH_SAMPLES % 2 else EGO_SMOOTH_SAMPLES + 1
        ev_s = _rolling_median(ev, win)
        exp_static = np.abs(ev_s) <= max_expansion
        ego_regime = ("static" if float(np.median(np.abs(ev_s))) <= max_expansion
                      else "forward_motion")

    def flags(frame_ids):
        f = np.asarray(frame_ids, dtype=float)
        ok = np.ones(f.shape, dtype=bool)
        if step_px is not None and step_frames.size:
            j = np.searchsorted(step_frames, f, side="right") - 1
            valid = j >= 0
            ok[valid] &= step_px[j[valid]] <= max_step_px
        if exp_static is not None and exp_frames.size:
            i = np.searchsorted(exp_frames, f, side="right") - 1
            valid = i >= 0
            ok[valid] &= exp_static[i[valid]]
            # before the first anchor: trust only a globally static clip
            ok[~valid] &= (ego_regime == "static")
        return ok

    return flags, ego_regime


def compute_vehicle_speeds(veh_df, ego_df=None, stripe_ab=None, lane_scale_px_per_m=None,
                           crosswalk_boxes=None, max_step_speed_mps=MAX_STEP_SPEED_MPS,
                           smooth_window=3, min_reliable_steps=MIN_RELIABLE_STEPS,
                           pan_limit_px=PAN_LIMIT_PX, car_length_m=CAR_LENGTH_M,
                           ped_plane_ab=None, lane_plane_ab=None, ped_df=None,
                           ego_max_step_px=EGO_STATIC_MAX_STEP_PX,
                           ego_max_expansion=EGO_STATIC_MAX_EXPANSION,
                           min_ego_static_frac=MIN_EGO_STATIC_FRAC,
                           stopped_speed_mps=STOPPED_SPEED_MPS):
    """Pure core: [V7]-shaped dataframe in -> list of per-track result dicts out.

    veh_df columns: frame_id, timestamp, track_id, x1, y1, x2, y2 (+ optional vtype).
    ego_df: [B3]-shaped (frame_id, cam_x, cam_y, step_px) or None.
    stripe_ab: (a, b) of the [S2] scale(y) = a*y + b fit; pass ONLY when quality=="good".
    ped_plane_ab: (a, b) of the pedestrian-height scale(y) fit (fit_ped_height_plane).
    lane_plane_ab: (a, b) of the [V5] lane-geometry scale(y) fit (fit_lane_width_plane).
    lane_scale_px_per_m: legacy global px/m from [V5], or None.
    ped_df: dense pedestrian tracks, used only for the forward-camera indicator.
    crosswalk_boxes: pre-padded [x1, y1, x2, y2] boxes (see _load_crosswalk_boxes).
    """
    crosswalk_boxes = crosswalk_boxes or []
    n_boxes = len(crosswalk_boxes)

    # --- ego motion -------------------------------------------------------------------
    # camera_moving keeps its historical meaning (reported flag only). The camera-still
    # test below is per-step and length-independent; the cumulative cam_x/cam_y series is
    # used ONLY through differences of consecutive samples, where its drift cancels.
    camera_moving = False
    ego_fr = ego_x = ego_y = None
    if ego_df is not None and len(ego_df) > 0:
        e = ego_df.sort_values("frame_id")
        if "step_px" in e.columns:
            camera_moving = bool(e["step_px"].median() >= 1.0)
        ego_fr = e["frame_id"].to_numpy(dtype=float)
        ego_x = e["cam_x"].to_numpy(dtype=float)
        ego_y = e["cam_y"].to_numpy(dtype=float)

    static_flags, ego_regime = _ego_static_series(
        ego_df, ped_df, ego_max_step_px, ego_max_expansion)

    def local_pan(f0, f1):
        """Max ego displacement WITHIN this track's own frame span (length-independent)."""
        if ego_fr is None or ego_fr.size == 0:
            return 0.0
        lo = int(np.searchsorted(ego_fr, f0, side="left"))
        hi = int(np.searchsorted(ego_fr, f1, side="right"))
        lo = max(0, min(lo, ego_fr.size - 1))
        hi = max(lo + 1, min(hi, ego_fr.size))
        xs, ys = ego_x[lo:hi], ego_y[lo:hi]
        if xs.size < 2:
            return 0.0
        return float(np.max(np.hypot(xs - xs[0], ys - ys[0])))

    rows = []
    for track_id, g in veh_df.groupby("track_id"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        if len(g) < 2:
            continue
        vtype = _majority_vtype(g)

        fr = g["frame_id"].to_numpy(dtype=float)
        t = g["timestamp"].to_numpy(dtype=float)
        gx_raw = (g["x1"].to_numpy(dtype=float) + g["x2"].to_numpy(dtype=float)) / 2.0
        gy_raw = g["y2"].to_numpy(dtype=float)          # ground contact row
        w_px = (g["x2"].to_numpy(dtype=float) - g["x1"].to_numpy(dtype=float))

        # Ego-compensate BEFORE smoothing (same rationale as [S1]): the rolling
        # median then attenuates camera jitter and box jitter together. Subtracting
        # is unconditional now - on a still camera the series is near-flat and
        # subtracting it is harmless, while the old all-or-nothing gate left partial
        # pans entirely uncompensated.
        if ego_fr is not None:
            cx = np.interp(fr, ego_fr, ego_x)
            cy = np.interp(fr, ego_fr, ego_y)
        else:
            cx = cy = np.zeros_like(fr)
        foot_x = _rolling_median(gx_raw - cx, smooth_window)
        foot_y = _rolling_median(gy_raw - cy, smooth_window)
        # Image-space series: scale(y) and the crosswalk band live in IMAGE
        # coordinates, so they are evaluated on the UNcompensated point.
        x_img = _rolling_median(gx_raw, smooth_window)
        y_img = _rolling_median(gy_raw, smooth_window)

        # --- scale-source resolution (track level) ---
        # Every plane is evaluated at the track's own median ground row so the
        # priority decision and the cross-check are about the rows we actually use.
        y_ref = float(np.median(y_img))

        def _plane_at(ab, y):
            return ab[0] * y + ab[1] if ab is not None else None

        ped_ref = _plane_at(ped_plane_ab, y_ref)
        lane_ref = _plane_at(lane_plane_ab, y_ref)
        if lane_ref is None and lane_scale_px_per_m is not None:
            lane_ref = float(lane_scale_px_per_m)
        cross_ratio = None
        if ped_ref and ped_ref > MIN_SCALE_PX_PER_M and lane_ref and lane_ref > 0:
            cross_ratio = float(lane_ref / ped_ref)
        lane_agrees = cross_ratio is None or (
            1.0 / SCALE_AGREEMENT_TOL <= cross_ratio <= SCALE_AGREEMENT_TOL)

        plane_ab = None
        const_scale = None
        if stripe_ab is not None:
            scale_source = "stripe_ground_plane"
            plane_ab = stripe_ab
        elif ped_plane_ab is not None:
            # Once the clip has a validated pedestrian plane it is THE scale. A track
            # sitting above its horizon is simply outside the calibrated range: its
            # steps fall out on the MIN_SCALE_PX_PER_M check below and it reports zero
            # valid steps, which is honest. Quietly serving it the lane scale instead
            # would hand back a number we have already measured to be ~4-7x wrong.
            scale_source = "ped_height_plane"
            plane_ab = ped_plane_ab
        elif lane_plane_ab is not None and lane_ref and lane_ref > MIN_SCALE_PX_PER_M:
            scale_source = "lane_width_plane"
            plane_ab = lane_plane_ab
        elif lane_scale_px_per_m is not None and lane_scale_px_per_m > MIN_SCALE_PX_PER_M:
            scale_source = "lane_width"
            const_scale = float(lane_scale_px_per_m)
        else:
            net_dx = abs(gx_raw[-1] - gx_raw[0])
            net_dy = abs(gy_raw[-1] - gy_raw[0])
            med_w = float(np.median(w_px))
            if (vtype in LENGTH_PRIOR_TYPES and net_dx > LATERAL_RATIO * net_dy
                    and med_w > 1.0):
                scale_source = "length_prior"
                const_scale = med_w / car_length_m
            else:
                scale_source = "none"

        # --- per-step speeds -------------------------------------------------------
        step_static = static_flags(fr[:-1]) if len(fr) > 1 else np.zeros(0, dtype=bool)
        speeds, cw_speeds, mb_speeds, used_scales = [], [], [], []
        n_considered = 0
        if scale_source != "none":
            for i in range(len(g) - 1):
                dt = t[i + 1] - t[i]
                if dt <= 0:
                    continue
                if plane_ab is not None:
                    a, b = plane_ab
                    scale = 0.5 * ((a * y_img[i] + b) + (a * y_img[i + 1] + b))
                else:
                    scale = const_scale
                if scale is None or scale <= MIN_SCALE_PX_PER_M:
                    continue
                dxp = foot_x[i + 1] - foot_x[i]
                dyp = foot_y[i + 1] - foot_y[i]
                v = (math.hypot(dxp, dyp) / scale) / dt
                if v > max_step_speed_mps:
                    continue
                n_considered += 1
                # A step measured while the camera translates is contaminated by
                # residual registration error, which inflates it (measured: 0.19 ->
                # 2.10 m/s across the step_px bins on Chicago). Drop it, as [S1] does.
                if i < step_static.size and not step_static[i]:
                    continue
                speeds.append(v)
                used_scales.append(scale)
                mx = 0.5 * (x_img[i] + x_img[i + 1])
                my = 0.5 * (y_img[i] + y_img[i + 1])
                if crosswalk_boxes and _point_in_boxes(mx, my, crosswalk_boxes):
                    cw_speeds.append(v)
                else:
                    mb_speeds.append(v)

        n_steps = len(speeds)
        ego_static_frac = (n_steps / n_considered) if n_considered else 0.0
        pan_px = local_pan(fr[0], fr[-1])
        pan_ok = pan_px <= pan_limit_px
        scale_ok = (scale_source in ("stripe_ground_plane", "ped_height_plane")
                    or (scale_source in ("lane_width_plane", "lane_width") and lane_agrees))
        reliable = bool(n_steps >= min_reliable_steps and scale_ok and pan_ok
                        and ego_static_frac >= min_ego_static_frac)

        moving = [v for v in speeds if v >= stopped_speed_mps]

        def _r(vals, fn, nd=3):
            return round(float(fn(vals)), nd) if len(vals) else None

        rows.append({
            "track_id": track_id,
            "veh_type": vtype,
            "n_valid_steps": n_steps,
            "median_speed_mps": _r(speeds, np.median),
            "p85_speed_mps": _r(speeds, lambda v: np.percentile(v, 85)),
            "max_speed_mps": _r(speeds, np.max),
            "speed_at_crosswalk_mps": _r(cw_speeds, np.median),
            "midblock_speed_mps": _r(mb_speeds, np.median),
            "scale_source": scale_source,
            "camera_moving": camera_moving,
            "reliable": reliable,
            "running_speed_mps": _r(moving, np.median),
            "stopped_frac": round(1.0 - len(moving) / n_steps, 3) if n_steps else None,
            "scale_px_per_m_median": _r(used_scales, np.median),
            "scale_cross_check_ratio": round(cross_ratio, 3) if cross_ratio else None,
            "ego_regime": ego_regime,
            "ego_static_frac": round(ego_static_frac, 3),
            "local_pan_px": round(pan_px, 1),
            "n_crosswalk_boxes": n_boxes,
        })
    return rows


def _read_ped_tracks(output_dir):
    """Dense pedestrian tracks for the scale fit + forward-camera indicator."""
    for name in PED_TRACK_SOURCES:
        p = os.path.join(output_dir, name)
        if not (os.path.exists(p) and os.path.getsize(p) > 0):
            continue
        try:
            d = pd.read_csv(p)
        except Exception as e:
            print(f"[vehicle_speed][warn] {name} unreadable: {e}")
            continue
        if not d.empty and {"track_id", "x1", "y1", "x2", "y2"}.issubset(d.columns):
            return d, name
    return None, None


def run_vehicle_speed(video_path, vehicle_tracks_csv=None, output_csv=None,
                      max_step_speed_mps=MAX_STEP_SPEED_MPS, smooth_window=3,
                      min_reliable_steps=MIN_RELIABLE_STEPS, pan_limit_px=PAN_LIMIT_PX,
                      mapping_csv="mapping.csv"):
    """Entry point (CSV-only; the video file itself is never opened)."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join("analysis_results", video_name)
    os.makedirs(output_dir, exist_ok=True)
    if output_csv is None:
        output_csv = os.path.join(output_dir, "[V8]vehicle_speed.csv")
    if vehicle_tracks_csv is None:
        vehicle_tracks_csv = os.path.join(output_dir, "[V7]vehicle_tracks.csv")

    def _write_empty(msg):
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(output_csv, index=False)
        print(f"[vehicle_speed] {msg} Empty results saved to {output_csv}")
        return output_csv

    if not (os.path.exists(vehicle_tracks_csv) and os.path.getsize(vehicle_tracks_csv) > 0):
        return _write_empty("[V7]vehicle_tracks.csv missing/empty.")
    try:
        veh_df = pd.read_csv(vehicle_tracks_csv)
    except Exception as e:
        return _write_empty(f"[V7] unreadable ({e}).")
    if veh_df.empty or not REQUIRED_V7_COLUMNS.issubset(veh_df.columns):
        return _write_empty("[V7] empty or malformed.")

    # [B3] ego motion (optional)
    ego_df = None
    ego_path = os.path.join(output_dir, "[B3]ego_motion.csv")
    if os.path.exists(ego_path) and os.path.getsize(ego_path) > 0:
        try:
            e = pd.read_csv(ego_path)
            if not e.empty and {"frame_id", "cam_x", "cam_y", "step_px"}.issubset(e.columns):
                ego_df = e
        except Exception as e:
            print(f"[vehicle_speed][warn] ego-motion read failed: {e}")

    # [S2] ground-plane scale, usable only when quality == "good"
    stripe_ab = None
    s2_path = os.path.join(output_dir, "[S2]scale_calibration.csv")
    if os.path.exists(s2_path) and os.path.getsize(s2_path) > 0:
        try:
            sc = pd.read_csv(s2_path)
            if not sc.empty and str(sc.iloc[0].get("quality")) == "good":
                stripe_ab = (float(sc.iloc[0]["a"]), float(sc.iloc[0]["b"]))
        except Exception as e:
            print(f"[vehicle_speed][warn] stripe calibration read failed: {e}")

    # Pedestrian-height ground plane: the physically defensible scale, and the same
    # 1.7 m prior [S1] validates at ~1.4 m/s median crossing speed.
    ped_df, ped_src = _read_ped_tracks(output_dir)
    assumed_height_m, height_source = _resolve_assumed_height_m(video_name, mapping_csv)
    ped_a, ped_b, n_ped_tracks = fit_ped_height_plane(ped_df, assumed_height_m)
    ped_plane_ab = (ped_a, ped_b) if ped_a is not None else None

    lane_csv = os.path.join(output_dir, "[V5]lane_detection.csv")
    lane_plane_ab = fit_lane_width_plane(lane_csv)
    # _lane_scale_px_per_m lives in [S1] and indexes the lane columns unguarded, so a
    # malformed [V5] raises KeyError there. This module must degrade to a header-only
    # CSV rather than crash the run, so the call is contained here.
    try:
        lane_scale = _lane_scale_px_per_m(lane_csv)
    except Exception as e:
        print(f"[vehicle_speed][warn] lane scale unavailable ({e}).")
        lane_scale = None
    crosswalk_boxes = _load_crosswalk_boxes(os.path.join(output_dir, "[E7]crosswalk_detection.csv"))

    rows = compute_vehicle_speeds(
        veh_df, ego_df=ego_df, stripe_ab=stripe_ab, lane_scale_px_per_m=lane_scale,
        crosswalk_boxes=crosswalk_boxes, max_step_speed_mps=max_step_speed_mps,
        smooth_window=smooth_window, min_reliable_steps=min_reliable_steps,
        pan_limit_px=pan_limit_px, ped_plane_ab=ped_plane_ab,
        lane_plane_ab=lane_plane_ab, ped_df=ped_df)

    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out.to_csv(output_csv, index=False)
    n_rel = int(out["reliable"].sum()) if not out.empty else 0
    srcs = out["scale_source"].value_counts().to_dict() if not out.empty else {}
    rel = out[out["reliable"]] if not out.empty else out
    med = rel["median_speed_mps"].median() if len(rel) else float("nan")
    run_med = rel["running_speed_mps"].median() if len(rel) else float("nan")
    ratio = out["scale_cross_check_ratio"].median() if not out.empty else float("nan")
    print(f"[vehicle_speed] {len(out)} vehicle tracks ({n_rel} reliable), "
          f"scale sources={srcs}, ped_plane={'%.4f*y%+.1f' % (ped_a, ped_b) if ped_plane_ab else 'n/a'} "
          f"from {n_ped_tracks} tracks in {ped_src or 'n/a'} ({height_source}, "
          f"{assumed_height_m:.2f} m), lane/ped scale ratio={ratio:.2f}, "
          f"crosswalk_boxes={len(crosswalk_boxes)}, reliable median={med:.2f} m/s "
          f"(running {run_med:.2f} m/s). Saved to {output_csv}")
    return output_csv


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Vehicle metric speed profiles ([V8]).")
    ap.add_argument("--source_video_path", required=True)
    ap.add_argument("--mapping_csv", default="mapping.csv")
    args = ap.parse_args()
    run_vehicle_speed(args.source_video_path, mapping_csv=args.mapping_csv)
