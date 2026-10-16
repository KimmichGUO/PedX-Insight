"""Score PedX outputs against JAAD ground truth.

Frame alignment
---------------
JAAD frames are 0-indexed. PedX producers differ (checked in each producer's code):

  1-indexed (frame_id += 1 BEFORE the first frame is processed, so frame_id 1 = JAAD 0):
    [B1]/[B2] new_track_id_with_imgs.py, [B3] modules/speed/ego_motion.py,
    [E9] modules/sidewalk/sidewalk_detect.py, and everything that copies [B1] frame ids:
    [C3] crossing_judge started/ended_frame, [C1]/[C5] windows.
  0-indexed (frame_id starts at -1 and is incremented per grabbed frame, or a 0-based
  loop): [E1] weather, [E2] traffic_light, [E3] traffic_sign, [E6] daytime,
    [E7] crosswalk, [E10] scene_seg, and the [X7] probe.

`to_jaad_frame(frame_id, code) = frame_id - FRAME_BASE[code]`. Sampling grids: the
tracker records [B1] at 1-indexed multiples of N_b1 = ceil(fps*interval) snapped down to a
multiple of the dense stride round(fps/15) (`tracker_grid`); the 0-indexed detectors
sample at multiples of int(ceil(fps)*interval) (`detector_grid`). At JAAD's 29.97/30/59.94
fps both give N = 30 or 60, so a [B1] row at frame_id 30 is JAAD frame 29 while an [E2] row
at frame_id 30 is JAAD frame 30.

Statistics
----------
Every metric carries n (units scored), n_clips and a 95% percentile CI from a CLUSTER
bootstrap over clips (frames/pedestrians within a clip are not independent): clips are
resampled with replacement, the metric is recomputed on the union of their rows.
B = 2000, seed 20261010. base_rate is the trivial-classifier reference: positive-class
prevalence for precision/recall/F1/PR-AUC, the majority-class share for accuracy, 0.5 for
ROC-AUC and 0 for Cohen's kappa.
"""

import math
import os
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.metrics import (average_precision_score, cohen_kappa_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)

N_BOOT = 2000
SEED = 20261010

FRAME_BASE = {"B1": 1, "B2": 1, "B3": 1, "E9": 1, "C3": 1, "C1": 1, "C5": 1,
              "E1": 0, "E2": 0, "E3": 0, "E6": 0, "E7": 0, "E10": 0, "X7": 0}

FILES = {
    "B0": "[B0]video_meta.csv", "B1": "[B1]tracked_pedestrians.csv",
    "B2": "[B2]dense_tracks.csv", "B3": "[B3]ego_motion.csv", "E1": "[E1]weather.csv",
    "E2": "[E2]traffic_light.csv", "E3": "[E3]traffic_sign.csv", "E6": "[E6]daytime.csv",
    "E7": "[E7]crosswalk_detection.csv", "E9": "[E9]sidewalk_detection.csv",
    "E10": "[E10]scene_segmentation.csv", "C1": "[C1]risky_crossing.csv",
    "C3": "[C3]crossing_judge.csv", "C5": "[C5]red_light_runner.csv",
    "P6": "[P6]age_gender.csv", "P12": "[P12]pose_behavior.csv",
    "S1": "[S1]pedestrian_speed.csv", "X7": "[X7]crosswalk_class_probe.csv",
}

# PedX -> JAAD label maps (documented in the report).
WEATHER_MAP = {"shine": "clear", "sunrise": "clear", "cloudy": "cloudy", "rain": "rain"}
WEATHER_LABELS = ["clear", "cloudy", "rain", "snow"]
DAYTIME_MAP = {"Day": "day", "Evening": "night", "Night": "night"}
DAYTIME_MAP_ALT = {"Day": "day", "Evening": "day", "Night": "night"}
# Age cut-points (years) -> JAAD groups. JAAD publishes the four group names but no
# boundaries; these are the report's assumption, with a sensitivity variant.
AGE_CUTS = [(0, 12, "child"), (13, 19, "young"), (20, 64, "adult"), (65, 200, "senior")]
AGE_CUTS_ALT = [(0, 12, "child"), (13, 24, "young"), (25, 59, "adult"), (60, 200, "senior")]
AGE_LABELS = ["child", "young", "adult", "senior"]
# Signs that [C1] risky_crossing.py treats as protective: imported, so the benchmark always
# scores exactly what production uses. (Before 2026-10 the set lacked 'ip' and scored F1 0.02 on
# JAAD's pedestrian sign; with 'ip' added the same detector scores ~0.7 -- see findings.md.)
from modules.risky_crossing.risky_crossing import (PROTECTIVE_SIGNS_1 as PED_SIGN_CLASSES_1,  # noqa: E402
                                                   PROTECTIVE_SIGNS_2 as PED_SIGN_CLASSES_2)
STOP_SIGN_CLASSES_2 = {"Stop_Sign"}
HEIGHT_BINS = [0, 50, 100, 200, 1e9]
HEIGHT_LABELS = ["<50px", "50-100px", "100-200px", ">=200px"]
MIN_TRACK_COVIS = 3
MIN_TRACK_MEAN_IOU = 0.3
DET_IOU = 0.5
LOOK_WINDOW_S = 5.0          # = modules/insights/pose_behavior.PRE_WINDOW_S


# =============================================================================================
# alignment
# =============================================================================================

def to_jaad_frame(frame_id, code):
    """PedX frame id of producer `code` -> 0-indexed JAAD frame."""
    return np.asarray(frame_id) - FRAME_BASE[code]


def tracker_grid(fps, num_frames, interval=1.0, tracking_fps=15.0):
    """(b1, b2) 1-indexed frame ids the tracker visits (new_track_id_with_imgs.py)."""
    fps = fps if fps and fps > 0 else 30
    every = max(1, math.ceil(fps * interval))
    stride = max(1, round(fps / tracking_fps)) if tracking_fps and tracking_fps > 0 else every
    if tracking_fps and tracking_fps > 0:
        every = max(stride, (every // stride) * stride)
    fids = np.arange(1, int(num_frames) + 1)
    b1 = fids[fids % every == 0]
    b2 = fids[(fids % stride == 0) | (fids % every == 0)]
    return b1, b2


def detector_grid(fps, num_frames, interval=1.0):
    """0-indexed frame ids sampled by the per-frame environment detectors ([E2] pattern)."""
    f = math.ceil(fps) if fps and fps > 0 else 30
    n = max(1, int(f * interval))
    return np.arange(0, int(num_frames), n)


# =============================================================================================
# geometry / matching
# =============================================================================================

def iou_matrix(a, b):
    """Pairwise IoU of boxes a (n,4) and b (m,4) in x1,y1,x2,y2."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def ioa_matrix(pred, region):
    """Intersection over PRED area (for 'people'-group ignore regions)."""
    p = np.asarray(pred, dtype=float).reshape(-1, 4)
    r = np.asarray(region, dtype=float).reshape(-1, 4)
    if len(p) == 0 or len(r) == 0:
        return np.zeros((len(p), len(r)))
    ix1 = np.maximum(p[:, None, 0], r[None, :, 0])
    iy1 = np.maximum(p[:, None, 1], r[None, :, 1])
    ix2 = np.minimum(p[:, None, 2], r[None, :, 2])
    iy2 = np.minimum(p[:, None, 3], r[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_p = np.clip(p[:, 2] - p[:, 0], 0, None) * np.clip(p[:, 3] - p[:, 1], 0, None)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(area_p[:, None] > 0, inter / area_p[:, None], 0.0)


def match_boxes(gt, pred, thr=DET_IOU):
    """One-to-one Hungarian matching maximising IoU; returns [(gi, pj, iou)] with iou>=thr."""
    m = iou_matrix(gt, pred)
    if m.size == 0:
        return []
    w = np.where(m >= thr, m, 0.0)
    r, c = linear_sum_assignment(-w)
    return [(int(i), int(j), float(m[i, j])) for i, j in zip(r, c) if m[i, j] >= thr]


def _pair_table(gt, pred):
    """Per (GT id, PedX id) pair: mean IoU over their CO-VISIBLE frames and the frame count."""
    m = gt[["frame", "id", "x1", "y1", "x2", "y2"]].merge(
        pred[["frame", "id", "x1", "y1", "x2", "y2"]], on="frame", suffixes=("_g", "_p"))
    if m.empty:
        return None
    ix1 = np.maximum(m["x1_g"], m["x1_p"]); iy1 = np.maximum(m["y1_g"], m["y1_p"])
    ix2 = np.minimum(m["x2_g"], m["x2_p"]); iy2 = np.minimum(m["y2_g"], m["y2_p"])
    inter = (ix2 - ix1).clip(lower=0) * (iy2 - iy1).clip(lower=0)
    ag = (m["x2_g"] - m["x1_g"]).clip(lower=0) * (m["y2_g"] - m["y1_g"]).clip(lower=0)
    ap = (m["x2_p"] - m["x1_p"]).clip(lower=0) * (m["y2_p"] - m["y1_p"]).clip(lower=0)
    union = ag + ap - inter
    m["iou"] = np.where(union > 0, inter / union.where(union > 0, 1), 0.0)
    return m.groupby(["id_g", "id_p"])["iou"].agg(["mean", "size"]).reset_index()


def overlap_tracks(gt, pred, min_mean_iou=MIN_TRACK_MEAN_IOU, min_covis=MIN_TRACK_COVIS):
    """ALL PedX tracks overlapping each GT track (not one-to-one): DataFrame gt_id, pred_id,
    mean_iou, n_covis. A person split into several track IDs (an ID switch / fragmentation)
    appears here once per fragment, which the one-to-one `match_tracks` hides."""
    cols = ["gt_id", "pred_id", "mean_iou", "n_covis"]
    if gt is None or pred is None or gt.empty or pred.empty:
        return pd.DataFrame(columns=cols)
    pair = _pair_table(gt, pred)
    if pair is None:
        return pd.DataFrame(columns=cols)
    pair = pair[(pair["size"] >= min_covis) & (pair["mean"] >= min_mean_iou)]
    return pair.rename(columns={"id_g": "gt_id", "id_p": "pred_id", "mean": "mean_iou",
                                "size": "n_covis"})[cols].reset_index(drop=True)


def match_tracks(gt, pred, min_mean_iou=MIN_TRACK_MEAN_IOU, min_covis=MIN_TRACK_COVIS):
    """Match GT pedestrian tracks to PedX tracks within one clip.

    gt, pred: DataFrames with columns frame (JAAD base), id, x1, y1, x2, y2.
    For every (gt, pred) pair the IoU is averaged over their CO-VISIBLE frames (frames on
    which both have a box; a non-overlapping co-visible frame contributes IoU 0). Pairs
    with fewer than `min_covis` co-visible frames are not candidates. One-to-one
    Hungarian assignment maximises the summed mean IoU; assignments below `min_mean_iou`
    are dropped. Returns DataFrame gt_id, pred_id, mean_iou, n_covis.
    """
    cols = ["gt_id", "pred_id", "mean_iou", "n_covis"]
    if gt is None or pred is None or gt.empty or pred.empty:
        return pd.DataFrame(columns=cols)
    pair = _pair_table(gt, pred)
    if pair is None:
        return pd.DataFrame(columns=cols)
    pair = pair[pair["size"] >= min_covis]
    if pair.empty:
        return pd.DataFrame(columns=cols)
    gids = sorted(pair["id_g"].unique(), key=str)
    pids = sorted(pair["id_p"].unique(), key=str)
    gi = {g: i for i, g in enumerate(gids)}
    pj = {p: j for j, p in enumerate(pids)}
    w = np.zeros((len(gids), len(pids)))
    cnt = np.zeros_like(w, dtype=int)
    for g, p, mu, n in pair[["id_g", "id_p", "mean", "size"]].itertuples(index=False):
        w[gi[g], pj[p]] = mu
        cnt[gi[g], pj[p]] = n
    r, c = linear_sum_assignment(-w)
    rows = [(gids[i], pids[j], float(w[i, j]), int(cnt[i, j]))
            for i, j in zip(r, c) if w[i, j] >= min_mean_iou]
    return pd.DataFrame(rows, columns=cols)


# =============================================================================================
# statistics
# =============================================================================================

def cluster_bootstrap(clusters, stat_fn, n_boot=N_BOOT, seed=SEED, alpha=0.05):
    """(point, lo, hi): percentile CI of stat_fn under resampling of whole clusters.

    clusters: per-row cluster id (array-like). stat_fn(idx) -> float computes the metric
    on the rows idx (an int array, may contain repeats). Non-finite replicates (e.g. a
    resample without positives) are dropped.
    """
    cl = np.asarray(clusters)
    n = len(cl)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    point = _f(stat_fn(np.arange(n)))
    order = np.argsort(cl, kind="stable")
    _, starts = np.unique(cl[order], return_index=True)
    groups = np.split(order, starts[1:])
    k = len(groups)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, k, k)
        idx = np.concatenate([groups[i] for i in pick])
        v = _f(stat_fn(idx))
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _div(a, b):
    return a / b if b else float("nan")


def precision(yt, yp):
    yt, yp = np.asarray(yt, bool), np.asarray(yp, bool)
    return _div(np.sum(yt & yp), np.sum(yp))


def recall(yt, yp):
    yt, yp = np.asarray(yt, bool), np.asarray(yp, bool)
    return _div(np.sum(yt & yp), np.sum(yt))


def f1(yt, yp):
    yt, yp = np.asarray(yt, bool), np.asarray(yp, bool)
    tp = np.sum(yt & yp)
    return _div(2 * tp, 2 * tp + np.sum(~yt & yp) + np.sum(yt & ~yp))


def accuracy(yt, yp):
    yt, yp = np.asarray(yt), np.asarray(yp)
    return float(np.mean(yt == yp)) if len(yt) else float("nan")


def kappa(yt, yp, labels=None):
    yt, yp = np.asarray(yt), np.asarray(yp)
    if len(yt) == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        k = cohen_kappa_score(yt, yp, labels=labels)
    return float(k) if np.isfinite(k) else float("nan")


def safe_auc(yt, s):
    yt = np.asarray(yt, bool)
    if yt.all() or (~yt).all():
        return float("nan")
    return float(roc_auc_score(yt, s))


def safe_ap(yt, s):
    yt = np.asarray(yt, bool)
    if not yt.any():
        return float("nan")
    return float(average_precision_score(yt, s))


def majority_share(yt):
    vc = pd.Series(np.asarray(yt)).dropna().value_counts(normalize=True)
    return float(vc.iloc[0]) if len(vc) else float("nan")


def best_f1_threshold(yt, score):
    """Threshold on `score` (predict positive when score >= t) maximising F1.
    Returns (threshold, f1, precision, recall)."""
    yt = np.asarray(yt, bool)
    score = np.asarray(score, dtype=float)
    if not yt.any() or yt.all():
        return float("nan"), float("nan"), float("nan"), float("nan")
    p, r, t = precision_recall_curve(yt, score)
    p, r = p[:-1], r[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    i = int(np.argmax(f))
    return float(t[i]), float(f[i]), float(p[i]), float(r[i])


def cv_threshold_predictions(yt, score, clusters, n_folds=5):
    """Out-of-fold predictions with the F1-optimal threshold chosen on the OTHER clips.
    Clips are assigned to folds by sorted id (deterministic). Returns bool array."""
    yt = np.asarray(yt, bool)
    score = np.asarray(score, dtype=float)
    cl = np.asarray(clusters)
    uniq = np.array(sorted(set(cl.tolist())))
    fold_of = {c: i % n_folds for i, c in enumerate(uniq)}
    folds = np.array([fold_of[c] for c in cl])
    out = np.zeros(len(yt), dtype=bool)
    for k in range(n_folds):
        te = folds == k
        tr = ~te
        if not te.any():
            continue
        t, *_ = best_f1_threshold(yt[tr], score[tr])
        out[te] = score[te] >= t if np.isfinite(t) else False
    return out


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


# =============================================================================================
# result rows
# =============================================================================================

METRIC_COLUMNS = ["task", "metric", "value", "ci_low", "ci_high", "n", "n_clips",
                  "base_rate", "note", "n_pos_clips"]


def mrow(task, metric, ci, n, n_clips, base_rate=float("nan"), note="", n_pos_clips=float("nan")):
    """One metric row. `n_clips` counts clips in the denominator; `n_pos_clips` (binary tasks)
    counts clips that contain at least one positive -- the verdict gate uses the smaller."""
    v, lo, hi = ci
    return {"task": task, "metric": metric, "value": v, "ci_low": lo, "ci_high": hi,
            "n": int(n), "n_clips": int(n_clips), "base_rate": base_rate, "note": note,
            "n_pos_clips": n_pos_clips}


def f1_reference(prevalence):
    """F1 of the trivial 'always positive' classifier: precision p, recall 1 -> 2p/(1+p).

    (Not the prevalence itself, which is the reference for precision and recall.)"""
    return float("nan") if not np.isfinite(prevalence) else 2 * prevalence / (1 + prevalence)


def _pos_clips(yt, clusters):
    yt = np.asarray(yt, bool)
    return len(set(np.asarray(clusters)[yt].tolist())) if len(yt) else 0


def binary_rows(task, yt, yp, clusters, metrics=("precision", "recall", "f1"), note="",
                n_boot=N_BOOT, seed=SEED):
    yt = np.asarray(yt, bool)
    yp = np.asarray(yp, bool)
    cl = np.asarray(clusters)
    fns = {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
           "kappa": lambda a, b: kappa(a, b, labels=[False, True])}
    prev = float(yt.mean()) if len(yt) else float("nan")
    base = {"precision": prev, "recall": prev, "f1": f1_reference(prev),
            "accuracy": majority_share(yt), "kappa": 0.0}
    npc = _pos_clips(yt, cl)
    out = []
    for m in metrics:
        fn = fns[m]
        ci = cluster_bootstrap(cl, lambda idx, fn=fn: fn(yt[idx], yp[idx]), n_boot, seed)
        out.append(mrow(task, m, ci, len(yt), len(set(cl.tolist())), base[m], note, npc))
    return out


def multiclass_rows(task, yt, yp, clusters, labels, note="", n_boot=N_BOOT, seed=SEED):
    yt = np.asarray(yt, dtype=object)
    yp = np.asarray(yp, dtype=object)
    cl = np.asarray(clusters)
    nc = len(set(cl.tolist()))
    acc = cluster_bootstrap(cl, lambda i: accuracy(yt[i], yp[i]), n_boot, seed)
    kap = cluster_bootstrap(cl, lambda i: kappa(yt[i], yp[i], labels=labels), n_boot, seed)
    return [mrow(task, "accuracy", acc, len(yt), nc, majority_share(yt), note),
            mrow(task, "kappa", kap, len(yt), nc, 0.0, note)]


def score_rows(task, yt, score, clusters, note="", n_boot=N_BOOT, seed=SEED, cv=True):
    """ROC-AUC, PR-AUC, in-sample best-F1 threshold, and clip-level cross-validated F1."""
    yt = np.asarray(yt, bool)
    s = np.asarray(score, dtype=float)
    cl = np.asarray(clusters)
    nc = len(set(cl.tolist()))
    prev = float(yt.mean()) if len(yt) else float("nan")
    npc = _pos_clips(yt, cl)
    out = [mrow(task, "roc_auc", cluster_bootstrap(cl, lambda i: safe_auc(yt[i], s[i]), n_boot, seed),
                len(yt), nc, 0.5, note, npc),
           mrow(task, "pr_auc", cluster_bootstrap(cl, lambda i: safe_ap(yt[i], s[i]), n_boot, seed),
                len(yt), nc, prev, note, npc)]
    t, fb, pb, rb = best_f1_threshold(yt, s)
    out.append(mrow(task, "best_f1_threshold", (t, float("nan"), float("nan")), len(yt), nc,
                    float("nan"), "in-sample argmax-F1 threshold on the score"))
    out.append(mrow(task, "f1_at_best_threshold",
                    cluster_bootstrap(cl, lambda i: f1(yt[i], s[i] >= t), n_boot, seed),
                    len(yt), nc, f1_reference(prev), f"in-sample (optimistic); P={pb:.3f} R={rb:.3f}",
                    npc))
    if cv and nc >= 5:
        yp = cv_threshold_predictions(yt, s, cl)
        out.append(mrow(task, "f1_cv_threshold",
                        cluster_bootstrap(cl, lambda i: f1(yt[i], yp[i]), n_boot, seed),
                        len(yt), nc, f1_reference(prev), "threshold chosen on other clips (5-fold by clip)",
                        npc))
    return out


def cm_frame(yt, yp, row_labels, col_labels, row_name="jaad", col_name="pedx"):
    yt = pd.Series(np.asarray(yt, dtype=object)).fillna("NA").astype(str)
    yp = pd.Series(np.asarray(yp, dtype=object)).fillna("NA").astype(str)
    ct = pd.crosstab(yt, yp)
    ct = ct.reindex(index=[str(r) for r in row_labels], columns=[str(c) for c in col_labels],
                    fill_value=0)
    ct.index.name = f"{row_name}\\{col_name}"
    return ct


# =============================================================================================
# I/O
# =============================================================================================

class Runs:
    """Read-only access to PedX outputs under an isolated results root."""

    def __init__(self, root):
        self.root = str(root)
        self._cache = {}

    def path(self, clip, code):
        return os.path.join(self.root, clip, FILES[code])

    def has(self, clip, code):
        return os.path.exists(self.path(clip, code))

    def read(self, clip, code):
        key = (clip, code)
        if key not in self._cache:
            p = self.path(clip, code)
            df = None
            if os.path.exists(p) and os.path.getsize(p) > 0:
                try:
                    df = pd.read_csv(p)
                except pd.errors.EmptyDataError:
                    df = None
            self._cache[key] = df
        return self._cache[key]


def _as_bool(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "yes", "1", "1.0"])


def _light_label(v):
    v = str(v).strip().lower()
    return v if v in ("red", "green", "yellow") else "None"


# =============================================================================================
# tasks
# =============================================================================================

def clips_with(runs, clips, code):
    return [c for c in clips if runs.read(c, code) is not None]


def score_weather(gt, runs, clips):
    """(a) [E1] video-level main class (mode of weather_label, as modules/summary/video_info.py)."""
    task = "a_weather"
    cdf = gt["clips"].set_index("clip")
    recs = []
    for c in clips_with(runs, clips, "E1"):
        df = runs.read(c, "E1")
        if df.empty or "weather_label" not in df.columns:
            continue
        raw = df["weather_label"].mode().iloc[0]
        recs.append({"clip": c, "jaad": cdf.at[c, "weather"], "pedx_raw": raw,
                     "pedx": WEATHER_MAP.get(raw)})
    d = pd.DataFrame(recs, columns=["clip", "jaad", "pedx_raw", "pedx"])
    d = d[d["jaad"].notna()]
    rows, cms = [], {}
    if d.empty:
        return rows, cms, d
    yt = d["jaad"].to_numpy(object)
    yp = d["pedx"].fillna("unknown").to_numpy(object)
    rows += multiclass_rows(task, yt, yp, d["clip"], WEATHER_LABELS + ["unknown"],
                            note="PedX shine/sunrise->clear; JAAD cloud->cloudy; JAAD n/a dropped")
    cl = d["clip"].to_numpy()
    rows += binary_rows(task + "_rain", yt == "rain", yp == "rain", cl,
                        note="'rain' one-vs-rest at clip level")
    rows.append(mrow(task, "pred_rain_share", (float(np.mean(yp == "rain")), np.nan, np.nan),
                     len(yt), len(cl), float(np.mean(yt == "rain")),
                     "base_rate here = JAAD rain share"))
    cms["weather_mapped"] = cm_frame(yt, yp, WEATHER_LABELS, ["clear", "cloudy", "rain", "unknown"])
    cms["weather_raw"] = cm_frame(yt, d["pedx_raw"], WEATHER_LABELS,
                                  ["cloudy", "rain", "shine", "sunrise", "unknown"])
    return rows, cms, d


def score_daynight(gt, runs, clips):
    """(b) [E6] video-level mode of daytime_label vs JAAD time_of_day."""
    task = "b_daynight"
    cdf = gt["clips"].set_index("clip")
    recs = []
    for c in clips_with(runs, clips, "E6"):
        df = runs.read(c, "E6")
        if df.empty or "daytime_label" not in df.columns:
            continue
        raw = df["daytime_label"].mode().iloc[0]
        recs.append({"clip": c, "jaad": cdf.at[c, "day_night"],
                     "jaad_raw": cdf.at[c, "time_of_day_raw"], "pedx_raw": raw,
                     "pedx": DAYTIME_MAP.get(raw), "pedx_alt": DAYTIME_MAP_ALT.get(raw),
                     "day_frac": float((df["daytime_label"] == "Day").mean())})
    d = pd.DataFrame(recs, columns=["clip", "jaad", "jaad_raw", "pedx_raw", "pedx",
                                    "pedx_alt", "day_frac"])
    d = d[d["jaad"].notna()]
    rows, cms = [], {}
    if d.empty:
        return rows, cms, d
    yt = d["jaad"].to_numpy(object)
    rows += multiclass_rows(task, yt, d["pedx"].to_numpy(object), d["clip"], ["day", "night"],
                            note="PedX Evening->night")
    rows += multiclass_rows(task + "_alt", yt, d["pedx_alt"].to_numpy(object), d["clip"],
                            ["day", "night"], note="sensitivity: PedX Evening->day")
    cms["daynight_raw"] = cm_frame(d["jaad_raw"], d["pedx_raw"],
                                   ["daytime", "afternoon", "evening", "nighttime"],
                                   ["Day", "Evening", "Night"])
    return rows, cms, d


def _frame_gt(gt):
    return gt["frames"].set_index(["clip", "frame"])


def light_table(gt, runs, clips):
    """[E2] sampled frames joined to JAAD per-frame traffic_light."""
    fr = gt["frames"]
    parts = []
    for c in clips_with(runs, clips, "E2"):
        df = runs.read(c, "E2")
        if df.empty:
            continue
        d = pd.DataFrame({"clip": c, "frame": to_jaad_frame(df["frame_id"].astype(int), "E2"),
                          "pedx": df["main_light_color"].map(_light_label)})
        parts.append(d)
    if not parts:
        return pd.DataFrame(columns=["clip", "frame", "pedx", "jaad"])
    d = pd.concat(parts, ignore_index=True).merge(
        fr[["clip", "frame", "traffic_light"]], on=["clip", "frame"], how="inner")
    return d.rename(columns={"traffic_light": "jaad"})


def score_traffic_light(gt, runs, clips):
    """(c) presence P/R/F1, state accuracy where both see a light, confusion incl. None."""
    task = "c_traffic_light"
    d = light_table(gt, runs, clips)
    rows, cms = [], {}
    if d.empty:
        return rows, cms, d
    cl = d["clip"].to_numpy()
    yt = (d["jaad"] != "n/a").to_numpy()
    yp = (d["pedx"] != "None").to_numpy()
    rows += binary_rows(task + "_presence", yt, yp, cl, note="frame-level, [E2] sampled frames")
    both = d[yt & yp]
    if not both.empty:
        rows += [mrow(task + "_state", "accuracy",
                      cluster_bootstrap(both["clip"].to_numpy(),
                                        lambda i: accuracy(both["jaad"].to_numpy()[i],
                                                           both["pedx"].to_numpy()[i])),
                      len(both), both["clip"].nunique(), majority_share(both["jaad"]),
                      "frames where both JAAD and PedX see a light; PedX yellow counts wrong")]
    cms["traffic_light"] = cm_frame(d["jaad"], d["pedx"], ["n/a", "red", "green"],
                                    ["None", "red", "green", "yellow"])
    # clip level: any light in the clip
    g = d.groupby("clip").agg(jt=("jaad", lambda s: (s != "n/a").any()),
                              jp=("pedx", lambda s: (s != "None").any()),
                              rt=("jaad", lambda s: (s != "n/a").mean()),
                              rp=("pedx", lambda s: (s != "None").mean())).reset_index()
    rows += binary_rows(task + "_presence_clip", g["jt"], g["jp"], g["clip"],
                        note="clip has any light")
    rows.append(_prevalence_corr(task, g))
    return rows, cms, d


def score_traffic_light_visible(gt, runs, clips):
    """(c2) [E2] restricted to frames where a light is plausibly IN VIEW.

    JAAD's per-frame traffic_light attribute is not a visibility label: in video_0030 it says
    'red' on all 180 frames although no light appears in any frame we inspected. Scoring
    "did PedX see a light" against it therefore blames the detector for lights nobody can see.
    This task keeps the frames where JAAD labels a light AND the independent Vistas segmenter
    ([E10] traffic_light_present) finds a traffic-light region. That is a visibility PROXY
    from a second model, not ground truth, and is reported as such; the rows also quantify how
    many labelled frames fall outside it.
    """
    task = "c_traffic_light_visible"
    d = light_table(gt, runs, clips)
    if d.empty:
        return [], {}, d
    parts = []
    for c in clips_with(runs, clips, "E10"):
        e = runs.read(c, "E10")
        if e is None or e.empty:
            continue
        parts.append(pd.DataFrame({
            "clip": c, "frame": to_jaad_frame(e["frame_id"].astype(int), "E10"),
            "seg_light": e["traffic_light_present"].astype(int).to_numpy() == 1}))
    if not parts:
        return [], {}, d
    d = d.merge(pd.concat(parts, ignore_index=True), on=["clip", "frame"], how="inner")
    if d.empty:
        return [], {}, d
    lab = (d["jaad"] != "n/a").to_numpy()
    seg = d["seg_light"].to_numpy()
    det = (d["pedx"] != "None").to_numpy()
    cl = d["clip"].to_numpy()
    rows = []

    lab_idx = np.flatnonzero(lab)
    if len(lab_idx):
        rows.append(mrow(
            task + "_label_validity", "labelled_frames_with_no_light_in_view",
            cluster_bootstrap(cl[lab_idx], lambda i, ix=lab_idx: float((~seg[ix][i]).mean())),
            len(lab_idx), len(set(cl[lab_idx])), np.nan,
            "share of JAAD traffic_light-labelled frames where [E10] sees no light region"))

    vis = lab & seg
    vi = np.flatnonzero(vis)
    if len(vi):
        rows.append(mrow(
            task + "_presence", "recall",
            cluster_bootstrap(cl[vi], lambda i, ix=vi: float(det[ix][i].mean())),
            len(vi), len(set(cl[vi])), 1.0,
            "frames with a JAAD light label AND a light region in view; [E2] reports any colour"))
        both = vis & det
        bi = np.flatnonzero(both)
        if len(bi):
            jt, pt = d["jaad"].to_numpy(), d["pedx"].to_numpy()
            rows.append(mrow(
                task + "_state", "accuracy",
                cluster_bootstrap(cl[bi], lambda i, ix=bi: float((jt[ix][i] == pt[ix][i]).mean())),
                len(bi), len(set(cl[bi])), majority_share(jt[bi]),
                "colour agreement where [E2] reports a colour on a visible, labelled light"))
    nolight = (~lab) & (~seg)
    ni = np.flatnonzero(nolight)
    if len(ni):
        rows.append(mrow(
            task + "_no_light", "false_positive_rate",
            cluster_bootstrap(cl[ni], lambda i, ix=ni: float(det[ix][i].mean())),
            len(ni), len(set(cl[ni])), 0.0,
            "[E2] reports a colour although JAAD says n/a and no light region is in view"))
    return rows, {}, d


def _prevalence_corr(task, g):
    """Spearman between per-clip PedX positive rate (rp) and JAAD positive rate (rt)."""
    rt, rp = g["rt"].to_numpy(float), g["rp"].to_numpy(float)
    cl = g["clip"].to_numpy()
    return mrow(task, "clip_rate_spearman",
                cluster_bootstrap(cl, lambda i: spearman(rp[i], rt[i])), len(g), len(g), 0.0,
                "per-clip detected-frame rate vs JAAD labelled-frame rate")


def crosswalk_table(gt, runs, clips, interval=1.0):
    """JAAD ped_crossing on the 0-indexed detector grid with [E7], [E10], [X7] columns."""
    cdf = gt["clips"].set_index("clip")
    fr = gt["frames"][["clip", "frame", "ped_crossing"]]
    parts = []
    for c in clips:
        fps = cdf.at[c, "fps"]
        grid = detector_grid(fps, cdf.at[c, "num_frames"], interval)
        d = pd.DataFrame({"clip": c, "frame": grid})
        e7 = runs.read(c, "E7")
        if e7 is not None and not e7.empty:
            m = dict(zip(to_jaad_frame(e7["frame_id"].astype(int), "E7"),
                         e7["crosswalk_detected"].astype(str).str.lower() == "yes"))
            d["e7"] = d["frame"].map(m)
        e10 = runs.read(c, "E10")
        if e10 is not None and not e10.empty:
            f10 = to_jaad_frame(e10["frame_id"].astype(int), "E10")
            d["e10_present"] = d["frame"].map(dict(zip(f10, _as_bool(e10["crosswalk_present"]))))
            d["e10_frac"] = d["frame"].map(dict(zip(f10, e10["crosswalk_frac"].astype(float))))
        x7 = runs.read(c, "X7")
        if x7 is not None and not x7.empty:
            f7 = to_jaad_frame(x7["frame_id"].astype(int), "X7")
            d["x7_cls4"] = d["frame"].map(dict(zip(f7, x7["conf_cls4"].astype(float))))
            d["x7_cls0"] = d["frame"].map(dict(zip(f7, x7["conf_cls0"].astype(float))))
        parts.append(d)
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts, ignore_index=True).merge(fr, on=["clip", "frame"], how="inner")
    for col in ("e7", "e10_present", "e10_frac", "x7_cls4", "x7_cls0"):
        if col not in d.columns:
            d[col] = np.nan
    return d


def majority_smooth(flags, window=5):
    """modules/crosswalk/crosswalk.py smoothing: strict majority in a centred window."""
    flags = np.asarray(flags, bool)
    n = len(flags)
    half = window // 2
    out = np.zeros(n, dtype=bool)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = flags[lo:hi].sum() * 2 > (hi - lo)
    return out


def score_crosswalk(gt, runs, clips, interval=1.0):
    """(d) [E7] vs [E10] vs [X7] crosswalk presence against JAAD ped_crossing."""
    task = "d_crosswalk"
    d = crosswalk_table(gt, runs, clips, interval)
    rows, cms, extra = [], {}, {}
    if d.empty:
        return rows, cms, d
    d["jaad"] = d["ped_crossing"].astype(int) == 1

    def clip_level(sub, col_pred):
        g = sub.groupby("clip").agg(jt=("jaad", "any"), jp=(col_pred, "any"),
                                    rt=("jaad", "mean"), rp=(col_pred, "mean")).reset_index()
        return g

    sources = []
    s7 = d[d["e7"].notna()].copy()
    if not s7.empty:
        s7["pred"] = s7["e7"].astype(bool)
        sources.append(("E7", s7, "pred", "production [E7] crosswalk_detected (smoothed)"))
    s10 = d[d["e10_present"].notna()].copy()
    if not s10.empty:
        s10["pred"] = s10["e10_present"].astype(bool)
        sources.append(("E10", s10, "pred", "[E10] crosswalk_present at shipped threshold"))
    sx = d[d["x7_cls4"].notna()].sort_values(["clip", "frame"]).copy()
    if not sx.empty:
        sx["pred"] = np.concatenate([majority_smooth(g["x7_cls4"].to_numpy() >= 0.35)
                                     for _, g in sx.groupby("clip", sort=True)])
        sources.append(("X7cls4", sx, "pred",
                        "COUNTERFACTUAL: same model, class 4, conf>=0.35 + module smoothing"))
    for name, sub, col, note in sources:
        cl = sub["clip"].to_numpy()
        rows += binary_rows(f"{task}_{name}_frame", sub["jaad"], sub[col], cl, note=note)
        g = clip_level(sub, col)
        rows += binary_rows(f"{task}_{name}_clip", g["jt"], g["jp"], g["clip"],
                            note="clip positive if any sampled frame positive")
        rows.append(_prevalence_corr(f"{task}_{name}", g))
        rows.append(mrow(f"{task}_{name}_frame", "pred_positive_rate",
                         cluster_bootstrap(cl, lambda i, s=sub, c=col: float(s[c].to_numpy()[i].mean())),
                         len(sub), len(g), float(sub["jaad"].mean()),
                         "share of sampled frames flagged; base_rate = JAAD share"))
        cms[f"crosswalk_{name}"] = cm_frame(sub["jaad"].map({True: "crossing", False: "none"}),
                                            sub[col].astype(bool).map({True: "yes", False: "no"}),
                                            ["crossing", "none"], ["yes", "no"])
    if not s10.empty:
        rows += score_rows(f"{task}_E10_frac", s10["jaad"], s10["e10_frac"], s10["clip"],
                           note="crosswalk_frac as a continuous score")
    if not sx.empty:
        rows += score_rows(f"{task}_X7cls4_conf", sx["jaad"], sx["x7_cls4"], sx["clip"],
                           note="COUNTERFACTUAL class-4 max confidence (raw, unsmoothed)")
        rows += score_rows(f"{task}_X7cls0_conf", sx["jaad"], sx["x7_cls0"], sx["clip"],
                           note="class-0 ('Flexible_delineator_post') max confidence = what [E7] thresholds",
                           cv=False)
        # probe sanity: class-0 @0.35 + smoothing must reproduce [E7]
        both = sx[sx["e7"].notna()].sort_values(["clip", "frame"])
        if not both.empty:
            rep = np.concatenate([majority_smooth(g["x7_cls0"].to_numpy() >= 0.35)
                                  for _, g in both.groupby("clip", sort=True)])
            agree = float(np.mean(rep == both["e7"].astype(bool).to_numpy()))
            rows.append(mrow(task + "_X7_probe_check", "agreement_with_E7",
                             (agree, np.nan, np.nan), len(both), both["clip"].nunique(), np.nan,
                             "probe class-0 reconstruction vs actual [E7] (should be ~1.0)"))
    # head-to-head on common frames: F1(E10) - F1(E7)
    hh = d[d["e7"].notna() & d["e10_present"].notna()]
    if not hh.empty:
        yt = hh["jaad"].to_numpy()
        a = hh["e7"].astype(bool).to_numpy()
        b = hh["e10_present"].astype(bool).to_numpy()
        cl = hh["clip"].to_numpy()
        for m, fn in (("f1", f1), ("recall", recall), ("precision", precision)):
            rows.append(mrow(task + "_E10_minus_E7", f"delta_{m}",
                             cluster_bootstrap(cl, lambda i, fn=fn: fn(yt[i], b[i]) - fn(yt[i], a[i])),
                             len(hh), hh["clip"].nunique(), 0.0,
                             "paired on common frames; CI excluding 0 = significant"))
    return rows, cms, d


def gt_box_index(gt, clips):
    b = gt["boxes"]
    return {c: g for c, g in b[b["clip"].isin(clips)].groupby("clip")}


def detection_eval(gt_boxes_clip, pred_clip, frames, iou_thr=DET_IOU):
    """Per-frame Hungarian matching on `frames` (JAAD base).

    gt_boxes_clip: JAAD boxes of one clip (all kinds); pred_clip: PedX boxes with JAAD
    'frame'. 'p' (people-group) boxes are not GT; unmatched predictions with IoA >= 0.5
    inside a 'p' box are ignored (neither TP nor FP). Returns (gt_records, n_fp, n_pred).
    """
    gtc = gt_boxes_clip[gt_boxes_clip["frame"].isin(frames)]
    g_by = {f: g for f, g in gtc.groupby("frame")}
    p_by = {f: g for f, g in pred_clip[pred_clip["frame"].isin(frames)].groupby("frame")}
    recs, n_fp, n_pred = [], 0, 0
    for f in frames:
        g = g_by.get(f)
        p = p_by.get(f)
        gpos = g[g["kind"] != "p"] if g is not None else None
        gign = g[g["kind"] == "p"] if g is not None else None
        pb = p[["x1", "y1", "x2", "y2"]].to_numpy(float) if p is not None else np.zeros((0, 4))
        gb = gpos[["x1", "y1", "x2", "y2"]].to_numpy(float) if gpos is not None else np.zeros((0, 4))
        matches = match_boxes(gb, pb, iou_thr)
        mg = {i for i, _, _ in matches}
        mp = {j for _, j, _ in matches}
        if gpos is not None:
            for k, r in enumerate(gpos.itertuples(index=False)):
                recs.append((f, r.ped_id, r.kind, r.occlusion, r.y2 - r.y1, k in mg))
        unmatched = [j for j in range(len(pb)) if j not in mp]
        if unmatched and gign is not None and len(gign):
            ioa = ioa_matrix(pb[unmatched], gign[["x1", "y1", "x2", "y2"]].to_numpy(float))
            unmatched = [j for j, row in zip(unmatched, ioa) if row.max() < 0.5]
        n_fp += len(unmatched)
        n_pred += len(pb)
    return recs, n_fp, n_pred


def score_detection(gt, runs, clips, interval=1.0):
    """(e) pedestrian detection recall/precision at IoU>=0.5 for [B1] (and dense [B2])."""
    cdf = gt["clips"].set_index("clip")
    gidx = gt_box_index(gt, clips)
    rows, cms = [], {}
    tables = {}
    for code, task in (("B1", "e_detection_B1"), ("B2", "e_detection_B2")):
        recs_all, per_clip = [], []
        for c in clips_with(runs, clips, code):
            pr = runs.read(c, code)
            b1, b2 = tracker_grid(cdf.at[c, "fps"], cdf.at[c, "num_frames"], interval)
            frames = to_jaad_frame(b1 if code == "B1" else b2, code)
            pred = pr.assign(frame=to_jaad_frame(pr["frame_id"].astype(int), code))
            g = gidx.get(c, pd.DataFrame(columns=gt["boxes"].columns))
            recs, n_fp, n_pred = detection_eval(g, pred, set(frames.tolist()))
            n_tp = sum(1 for r in recs if r[5])
            per_clip.append({"clip": c, "tp": n_tp, "n_gt": len(recs), "fp": n_fp,
                             "n_pred": n_pred})
            recs_all += [(c,) + r for r in recs]
        if not per_clip:
            continue
        rec = pd.DataFrame(recs_all, columns=["clip", "frame", "ped_id", "kind", "occlusion",
                                              "height", "matched"])
        pc = pd.DataFrame(per_clip)
        tables[code] = rec
        cl = pc["clip"].to_numpy()
        tp, ng, fp = pc["tp"].to_numpy(), pc["n_gt"].to_numpy(), pc["fp"].to_numpy()
        rows.append(mrow(task, "recall", cluster_bootstrap(cl, lambda i: _div(tp[i].sum(), ng[i].sum())),
                         ng.sum(), len(pc), np.nan, "all non-'p' JAAD boxes on the grid"))
        rows.append(mrow(task, "precision_lower_bound",
                         cluster_bootstrap(cl, lambda i: _div(tp[i].sum(), tp[i].sum() + fp[i].sum())),
                         int(tp.sum() + fp.sum()), len(pc), np.nan,
                         "JAAD does not box every pedestrian -> precision is a LOWER bound"))
        if not rec.empty:
            rc = rec["clip"].to_numpy()
            mt = rec["matched"].to_numpy(bool)
            for col, levels, label in (("occlusion", ["none", "part", "full"], "occlusion"),
                                       ("hbin", HEIGHT_LABELS, "height"),
                                       ("kind", ["b", "ped"], "kind")):
                if col == "hbin":
                    vals = pd.cut(rec["height"], HEIGHT_BINS, right=False,
                                  labels=HEIGHT_LABELS).astype(str).to_numpy()
                else:
                    vals = rec[col].astype(str).to_numpy()
                for lv in levels:
                    sel = vals == lv
                    if not sel.any():
                        continue
                    rows.append(mrow(task, f"recall_{label}={lv}",
                                     cluster_bootstrap(rc[sel], lambda i, m=mt[sel]: float(m[i].mean())),
                                     int(sel.sum()), len(set(rc[sel].tolist())), np.nan,
                                     f"recall stratified by GT {label}"))
            # Combined sub-populations. These were defined AFTER inspecting interim results, so
            # they are exploratory: the verdict's primary population is the a-priori
            # recall_kind=b (JAAD's behaviourally annotated, road-interacting pedestrians).
            hh = rec["height"].to_numpy(float)
            oc = rec["occlusion"].astype(str).to_numpy()
            for label, sel in (("unoccluded_h>=50px", (oc == "none") & (hh >= 50)),
                               ("unoccluded_h>=100px", (oc == "none") & (hh >= 100)),
                               ("not_fully_occluded_h>=50px", (oc != "full") & (hh >= 50))):
                if sel.any():
                    rows.append(mrow(task, f"recall_{label}",
                                     cluster_bootstrap(rc[sel], lambda i, m=mt[sel]: float(m[i].mean())),
                                     int(sel.sum()), len(set(rc[sel].tolist())), np.nan,
                                     "combined GT sub-population (exploratory: chosen after interim results)"))
    return rows, cms, tables


def track_matches(gt, runs, clips, mode="match"):
    """Per clip: GT (non-'p') tracks matched to PedX tracks ([B2] dense, fallback [B1]).

    mode='match': one-to-one Hungarian assignment (`match_tracks`).
    mode='overlap': every overlapping PedX track (`overlap_tracks`), for fragmentation and for
    'was the person flagged in ANY of their tracks' end-to-end scoring."""
    fn = match_tracks if mode == "match" else overlap_tracks
    gidx = gt_box_index(gt, clips)
    out = []
    for c in clips:
        code = "B2" if runs.read(c, "B2") is not None and not runs.read(c, "B2").empty else "B1"
        pr = runs.read(c, code)
        if pr is None or c not in gidx:
            continue
        g = gidx[c]
        g = g[g["kind"] != "p"].rename(columns={"ped_id": "id"})
        p = pr.assign(frame=to_jaad_frame(pr["frame_id"].astype(int), code),
                      id=pr["track_id"])
        m = fn(g, p)
        if not m.empty:
            m.insert(0, "clip", c)
            m["source"] = code
            out.append(m)
    if not out:
        return pd.DataFrame(columns=["clip", "gt_id", "pred_id", "mean_iou", "n_covis", "source"])
    return pd.concat(out, ignore_index=True)


def _truthy(v):
    return str(v).strip().lower() in ("true", "yes", "1", "1.0")


def score_crossing(gt, runs, clips, matches, fps_by_clip, overlaps=None):
    """(f) [C3] crossed vs JAAD crossing attribute, with the choices the audit flagged made explicit.

    Population: behavioural pedestrians ('b') labelled crossing 1 (crosses) or 0 (does not).
      f_crossing                     one-to-one matched tracks only
      f_crossing_incl_irrelevant     matched; JAAD crossing = -1 ('irrelevant') counted as NOT crossing
      f_crossing_any_track           END TO END over every labelled pedestrian: flagged if ANY PedX track
                                     overlapping them is crossed (robust to ID fragmentation); a
                                     pedestrian with no overlapping track counts as not crossed
      f_crossing_end_to_end          one-to-one matching, unmatched counted as not crossed
      f_track_fragmentation          PedX track ids overlapping each GT pedestrian
    A matched track with no [C3] row (never judged) is scored 'not crossed'.
    """
    task = "f_crossing"
    peds = gt["peds"]
    c3_clips = set(clips_with(runs, clips, "C3"))
    beh = peds[(peds["kind"] == "b") & peds["clip"].isin(c3_clips)]
    beh = beh[beh["crossing"].isin([-1, 0, 1])]
    rows, cms = [], {}
    if beh.empty:
        return rows, cms, pd.DataFrame()
    mm = matches.set_index(["clip", "gt_id"])
    ov = {}
    if overlaps is not None and not overlaps.empty:
        ov = overlaps.groupby(["clip", "gt_id"])["pred_id"].apply(list).to_dict()

    def c3_row(clip, tid):
        c3 = runs.read(clip, "C3")
        if c3 is None or c3.empty:
            return None
        hit = c3[c3["track_id"] == tid]
        return None if hit.empty else hit.iloc[0]

    recs = []
    for r in beh.itertuples(index=False):
        key = (r.clip, r.ped_id)
        matched = key in mm.index
        crossed, start = False, np.nan
        tid = mm.loc[key, "pred_id"] if matched else None
        if matched:
            row = c3_row(r.clip, tid)
            if row is not None:
                crossed = _truthy(row["crossed"])
                sf = row["started_frame"]
                start = float(to_jaad_frame(sf, "C3")) if pd.notna(sf) else np.nan
        tids = ov.get(key, [])
        any_crossed = False
        for t in tids:
            row = c3_row(r.clip, t)
            if row is not None and _truthy(row["crossed"]):
                any_crossed = True
                break
        recs.append({"clip": r.clip, "ped_id": r.ped_id, "crossing_raw": int(r.crossing),
                     "jaad": int(r.crossing) == 1, "matched": matched, "track_id": tid,
                     "pedx": crossed, "pedx_start_frame": start, "crossing_point": r.crossing_point,
                     "any_crossed": any_crossed, "n_overlap_ids": len(tids)})
    d_all = pd.DataFrame(recs)
    d = d_all[d_all["crossing_raw"].isin([0, 1])].reset_index(drop=True)
    if d.empty:
        return rows, cms, d_all
    cl_all = d["clip"].to_numpy()
    mt = d["matched"].to_numpy(bool)
    rows.append(mrow(task, "match_rate", cluster_bootstrap(cl_all, lambda i: float(mt[i].mean())),
                     len(d), d["clip"].nunique(), np.nan,
                     f"GT behavioural peds with a PedX track (mean IoU>={MIN_TRACK_MEAN_IOU}, "
                     f">={MIN_TRACK_COVIS} co-visible frames)"))
    m = d[d["matched"]]
    if not m.empty:
        rows += binary_rows(task, m["jaad"], m["pedx"], m["clip"],
                            metrics=("precision", "recall", "f1", "accuracy", "kappa"),
                            note="matched pedestrians only (one-to-one assignment)")
        cms["crossing_matched"] = cm_frame(m["jaad"].map({True: "crossed", False: "not"}),
                                           m["pedx"].map({True: "crossed", False: "not"}),
                                           ["crossed", "not"], ["crossed", "not"])
        tp = m[m["jaad"] & m["pedx"] & m["pedx_start_frame"].notna() & (m["crossing_point"] >= 0)]
        if not tp.empty:
            fps = tp["clip"].map(fps_by_clip).to_numpy(float)
            err = (tp["pedx_start_frame"].to_numpy(float) - tp["crossing_point"].to_numpy(float)) / fps
            ec = tp["clip"].to_numpy()
            rows.append(mrow(task, "onset_error_median_s",
                             cluster_bootstrap(ec, lambda i: float(np.median(err[i]))), len(tp),
                             tp["clip"].nunique(), np.nan,
                             "PedX started_frame minus JAAD crossing_point (s), true positives"))
            rows.append(mrow(task, "onset_abs_error_median_s",
                             cluster_bootstrap(ec, lambda i: float(np.median(np.abs(err[i])))),
                             len(tp), tp["clip"].nunique(), np.nan, ""))
    mi = d_all[d_all["matched"]]
    if not mi.empty:
        rows += binary_rows(task + "_incl_irrelevant", mi["jaad"], mi["pedx"], mi["clip"],
                            metrics=("precision", "recall", "f1", "kappa"),
                            note="matched; JAAD crossing=-1 ('irrelevant') counted as NOT crossing")
    rows += binary_rows(task + "_end_to_end", d["jaad"], d["pedx"] & d["matched"], cl_all,
                        metrics=("precision", "recall", "f1"),
                        note="one-to-one matching; unmatched GT counted as 'not crossed'")
    rows += binary_rows(task + "_any_track", d["jaad"], d["any_crossed"], cl_all,
                        metrics=("precision", "recall", "f1", "kappa"),
                        note="all labelled pedestrians; flagged if ANY overlapping PedX track is "
                             "crossed; no overlapping track = not crossed")
    seen = d_all[d_all["n_overlap_ids"] >= 1]
    if not seen.empty:
        k = seen["n_overlap_ids"].to_numpy(float)
        sc = seen["clip"].to_numpy()
        rows.append(mrow("f_track_fragmentation", "ids_per_gt_pedestrian_mean",
                         cluster_bootstrap(sc, lambda i: float(k[i].mean())), len(seen),
                         seen["clip"].nunique(), 1.0,
                         "PedX track ids overlapping a GT behavioural pedestrian; 1.0 = no ID splits"))
        rows.append(mrow("f_track_fragmentation", "share_with_2plus_ids",
                         cluster_bootstrap(sc, lambda i: float((k[i] >= 2).mean())), len(seen),
                         seen["clip"].nunique(), 0.0, "share of GT pedestrians split over >=2 PedX ids"))
    return rows, cms, d_all


def ego_table(gt, runs, clips):
    """[B3] rows with JAAD vehicle action and the production static-camera gate signals."""
    from modules.speed.speed_estimation import (EGO_SMOOTH_SAMPLES, EGO_STATIC_MAX_EXPANSION,
                                                EGO_STATIC_MAX_STEP_PX, _rolling_median,
                                                estimate_ego_expansion)
    fr = gt["frames"][["clip", "frame", "vehicle_action", "ego_stopped"]]
    parts = []
    for c in clips_with(runs, clips, "B3"):
        e = runs.read(c, "B3")
        if e.empty:
            continue
        e = e.sort_values("frame_id").reset_index(drop=True)
        fids = e["frame_id"].to_numpy(float)
        step = e["step_px"].to_numpy(float)
        step_med = _rolling_median(step, 5)               # as run_speed_estimation
        dense = runs.read(c, "B2")
        exp_map = estimate_ego_expansion(dense) if dense is not None else {}
        exp_val = np.full(len(e), np.nan)
        exp_static = np.full(len(e), np.nan)
        if exp_map:
            ef = np.array(sorted(exp_map), dtype=float)
            ev = np.array([exp_map[int(f)] for f in ef], dtype=float)
            win = EGO_SMOOTH_SAMPLES if EGO_SMOOTH_SAMPLES % 2 else EGO_SMOOTH_SAMPLES + 1
            ev_s = _rolling_median(ev, win)
            regime_static = float(np.median(np.abs(ev_s))) <= EGO_STATIC_MAX_EXPANSION
            flags = np.abs(ev_s) <= EGO_STATIC_MAX_EXPANSION
            idx = np.searchsorted(ef, fids, side="right") - 1
            ok = idx >= 0
            exp_val[ok] = np.abs(ev_s[idx[ok]])
            exp_static[ok] = flags[idx[ok]]
            exp_static[~ok] = float(regime_static)
        # production gate (step_is_ego_static in run_speed_estimation), evaluated per row
        gate = (step_med <= EGO_STATIC_MAX_STEP_PX) & np.where(np.isnan(exp_static), True,
                                                                exp_static.astype(bool))
        parts.append(pd.DataFrame({
            "clip": c, "frame": to_jaad_frame(e["frame_id"].astype(int), "B3"),
            "step_px": step, "step_med": step_med, "exp_abs": exp_val,
            "exp_available": ~np.isnan(exp_val), "gate_static": gate}))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).merge(fr, on=["clip", "frame"], how="inner")


def score_ego(gt, runs, clips):
    """(g) static-camera gate vs JAAD vehicle action (stopped vs moving)."""
    task = "g_ego"
    d = ego_table(gt, runs, clips)
    rows, cms = [], {}
    if d.empty:
        return rows, cms, d
    yt = d["ego_stopped"].to_numpy(bool)
    cl = d["clip"].to_numpy()
    for col, note in (("step_px", "raw [B3] step_px (lower = stopped)"),
                      ("step_med", "rolling-median(5) step_px used by the gate")):
        s = -d[col].to_numpy(float)
        rows.append(mrow(f"{task}_{col}", "roc_auc_stopped",
                         cluster_bootstrap(cl, lambda i, s=s: safe_auc(yt[i], s[i])),
                         len(d), d["clip"].nunique(), 0.5, note))
    s_med = -d["step_med"].to_numpy(float)
    ok = np.isfinite(s_med)
    if ok.any() and len(set(cl[ok].tolist())) >= 5 and yt[ok].any() and not yt[ok].all():
        yp_cv = cv_threshold_predictions(yt[ok], s_med[ok], cl[ok])
        prev = float(yt[ok].mean())
        rows.append(mrow(f"{task}_step_med", "f1_cv_threshold",
                         cluster_bootstrap(cl[ok], lambda i: f1(yt[ok][i], yp_cv[i])),
                         int(ok.sum()), len(set(cl[ok].tolist())), f1_reference(prev),
                         "threshold on smoothed step chosen on other clips (5-fold by clip)",
                         _pos_clips(yt[ok], cl[ok])))
    ea = d[d["exp_available"]]
    if not ea.empty:
        s = -ea["exp_abs"].to_numpy(float)
        yte = ea["ego_stopped"].to_numpy(bool)
        rows.append(mrow(f"{task}_expansion", "roc_auc_stopped",
                         cluster_bootstrap(ea["clip"].to_numpy(), lambda i: safe_auc(yte[i], s[i])),
                         len(ea), ea["clip"].nunique(), 0.5,
                         "|bbox expansion rate|, frames with >=3 co-tracked pedestrians only"))
    rows.append(mrow(f"{task}_expansion", "coverage",
                     (float(d["exp_available"].mean()), np.nan, np.nan), len(d),
                     d["clip"].nunique(), np.nan, "share of [B3] rows with an expansion anchor"))
    rows += binary_rows(f"{task}_gate", yt, d["gate_static"], cl,
                        metrics=("precision", "recall", "f1"),
                        note="positive = camera static/stopped; precision = P(stopped | gate says static)")
    cms["ego_gate_by_action"] = cm_frame(d["vehicle_action"], d["gate_static"].map(
        {True: "static", False: "moving"}),
        ["stopped", "moving_slow", "decelerating", "accelerating", "moving_fast"],
        ["static", "moving"], row_name="jaad_action", col_name="gate")
    return rows, cms, d


def score_looking(gt, runs, clips, matches, fps_by_clip):
    """(h) [P12] looked_left|looked_right vs JAAD 'look' before the crossing."""
    task = "h_looking"
    boxes = gt["boxes"]
    peds = gt["peds"].set_index(["clip", "ped_id"])
    bm = matches[matches["gt_id"].astype(str).str.contains("b")]
    recs = []
    for r in bm.itertuples(index=False):
        p12 = runs.read(r.clip, "P12")
        c3 = runs.read(r.clip, "C3")
        if p12 is None or p12.empty or c3 is None or c3.empty:
            continue
        row = p12[p12["track_id"] == r.pred_id]
        win = c3[(c3["track_id"] == r.pred_id) & _as_bool(c3["crossed"])]
        if row.empty or win.empty or pd.isna(win["started_frame"].iloc[0]):
            continue
        row = row.iloc[0]
        if int(row["n_pose_frames"]) <= 0:
            continue
        fps = fps_by_clip[r.clip]
        start = int(to_jaad_frame(int(win["started_frame"].iloc[0]), "C3"))
        lo = start - int(round(LOOK_WINDOW_S * fps))
        g = boxes[(boxes["clip"] == r.clip) & (boxes["ped_id"] == r.gt_id)]
        gw = g[(g["frame"] >= lo) & (g["frame"] < start)]
        cp = peds.at[(r.clip, r.gt_id), "crossing_point"]
        gcp = g[(g["frame"] >= cp - int(round(LOOK_WINDOW_S * fps))) & (g["frame"] < cp)] \
            if cp is not None and cp >= 0 else g.iloc[0:0]
        looked = bool(_as_bool(pd.Series([row["looked_left"]])).iloc[0]
                      or _as_bool(pd.Series([row["looked_right"]])).iloc[0])
        recs.append({"clip": r.clip, "gt_id": r.gt_id, "track_id": r.pred_id,
                     "pedx_looked": looked, "reliable": bool(_as_bool(pd.Series([row["reliable"]])).iloc[0]),
                     "jaad_look_pedx_window": (gw["look"] == 1).any() if not gw.empty else np.nan,
                     "jaad_look_before_cp": (gcp["look"] == 1).any() if not gcp.empty else np.nan})
    d = pd.DataFrame(recs)
    rows, cms = [], {}
    if d.empty:
        return rows, cms, d
    for col, note in (("jaad_look_pedx_window", "JAAD look in PedX's own pre-crossing window"),
                      ("jaad_look_before_cp", "JAAD look in the 5 s before JAAD crossing_point")):
        s = d[d[col].notna()]
        if s.empty:
            continue
        rows += binary_rows(f"{task}_{col.replace('jaad_look_', '')}", s[col].astype(bool),
                            s["pedx_looked"], s["clip"],
                            metrics=("precision", "recall", "f1", "accuracy", "kappa"),
                            note=note + "; JAAD look = looking at the ego-vehicle")
    s = d[d["jaad_look_pedx_window"].notna()]
    if not s.empty:
        cms["looking"] = cm_frame(s["jaad_look_pedx_window"].map({True: "looking", False: "not"}),
                                  s["pedx_looked"].map({True: "head_turn", False: "none"}),
                                  ["looking", "not"], ["head_turn", "none"])
    return rows, cms, d


def age_group(age, cuts=AGE_CUTS):
    if age is None or (isinstance(age, float) and np.isnan(age)):
        return None
    a = int(round(float(age)))
    for lo, hi, lab in cuts:
        if lo <= a <= hi:
            return lab
    return None


def score_age_gender(gt, runs, clips, matches):
    """(i) [P6] gender / age group for matched behavioural pedestrians."""
    task = "i_age_gender"
    peds = gt["peds"].set_index(["clip", "ped_id"])
    bm = matches[matches["gt_id"].astype(str).str.contains("b")]
    recs = []
    for r in bm.itertuples(index=False):
        p6 = runs.read(r.clip, "P6")
        hit = p6[p6["id"].astype(str) == str(r.pred_id)] if p6 is not None and not p6.empty \
            else pd.DataFrame()
        recs.append({"clip": r.clip, "gt_id": r.gt_id, "has_p6": not hit.empty,
                     "jaad_gender": peds.at[(r.clip, r.gt_id), "gender"],
                     "jaad_age": peds.at[(r.clip, r.gt_id), "age"],
                     "pedx_gender": hit["gender"].iloc[0] if not hit.empty else None,
                     "pedx_age_years": float(hit["age"].iloc[0]) if not hit.empty else np.nan})
    d = pd.DataFrame(recs)
    rows, cms = [], {}
    if d.empty:
        return rows, cms, d
    hp = d["has_p6"].to_numpy(bool)
    rows.append(mrow(task, "coverage", cluster_bootstrap(d["clip"].to_numpy(), lambda i: float(hp[i].mean())),
                     len(d), d["clip"].nunique(), np.nan,
                     "matched behavioural peds whose PedX track has a [P6] row (face found)"))
    s = d[d["has_p6"] & d["jaad_gender"].isin(["female", "male"])]
    if not s.empty:
        rows += multiclass_rows(task + "_gender", s["jaad_gender"].to_numpy(object),
                                s["pedx_gender"].to_numpy(object), s["clip"], ["female", "male"])
        cms["gender"] = cm_frame(s["jaad_gender"], s["pedx_gender"], ["female", "male"],
                                 ["female", "male"])
    s = d[d["has_p6"] & d["jaad_age"].isin(AGE_LABELS)].copy()
    if not s.empty:
        for cuts, name, note in ((AGE_CUTS, "", "cuts child<=12, young 13-19, adult 20-64, senior>=65"),
                                 (AGE_CUTS_ALT, "_alt", "cuts child<=12, young 13-24, adult 25-59, senior>=60")):
            s["pedx_age"] = s["pedx_age_years"].map(lambda a: age_group(a, cuts))
            rows += multiclass_rows(task + "_age" + name, s["jaad_age"].to_numpy(object),
                                    s["pedx_age"].to_numpy(object), s["clip"], AGE_LABELS, note=note)
        s["pedx_age"] = s["pedx_age_years"].map(age_group)
        cms["age"] = cm_frame(s["jaad_age"], s["pedx_age"], AGE_LABELS, AGE_LABELS)
    return rows, cms, d


def score_signal_coverage(gt, runs, clips):
    """(j) [C1] signal-observation coverage vs JAAD traffic_light labels (no risky label)."""
    task = "j_risky_coverage"
    fr = gt["frames"].set_index(["clip", "frame"])["traffic_light"]
    recs = []
    for c in clips_with(runs, clips, "C1"):
        c1 = runs.read(c, "C1")
        c5 = runs.read(c, "C5")
        if c1.empty:
            continue
        for r in c1.itertuples(index=False):
            s = int(to_jaad_frame(int(r.started_frame), "C1"))
            e = int(to_jaad_frame(int(r.ended_frame), "C1"))
            labs = [fr.get((c, f)) for f in range(s, e + 1)]
            labs = [x for x in labs if x is not None]
            gl = np.mean([x != "n/a" for x in labs]) if labs else np.nan
            ran = np.nan
            if c5 is not None and not c5.empty and "ran_red_light" in c5.columns:
                h = c5[c5["track_id"] == r.track_id]
                ran = bool(_as_bool(h["ran_red_light"]).iloc[0]) if not h.empty else np.nan
            recs.append({"clip": c, "track_id": r.track_id, "pedx_coverage": float(r.signal_coverage),
                         "pedx_determined": pd.notna(r.risk), "jaad_light_frac": gl,
                         "jaad_any_light": bool(gl > 0) if gl == gl else False,
                         "jaad_red_frac": np.mean([x == "red" for x in labs]) if labs else np.nan,
                         "jaad_green_frac": np.mean([x == "green" for x in labs]) if labs else np.nan,
                         "ran_red_light": ran})
    d = pd.DataFrame(recs)
    rows = []
    if d.empty:
        return rows, {}, d
    cl = d["clip"].to_numpy()
    nc = d["clip"].nunique()
    det = d["pedx_determined"].to_numpy(bool)
    anyl = d["jaad_any_light"].to_numpy(bool)
    pc = d["pedx_coverage"].to_numpy(float)
    gl = d["jaad_light_frac"].to_numpy(float)
    rows.append(mrow(task, "pedx_determined_share", cluster_bootstrap(cl, lambda i: float(det[i].mean())),
                     len(d), nc, float(anyl.mean()),
                     "[C1] crossings with an observed signal; base_rate = share whose window has a JAAD light"))
    rows.append(mrow(task, "pedx_mean_signal_coverage", cluster_bootstrap(cl, lambda i: float(pc[i].mean())),
                     len(d), nc, float(np.nanmean(gl)),
                     "base_rate = mean JAAD light-labelled share of the same windows"))
    rows += binary_rows(task + "_observed", anyl, det, cl, metrics=("precision", "recall"),
                        note="PedX 'signal observed' vs JAAD 'light labelled' in the crossing window")
    if d["ran_red_light"].notna().any():
        rr = d[d["ran_red_light"] == True]  # noqa: E712
        rows.append(mrow(task, "C5_ran_red_count", (float(len(rr)), np.nan, np.nan), len(d), nc,
                         np.nan, "count only; JAAD green share in those windows: "
                         + (f"{rr['jaad_green_frac'].mean():.2f}" if len(rr) else "n/a")))
    return rows, {}, d


def score_signs(gt, runs, clips):
    """(k) [E3] pedestrian-crossing / stop-sign evidence vs JAAD ped_sign / stop_sign.

    The scored class sets are the ones production risky_crossing.py uses. Because a "recall 0"
    can mean "detector cannot see the sign" OR "the production class list is wrong", the rows
    also give, per [E3] class, how often JAAD labels the sign on frames where that class fires
    (`k_signs_class[<name>]`), and an exploratory score for class 'ip' alone.
    """
    task = "k_signs"
    fr = gt["frames"][["clip", "frame", "ped_sign", "stop_sign"]]
    parts = []
    for c in clips_with(runs, clips, "E3"):
        e = runs.read(c, "E3")
        if e.empty:
            continue
        c1 = e["sign_classes_1"].fillna("").astype(str).str.split(";")
        c2 = e["sign_classes_2"].fillna("").astype(str).str.split(";")
        parts.append(pd.DataFrame({
            "clip": c, "frame": to_jaad_frame(e["frame_id"].astype(int), "E3"),
            "ped": [bool(set(a) & PED_SIGN_CLASSES_1) or bool(set(b) & PED_SIGN_CLASSES_2)
                    for a, b in zip(c1, c2)],
            "stop": [bool(set(b) & STOP_SIGN_CLASSES_2) for b in c2],
            "ped_ip": [("ip" in set(a)) for a in c1],
            "classes": [frozenset(x for x in (set(a) | set(b)) if x) for a, b in zip(c1, c2)]}))
    if not parts:
        return [], {}, pd.DataFrame()
    d = pd.concat(parts, ignore_index=True).merge(fr, on=["clip", "frame"], how="inner")
    cl = d["clip"].to_numpy()
    rows = binary_rows(task + "_ped_sign", d["ped_sign"] == 1, d["ped"], cl,
                       note="classes risky_crossing.py treats as protective")
    rows += binary_rows(task + "_stop_sign", d["stop_sign"] == 1, d["stop"], cl,
                        note="best_new.pt 'Stop_Sign'")
    rows += binary_rows(task + "_ped_sign_ip_only", d["ped_sign"] == 1, d["ped_ip"], cl,
                        note="EXPLORATORY: class 'ip' alone, chosen after inspecting interim results")
    counts = {}
    for cs in d["classes"]:
        for k in cs:
            counts[k] = counts.get(k, 0) + 1
    for k, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        if n < 10:
            continue
        sel = np.array([k in cs for cs in d["classes"]])
        for lab in ("ped_sign", "stop_sign"):
            v = (d[lab].to_numpy() == 1)
            rows.append(mrow(f"k_signs_class[{k}]", f"precision_{lab}",
                             cluster_bootstrap(cl[sel], lambda i, v=v[sel]: float(v[i].mean())),
                             int(sel.sum()), len(set(cl[sel].tolist())), float(v.mean()),
                             "descriptive: share of frames where [E3] flags this class that JAAD "
                             "labels with the sign; base_rate = overall prevalence"))
    return rows, {}, d


def coverage_rows(gt, runs, clips):
    """Descriptive: which outputs exist, and [S1] reliability (JAAD has no speed GT)."""
    rows = []
    for code in ("B1", "B3", "E1", "E2", "E3", "E6", "E7", "E9", "E10", "C3", "C1", "C5",
                 "P6", "P12", "S1", "X7"):
        n = sum(1 for c in clips if runs.has(c, code))
        rows.append(mrow("z_coverage", f"clips_with_{code}", (float(n), np.nan, np.nan),
                         len(clips), len(clips), np.nan, FILES[code]))
    s1 = [runs.read(c, "S1") for c in clips]
    s1 = [x for x in s1 if x is not None and not x.empty]
    if s1:
        s = pd.concat(s1, ignore_index=True)
        rows.append(mrow("z_coverage", "S1_reliable_share",
                         (float(_as_bool(s["reliable"]).mean()), np.nan, np.nan), len(s),
                         len(s1), np.nan, "[S1] tracks flagged reliable (no speed GT in JAAD)"))
    return rows


# =============================================================================================
# driver
# =============================================================================================

def score_all(gt, runs, clips, interval=1.0):
    """Run every task. Returns (metrics DataFrame, {name: confusion DataFrame}, details)."""
    clips = sorted(clips)
    fps_by_clip = gt["clips"].set_index("clip")["fps"].to_dict()
    rows, cms, details = [], {}, {}

    def add(name, res):
        r, c, d = res
        rows.extend(r)
        cms.update(c)
        details[name] = d

    add("weather", score_weather(gt, runs, clips))
    add("daynight", score_daynight(gt, runs, clips))
    add("traffic_light", score_traffic_light(gt, runs, clips))
    add("traffic_light_visible", score_traffic_light_visible(gt, runs, clips))
    add("crosswalk", score_crosswalk(gt, runs, clips, interval))
    r, c, t = score_detection(gt, runs, clips, interval)
    rows.extend(r)
    details["detection"] = t
    matches = track_matches(gt, runs, clips)
    overlaps = track_matches(gt, runs, clips, mode="overlap")
    details["matches"] = matches
    details["overlaps"] = overlaps
    add("crossing", score_crossing(gt, runs, clips, matches, fps_by_clip, overlaps))
    add("ego", score_ego(gt, runs, clips))
    add("looking", score_looking(gt, runs, clips, matches, fps_by_clip))
    add("age_gender", score_age_gender(gt, runs, clips, matches))
    add("signal_coverage", score_signal_coverage(gt, runs, clips))
    add("signs", score_signs(gt, runs, clips))
    rows.extend(coverage_rows(gt, runs, clips))
    return pd.DataFrame(rows, columns=METRIC_COLUMNS), cms, details
