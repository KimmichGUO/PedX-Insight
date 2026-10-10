"""Unit tests for the JAAD scoring harness (validation/jaad) on SYNTHETIC data.

No dataset, GPU or PedX run is needed. Each test pins a failure mode that would silently
corrupt a validation result:

  * frame alignment (1-indexed tracking vs 0-indexed detectors) -- an off-by-one here turns
    every frame-level metric into noise
  * JAAD's literal label 'n/a' being read as missing (it made 94% of traffic_light NaN and
    unlabelled frames count as "a light is present")
  * one-to-one Hungarian matching, IoU, track matching on co-visible frames
  * the cluster bootstrap (resampling whole clips; reproducible; wider than i.i.d.)
  * verdicts: "fail" only when the whole CI is below the bar (a lower bound just under it is
    "inconclusive")

Run: python -m pytest -q tests/test_jaad_scoring.py   (or python tests/test_jaad_scoring.py)
"""

import os
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np
import pandas as pd

from validation.jaad import gt as G
from validation.jaad import report as R
from validation.jaad import score as S


# ------------------------------------------------------------------ frame alignment

def test_frame_base_and_grids():
    # [B1] is 1-indexed: frame_id 30 is JAAD frame 29. [E2] is 0-indexed: 30 is 30.
    assert S.to_jaad_frame([1, 30], "B1").tolist() == [0, 29]
    assert S.to_jaad_frame([0, 30], "E2").tolist() == [0, 30]
    assert S.to_jaad_frame([0, 30], "E10").tolist() == [0, 30]
    # 30 fps, interval 1 s: detectors sample 0, 30, 60 ...; tracker B1 records 30, 60 ...
    det = S.detector_grid(30.0, 100)
    assert det.tolist() == [0, 30, 60, 90]
    b1, b2 = S.tracker_grid(30.0, 100)
    assert b1.tolist() == [30, 60, 90]
    assert set(b1) <= set(b2) and len(b2) > len(b1)            # dense track is a superset
    assert S.to_jaad_frame(b1, "B1").tolist() == [29, 59, 89]  # one before the detector frame
    # 29.97 fps rounds UP to 30 for both grids
    assert S.detector_grid(29.97, 61).tolist() == [0, 30, 60]


# ------------------------------------------------------------------ JAAD 'n/a' label

def test_load_tables_keeps_na_label():
    d = tempfile.mkdtemp(prefix="pedx_jaad_")
    try:
        pd.DataFrame({"clip": ["video_0001"] * 3, "frame": [0, 1, 2],
                      "traffic_light": ["n/a", "red", None], "ped_crossing": [0, 1, 0],
                      "ped_sign": [0, 0, 0], "stop_sign": [0, 0, 0],
                      "vehicle_action": ["stopped", "stopped", "stopped"],
                      "ego_stopped": [True, True, True]}
                     ).to_csv(os.path.join(d, "jaad_gt_frames.csv"), index=False)
        for name in ("clips", "peds", "boxes"):
            pd.DataFrame({"clip": ["video_0001"]}).to_csv(os.path.join(d, G.TABLE_FILES[name]), index=False)
        # A plain read_csv turns 'n/a' into NaN -- the bug.
        assert pd.read_csv(os.path.join(d, "jaad_gt_frames.csv"))["traffic_light"].isna().sum() == 2
        fr = G.load_tables(d)["frames"]
        assert fr["traffic_light"].tolist()[:2] == ["n/a", "red"]
        assert fr["traffic_light"].isna().tolist() == [False, False, True]   # only the empty cell
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_majority_share_ignores_nan():
    assert np.isnan(S.majority_share([np.nan, np.nan]))            # no crash, no number
    assert S.majority_share(["a", "a", "b", None]) == 2 / 3        # NaN excluded from the base


# ------------------------------------------------------------------ matching

def test_iou_and_hungarian_one_to_one():
    a = np.array([[0, 0, 10, 10], [20, 0, 30, 10]], float)
    b = np.array([[1, 0, 11, 10], [100, 100, 110, 110]], float)
    m = S.iou_matrix(a, b)
    assert abs(m[0, 0] - 9 / 11) < 1e-9 and m[1, 0] == 0 and m[0, 1] == 0
    pairs = S.match_boxes(a, b, thr=0.5)
    assert pairs == [(0, 0, pairs[0][2])] and pairs[0][2] > 0.8
    # two GT boxes compete for one prediction: only the better one gets it (one-to-one)
    gt = np.array([[0, 0, 10, 10], [0, 0, 10, 9]], float)
    pr = np.array([[0, 0, 10, 10]], float)
    got = S.match_boxes(gt, pr, thr=0.5)
    assert len(got) == 1 and got[0][:2] == (0, 0)
    assert S.match_boxes(np.zeros((0, 4)), pr) == []


def test_match_tracks_uses_covisible_frames_and_one_to_one():
    frames = list(range(10))
    def track(tid, x0):
        return pd.DataFrame({"frame": frames, "id": tid, "x1": x0, "y1": 0, "x2": x0 + 10, "y2": 30})
    gt = pd.concat([track("g1", 0), track("g2", 100)], ignore_index=True)
    pred = pd.concat([track(7, 1), track(8, 101), track(9, 500)], ignore_index=True)  # 9 = false track
    out = S.match_tracks(gt, pred).set_index("gt_id")
    assert out.loc["g1", "pred_id"] == 7 and out.loc["g2", "pred_id"] == 8
    assert (out["mean_iou"] > 0.7).all() and (out["n_covis"] == 10).all()
    # a prediction visible on fewer than min_covis frames is not a candidate
    short = pred[(pred["id"] != 7) | (pred["frame"] < 2)]
    assert "g1" not in set(S.match_tracks(gt, short)["gt_id"])
    assert S.match_tracks(gt, pd.DataFrame(columns=gt.columns)).empty


# ------------------------------------------------------------------ statistics

def test_cluster_bootstrap_resamples_clips_and_is_reproducible():
    # 8 clips; within a clip every row is identical, so the effective sample size is 8.
    clusters = np.repeat([f"c{i}" for i in range(8)], 50)
    y = np.repeat([1, 0, 1, 0, 1, 0, 1, 0], 50).astype(float)
    p, lo, hi = S.cluster_bootstrap(clusters, lambda i: y[i].mean())
    assert p == 0.5 and lo < 0.5 < hi
    assert hi - lo > 0.3                                  # i.i.d. over 400 rows would give ~0.1
    assert S.cluster_bootstrap(clusters, lambda i: y[i].mean()) == (p, lo, hi)   # seeded
    assert np.isnan(S.cluster_bootstrap([], lambda i: 0.0)[0])


def test_metric_wiring_matches_definitions():
    yt = np.array([1, 1, 1, 0, 0, 0, 0, 1], bool)
    yp = np.array([1, 1, 0, 0, 0, 1, 0, 1], bool)
    assert S.precision(yt, yp) == 3 / 4 and S.recall(yt, yp) == 3 / 4
    assert abs(S.f1(yt, yp) - 0.75) < 1e-12
    assert S.accuracy(yt, yp) == 6 / 8
    from sklearn.metrics import cohen_kappa_score
    assert abs(S.kappa(yt, yp) - cohen_kappa_score(yt, yp)) < 1e-12
    assert S.majority_share(yt) == 0.5
    assert np.isnan(S.safe_auc(np.ones(5, bool), np.arange(5)))   # one class: undefined, not a crash


def test_best_threshold_and_cv_are_out_of_fold():
    rng = np.random.default_rng(0)
    clips = np.repeat([f"c{i}" for i in range(10)], 30)
    yt = rng.random(300) < 0.4
    score = np.where(yt, rng.normal(0.7, 0.1, 300), rng.normal(0.3, 0.1, 300))
    t, f1, p, r = S.best_f1_threshold(yt, score)
    assert 0.4 < t < 0.65 and f1 > 0.95
    pred = S.cv_threshold_predictions(yt, score, clips)
    assert pred.shape == yt.shape and S.f1(yt, pred) > 0.9
    # degenerate labels: no threshold exists
    assert np.isnan(S.best_f1_threshold(np.zeros(5, bool), np.arange(5))[0])


# ------------------------------------------------------------------ a task end to end

def test_traffic_light_table_alignment_and_na_label():
    root = tempfile.mkdtemp(prefix="pedx_jaad_runs_")
    try:
        clip = os.path.join(root, "video_0001")
        os.makedirs(clip)
        # [E2] is 0-indexed: rows at frame_id 0, 30, 60 are JAAD frames 0, 30, 60.
        pd.DataFrame({"frame_id": [0, 30, 60], "main_light_color": ["None", "red", "green"]}
                     ).to_csv(os.path.join(clip, S.FILES["E2"]), index=False)
        frames = pd.DataFrame({"clip": "video_0001", "frame": [0, 30, 60],
                               "traffic_light": ["n/a", "red", "n/a"]})
        gt = {"frames": frames}
        d = S.light_table(gt, S.Runs(root), ["video_0001"])
        assert d["frame"].tolist() == [0, 30, 60]
        assert d["jaad"].tolist() == ["n/a", "red", "n/a"] and d["pedx"].tolist() == ["None", "red", "green"]
        # presence confusion: frame 60 is a false positive, frame 30 a true positive
        yt = (d["jaad"] != "n/a").to_numpy()
        yp = (d["pedx"] != "None").to_numpy()
        assert S.precision(yt, yp) == 0.5 and S.recall(yt, yp) == 1.0
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_f1_reference_is_the_always_positive_f1():
    """The trivial classifier's F1 is 2p/(1+p), not the prevalence p (audit finding)."""
    assert abs(S.f1_reference(0.27) - 2 * 0.27 / 1.27) < 1e-12
    assert np.isnan(S.f1_reference(float("nan")))
    yt = np.array([1, 0, 0, 0] * 5, bool)
    rows = {r["metric"]: r for r in S.binary_rows("t", yt, yt, np.repeat(list("abcde"), 4))}
    assert abs(rows["f1"]["base_rate"] - 2 * 0.25 / 1.25) < 1e-12
    assert rows["precision"]["base_rate"] == 0.25 and rows["recall"]["base_rate"] == 0.25
    assert rows["f1"]["n_pos_clips"] == 5                       # clips that contain a positive
    # ...which is not the same as the number of clips:
    rows2 = {r["metric"]: r for r in S.binary_rows("t", [1, 0, 0, 0], [1, 0, 0, 0], list("aabb"))}
    assert rows2["f1"]["n_clips"] == 2 and rows2["f1"]["n_pos_clips"] == 1


def test_judge_gates_on_positive_clips_and_nan():
    ok = _row("t", "m", 0.9, 0.85, 0.95, clips=40)
    assert R.judge(ok, 0.7) == R.PASS
    # 40 clips in the denominator but only 5 contain a positive: not powered -> few, never fail
    thin = dict(ok, value=0.0, ci_low=0.0, ci_high=0.0, n_pos_clips=5)
    assert R.judge(thin, 0.7) == R.FEW
    # a metric that could not be computed (NaN) is 'cannot tell', not a failure
    nan = dict(ok, value=float("nan"), ci_low=float("nan"), ci_high=float("nan"))
    assert R.judge(nan, 0.7) == R.FEW


def test_overlap_tracks_exposes_id_fragmentation():
    frames = list(range(10))
    gt = pd.DataFrame({"frame": frames, "id": "g", "x1": 0, "y1": 0, "x2": 10, "y2": 30})
    a = pd.DataFrame({"frame": frames[:5], "id": 1, "x1": 0, "y1": 0, "x2": 10, "y2": 30})
    b = pd.DataFrame({"frame": frames[5:], "id": 2, "x1": 1, "y1": 0, "x2": 11, "y2": 30})
    pred = pd.concat([a, b], ignore_index=True)
    ov = S.overlap_tracks(gt, pred)
    assert sorted(ov["pred_id"]) == [1, 2]                     # both fragments are reported...
    assert len(S.match_tracks(gt, pred)) == 1                  # ...but the one-to-one match keeps one


def test_crossing_any_track_and_irrelevant_negatives():
    """One-to-one matching picks a fragment that was never judged 'crossed'; the any-track
    variant credits the pedestrian, and JAAD 'irrelevant' (-1) counts against precision."""
    root = tempfile.mkdtemp(prefix="pedx_jaad_runs_")
    try:
        os.makedirs(os.path.join(root, "video_0001"))
        pd.DataFrame({"track_id": [7, 8, 9], "crossed": [False, True, True],
                      "started_frame": [np.nan, 40, 50], "ended_frame": [np.nan, 80, 90]}
                     ).to_csv(os.path.join(root, "video_0001", S.FILES["C3"]), index=False)
        peds = pd.DataFrame({"clip": "video_0001", "ped_id": ["b1", "b2"], "kind": "b",
                             "crossing": [1, -1], "crossing_point": [45, -1]})
        gt = {"peds": peds}
        matches = pd.DataFrame({"clip": "video_0001", "gt_id": ["b1", "b2"], "pred_id": [7, 9],
                                "mean_iou": 0.9, "n_covis": 10, "source": "B2"})
        overlaps = pd.DataFrame({"clip": "video_0001", "gt_id": ["b1", "b1", "b2"],
                                 "pred_id": [7, 8, 9], "mean_iou": 0.8, "n_covis": 6})
        rows, _, d = S.score_crossing(gt, S.Runs(root), ["video_0001"], matches,
                                      {"video_0001": 30.0}, overlaps)
        by = {(r["task"], r["metric"]): r["value"] for r in rows}
        assert by[("f_crossing", "recall")] == 0.0                      # matched fragment 7 not crossed
        assert by[("f_crossing_any_track", "recall")] == 1.0            # fragment 8 is crossed
        # b2 is JAAD 'irrelevant' (-1): ignored by the main population, a false positive here
        assert by[("f_crossing_incl_irrelevant", "precision")] == 0.0   # matched crossers: b1 no, b2 yes(-1)
        assert by[("f_track_fragmentation", "ids_per_gt_pedestrian_mean")] == 1.5
        assert by[("f_track_fragmentation", "share_with_2plus_ids")] == 0.5
        assert len(d) == 2 and set(d["crossing_raw"]) == {1, -1}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_traffic_light_visible_subset():
    """JAAD labels a light on frames where none is in view; those must not count against [E2]."""
    root = tempfile.mkdtemp(prefix="pedx_jaad_runs_")
    try:
        clip = os.path.join(root, "video_0001")
        os.makedirs(clip)
        # JAAD: red on frames 0 and 30 (the light is only IN VIEW at frame 0), n/a at 60 and 90.
        pd.DataFrame({"frame_id": [0, 30, 60, 90],
                      "main_light_color": ["red", "None", "green", "None"]}
                     ).to_csv(os.path.join(clip, S.FILES["E2"]), index=False)
        pd.DataFrame({"frame_id": [0, 30, 60, 90], "traffic_light_frac": [0.001, 0.0, 0.0, 0.0],
                      "traffic_light_present": [1, 0, 0, 0]}
                     ).to_csv(os.path.join(clip, S.FILES["E10"]), index=False)
        gt = {"frames": pd.DataFrame({"clip": "video_0001", "frame": [0, 30, 60, 90],
                                      "traffic_light": ["red", "red", "n/a", "n/a"]})}
        rows, _, d = S.score_traffic_light_visible(gt, S.Runs(root), ["video_0001"])
        by = {(r["task"], r["metric"]): r for r in rows}
        inval = by[("c_traffic_light_visible_label_validity", "labelled_frames_with_no_light_in_view")]
        assert inval["value"] == 0.5 and inval["n"] == 2          # frame 30 labelled, nothing in view
        rec = by[("c_traffic_light_visible_presence", "recall")]
        assert rec["value"] == 1.0 and rec["n"] == 1              # only frame 0 is labelled AND in view
        assert by[("c_traffic_light_visible_state", "accuracy")]["value"] == 1.0
        fpr = by[("c_traffic_light_visible_no_light", "false_positive_rate")]
        assert fpr["n"] == 2 and fpr["value"] == 0.5              # frames 60 (green, wrong) and 90
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ------------------------------------------------------------------ verdict logic

def _row(task, metric, value, lo, hi, n=500, clips=30, base=0.3):
    return {"task": task, "metric": metric, "value": value, "ci_low": lo, "ci_high": hi,
            "n": n, "n_clips": clips, "base_rate": base, "note": ""}


def test_judge_is_three_way():
    assert R.judge(_row("t", "m", 0.9, 0.75, 0.95), 0.7) == R.PASS
    assert R.judge(_row("t", "m", 0.5, 0.3, 0.6), 0.7) == R.FAIL            # CI wholly below
    assert R.judge(_row("t", "m", 0.8, 0.6, 0.9), 0.7) == R.STRADDLE        # lower bound just under
    assert R.judge(_row("t", "m", 0.9, 0.8, 0.95, clips=5), 0.7) == R.FEW   # too few clips
    assert R.judge(_row("t", "m", 0.9, 0.8, 0.95, n=50, base=0.2), 0.7, min_positives=30) == R.FEW
    assert R.judge(None, 0.7) == R.FEW
    assert R.combine(R.PASS, R.FAIL) == R.FAIL and R.combine(R.PASS, R.STRADDLE) == R.STRADDLE
    assert R.combine(R.PASS, R.PASS) == R.PASS and R.combine(R.FEW, R.STRADDLE) == R.FEW


def _metrics(rows):
    return pd.DataFrame(rows, columns=S.METRIC_COLUMNS)


def test_verdicts_distinguish_fail_from_cannot_tell():
    # kappa CI wholly under the bar -> not usable; CI straddling -> inconclusive.
    bad = _metrics([_row("a_weather", "accuracy", 0.30, 0.15, 0.45, base=0.6),
                    _row("a_weather", "kappa", -0.1, -0.3, 0.1, base=0.0)])
    assert [x for x in R.verdicts(bad) if x["module"] == "[E1] weather"][0]["verdict"] == R.NOT_USABLE
    unsure = _metrics([_row("b_daynight", "accuracy", 0.95, 0.85, 1.0, base=0.9),
                       _row("b_daynight", "kappa", 0.64, 0.0, 1.0, base=0.0)])
    v = [x for x in R.verdicts(unsure) if x["module"] == "[E6] day/night"][0]
    assert v["verdict"] == R.INCONCLUSIVE and "straddles" in v["evidence"]


def test_verdict_recalibrated_threshold_and_counterfactual():
    # Shipped crosswalk fails at its operating point; the continuous score is excellent and
    # a cross-validated threshold fixes it -> "recalibrated threshold", not "not usable".
    rows = [_row("d_crosswalk_E10_frame", "f1", 0.4, 0.3, 0.5, base=0.35),
            _row("d_crosswalk_E10_frame", "recall", 0.3, 0.2, 0.4, base=0.35),
            _row("d_crosswalk_E10_frame", "precision", 0.9, 0.8, 1.0, base=0.35),
            _row("d_crosswalk_E10_frac", "roc_auc", 0.93, 0.88, 0.97, base=0.5),
            _row("d_crosswalk_E10_frac", "f1_cv_threshold", 0.82, 0.75, 0.88, base=0.35)]
    e10 = [v for v in R.verdicts(_metrics(rows)) if v["module"].startswith("[E10]")][0]
    assert e10["verdict"] == R.RECAL
    # A detector that never fires: recall 0 with a zero-width CI -> demonstrably fails.
    dead = [_row("d_crosswalk_E7_frame", "f1", 0.0, 0.0, 0.0, base=0.36),
            _row("d_crosswalk_E7_frame", "recall", 0.0, 0.0, 0.0, base=0.36)]
    e7 = [v for v in R.verdicts(_metrics(dead)) if v["module"].startswith("[E7] crosswalk as shipped")][0]
    assert e7["verdict"] == R.NOT_USABLE


def test_report_runs_on_a_tiny_synthetic_run():
    """write_report must not crash on the smallest possible inputs (no verdict rows, one task)."""
    out = tempfile.mkdtemp(prefix="pedx_jaad_out_")
    try:
        m = _metrics([_row("a_weather", "accuracy", 0.7, 0.5, 0.9, base=0.6)])
        gt = {"clips": pd.DataFrame({"clip": ["video_0001"]})}
        R.write_report(out, m, {}, {}, ["video_0001"], gt, out, 1.0, R.verdicts(m), 1)
        text = open(os.path.join(out, "JAAD_REPORT.md"), encoding="utf-8").read()
        assert "## Protocol" in text and "## Verdicts" in text and "a_weather" in text
    finally:
        shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
