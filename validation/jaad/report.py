"""Score a JAAD run and write validation/results/{jaad_metrics.csv, cm_*.csv, JAAD_REPORT.md}.

    python -m validation.jaad.report                      # every clip with outputs
    python -m validation.jaad.report --clips_file validation/jaad/sample_clips.txt

Verdicts are produced by the explicit rules in `verdicts()` -- thresholds are named
constants below and are printed in the report -- so the prose cannot drift from the numbers.
A verdict is one of:

  usable as-is                     the lower CI bound clears the bar on the key metric
  usable with a recalibrated threshold   the continuous score discriminates well (CI lower
                                   bound of ROC-AUC clears the bar) but the shipped
                                   operating point does not
  usable with restrictions         clears the bar only in a stated sub-population
  not usable for cross-city claims the metric is not distinguishable from (or is worse
                                   than) the trivial baseline, or fails the bar
  inconclusive                     fewer than MIN_CLIPS clips or MIN_POSITIVES positives, OR the
                                   95% CI straddles the bar (cannot tell yet)

Each bar is tested three ways (`judge`): pass when the whole CI is above it, fail only when
the whole CI is below it. A lower bound just under the bar is "inconclusive", never "fail".

"Not usable" is a statement about THIS dataset: JAAD is dashcam footage from a handful of
cities (mainly Canada, Ukraine, USA, Germany), so a pass here is necessary, not sufficient,
for the global dataset.
"""

import argparse
import datetime as dt
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from validation.jaad import gt as G  # noqa: E402
from validation.jaad import score as S  # noqa: E402
from validation.jaad.run_pedx import DEFAULT_RESULTS_ROOT  # noqa: E402

DEFAULT_GT_DIR = os.path.join(os.path.dirname(DEFAULT_RESULTS_ROOT.rstrip("/\\")), "jaad_gt")
DEFAULT_OUT = REPO / "validation" / "results"

USABLE = "usable as-is"
RECAL = "usable with a recalibrated threshold"
RESTRICTED = "usable with restrictions"
NOT_USABLE = "not usable for cross-city claims"
INCONCLUSIVE = "inconclusive"

# ---- verdict thresholds (the bar a module must clear) -------------------------------------
MIN_CLIPS = 15           # fewer clips than this -> inconclusive
MIN_POSITIVES = 30       # fewer positive units than this -> inconclusive
BAR_F1 = 0.70            # detection-style tasks: lower CI bound of F1
BAR_AUC = 0.80           # score-based tasks: lower CI bound of ROC-AUC
BAR_KAPPA_WEATHER = 0.40
BAR_KAPPA_DAYNIGHT = 0.60
BAR_KAPPA_GENDER = 0.40
BAR_RECALL_TRACK = 0.80
BAR_AUC_EGO = 0.90
BAR_COVERAGE_RISKY = 0.50


# ------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------

def fmt(v, nd=3):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def cell(row):
    """'0.731 [0.512, 0.900]' for a metric row, '0.731' when it has no CI."""
    if row is None:
        return "n/a"
    lo, hi = row["ci_low"], row["ci_high"]
    if pd.isna(lo) or pd.isna(hi):
        return fmt(row["value"])
    return f"{fmt(row['value'])} [{fmt(lo)}, {fmt(hi)}]"


def get(m, task, metric):
    sub = m[(m["task"] == task) & (m["metric"] == metric)]
    return None if sub.empty else sub.iloc[0].to_dict()


def positives(row):
    """Approximate number of positive units behind a binary metric (n * prevalence)."""
    if row is None or pd.isna(row.get("base_rate")):
        return float("nan")
    return float(row["n"]) * float(row["base_rate"])


def _entry(module, verdict, evidence, scope=""):
    return {"module": module, "verdict": verdict, "evidence": evidence, "scope": scope}


# ------------------------------------------------------------------------------------------
# verdict rules
# ------------------------------------------------------------------------------------------

PASS, FAIL, STRADDLE, FEW = "pass", "fail", "straddle", "few"


def judge(row, bar, min_positives=0):
    """Three-way test of a metric's 95% CI against a bar (higher is better).

    pass      lower CI bound >= bar            -> the module demonstrably clears it
    fail      upper CI bound <  bar            -> it demonstrably does not
    straddle  the CI contains the bar          -> cannot tell yet (needs more clips)
    few       too few clips / positives to say anything

    A lower bound just under the bar is NOT a failure: calling it one would turn
    'not enough data' into a negative result.
    """
    if row is None or pd.isna(row["value"]):
        return FEW
    clips = row["n_clips"]
    npc = row.get("n_pos_clips")
    if npc is not None and pd.notna(npc):
        clips = min(clips, npc)          # a task with few positive-bearing clips is not powered
    if clips < MIN_CLIPS or positives(row) < min_positives:
        return FEW
    lo, hi = row["ci_low"], row["ci_high"]
    if pd.isna(lo) or pd.isna(hi):
        return PASS if row["value"] >= bar else FAIL
    if lo >= bar:
        return PASS
    if hi < bar:
        return FAIL
    return STRADDLE


def combine(*states):
    """Any fail -> fail; else any few -> few; else any straddle -> straddle; else pass."""
    if FAIL in states:
        return FAIL
    if FEW in states:
        return FEW
    if STRADDLE in states:
        return STRADDLE
    return PASS


def _verdict(state, on_pass=USABLE, on_fail=NOT_USABLE):
    return {PASS: on_pass, FAIL: on_fail}.get(state, INCONCLUSIVE)


def _note(state, ev):
    if state == STRADDLE:
        return ev + "; CI straddles the bar -- the data cannot separate the module from the bar"
    if state == FEW:
        return ev + "; too few clips or positives"
    return ev


def verdicts(m):
    out = []

    def add(module, state, ev, scope="", on_pass=USABLE, on_fail=NOT_USABLE):
        out.append(_entry(module, _verdict(state, on_pass, on_fail), _note(state, ev), scope))

    # [E1] weather: must beat the majority-class share AND show chance-corrected agreement.
    acc, kap = get(m, "a_weather", "accuracy"), get(m, "a_weather", "kappa")
    rain = get(m, "a_weather_rain", "recall")
    if acc is None:
        out.append(_entry("[E1] weather", INCONCLUSIVE, "no [E1] outputs scored"))
    else:
        ev = (f"accuracy {cell(acc)} vs majority-class {fmt(acc['base_rate'])}; kappa {cell(kap)}; "
              f"rain recall {cell(rain)} on {acc['n_clips']:.0f} clips")
        add("[E1] weather", combine(judge(acc, acc["base_rate"]), judge(kap, BAR_KAPPA_WEATHER)), ev)

    # [E6] day/night: kappa (JAAD is mostly daytime, so accuracy alone is weak)
    acc, kap = get(m, "b_daynight", "accuracy"), get(m, "b_daynight", "kappa")
    if acc is not None:
        kalt = get(m, "b_daynight_alt", "kappa")
        ev = (f"accuracy {cell(acc)} vs majority-class {fmt(acc['base_rate'])}; kappa {cell(kap)} with "
              f"PedX 'Evening' counted as night (the stricter mapping, used for the verdict)"
              + (f", {cell(kalt)} with Evening counted as day" if kalt else "")
              + f", on {acc['n_clips']:.0f} clips")
        add("[E6] day/night", judge(kap, BAR_KAPPA_DAYNIGHT), ev)

    # [E2] traffic light. JAAD's traffic_light label is not a visibility label (a clip labelled
    # red on every frame shows no light), so the verdict is judged on frames where a light is
    # plausibly in view; the all-frames presence numbers stay in the table for transparency.
    vis_rec = get(m, "c_traffic_light_visible_presence", "recall")
    vis_st = get(m, "c_traffic_light_visible_state", "accuracy")
    inval = get(m, "c_traffic_light_visible_label_validity", "labelled_frames_with_no_light_in_view")
    fpr = get(m, "c_traffic_light_visible_no_light", "false_positive_rate")
    allf = get(m, "c_traffic_light_presence", "recall")
    if vis_rec is not None:
        ev = (f"on frames with a labelled light that is also in view (Vistas proxy): detects a light in "
              f"{cell(vis_rec)} of frames"
              + (f", colour agreement {cell(vis_st)}" if vis_st is not None else "")
              + (f"; false-positive rate where no light is in view {cell(fpr)}" if fpr is not None else "")
              + (f". JAAD labels a light on frames where none is in view: {cell(inval)} of labelled frames"
                 if inval is not None else "")
              + (f". All labelled frames (invalid as visibility): recall {cell(allf)}" if allf else ""))
        s_rec = judge(vis_rec, 0.7)
        s_st = judge(vis_st, 0.8) if vis_st is not None else FEW
        add("[E2] traffic light", combine(s_rec, s_st), ev)
    else:
        f1p = get(m, "c_traffic_light_presence", "f1")
        if f1p is not None:
            add("[E2] traffic light", FEW, "visible-light subset unavailable (needs [E10])")

    # crosswalk: shipped [E7], counterfactual class-4 [X7], and [E10]
    def crosswalk_entry(label, src, score_task):
        f1r = get(m, f"d_crosswalk_{src}_frame", "f1")
        if f1r is None:
            return
        rec = get(m, f"d_crosswalk_{src}_frame", "recall")
        pre = get(m, f"d_crosswalk_{src}_frame", "precision")
        ev = (f"frame-level precision {cell(pre)}, recall {cell(rec)}, F1 {cell(f1r)} "
              f"on {f1r['n_clips']:.0f} clips")
        auc = get(m, score_task, "roc_auc")
        cvf1 = get(m, score_task, "f1_cv_threshold")
        if auc is not None:
            ev += f"; continuous-score ROC-AUC {cell(auc)}"
        if cvf1 is not None:
            ev += f", cross-validated best-threshold F1 {cell(cvf1)}"
        s_f1 = judge(f1r, BAR_F1, MIN_POSITIVES)
        if (s_f1 == FAIL and auc is not None and judge(auc, BAR_AUC) == PASS
                and cvf1 is not None and judge(cvf1, BAR_F1) == PASS):
            add(label, PASS, ev, on_pass=RECAL)
        else:
            add(label, s_f1, ev)

    crosswalk_entry("[E7] crosswalk as shipped (class 0)", "E7", "d_crosswalk_X7cls0_conf")
    crosswalk_entry("[E7] crosswalk, same weights, class 4 (COUNTERFACTUAL)", "X7cls4",
                    "d_crosswalk_X7cls4_conf")
    crosswalk_entry("[E10] scene_seg crosswalk (Mask2Former / Vistas)", "E10", "d_crosswalk_E10_frac")

    # pedestrian detection / tracking. The PRIMARY population is fixed a priori: JAAD's
    # behaviourally annotated pedestrians ('b'), i.e. the ones that interact with the road and
    # are what the crossing analysis is about. All boxes also include distant bystanders, so that
    # recall is shown but not judged. The restricted-scope fallback (unoccluded and >= 50 px) was
    # defined after inspecting interim results and is labelled exploratory.
    src = "B2" if get(m, "e_detection_B2", "recall") is not None else "B1"
    r_all = get(m, f"e_detection_{src}", "recall")
    if r_all is not None:
        r_b = get(m, f"e_detection_{src}", "recall_kind=b") or r_all
        r_ped = get(m, f"e_detection_{src}", "recall_kind=ped")
        r_none = get(m, f"e_detection_{src}", "recall_occlusion=none")
        r_u50 = get(m, f"e_detection_{src}", "recall_unoccluded_h>=50px")
        r_u100 = get(m, f"e_detection_{src}", "recall_unoccluded_h>=100px")
        pr = get(m, f"e_detection_{src}", "precision_lower_bound")
        ev = (f"[{src}] recall on behaviourally annotated pedestrians (PRIMARY, a priori) {cell(r_b)}; "
              f"all JAAD boxes {cell(r_all)} (distant bystanders alone: {cell(r_ped)}); "
              f"unoccluded {cell(r_none)}; unoccluded and >=50 px {cell(r_u50)}, >=100 px {cell(r_u100)} "
              f"(exploratory sub-populations); precision (lower bound; JAAD does not box everyone) {cell(pr)}")
        s_b = judge(r_b, BAR_RECALL_TRACK)
        if s_b == FAIL and judge(r_u50, BAR_RECALL_TRACK) == PASS:
            add("[B1]/[B2] pedestrian detection", PASS, ev, on_pass=RESTRICTED,
                scope="unoccluded pedestrians of >= 50 px (scope defined after interim results)")
        else:
            add("[B1]/[B2] pedestrian detection", s_b, ev)

    # crossing judge. Judged END TO END (a pedestrian counts as flagged if ANY PedX track
    # overlapping them is crossed, which removes the penalty for ID fragmentation) and with JAAD's
    # 'irrelevant' pedestrians counted as not crossing for precision (conservative).
    f1c, kc = get(m, "f_crossing_any_track", "f1"), get(m, "f_crossing_any_track", "kappa")
    rc, pc = get(m, "f_crossing_any_track", "recall"), get(m, "f_crossing_any_track", "precision")
    pi = get(m, "f_crossing_incl_irrelevant", "precision")
    m1 = get(m, "f_crossing", "recall")
    frag = get(m, "f_track_fragmentation", "share_with_2plus_ids")
    mr = get(m, "f_crossing", "match_rate")
    if f1c is not None:
        ev = (f"end to end over labelled behavioural pedestrians (flagged if any overlapping track is "
              f"crossed): precision {cell(pc)}, recall {cell(rc)}, F1 {cell(f1c)}, kappa {cell(kc)}; "
              f"precision with JAAD 'irrelevant' pedestrians counted as not crossing {cell(pi)}; "
              f"one-to-one-matched recall {cell(m1)} is lower because {cell(frag)} of pedestrians are "
              f"split over >= 2 PedX track ids; match rate {cell(mr)}; JAAD behavioural pedestrians cross "
              f"{fmt(f1c['base_rate'])} of the time, so kappa is the informative number")
        add("[C3] crossing judge", combine(judge(f1c, 0.8), judge(kc, 0.4), judge(pi, 0.8)), ev)

    # ego-motion gate
    a1, a2 = get(m, "g_ego_step_px", "roc_auc_stopped"), get(m, "g_ego_step_med", "roc_auc_stopped")
    a3 = get(m, "g_ego_expansion", "roc_auc_stopped")
    gf = get(m, "g_ego_gate", "f1")
    gp, gr = get(m, "g_ego_gate", "precision"), get(m, "g_ego_gate", "recall")
    if a1 is not None:
        ev = (f"ROC-AUC for 'ego vehicle stopped': translation step {cell(a1)}"
              + (f", smoothed step {cell(a2)}" if a2 else "")
              + (f", bbox-expansion {cell(a3)}" if a3 else "")
              + (f"; production gate precision {cell(gp)}, recall {cell(gr)}, F1 {cell(gf)}" if gf else ""))
        cvf = get(m, "g_ego_step_med", "f1_cv_threshold")
        if cvf is not None:
            ev += f"; cross-validated best-threshold F1 on the smoothed step {cell(cvf)}"
        s_auc = judge(a1, BAR_AUC_EGO)
        s_gate = judge(gf, BAR_F1) if gf is not None else FEW
        s_cv = judge(cvf, BAR_F1) if cvf is not None else FEW
        if s_auc == PASS and s_gate in (FAIL, STRADDLE) and s_cv == PASS:
            add("[B3] ego-motion / static-camera gate", PASS, ev, on_pass=RECAL)
        else:
            add("[B3] ego-motion / static-camera gate", combine(s_auc, s_gate), ev)

    # looking behaviour
    lf = get(m, "h_looking_pedx_window", "f1")
    if lf is not None:
        lk = get(m, "h_looking_pedx_window", "kappa")
        ev = (f"look-before-crossing vs JAAD 'look': F1 {cell(lf)}, kappa {cell(lk)} "
              f"on {lf['n']:.0f} pedestrians / {lf['n_clips']:.0f} clips")
        s = combine(judge(lf, BAR_F1, MIN_POSITIVES), judge(lk, 0.4))
        add("[P12] looking behaviour", FEW if lf["n"] < MIN_POSITIVES else s, ev)

    # gender
    ga, gk = get(m, "i_age_gender_gender", "accuracy"), get(m, "i_age_gender_gender", "kappa")
    cov = get(m, "i_age_gender", "coverage")
    if ga is not None:
        ev = (f"gender accuracy {cell(ga)} vs majority {fmt(ga['base_rate'])}, kappa {cell(gk)} on "
              f"{ga['n']:.0f} matched pedestrians; face detected for {cell(cov)} of tracked pedestrians")
        s = combine(judge(ga, ga["base_rate"]), judge(gk, BAR_KAPPA_GENDER))
        add("[P6] gender (inferred)", FEW if ga["n"] < MIN_POSITIVES else s, ev)

    # pedestrian sign detector
    sf = get(m, "k_signs_ped_sign", "f1")
    if sf is not None:
        sr, sp = get(m, "k_signs_ped_sign", "recall"), get(m, "k_signs_ped_sign", "precision")
        ip = get(m, "k_signs_ped_sign_ip_only", "f1")
        ipp, ipr = get(m, "k_signs_ped_sign_ip_only", "precision"), get(m, "k_signs_ped_sign_ip_only", "recall")
        ev = (f"JAAD pedestrian-crossing sign, with the class set production risky_crossing.py uses "
              f"('ip' was added after this benchmark found the earlier set {{w57, pg, i1}} scored "
              f"F1 0.02; see findings): precision {cell(sp)}, recall {cell(sr)}, F1 {cell(sf)}"
              + (f". Exploratory (class 'ip' alone, chosen after interim results): precision {cell(ipp)}, "
                 f"recall {cell(ipr)}, F1 {cell(ip)} -- the other classes in the set mostly add false "
                 f"positives; see the per-class `k_signs_class[...]` rows" if ip else ""))
        add("[E3] traffic sign (pedestrian sign)", judge(sf, BAR_F1, MIN_POSITIVES), ev)

    # risky-crossing signal coverage. The unconditional share is capped by how often JAAD labels a
    # light at all (a few % of windows), so it cannot be judged against an absolute bar. The
    # verdict uses coverage GIVEN a JAAD-labelled light in the crossing window.
    cv = get(m, "j_risky_coverage_observed", "recall")
    un = get(m, "j_risky_coverage", "pedx_determined_share")
    if un is not None:
        ev = (f"coverage given a JAAD-labelled light in the crossing window: {cell(cv)}"
              + (f" ({positives(cv):.0f} such crossings)" if cv is not None and pd.notna(positives(cv)) else "")
              + f"; unconditional share of crossings with an observed signal {cell(un)} on "
                f"{un['n']:.0f} crossings is capped by JAAD's light prevalence "
                f"({fmt(un['base_rate'])} of windows). JAAD has no risky label, so only coverage can be "
                f"validated")
        add("[C1] risky crossing (coverage only)", judge(cv, BAR_COVERAGE_RISKY, MIN_POSITIVES), ev)
    return out


# ------------------------------------------------------------------------------------------
# rendering
# ------------------------------------------------------------------------------------------

TASK_TITLES = [
    ("a_", "a. Weather ([E1], clip level)"),
    ("b_", "b. Day / night ([E6], clip level)"),
    ("c_", "c. Traffic light ([E2], sampled frames)"),
    ("d_", "d. Crosswalk presence ([E7] shipped, [X7] counterfactual, [E10] scene_seg)"),
    ("e_", "e. Pedestrian detection ([B1] 1 Hz / [B2] dense) at IoU >= 0.5"),
    ("f_", "f. Crossing judge ([C3])"),
    ("g_", "g. Ego-motion / static-camera gate ([B3])"),
    ("h_", "h. Looking behaviour ([P12])"),
    ("i_", "i. Age / gender ([P6])"),
    ("j_", "j. Risky crossing coverage ([C1])"),
    ("k_", "k. Traffic signs ([E3])"),
]


def md_table(df):
    lines = ["| task | metric | value [95% CI] | n | clips | base rate | note |",
             "|---|---|---|---:|---:|---:|---|"]
    for r in df.itertuples():
        row = {"value": r.value, "ci_low": r.ci_low, "ci_high": r.ci_high}
        note = str(r.note) if isinstance(r.note, str) else ""
        lines.append(f"| {r.task} | {r.metric} | {cell(row)} | {int(r.n) if pd.notna(r.n) else ''} | "
                     f"{int(r.n_clips) if pd.notna(r.n_clips) else ''} | {fmt(r.base_rate)} | {note} |")
    return "\n".join(lines)


def run_log_summary(results_root):
    p = os.path.join(results_root, "_harness", "run_log.csv")
    if not os.path.exists(p):
        return None
    log = pd.read_csv(p)
    last = log.sort_values("started_utc").groupby(["clip", "mode"]).tail(1)
    return last


SENSITIVITY_KEYS = [
    ("a_weather", "accuracy"), ("a_weather", "kappa"), ("b_daynight", "kappa"),
    ("c_traffic_light_visible_presence", "recall"), ("d_crosswalk_E7_frame", "f1"),
    ("d_crosswalk_E10_frame", "f1"), ("d_crosswalk_E10_frac", "roc_auc"),
    ("e_detection_B2", "recall_kind=b"), ("f_crossing_any_track", "f1"),
    ("f_crossing_any_track", "kappa"), ("g_ego_step_px", "roc_auc_stopped"),
    ("h_looking_pedx_window", "f1"), ("i_age_gender_gender", "accuracy"),
]


def sensitivity_table(m_all, m_street):
    lines = ["| task | metric | all clips | street clips only |", "|---|---|---|---|"]
    for task, metric in SENSITIVITY_KEYS:
        a, b = get(m_all, task, metric), get(m_street, task, metric)
        if a is None and b is None:
            continue
        fa = f"{cell(a)} (clips {a['n_clips']:.0f})" if a else "n/a"
        fb = f"{cell(b)} (clips {b['n_clips']:.0f})" if b else "n/a"
        lines.append(f"| {task} | {metric} | {fa} | {fb} |")
    return "\n".join(lines)


def write_report(out_dir, m, cms, details, clips, gt, results_root, interval, verdict_rows,
                 requested_clips, m_street=None, n_street=0):
    from modules.provenance import git_state, package_versions
    sha, dirty = git_state()
    pk = package_versions()
    log = run_log_summary(results_root)
    lines = []
    A = lines.append
    A("# JAAD validation of PedX detectors\n")
    A(f"Generated {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by "
      "`python -m validation.jaad.report`.\n")
    A("## Protocol\n")
    A(f"- Data: JAAD (Joint Attention in Autonomous Driving; Rasouli, Kotseruba, Tsotsos), "
      f"{len(gt['clips'])} dashcam clips, annotations parsed with the dataset's own `jaad_data.py`.")
    A(f"- Clips scored: **{len(clips)}** of {requested_clips} requested "
      f"({len(gt['clips'])} in JAAD). Each task uses the clips for which its producer output exists; "
      "`clips` in every table is the per-metric count.")
    A(f"- PedX run at the production setting `--analysis_interval {interval}`, with the production "
      "module code, in an isolated results root (`PEDX_RESULTS_ROOT`) so nothing reaches the "
      "published dataset.")
    A(f"- Code: git `{sha}`{' (working tree dirty)' if dirty else ''}; python {pk.get('python')}, "
      f"torch {pk.get('torch')}, ultralytics {pk.get('ultralytics')}, opencv {pk.get('opencv-python')}.")
    A(f"- Statistics: every metric is a point estimate with a 95% percentile CI from a **cluster "
      f"bootstrap over clips** (B = {S.N_BOOT}, seed {S.SEED}); frames and pedestrians inside a clip "
      "are not independent. `base rate` is the trivial-classifier reference (positive prevalence for "
      "P/R/F1/PR-AUC, majority-class share for accuracy, 0.5 for ROC-AUC, 0 for kappa).")
    A("- Frame alignment: JAAD frames are 0-indexed; tracking outputs are 1-indexed, detectors "
      "0-indexed (`validation/jaad/score.py`, unit-tested).")
    A("- Domain caveat: JAAD is dashcam footage from a few countries; passing here is necessary, "
      "not sufficient, for the global dataset. Failing here is informative everywhere.\n")
    if log is not None:
        bad = log[log["status"] != "ok"]
        A(f"- Run log: {len(log)} clip x mode runs, {len(bad)} not ok"
          + ("." if bad.empty else
             ": " + ", ".join(f"{k} x{v}" for k, v in bad.groupby('mode').size().items()) + "."))
        A("")

    A("## Verdicts\n")
    A(f"Rules (`validation/jaad/report.py::verdicts`): each metric's 95% CI is tested against a bar -- "
      f"*pass* when the whole CI is above it, *fail* only when the whole CI is below it, otherwise "
      f"*inconclusive* (the CI straddles the bar). A module is *usable as-is* when its key metric "
      f"passes (F1 >= {BAR_F1}; weather kappa >= {BAR_KAPPA_WEATHER} and accuracy above "
      f"the majority class; day/night kappa >= {BAR_KAPPA_DAYNIGHT}; detection recall >= "
      f"{BAR_RECALL_TRACK}; ego-motion AUC >= {BAR_AUC_EGO}); *recalibrated threshold* when the "
      f"continuous score has AUC lower bound >= {BAR_AUC} and a cross-validated threshold reaches "
      f"F1 >= {BAR_F1}; *inconclusive* below {MIN_CLIPS} clips or {MIN_POSITIVES} positives.\n")
    A("| module | verdict | evidence |")
    A("|---|---|---|")
    for v in verdict_rows:
        scope = f" (**scope:** {v['scope']})" if v["scope"] else ""
        A(f"| {v['module']} | **{v['verdict']}**{scope} | {v['evidence']} |")
    A("")

    A("## Results\n")
    body = m[~m["task"].str.startswith("z_")]
    for prefix, title in TASK_TITLES:
        sub = body[body["task"].str.startswith(prefix)]
        if sub.empty:
            continue
        A(f"### {title}\n")
        A(md_table(sub))
        A("")

    if m_street is not None:
        A("## Sensitivity: street clips only\n")
        A(f"JAAD includes parking-lot and garage clips ({len(clips) - n_street} of the {len(clips)} "
          f"scored), a scene type PedX is not designed for. The same metrics on the {n_street} "
          "street clips:\n")
        A(sensitivity_table(m, m_street))
        A("")
    A("## Confusion matrices\n")
    A("Rows = JAAD label, columns = PedX. Also written as `cm_*.csv`.\n")
    for name, cm in cms.items():
        A(f"**{name}**\n")
        A("```")
        A(cm.to_string())
        A("```\n")

    A("## Coverage\n")
    cov = m[m["task"].str.startswith("z_")]
    if not cov.empty:
        A(md_table(cov))
        A("")
    findings = Path(__file__).with_name("findings.md")
    if findings.exists():
        A("## Findings (hand-written; bugs located while validating)\n")
        A(findings.read_text(encoding="utf-8").strip())
        A("")
    with open(os.path.join(out_dir, "JAAD_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_root", default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--gt_dir", default=DEFAULT_GT_DIR)
    ap.add_argument("--clips_file", default=None, help="score only these clips")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    warnings.simplefilter("ignore")

    gt = G.load_tables(args.gt_dir)
    runs = S.Runs(args.results_root)
    if args.clips_file:
        requested = [ln.split("#")[0].strip() for ln in open(args.clips_file, encoding="utf-8")]
        requested = [c for c in requested if c]
    else:
        requested = sorted(gt["clips"]["clip"])
    clips = [c for c in requested if runs.has(c, "B1") or runs.has(c, "E1") or runs.has(c, "E2")]
    if not clips:
        print("No clips with outputs under", args.results_root)
        return 1
    print(f"scoring {len(clips)} of {len(requested)} clips")

    m, cms, details = S.score_all(gt, runs, clips, args.interval)
    os.makedirs(args.out, exist_ok=True)
    m.to_csv(os.path.join(args.out, "jaad_metrics.csv"), index=False)
    for name, cm in cms.items():
        cm.to_csv(os.path.join(args.out, f"cm_{name}.csv"))
    street = set(gt["clips"].loc[gt["clips"]["road_type"] == "street", "clip"])
    clips_st = [c for c in clips if c in street]
    m_st = None
    if 0 < len(clips_st) < len(clips):
        m_st, _, _ = S.score_all(gt, runs, clips_st, args.interval)
        m_st.to_csv(os.path.join(args.out, "jaad_metrics_street.csv"), index=False)
    write_report(args.out, m, cms, details, clips, gt, args.results_root, args.interval,
                 verdicts(m), len(requested), m_st, len(clips_st))
    print(f"wrote {args.out}/jaad_metrics.csv, {len(cms)} confusion matrices, JAAD_REPORT.md")
    for v in verdicts(m):
        print(f"  {v['module']:<55} {v['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
