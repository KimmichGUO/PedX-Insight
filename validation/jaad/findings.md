Bugs and validity problems located while validating (all numbers are from the 346-clip run
above; "fixed" names the commit). Verdict rules are in `report.py`; this section is the part a
rule cannot say.

### 1. The crosswalk detector never detected crosswalks (fixed: replaced, 02019ba)
`modules/crosswalk/crosswalk.py:78` keeps `cls_id == 0`, but the shipped `best.pt` names class 0
`Flexible_delineator_post` (class 4 is `crosswalk`). The weights have been byte-identical since the
repository's first commit, so every crosswalk-derived number in the published dataset
(`crosswalk_prob`, `crosswalk_usage_ratio`, per-pedestrian `crosswalk` / `crosswalk_use_or_not`,
`crosswalk_coeff.csv`, and the `[S2]` / `[V8]` crosswalk boxes) came from a detector that does not
detect crosswalks. Against JAAD's per-frame `ped_crossing` label: class 0 (as shipped) ROC-AUC 0.504
[0.492, 0.515], never fires at its threshold; class 4 (counterfactual `[X7]` probe, same weights and
size gate) ROC-AUC 0.490 [0.437, 0.545] -- so correcting the class index would not have helped. The
cross-validated "best-threshold F1" of 0.657 for class 4 is the always-positive classifier (2p/(1+p)
with p = 0.489), not skill. Replaced by Mask2Former / Mapillary Vistas (`modules/crosswalk/
crosswalk_seg.py`): ROC-AUC 0.947, F1 0.91 with a threshold chosen out of fold. Calibrated on JAAD
(Canada/Ukraine): transfer to other cities is unverified. Already-analysed folders keep the invalid
values until their videos are re-analysed.

### 2. `cv2.destroyAllWindows()` crashed crosswalk and traffic-sign runs (fixed: 889f8f0)
With opencv-python, -contrib and -headless 5.0.0.93 installed side by side, `destroyAllWindows()`
raises `cv2.error` once torch/ultralytics is imported. In `crosswalk.py:115` and `traffic_sign.py:80`
it ran before the CSV was written, so those modes lost their whole output on any fresh run since the
packages were installed (2026-08-11); `sidewalk_detect.py:136` is guarded by `--noshow`. The harness
shipped a shim until the modules were guarded.

### 3. The protective-sign class list was wrong (fixed: `risky_crossing.py`, `PROTECTIVE_SIGNS_1`)
`risky_crossing.py` treated `{w57, pg, i1}` as the "marked crossing" signs. TT100K's pedestrian-
crossing class is `ip`: on frames where `[E3]` flags it, JAAD labels a pedestrian sign 97% of the
time (453 frames, 99 clips; base rate 29%). The original set scored F1 0.02 (recall 0.01); `ip` alone
scores F1 0.72. `pg` fires at base rate (precision 0.19, n = 42); `w57` / `i1` fire on too few frames
to judge. So that failure was the class list, not the detector.

### 4. JAAD's `traffic_light` attribute is not a visibility label (scoring corrected)
In video_0030 the label is "red" on all 180 frames and no light is in view in any frame inspected.
Scoring "did PedX see a light" against it blames the detector for invisible lights, so `[E2]` is also
scored on frames where JAAD labels a light AND the independent Vistas segmenter sees a light region
(a proxy, not ground truth). There, `[E2]` detects a light in 42% [17%, 64%] of frames (19 clips); when
it reports a colour it is right 56/56 times, and it almost never fires where no light is in view (0.5%).
JAAD labels a light on 5.5% of clips only, so this task can never be well powered.

### 5. The weather classifier is not usable (`modules/weather`)
Kappa -0.055 [-0.115, 0.006]; accuracy 0.32 against 0.48 for always guessing the majority class. It
predicts "rain" for 29% of clips (100 of 343) while 8% (28) are rainy, and finds 7 of those 28. The
published dataset's "rain" label (14 of 28 videos) should be read in this light.

### 4b. Day / night: the test is underpowered, and "Evening" is overused
JAAD has 4 night clips of 346. PedX labels all 4 "Night", but also labels 84 of 336 daytime clips
"Evening" and 4 daytime clips "Night". Treating Evening as night gives kappa 0.08 (the verdict's
stricter mapping); treating it as day gives 0.61 [0.19, 0.89], whose interval is wide because there
are only 4 night clips. Whether PedX can detect night cannot be settled here; the three-way
Day/Evening/Night label is not usable.

### 6. The crossing judge undercounts crossings; tracks are fragmented (`[C3]`, `[B2]`)
End to end, `[C3]` flags 56% of JAAD's crossing pedestrians (recall 0.56 [0.51, 0.62]) at precision
0.98 (0.93 if JAAD's "irrelevant" pedestrians count as non-crossers), so published crossing counts
are lower bounds. 40% of pedestrians are split over two or more PedX track ids (mean 1.63 ids per
pedestrian [1.54, 1.73]), which is also why one-to-one matching understates recall (0.47) and
why any per-track quantity (duration, speed, waiting time) is measured on fragments.

### 7. Inferred gender and age are not reliable (`[P6]`)
A face is found for only 41% of tracked pedestrians; gender accuracy is 0.548 against 0.548 for
always guessing the majority class (kappa 0.14 [0.04, 0.24], 272 matched pedestrians); age-group accuracy is 0.783 against 0.786 for
always guessing the majority group. Do not use the gender or age columns, or statistics split by
them, for cross-city claims.

### 8. Pedestrian detection and the static-camera gate (`[B2]`, `[B3]`)
Recall on JAAD's behaviourally annotated (road-interacting) pedestrians is 0.81 [0.78, 0.83]; it is
0.08 for fully occluded and 0.20 for sub-50 px boxes. Precision is at least 0.87 (JAAD does not box
everyone). The ego-motion step separates "ego vehicle stopped" well (ROC-AUC 0.945), but the
production gate that decides when speed is measured has precision 0.72: about 28% of the frames it
calls static are not stopped.

### 9. Harness notes
JAAD's own "no traffic light" label is the text `n/a`, which `pandas.read_csv` turns into NaN; read
the ground-truth tables with `validation.jaad.gt.load_tables`. Frame ids are 1-indexed for tracking
outputs and 0-indexed for detectors; both are covered by unit tests. Parking lots and garages (56 of
346 clips) do not change any conclusion (street-only table above).
