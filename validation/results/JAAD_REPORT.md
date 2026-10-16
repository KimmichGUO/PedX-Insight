# JAAD validation of PedX detectors

Generated 2026-10-16 19:27 UTC by `python -m validation.jaad.report`.

## Protocol

- Data: JAAD (Joint Attention in Autonomous Driving; Rasouli, Kotseruba, Tsotsos), 346 dashcam clips, annotations parsed with the dataset's own `jaad_data.py`.
- Clips scored: **346** of 346 requested (346 in JAAD). Each task uses the clips for which its producer output exists; `clips` in every table is the per-metric count.
- PedX run at the production setting `--analysis_interval 1.0`, with the production module code, in an isolated results root (`PEDX_RESULTS_ROOT`) so nothing reaches the published dataset.
- Code: git `8be2822ecb43ccf0d3bc5af29a4e975a03b85bfa`; python 3.12.9, torch 2.13.0+cu126, ultralytics 8.4.117, opencv 5.0.0.93.
- Statistics: every metric is a point estimate with a 95% percentile CI from a **cluster bootstrap over clips** (B = 2000, seed 20261010); frames and pedestrians inside a clip are not independent. `base rate` is the trivial-classifier reference (positive prevalence for P/R/F1/PR-AUC, majority-class share for accuracy, 0.5 for ROC-AUC, 0 for kappa).
- Frame alignment: JAAD frames are 0-indexed; tracking outputs are 1-indexed, detectors 0-indexed (`validation/jaad/score.py`, unit-tested).
- Domain caveat: JAAD is dashcam footage from a few countries; passing here is necessary, not sufficient, for the global dataset. Failing here is informative everywhere.

- Run log: 5481 clip x mode runs, 0 not ok.

## Verdicts

Rules (`validation/jaad/report.py::verdicts`): each metric's 95% CI is tested against a bar -- *pass* when the whole CI is above it, *fail* only when the whole CI is below it, otherwise *inconclusive* (the CI straddles the bar). A module is *usable as-is* when its key metric passes (F1 >= 0.7; weather kappa >= 0.4 and accuracy above the majority class; day/night kappa >= 0.6; detection recall >= 0.8; ego-motion AUC >= 0.9); *recalibrated threshold* when the continuous score has AUC lower bound >= 0.8 and a cross-validated threshold reaches F1 >= 0.7; *inconclusive* below 15 clips or 30 positives.

| module | verdict | evidence |
|---|---|---|
| [E1] weather | **not usable for cross-city claims** | accuracy 0.320 [0.273, 0.366] vs majority-class 0.483; kappa -0.055 [-0.115, 0.006]; rain recall 0.250 [0.097, 0.429] on 344 clips |
| [E6] day/night | **not usable for cross-city claims** | accuracy 0.745 [0.699, 0.791] vs majority-class 0.986; kappa 0.077 [0.017, 0.143] with PedX 'Evening' counted as night (the stricter mapping, used for the verdict), 0.608 [0.189, 0.887] with Evening counted as day, on 345 clips |
| [E2] traffic light | **not usable for cross-city claims** | on frames with a labelled light that is also in view (Vistas proxy): detects a light in 0.424 [0.170, 0.642] of frames, colour agreement 1.000 [1.000, 1.000]; false-positive rate where no light is in view 0.005 [0.000, 0.013]. JAAD labels a light on frames where none is in view: 0.077 [0.017, 0.174] of labelled frames. All labelled frames (invalid as visibility): recall 0.392 [0.146, 0.628] |
| [E7] crosswalk as shipped (class 0) | **not usable for cross-city claims** | frame-level precision n/a, recall 0.000 [0.000, 0.000], F1 0.000 [0.000, 0.000] on 346 clips; continuous-score ROC-AUC 0.504 [0.492, 0.515] |
| [E7] crosswalk, same weights, class 4 (COUNTERFACTUAL) | **not usable for cross-city claims** | frame-level precision 0.286 [0.000, 0.835], recall 0.006 [0.000, 0.016], F1 0.012 [0.000, 0.032] on 346 clips; continuous-score ROC-AUC 0.490 [0.437, 0.545], cross-validated best-threshold F1 0.657 [0.606, 0.702] |
| [E10] scene_seg crosswalk (Mask2Former / Vistas) | **usable as-is** | frame-level precision 0.928 [0.901, 0.951], recall 0.824 [0.788, 0.857], F1 0.873 [0.847, 0.895] on 346 clips; continuous-score ROC-AUC 0.947 [0.932, 0.960], cross-validated best-threshold F1 0.911 [0.888, 0.931] |
| [B1]/[B2] pedestrian detection | **inconclusive** | [B2] recall on behaviourally annotated pedestrians (PRIMARY, a priori) 0.806 [0.783, 0.827]; all JAAD boxes 0.628 [0.610, 0.647] (distant bystanders alone: 0.532 [0.508, 0.555]); unoccluded 0.774 [0.756, 0.792]; unoccluded and >=50 px 0.836 [0.824, 0.848], >=100 px 0.949 [0.941, 0.957] (exploratory sub-populations); precision (lower bound; JAAD does not box everyone) 0.871 [0.860, 0.882]; CI straddles the bar -- the data cannot separate the module from the bar |
| [C3] crossing judge | **not usable for cross-city claims** | end to end over labelled behavioural pedestrians (flagged if any overlapping track is crossed): precision 0.979 [0.957, 0.996], recall 0.564 [0.509, 0.619], F1 0.715 [0.669, 0.760], kappa 0.256 [0.190, 0.328]; precision with JAAD 'irrelevant' pedestrians counted as not crossing 0.934 [0.896, 0.967]; one-to-one-matched recall 0.467 [0.416, 0.519] is lower because 0.397 [0.349, 0.446] of pedestrians are split over >= 2 PedX track ids; match rate 0.986 [0.977, 0.995]; JAAD behavioural pedestrians cross 0.916 of the time, so kappa is the informative number |
| [B3] ego-motion / static-camera gate | **inconclusive** | ROC-AUC for 'ego vehicle stopped': translation step 0.945 [0.928, 0.958], smoothed step 0.950 [0.933, 0.963], bbox-expansion 0.866 [0.841, 0.889]; production gate precision 0.717 [0.633, 0.785], recall 0.682 [0.603, 0.746], F1 0.699 [0.629, 0.750]; cross-validated best-threshold F1 on the smoothed step 0.740 [0.675, 0.789]; CI straddles the bar -- the data cannot separate the module from the bar |
| [P12] looking behaviour | **inconclusive** | look-before-crossing vs JAAD 'look': F1 0.735 [0.669, 0.794], kappa 0.393 [0.265, 0.517] on 240 pedestrians / 173 clips; CI straddles the bar -- the data cannot separate the module from the bar |
| [P6] gender (inferred) | **not usable for cross-city claims** | gender accuracy 0.548 [0.481, 0.612] vs majority 0.548, kappa 0.139 [0.036, 0.241] on 272 matched pedestrians; face detected for 0.414 [0.367, 0.462] of tracked pedestrians |
| [E3] traffic sign (pedestrian sign) | **inconclusive** | JAAD pedestrian-crossing sign, with the class set production risky_crossing.py uses ('ip' was added after this benchmark found the earlier set {w57, pg, i1} scored F1 0.02; see findings): precision 0.898 [0.843, 0.942], recall 0.571 [0.493, 0.650], F1 0.698 [0.632, 0.756]. Exploratory (class 'ip' alone, chosen after interim results): precision 0.969 [0.943, 0.989], recall 0.569 [0.490, 0.647], F1 0.717 [0.652, 0.777] -- the other classes in the set mostly add false positives; see the per-class `k_signs_class[...]` rows; CI straddles the bar -- the data cannot separate the module from the bar |
| [C1] risky crossing (coverage only) | **inconclusive** | coverage given a JAAD-labelled light in the crossing window: 0.250 [0.000, 0.556] (16 such crossings); unconditional share of crossings with an observed signal 0.022 [0.006, 0.044] on 314 crossings is capped by JAAD's light prevalence (0.051 of windows). JAAD has no risky label, so only coverage can be validated; too few clips or positives |

## Results

### a. Weather ([E1], clip level)

| task | metric | value [95% CI] | n | clips | base rate | note |
|---|---|---|---:|---:|---:|---|
| a_weather | accuracy | 0.320 [0.273, 0.366] | 344 | 344 | 0.483 | PedX shine/sunrise->clear; JAAD cloud->cloudy; JAAD n/a dropped |
| a_weather | kappa | -0.055 [-0.115, 0.006] | 344 | 344 | 0.000 | PedX shine/sunrise->clear; JAAD cloud->cloudy; JAAD n/a dropped |
| a_weather_rain | precision | 0.070 [0.022, 0.125] | 344 | 344 | 0.081 | 'rain' one-vs-rest at clip level |
| a_weather_rain | recall | 0.250 [0.097, 0.429] | 344 | 344 | 0.081 | 'rain' one-vs-rest at clip level |
| a_weather_rain | f1 | 0.109 [0.036, 0.188] | 344 | 344 | 0.151 | 'rain' one-vs-rest at clip level |
| a_weather | pred_rain_share | 0.291 | 344 | 344 | 0.081 | base_rate here = JAAD rain share |

### b. Day / night ([E6], clip level)

| task | metric | value [95% CI] | n | clips | base rate | note |
|---|---|---|---:|---:|---:|---|
| b_daynight | accuracy | 0.745 [0.699, 0.791] | 345 | 345 | 0.986 | PedX Evening->night |
| b_daynight | kappa | 0.077 [0.017, 0.143] | 345 | 345 | 0.000 | PedX Evening->night |
| b_daynight_alt | accuracy | 0.986 [0.971, 0.997] | 345 | 345 | 0.986 | sensitivity: PedX Evening->day |
| b_daynight_alt | kappa | 0.608 [0.189, 0.887] | 345 | 345 | 0.000 | sensitivity: PedX Evening->day |

### c. Traffic light ([E2], sampled frames)

| task | metric | value [95% CI] | n | clips | base rate | note |
|---|---|---|---:|---:|---:|---|
| c_traffic_light_presence | precision | 0.718 [0.358, 0.945] | 2691 | 346 | 0.053 | frame-level, [E2] sampled frames |
| c_traffic_light_presence | recall | 0.392 [0.146, 0.628] | 2691 | 346 | 0.053 | frame-level, [E2] sampled frames |
| c_traffic_light_presence | f1 | 0.507 [0.206, 0.718] | 2691 | 346 | 0.101 | frame-level, [E2] sampled frames |
| c_traffic_light_state | accuracy | 1.000 [1.000, 1.000] | 56 | 8 | 0.589 | frames where both JAAD and PedX see a light; PedX yellow counts wrong |
| c_traffic_light_presence_clip | precision | 0.667 [0.375, 0.917] | 346 | 346 | 0.055 | clip has any light |
| c_traffic_light_presence_clip | recall | 0.421 [0.208, 0.647] | 346 | 346 | 0.055 | clip has any light |
| c_traffic_light_presence_clip | f1 | 0.516 [0.276, 0.714] | 346 | 346 | 0.104 | clip has any light |
| c_traffic_light | clip_rate_spearman | 0.510 [0.268, 0.713] | 346 | 346 | 0.000 | per-clip detected-frame rate vs JAAD labelled-frame rate |
| c_traffic_light_visible_label_validity | labelled_frames_with_no_light_in_view | 0.077 [0.017, 0.174] | 143 | 19 | n/a | share of JAAD traffic_light-labelled frames where [E10] sees no light region |
| c_traffic_light_visible_presence | recall | 0.424 [0.170, 0.642] | 132 | 19 | 1.000 | frames with a JAAD light label AND a light region in view; [E2] reports any colour |
| c_traffic_light_visible_state | accuracy | 1.000 [1.000, 1.000] | 56 | 8 | 0.589 | colour agreement where [E2] reports a colour on a visible, labelled light |
| c_traffic_light_visible_no_light | false_positive_rate | 0.005 [0.000, 0.013] | 2320 | 322 | 0.000 | [E2] reports a colour although JAAD says n/a and no light region is in view |

### d. Crosswalk presence ([E7] shipped, [X7] counterfactual, [E10] scene_seg)

| task | metric | value [95% CI] | n | clips | base rate | note |
|---|---|---|---:|---:|---:|---|
| d_crosswalk_E7_frame | precision | n/a | 2691 | 346 | 0.489 | production [E7] crosswalk_detected (smoothed) |
| d_crosswalk_E7_frame | recall | 0.000 [0.000, 0.000] | 2691 | 346 | 0.489 | production [E7] crosswalk_detected (smoothed) |
| d_crosswalk_E7_frame | f1 | 0.000 [0.000, 0.000] | 2691 | 346 | 0.657 | production [E7] crosswalk_detected (smoothed) |
| d_crosswalk_E7_clip | precision | n/a | 346 | 346 | 0.546 | clip positive if any sampled frame positive |
| d_crosswalk_E7_clip | recall | 0.000 [0.000, 0.000] | 346 | 346 | 0.546 | clip positive if any sampled frame positive |
| d_crosswalk_E7_clip | f1 | 0.000 [0.000, 0.000] | 346 | 346 | 0.707 | clip positive if any sampled frame positive |
| d_crosswalk_E7 | clip_rate_spearman | n/a | 346 | 346 | 0.000 | per-clip detected-frame rate vs JAAD labelled-frame rate |
| d_crosswalk_E7_frame | pred_positive_rate | 0.000 [0.000, 0.000] | 2691 | 346 | 0.489 | share of sampled frames flagged; base_rate = JAAD share |
| d_crosswalk_E10_frame | precision | 0.928 [0.901, 0.951] | 2691 | 346 | 0.489 | [E10] crosswalk_present at shipped threshold |
| d_crosswalk_E10_frame | recall | 0.824 [0.788, 0.857] | 2691 | 346 | 0.489 | [E10] crosswalk_present at shipped threshold |
| d_crosswalk_E10_frame | f1 | 0.873 [0.847, 0.895] | 2691 | 346 | 0.657 | [E10] crosswalk_present at shipped threshold |
| d_crosswalk_E10_clip | precision | 0.860 [0.812, 0.905] | 346 | 346 | 0.546 | clip positive if any sampled frame positive |
| d_crosswalk_E10_clip | recall | 0.974 [0.948, 0.994] | 346 | 346 | 0.546 | clip positive if any sampled frame positive |
| d_crosswalk_E10_clip | f1 | 0.913 [0.882, 0.940] | 346 | 346 | 0.707 | clip positive if any sampled frame positive |
| d_crosswalk_E10 | clip_rate_spearman | 0.859 [0.823, 0.890] | 346 | 346 | 0.000 | per-clip detected-frame rate vs JAAD labelled-frame rate |
| d_crosswalk_E10_frame | pred_positive_rate | 0.434 [0.386, 0.482] | 2691 | 346 | 0.489 | share of sampled frames flagged; base_rate = JAAD share |
| d_crosswalk_X7cls4_frame | precision | 0.286 [0.000, 0.835] | 2691 | 346 | 0.489 | COUNTERFACTUAL: same model, class 4, conf>=0.35 + module smoothing |
| d_crosswalk_X7cls4_frame | recall | 0.006 [0.000, 0.016] | 2691 | 346 | 0.489 | COUNTERFACTUAL: same model, class 4, conf>=0.35 + module smoothing |
| d_crosswalk_X7cls4_frame | f1 | 0.012 [0.000, 0.032] | 2691 | 346 | 0.657 | COUNTERFACTUAL: same model, class 4, conf>=0.35 + module smoothing |
| d_crosswalk_X7cls4_clip | precision | 0.286 [0.000, 0.667] | 346 | 346 | 0.546 | clip positive if any sampled frame positive |
| d_crosswalk_X7cls4_clip | recall | 0.011 [0.000, 0.027] | 346 | 346 | 0.546 | clip positive if any sampled frame positive |
| d_crosswalk_X7cls4_clip | f1 | 0.020 [0.000, 0.052] | 346 | 346 | 0.707 | clip positive if any sampled frame positive |
| d_crosswalk_X7cls4 | clip_rate_spearman | -0.061 [-0.149, 0.049] | 346 | 346 | 0.000 | per-clip detected-frame rate vs JAAD labelled-frame rate |
| d_crosswalk_X7cls4_frame | pred_positive_rate | 0.010 [0.002, 0.023] | 2691 | 346 | 0.489 | share of sampled frames flagged; base_rate = JAAD share |
| d_crosswalk_E10_frac | roc_auc | 0.947 [0.932, 0.960] | 2691 | 346 | 0.500 | crosswalk_frac as a continuous score |
| d_crosswalk_E10_frac | pr_auc | 0.942 [0.923, 0.958] | 2691 | 346 | 0.489 | crosswalk_frac as a continuous score |
| d_crosswalk_E10_frac | best_f1_threshold | 0.000 | 2691 | 346 | n/a | in-sample argmax-F1 threshold on the score |
| d_crosswalk_E10_frac | f1_at_best_threshold | 0.914 [0.891, 0.933] | 2691 | 346 | 0.657 | in-sample (optimistic); P=0.902 R=0.926 |
| d_crosswalk_E10_frac | f1_cv_threshold | 0.911 [0.888, 0.931] | 2691 | 346 | 0.657 | threshold chosen on other clips (5-fold by clip) |
| d_crosswalk_X7cls4_conf | roc_auc | 0.490 [0.437, 0.545] | 2691 | 346 | 0.500 | COUNTERFACTUAL class-4 max confidence (raw, unsmoothed) |
| d_crosswalk_X7cls4_conf | pr_auc | 0.464 [0.403, 0.535] | 2691 | 346 | 0.489 | COUNTERFACTUAL class-4 max confidence (raw, unsmoothed) |
| d_crosswalk_X7cls4_conf | best_f1_threshold | 0.000 | 2691 | 346 | n/a | in-sample argmax-F1 threshold on the score |
| d_crosswalk_X7cls4_conf | f1_at_best_threshold | 0.657 [0.606, 0.702] | 2691 | 346 | 0.657 | in-sample (optimistic); P=0.489 R=1.000 |
| d_crosswalk_X7cls4_conf | f1_cv_threshold | 0.657 [0.606, 0.702] | 2691 | 346 | 0.657 | threshold chosen on other clips (5-fold by clip) |
| d_crosswalk_X7cls0_conf | roc_auc | 0.504 [0.492, 0.515] | 2691 | 346 | 0.500 | class-0 ('Flexible_delineator_post') max confidence = what [E7] thresholds |
| d_crosswalk_X7cls0_conf | pr_auc | 0.491 [0.438, 0.545] | 2691 | 346 | 0.489 | class-0 ('Flexible_delineator_post') max confidence = what [E7] thresholds |
| d_crosswalk_X7cls0_conf | best_f1_threshold | 0.000 | 2691 | 346 | n/a | in-sample argmax-F1 threshold on the score |
| d_crosswalk_X7cls0_conf | f1_at_best_threshold | 0.657 [0.606, 0.702] | 2691 | 346 | 0.657 | in-sample (optimistic); P=0.489 R=1.000 |
| d_crosswalk_X7_probe_check | agreement_with_E7 | 1.000 | 2691 | 346 | n/a | probe class-0 reconstruction vs actual [E7] (should be ~1.0) |
| d_crosswalk_E10_minus_E7 | delta_f1 | 0.873 [0.847, 0.895] | 2691 | 346 | 0.000 | paired on common frames; CI excluding 0 = significant |
| d_crosswalk_E10_minus_E7 | delta_recall | 0.824 [0.788, 0.857] | 2691 | 346 | 0.000 | paired on common frames; CI excluding 0 = significant |
| d_crosswalk_E10_minus_E7 | delta_precision | n/a | 2691 | 346 | 0.000 | paired on common frames; CI excluding 0 = significant |

### e. Pedestrian detection ([B1] 1 Hz / [B2] dense) at IoU >= 0.5

| task | metric | value [95% CI] | n | clips | base rate | note |
|---|---|---|---:|---:|---:|---|
| e_detection_B1 | recall | 0.633 [0.615, 0.653] | 12185 | 346 | n/a | all non-'p' JAAD boxes on the grid |
| e_detection_B1 | precision_lower_bound | 0.866 [0.854, 0.878] | 8908 | 346 | n/a | JAAD does not box every pedestrian -> precision is a LOWER bound |
| e_detection_B1 | recall_occlusion=none | 0.780 [0.761, 0.798] | 8914 | 333 | n/a | recall stratified by GT occlusion |
| e_detection_B1 | recall_occlusion=part | 0.382 [0.350, 0.416] | 1740 | 271 | n/a | recall stratified by GT occlusion |
| e_detection_B1 | recall_occlusion=full | 0.067 [0.052, 0.084] | 1531 | 251 | n/a | recall stratified by GT occlusion |
| e_detection_B1 | recall_height=<50px | 0.194 [0.159, 0.231] | 1221 | 184 | n/a | recall stratified by GT height |
| e_detection_B1 | recall_height=50-100px | 0.531 [0.505, 0.557] | 5117 | 296 | n/a | recall stratified by GT height |
| e_detection_B1 | recall_height=100-200px | 0.770 [0.742, 0.796] | 3417 | 313 | n/a | recall stratified by GT height |
| e_detection_B1 | recall_height=>=200px | 0.877 [0.856, 0.898] | 2430 | 298 | n/a | recall stratified by GT height |
| e_detection_B1 | recall_kind=b | 0.814 [0.790, 0.836] | 4258 | 320 | n/a | recall stratified by GT kind |
| e_detection_B1 | recall_kind=ped | 0.537 [0.510, 0.560] | 7927 | 297 | n/a | recall stratified by GT kind |
| e_detection_B1 | recall_unoccluded_h>=50px | 0.839 [0.825, 0.852] | 8022 | 331 | n/a | combined GT sub-population (exploratory: chosen after interim results) |
| e_detection_B1 | recall_unoccluded_h>=100px | 0.947 [0.936, 0.956] | 4497 | 326 | n/a | combined GT sub-population (exploratory: chosen after interim results) |
| e_detection_B1 | recall_not_fully_occluded_h>=50px | 0.766 [0.751, 0.781] | 9637 | 334 | n/a | combined GT sub-population (exploratory: chosen after interim results) |
| e_detection_B2 | recall | 0.628 [0.610, 0.647] | 186785 | 346 | n/a | all non-'p' JAAD boxes on the grid |
| e_detection_B2 | precision_lower_bound | 0.871 [0.860, 0.882] | 134677 | 346 | n/a | JAAD does not box every pedestrian -> precision is a LOWER bound |
| e_detection_B2 | recall_occlusion=none | 0.774 [0.756, 0.792] | 136367 | 335 | n/a | recall stratified by GT occlusion |
| e_detection_B2 | recall_occlusion=part | 0.374 [0.345, 0.404] | 26624 | 328 | n/a | recall stratified by GT occlusion |
| e_detection_B2 | recall_occlusion=full | 0.078 [0.066, 0.092] | 23794 | 306 | n/a | recall stratified by GT occlusion |
| e_detection_B2 | recall_height=<50px | 0.201 [0.171, 0.233] | 19872 | 224 | n/a | recall stratified by GT height |
| e_detection_B2 | recall_height=50-100px | 0.529 [0.505, 0.553] | 79109 | 314 | n/a | recall stratified by GT height |
| e_detection_B2 | recall_height=100-200px | 0.768 [0.743, 0.792] | 52191 | 321 | n/a | recall stratified by GT height |
| e_detection_B2 | recall_height=>=200px | 0.882 [0.865, 0.901] | 35613 | 313 | n/a | recall stratified by GT height |
| e_detection_B2 | recall_kind=b | 0.806 [0.783, 0.827] | 65692 | 320 | n/a | recall stratified by GT kind |
| e_detection_B2 | recall_kind=ped | 0.532 [0.508, 0.555] | 121093 | 303 | n/a | recall stratified by GT kind |
| e_detection_B2 | recall_unoccluded_h>=50px | 0.836 [0.824, 0.848] | 121679 | 334 | n/a | combined GT sub-population (exploratory: chosen after interim results) |
| e_detection_B2 | recall_unoccluded_h>=100px | 0.949 [0.941, 0.957] | 67167 | 328 | n/a | combined GT sub-population (exploratory: chosen after interim results) |
| e_detection_B2 | recall_not_fully_occluded_h>=50px | 0.763 [0.749, 0.777] | 146244 | 336 | n/a | combined GT sub-population (exploratory: chosen after interim results) |

### f. Crossing judge ([C3])

| task | metric | value [95% CI] | n | clips | base rate | note |
|---|---|---|---:|---:|---:|---|
| f_crossing | match_rate | 0.986 [0.977, 0.995] | 586 | 296 | n/a | GT behavioural peds with a PedX track (mean IoU>=0.3, >=3 co-visible frames) |
| f_crossing | precision | 0.983 [0.964, 0.996] | 578 | 293 | 0.844 | matched pedestrians only (one-to-one assignment) |
| f_crossing | recall | 0.467 [0.416, 0.519] | 578 | 293 | 0.844 | matched pedestrians only (one-to-one assignment) |
| f_crossing | f1 | 0.633 [0.585, 0.679] | 578 | 293 | 0.916 | matched pedestrians only (one-to-one assignment) |
| f_crossing | accuracy | 0.543 [0.495, 0.591] | 578 | 293 | 0.844 | matched pedestrians only (one-to-one assignment) |
| f_crossing | kappa | 0.196 [0.145, 0.251] | 578 | 293 | 0.000 | matched pedestrians only (one-to-one assignment) |
| f_crossing | onset_error_median_s | 1.617 [1.350, 1.969] | 168 | 124 | n/a | PedX started_frame minus JAAD crossing_point (s), true positives |
| f_crossing | onset_abs_error_median_s | 1.967 [1.633, 2.333] | 168 | 124 | n/a |  |
| f_crossing_incl_irrelevant | precision | 0.934 [0.896, 0.967] | 678 | 317 | 0.720 | matched; JAAD crossing=-1 ('irrelevant') counted as NOT crossing |
| f_crossing_incl_irrelevant | recall | 0.467 [0.416, 0.519] | 678 | 317 | 0.720 | matched; JAAD crossing=-1 ('irrelevant') counted as NOT crossing |
| f_crossing_incl_irrelevant | f1 | 0.623 [0.573, 0.669] | 678 | 317 | 0.837 | matched; JAAD crossing=-1 ('irrelevant') counted as NOT crossing |
| f_crossing_incl_irrelevant | kappa | 0.275 [0.207, 0.343] | 678 | 317 | 0.000 | matched; JAAD crossing=-1 ('irrelevant') counted as NOT crossing |
| f_crossing_end_to_end | precision | 0.983 [0.963, 0.996] | 586 | 296 | 0.845 | one-to-one matching; unmatched GT counted as 'not crossed' |
| f_crossing_end_to_end | recall | 0.461 [0.411, 0.514] | 586 | 296 | 0.845 | one-to-one matching; unmatched GT counted as 'not crossed' |
| f_crossing_end_to_end | f1 | 0.627 [0.579, 0.676] | 586 | 296 | 0.916 | one-to-one matching; unmatched GT counted as 'not crossed' |
| f_crossing_any_track | precision | 0.979 [0.957, 0.996] | 586 | 296 | 0.845 | all labelled pedestrians; flagged if ANY overlapping PedX track is crossed; no overlapping track = not crossed |
| f_crossing_any_track | recall | 0.564 [0.509, 0.619] | 586 | 296 | 0.845 | all labelled pedestrians; flagged if ANY overlapping PedX track is crossed; no overlapping track = not crossed |
| f_crossing_any_track | f1 | 0.715 [0.669, 0.760] | 586 | 296 | 0.916 | all labelled pedestrians; flagged if ANY overlapping PedX track is crossed; no overlapping track = not crossed |
| f_crossing_any_track | kappa | 0.256 [0.190, 0.328] | 586 | 296 | 0.000 | all labelled pedestrians; flagged if ANY overlapping PedX track is crossed; no overlapping track = not crossed |
| f_track_fragmentation | ids_per_gt_pedestrian_mean | 1.631 [1.536, 1.734] | 683 | 317 | 1.000 | PedX track ids overlapping a GT behavioural pedestrian; 1.0 = no ID splits |
| f_track_fragmentation | share_with_2plus_ids | 0.397 [0.349, 0.446] | 683 | 317 | 0.000 | share of GT pedestrians split over >=2 PedX ids |

### g. Ego-motion / static-camera gate ([B3])

| task | metric | value [95% CI] | n | clips | base rate | note |
|---|---|---|---:|---:|---:|---|
| g_ego_step_px | roc_auc_stopped | 0.945 [0.928, 0.958] | 40010 | 346 | 0.500 | raw [B3] step_px (lower = stopped) |
| g_ego_step_med | roc_auc_stopped | 0.950 [0.933, 0.963] | 40010 | 346 | 0.500 | rolling-median(5) step_px used by the gate |
| g_ego_step_med | f1_cv_threshold | 0.740 [0.675, 0.789] | 40010 | 346 | 0.243 | threshold on smoothed step chosen on other clips (5-fold by clip) |
| g_ego_expansion | roc_auc_stopped | 0.866 [0.841, 0.889] | 25999 | 255 | 0.500 | |bbox expansion rate|, frames with >=3 co-tracked pedestrians only |
| g_ego_expansion | coverage | 0.650 | 40010 | 346 | n/a | share of [B3] rows with an expansion anchor |
| g_ego_gate | precision | 0.717 [0.633, 0.785] | 40010 | 346 | 0.139 | positive = camera static/stopped; precision = P(stopped | gate says static) |
| g_ego_gate | recall | 0.682 [0.603, 0.746] | 40010 | 346 | 0.139 | positive = camera static/stopped; precision = P(stopped | gate says static) |
| g_ego_gate | f1 | 0.699 [0.629, 0.750] | 40010 | 346 | 0.243 | positive = camera static/stopped; precision = P(stopped | gate says static) |

### h. Looking behaviour ([P12])

| task | metric | value [95% CI] | n | clips | base rate | note |
|---|---|---|---:|---:|---:|---|
| h_looking_pedx_window | precision | 0.794 [0.709, 0.873] | 240 | 173 | 0.608 | JAAD look in PedX's own pre-crossing window; JAAD look = looking at the ego-vehicle |
| h_looking_pedx_window | recall | 0.685 [0.601, 0.762] | 240 | 173 | 0.608 | JAAD look in PedX's own pre-crossing window; JAAD look = looking at the ego-vehicle |
| h_looking_pedx_window | f1 | 0.735 [0.669, 0.794] | 240 | 173 | 0.756 | JAAD look in PedX's own pre-crossing window; JAAD look = looking at the ego-vehicle |
| h_looking_pedx_window | accuracy | 0.700 [0.636, 0.760] | 240 | 173 | 0.608 | JAAD look in PedX's own pre-crossing window; JAAD look = looking at the ego-vehicle |
| h_looking_pedx_window | kappa | 0.393 [0.265, 0.517] | 240 | 173 | 0.000 | JAAD look in PedX's own pre-crossing window; JAAD look = looking at the ego-vehicle |
| h_looking_before_cp | precision | 0.848 [0.759, 0.929] | 128 | 97 | 0.773 | JAAD look in the 5 s before JAAD crossing_point; JAAD look = looking at the ego-vehicle |
| h_looking_before_cp | recall | 0.677 [0.570, 0.768] | 128 | 97 | 0.773 | JAAD look in the 5 s before JAAD crossing_point; JAAD look = looking at the ego-vehicle |
| h_looking_before_cp | f1 | 0.753 [0.667, 0.819] | 128 | 97 | 0.872 | JAAD look in the 5 s before JAAD crossing_point; JAAD look = looking at the ego-vehicle |
| h_looking_before_cp | accuracy | 0.656 [0.564, 0.735] | 128 | 97 | 0.773 | JAAD look in the 5 s before JAAD crossing_point; JAAD look = looking at the ego-vehicle |
| h_looking_before_cp | kappa | 0.211 [0.035, 0.368] | 128 | 97 | 0.000 | JAAD look in the 5 s before JAAD crossing_point; JAAD look = looking at the ego-vehicle |

### i. Age / gender ([P6])

| task | metric | value [95% CI] | n | clips | base rate | note |
|---|---|---|---:|---:|---:|---|
| i_age_gender | coverage | 0.414 [0.367, 0.462] | 678 | 317 | n/a | matched behavioural peds whose PedX track has a [P6] row (face found) |
| i_age_gender_gender | accuracy | 0.548 [0.481, 0.612] | 272 | 167 | 0.548 |  |
| i_age_gender_gender | kappa | 0.139 [0.036, 0.241] | 272 | 167 | 0.000 |  |
| i_age_gender_age | accuracy | 0.783 [0.736, 0.831] | 281 | 168 | 0.786 | cuts child<=12, young 13-19, adult 20-64, senior>=65 |
| i_age_gender_age | kappa | 0.028 [-0.011, 0.083] | 281 | 168 | 0.000 | cuts child<=12, young 13-19, adult 20-64, senior>=65 |
| i_age_gender_age_alt | accuracy | 0.765 [0.718, 0.814] | 281 | 168 | 0.786 | cuts child<=12, young 13-24, adult 25-59, senior>=60 |
| i_age_gender_age_alt | kappa | 0.025 [-0.034, 0.102] | 281 | 168 | 0.000 | cuts child<=12, young 13-24, adult 25-59, senior>=60 |

### j. Risky crossing coverage ([C1])

| task | metric | value [95% CI] | n | clips | base rate | note |
|---|---|---|---:|---:|---:|---|
| j_risky_coverage | pedx_determined_share | 0.022 [0.006, 0.044] | 314 | 194 | 0.051 | [C1] crossings with an observed signal; base_rate = share whose window has a JAAD light |
| j_risky_coverage | pedx_mean_signal_coverage | 0.018 [0.003, 0.037] | 314 | 194 | 0.049 | base_rate = mean JAAD light-labelled share of the same windows |
| j_risky_coverage_observed | precision | 0.571 [0.000, 1.000] | 314 | 194 | 0.051 | PedX 'signal observed' vs JAAD 'light labelled' in the crossing window |
| j_risky_coverage_observed | recall | 0.250 [0.000, 0.556] | 314 | 194 | 0.051 | PedX 'signal observed' vs JAAD 'light labelled' in the crossing window |
| j_risky_coverage | C5_ran_red_count | 1.000 | 314 | 194 | n/a | count only; JAAD green share in those windows: 1.00 |

### k. Traffic signs ([E3])

| task | metric | value [95% CI] | n | clips | base rate | note |
|---|---|---|---:|---:|---:|---|
| k_signs_ped_sign | precision | 0.898 [0.843, 0.942] | 2691 | 346 | 0.287 | classes risky_crossing.py treats as protective |
| k_signs_ped_sign | recall | 0.571 [0.493, 0.650] | 2691 | 346 | 0.287 | classes risky_crossing.py treats as protective |
| k_signs_ped_sign | f1 | 0.698 [0.632, 0.756] | 2691 | 346 | 0.446 | classes risky_crossing.py treats as protective |
| k_signs_stop_sign | precision | 0.111 [0.000, 0.400] | 2691 | 346 | 0.036 | best_new.pt 'Stop_Sign' |
| k_signs_stop_sign | recall | 0.010 [0.000, 0.041] | 2691 | 346 | 0.036 | best_new.pt 'Stop_Sign' |
| k_signs_stop_sign | f1 | 0.019 [0.000, 0.070] | 2691 | 346 | 0.069 | best_new.pt 'Stop_Sign' |
| k_signs_ped_sign_ip_only | precision | 0.969 [0.943, 0.989] | 2691 | 346 | 0.287 | EXPLORATORY: class 'ip' alone, chosen after inspecting interim results |
| k_signs_ped_sign_ip_only | recall | 0.569 [0.490, 0.647] | 2691 | 346 | 0.287 | EXPLORATORY: class 'ip' alone, chosen after inspecting interim results |
| k_signs_ped_sign_ip_only | f1 | 0.717 [0.652, 0.777] | 2691 | 346 | 0.446 | EXPLORATORY: class 'ip' alone, chosen after inspecting interim results |
| k_signs_class[ip] | precision_ped_sign | 0.969 [0.943, 0.989] | 453 | 99 | 0.287 | descriptive: share of frames where [E3] flags this class that JAAD labels with the sign; base_rate = overall prevalence |
| k_signs_class[ip] | precision_stop_sign | 0.002 [0.000, 0.007] | 453 | 99 | 0.036 | descriptive: share of frames where [E3] flags this class that JAAD labels with the sign; base_rate = overall prevalence |
| k_signs_class[pn] | precision_ped_sign | 0.622 [0.475, 0.765] | 188 | 64 | 0.287 | descriptive: share of frames where [E3] flags this class that JAAD labels with the sign; base_rate = overall prevalence |
| k_signs_class[pn] | precision_stop_sign | 0.021 [0.000, 0.070] | 188 | 64 | 0.036 | descriptive: share of frames where [E3] flags this class that JAAD labels with the sign; base_rate = overall prevalence |
| k_signs_class[pg] | precision_ped_sign | 0.190 [0.051, 0.400] | 42 | 20 | 0.287 | descriptive: share of frames where [E3] flags this class that JAAD labels with the sign; base_rate = overall prevalence |
| k_signs_class[pg] | precision_stop_sign | 0.000 [0.000, 0.000] | 42 | 20 | 0.036 | descriptive: share of frames where [E3] flags this class that JAAD labels with the sign; base_rate = overall prevalence |
| k_signs_class[p5] | precision_ped_sign | 0.310 [0.042, 0.593] | 29 | 18 | 0.287 | descriptive: share of frames where [E3] flags this class that JAAD labels with the sign; base_rate = overall prevalence |
| k_signs_class[p5] | precision_stop_sign | 0.000 [0.000, 0.000] | 29 | 18 | 0.036 | descriptive: share of frames where [E3] flags this class that JAAD labels with the sign; base_rate = overall prevalence |
| k_signs_class[pl5] | precision_ped_sign | 0.545 [0.000, 0.875] | 11 | 6 | 0.287 | descriptive: share of frames where [E3] flags this class that JAAD labels with the sign; base_rate = overall prevalence |
| k_signs_class[pl5] | precision_stop_sign | 0.000 [0.000, 0.000] | 11 | 6 | 0.036 | descriptive: share of frames where [E3] flags this class that JAAD labels with the sign; base_rate = overall prevalence |
| k_signs_class[pne] | precision_ped_sign | 0.600 [0.143, 0.917] | 10 | 7 | 0.287 | descriptive: share of frames where [E3] flags this class that JAAD labels with the sign; base_rate = overall prevalence |
| k_signs_class[pne] | precision_stop_sign | 0.000 [0.000, 0.000] | 10 | 7 | 0.036 | descriptive: share of frames where [E3] flags this class that JAAD labels with the sign; base_rate = overall prevalence |

## Sensitivity: street clips only

JAAD includes parking-lot and garage clips (56 of the 346 scored), a scene type PedX is not designed for. The same metrics on the 290 street clips:

| task | metric | all clips | street clips only |
|---|---|---|---|
| a_weather | accuracy | 0.320 [0.273, 0.366] (clips 344) | 0.317 [0.266, 0.372] (clips 290) |
| a_weather | kappa | -0.055 [-0.115, 0.006] (clips 344) | -0.071 [-0.144, 0.002] (clips 290) |
| b_daynight | kappa | 0.077 [0.017, 0.143] (clips 345) | 0.079 [0.018, 0.149] (clips 290) |
| c_traffic_light_visible_presence | recall | 0.424 [0.170, 0.642] (clips 19) | 0.427 [0.167, 0.638] (clips 18) |
| d_crosswalk_E7_frame | f1 | 0.000 [0.000, 0.000] (clips 346) | 0.000 [0.000, 0.000] (clips 290) |
| d_crosswalk_E10_frame | f1 | 0.873 [0.847, 0.895] (clips 346) | 0.879 [0.853, 0.904] (clips 290) |
| d_crosswalk_E10_frac | roc_auc | 0.947 [0.932, 0.960] (clips 346) | 0.949 [0.934, 0.963] (clips 290) |
| e_detection_B2 | recall_kind=b | 0.806 [0.783, 0.827] (clips 320) | 0.802 [0.778, 0.825] (clips 270) |
| f_crossing_any_track | f1 | 0.715 [0.669, 0.760] (clips 296) | 0.721 [0.673, 0.765] (clips 259) |
| f_crossing_any_track | kappa | 0.256 [0.190, 0.328] (clips 296) | 0.256 [0.186, 0.338] (clips 259) |
| g_ego_step_px | roc_auc_stopped | 0.945 [0.928, 0.958] (clips 346) | 0.948 [0.931, 0.961] (clips 290) |
| h_looking_pedx_window | f1 | 0.735 [0.669, 0.794] (clips 173) | 0.744 [0.678, 0.800] (clips 152) |
| i_age_gender_gender | accuracy | 0.548 [0.481, 0.612] (clips 167) | 0.562 [0.489, 0.635] (clips 138) |

## Confusion matrices

Rows = JAAD label, columns = PedX. Also written as `cm_*.csv`.

**weather_mapped**

```
col_0      clear  cloudy  rain  unknown
jaad\pedx                              
clear         83      30    53        0
cloudy        85      20    39        0
rain          16       5     7        0
snow           2       3     1        0
```

**weather_raw**

```
col_0      cloudy  rain  shine  sunrise  unknown
jaad\pedx                                       
clear          30    53     47       36        0
cloudy         20    39     50       35        0
rain            5     7     12        4        0
snow            3     1      2        0        0
```

**daynight_raw**

```
col_0      Day  Evening  Night
jaad\pedx                     
daytime    248       84      4
afternoon    4        0      0
evening      0        1      0
nighttime    0        0      4
```

**traffic_light**

```
col_0      None  red  green  yellow
jaad\pedx                          
n/a        2526    6      0      16
red          41   33      0       0
green        46    0     23       0
```

**crosswalk_E7**

```
col_0      yes    no
jaad\pedx           
crossing     0  1315
none         0  1376
```

**crosswalk_E10**

```
col_0       yes    no
jaad\pedx            
crossing   1083   232
none         84  1292
```

**crosswalk_X7cls4**

```
col_0      yes    no
jaad\pedx           
crossing     8  1307
none        20  1356
```

**crossing_matched**

```
col_0      crossed  not
jaad\pedx              
crossed        228  260
not              4   86
```

**ego_gate_by_action**

```
col_0             static  moving
jaad_action\gate                
stopped             3779    1765
moving_slow           31     958
decelerating         830   18487
accelerating         620   12349
moving_fast           12    1179
```

**looking**

```
col_0      head_turn  none
jaad\pedx                 
looking          100    46
not               26    68
```

**gender**

```
col_0      female  male
jaad\pedx              
female         45   104
male           19   104
```

**age**

```
col_0      child  young  adult  senior
jaad\pedx                             
child          0      1      8       0
young          0      1     20       0
adult          0      0    219       2
senior         0      0     30       0
```

## Coverage

| task | metric | value [95% CI] | n | clips | base rate | note |
|---|---|---|---:|---:|---:|---|
| z_coverage | clips_with_B1 | 346.000 | 346 | 346 | n/a | [B1]tracked_pedestrians.csv |
| z_coverage | clips_with_B3 | 346.000 | 346 | 346 | n/a | [B3]ego_motion.csv |
| z_coverage | clips_with_E1 | 346.000 | 346 | 346 | n/a | [E1]weather.csv |
| z_coverage | clips_with_E2 | 346.000 | 346 | 346 | n/a | [E2]traffic_light.csv |
| z_coverage | clips_with_E3 | 346.000 | 346 | 346 | n/a | [E3]traffic_sign.csv |
| z_coverage | clips_with_E6 | 346.000 | 346 | 346 | n/a | [E6]daytime.csv |
| z_coverage | clips_with_E7 | 346.000 | 346 | 346 | n/a | [E7]crosswalk_detection.csv |
| z_coverage | clips_with_E9 | 346.000 | 346 | 346 | n/a | [E9]sidewalk_detection.csv |
| z_coverage | clips_with_E10 | 346.000 | 346 | 346 | n/a | [E10]scene_segmentation.csv |
| z_coverage | clips_with_C3 | 346.000 | 346 | 346 | n/a | [C3]crossing_judge.csv |
| z_coverage | clips_with_C1 | 346.000 | 346 | 346 | n/a | [C1]risky_crossing.csv |
| z_coverage | clips_with_C5 | 346.000 | 346 | 346 | n/a | [C5]red_light_runner.csv |
| z_coverage | clips_with_P6 | 346.000 | 346 | 346 | n/a | [P6]age_gender.csv |
| z_coverage | clips_with_P12 | 346.000 | 346 | 346 | n/a | [P12]pose_behavior.csv |
| z_coverage | clips_with_S1 | 346.000 | 346 | 346 | n/a | [S1]pedestrian_speed.csv |
| z_coverage | clips_with_X7 | 346.000 | 346 | 346 | n/a | [X7]crosswalk_class_probe.csv |
| z_coverage | S1_reliable_share | 0.203 | 942 | 113 | n/a | [S1] tracks flagged reliable (no speed GT in JAAD) |

## Findings (hand-written; bugs located while validating)

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
