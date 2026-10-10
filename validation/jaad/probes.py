"""Counterfactual probes: quantify a suspected module bug WITHOUT editing the module.

These are NOT PedX outputs and never feed the production pipeline. They re-run a
module's own model with its own sampling and gates, varying exactly one thing, and
write `[X*]` files next to the PedX outputs in the isolated results root.

[X7] crosswalk class probe
--------------------------
modules/crosswalk/crosswalk.py keeps detections with `cls_id == 0`, but the weights it
loads (modules/crosswalk/best.pt) name class 0 'Flexible_delineator_post' and class 4
'crosswalk'. The probe runs the same model on the same frames (0-indexed frame_id,
fps = ceil(CAP_PROP_FPS), every int(fps * interval) frames, imgsz=640) with a low
confidence floor and records, per sampled frame, the maximum confidence of a class-0 and
of a class-4 box that passes the module's own minimum-size gate
(MIN_CROSSWALK_BOX_W_PX / _H_PX, imported from the module). Thresholding `conf_cls0` at
the production 0.35 and applying the module's 5-sample majority vote reproduces [E7]
(checked in score.py); `conf_cls4` is the score the detector would emit if the class
index were 4.

    python -m validation.jaad.probes --clips_file validation/jaad/sample_clips.txt
"""

import argparse
import math
import os
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from validation.jaad.run_pedx import (DEFAULT_JAAD_ROOT, DEFAULT_RESULTS_ROOT,  # noqa: E402
                                      assert_isolated, read_clips_file)

PROBE_X7 = "[X7]crosswalk_class_probe.csv"
X7_COLUMNS = ["frame_id", "conf_cls0", "conf_cls4", "n_cls0", "n_cls4"]
PROBE_CONF_FLOOR = 0.01


def crosswalk_class_probe(video_path, out_csv, model, min_w, min_h, interval=1.0, device=None):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    rows = []
    if not cap.isOpened():
        pd.DataFrame(columns=X7_COLUMNS).to_csv(out_csv, index=False)
        return 0
    fps_raw = cap.get(cv2.CAP_PROP_FPS)
    fps = math.ceil(fps_raw) if fps_raw > 0 else 30
    frame_interval = max(1, int(fps * interval))
    frame_id = -1
    while True:
        if not cap.grab():
            break
        frame_id += 1
        if frame_id % frame_interval:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            rows.append({"frame_id": frame_id, "conf_cls0": 0.0, "conf_cls4": 0.0,
                         "n_cls0": 0, "n_cls4": 0})
            continue
        res = model(frame, imgsz=640, conf=PROBE_CONF_FLOOR, verbose=False, device=device)[0]
        best = {0: 0.0, 4: 0.0}
        cnt = {0: 0, 4: 0}
        if res.boxes is not None and len(res.boxes):
            for cls_t, conf_t, xyxy in zip(res.boxes.cls, res.boxes.conf, res.boxes.xyxy):
                c = int(cls_t)
                if c not in best:
                    continue
                x1, y1, x2, y2 = xyxy.tolist()
                if (x2 - x1) < min_w or (y2 - y1) < min_h:   # module's own size gate
                    continue
                best[c] = max(best[c], float(conf_t))
                cnt[c] += 1
        rows.append({"frame_id": frame_id, "conf_cls0": round(best[0], 4),
                     "conf_cls4": round(best[4], 4), "n_cls0": cnt[0], "n_cls4": cnt[4]})
    cap.release()
    pd.DataFrame(rows, columns=X7_COLUMNS).to_csv(out_csv, index=False)
    return len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Counterfactual crosswalk class probe ([X7]).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--clips", nargs="+")
    g.add_argument("--clips_file")
    ap.add_argument("--results_root", default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--jaad_root", default=DEFAULT_JAAD_ROOT)
    ap.add_argument("--analysis_interval", type=float, default=1.0)
    args = ap.parse_args(argv)

    root = assert_isolated(args.results_root)
    clips_dir = Path(args.jaad_root) / "JAAD_clips"
    if args.clips_file:
        clips = read_clips_file(args.clips_file)
    elif args.clips == ["all"]:
        clips = sorted(p.stem for p in clips_dir.glob("video_*.mp4"))
    else:
        clips = sorted(set(args.clips))

    os.chdir(REPO)                       # weight paths are repo-relative, as in main.py
    import torch
    from ultralytics import YOLO
    from modules.crosswalk.crosswalk import MIN_CROSSWALK_BOX_W_PX, MIN_CROSSWALK_BOX_H_PX
    model = YOLO("modules/crosswalk/best.pt")
    names = model.names
    assert names.get(4) == "crosswalk", f"unexpected class map {names}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    done = 0
    for c in clips:
        out = root / c / PROBE_X7
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        n = crosswalk_class_probe(clips_dir / f"{c}.mp4", out, model,
                                  MIN_CROSSWALK_BOX_W_PX, MIN_CROSSWALK_BOX_H_PX,
                                  args.analysis_interval)
        done += 1
        print(f"[probe X7] {c}: {n} sampled frames", flush=True)
    print(f"[probe X7] {done} clips probed; class map {names}")


if __name__ == "__main__":
    main()
