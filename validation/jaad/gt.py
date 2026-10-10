"""Tidy ground-truth tables for the JAAD benchmark.

All annotation parsing is delegated to JAAD's official parser (`jaad_data.py`, MIT,
github.com/ykotseruba/JAAD), loaded with importlib from the dataset checkout without
copying or modifying it. Its `JAAD(data_path).generate_database()` caches the parsed
database at `<jaad_root>/data_cache/jaad_database.pkl`.

The one exception: `generate_database()` does not read the clip-level
`<meta><task><video_attributes>` block (weather / time_of_day / location), so those
three fields are read here with ElementTree. Nothing else is re-parsed.

Encodings (checked against jaad_data.py and the XML, not assumed):
  * frames are 0-indexed (`<box frame="0">`, traffic/vehicle `<frame id="0">`);
  * pedestrian ids: 'b' suffix = behavioural annotation (crossing/look/... labels),
    'p' suffix = "people" (group boxes, treated as `difficult` by jaad_data's own
    detection export), no suffix = pedestrian with a box but no behaviour labels;
  * attributes are ints after jaad_data's `_map_text_to_scalar`; `crossing` is
    1 (crossed), 0 (did not cross), -1 (irrelevant: never intended to cross);
    `crossing_point` is the frame of the crossing onset, -1 when undefined;
  * per-frame `cross` takes only 0/1 in the released XML (no -1), `look` is
    "looking at the ego-vehicle" (not a generic head turn);
  * clip `weather` contains a typo class 'cloud' (1 clip) and 'n/a' (2 clips);
    `time_of_day` uses daytime / afternoon / evening / nighttime / n/a, not just
    day/night.  Normalisations are in WEATHER_NORM and DAY_NIGHT below.
"""

import argparse
import importlib.util
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

DEFAULT_JAAD_ROOT = os.environ.get("JAAD_ROOT", r"E:/PedX-validation-data/JAAD")

# Clip-level label normalisation (documented in the report).
WEATHER_NORM = {"clear": "clear", "cloudy": "cloudy", "cloud": "cloudy",
                "rain": "rain", "snow": "snow", "n/a": None}
# JAAD 'afternoon' is daylight; 'evening' (1 clip) is dusk -> grouped with night
# because PedX's 'Day' is the only label that asserts daylight.
DAY_NIGHT = {"daytime": "day", "afternoon": "day", "evening": "night",
             "nighttime": "night", "n/a": None}

CLIP_COLUMNS = ["clip", "weather_raw", "weather", "time_of_day_raw", "day_night",
                "location", "road_type", "num_frames", "width", "height", "fps",
                "video_path"]
FRAME_COLUMNS = ["clip", "frame", "traffic_light", "ped_crossing", "ped_sign",
                 "stop_sign", "vehicle_action", "ego_stopped"]
BOX_COLUMNS = ["clip", "ped_id", "kind", "frame", "x1", "y1", "x2", "y2",
               "occlusion", "look", "cross"]
PED_COLUMNS = ["clip", "ped_id", "kind", "old_id", "n_frames", "first_frame",
               "last_frame", "crossing", "crossing_point", "decision_point", "age",
               "gender", "designated", "signalized", "intersection", "group_size",
               "motion_direction"]


def load_jaad_module(jaad_root=DEFAULT_JAAD_ROOT):
    """Import the official jaad_data.py from the dataset checkout, by path.

    spec_from_file_location does not put the dataset directory on sys.path, so no
    other file in the (untrusted) download can shadow an import.
    """
    path = os.path.join(jaad_root, "jaad_data.py")
    spec = importlib.util.spec_from_file_location("jaad_data", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_database(jaad_root=DEFAULT_JAAD_ROOT, regen=False):
    """(JAAD instance, database dict) via JAAD(data_path).generate_database()."""
    mod = load_jaad_module(jaad_root)
    jaad = mod.JAAD(data_path=jaad_root, regen_pkl=regen)
    return jaad, jaad.generate_database()


class _Decoder:
    """Memoised wrapper around jaad_data's own scalar->text map (reuse, not re-encode)."""

    def __init__(self, jaad):
        self._jaad = jaad
        self._cache = {}

    def __call__(self, label, value):
        key = (label, value)
        if key not in self._cache:
            try:
                self._cache[key] = self._jaad._map_scalar_to_text(label, value)
            except KeyError:
                self._cache[key] = None
        return self._cache[key]


def ped_kind(ped_id):
    """'b' (behavioural), 'p' (people group) or 'ped' -- the same substring tests
    jaad_data.py uses ('b' in pid / 'p' in pid)."""
    if "b" in ped_id:
        return "b"
    if "p" in ped_id:
        return "p"
    return "ped"


def video_attributes(jaad_root, clip):
    """{'weather','time_of_day','location'} from the annotation XML's meta block."""
    tree = ET.parse(os.path.join(jaad_root, "annotations", clip + ".xml"))
    out = {}
    for k in ("weather", "time_of_day", "location"):
        e = tree.find("./meta/task/video_attributes/" + k)
        out[k] = e.text.strip() if (e is not None and e.text) else None
    return out


def _clip_fps(video_path):
    if not os.path.exists(video_path):
        return float("nan")
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else float("nan")
    cap.release()
    return float(fps) if fps and fps > 0 else float("nan")


def build_tables(jaad_root=DEFAULT_JAAD_ROOT, clips=None, with_fps=True, db=None, jaad=None):
    """Return dict of tidy DataFrames: clips, frames, boxes, peds.

    `clips` restricts to a subset of clip ids (e.g. 'video_0001'); None = all 346.
    """
    if db is None or jaad is None:
        jaad, db = load_database(jaad_root)
    dec = _Decoder(jaad)
    clip_ids = sorted(db) if clips is None else [c for c in sorted(set(clips)) if c in db]
    clips_dir = os.path.join(jaad_root, "JAAD_clips")

    clip_rows, frame_parts, box_parts, ped_rows = [], [], [], []
    for clip in clip_ids:
        d = db[clip]
        va = video_attributes(jaad_root, clip)
        vpath = os.path.join(clips_dir, clip + ".mp4")
        ta = d["traffic_annotations"]
        clip_rows.append({
            "clip": clip,
            "weather_raw": va["weather"],
            "weather": WEATHER_NORM.get(va["weather"]),
            "time_of_day_raw": va["time_of_day"],
            "day_night": DAY_NIGHT.get(va["time_of_day"]),
            "location": va["location"],
            "road_type": dec("road_type", ta["road_type"]),
            "num_frames": int(d["num_frames"]),
            "width": int(d["width"]),
            "height": int(d["height"]),
            "fps": _clip_fps(vpath) if with_fps else float("nan"),
            "video_path": vpath,
        })

        frames = sorted(k for k in ta if isinstance(k, int))
        va_act = d["vehicle_annotations"]
        frame_parts.append(pd.DataFrame({
            "clip": clip,
            "frame": frames,
            "traffic_light": [dec("traffic_light", ta[f]["traffic_light"]) for f in frames],
            "ped_crossing": [int(ta[f]["ped_crossing"]) for f in frames],
            "ped_sign": [int(ta[f]["ped_sign"]) for f in frames],
            "stop_sign": [int(ta[f]["stop_sign"]) for f in frames],
            "vehicle_action": [dec("vehicle", va_act[f]) if f in va_act else None for f in frames],
        }))

        for pid, p in d["ped_annotations"].items():
            kind = ped_kind(pid)
            bb = np.asarray(p["bbox"], dtype=float).reshape(-1, 4)
            n = len(p["frames"])
            beh = p.get("behavior") or {}
            look = beh.get("look") if beh.get("look") else [np.nan] * n
            cross = beh.get("cross") if beh.get("cross") else [np.nan] * n
            box_parts.append(pd.DataFrame({
                "clip": clip, "ped_id": pid, "kind": kind,
                "frame": np.asarray(p["frames"], dtype=int),
                "x1": bb[:, 0], "y1": bb[:, 1], "x2": bb[:, 2], "y2": bb[:, 3],
                "occlusion": [dec("occlusion", o) for o in p["occlusion"]],
                "look": look, "cross": cross,
            }))
            a = p.get("attributes") or {}
            ped_rows.append({
                "clip": clip, "ped_id": pid, "kind": kind, "old_id": p.get("old_id"),
                "n_frames": n,
                "first_frame": int(min(p["frames"])) if n else -1,
                "last_frame": int(max(p["frames"])) if n else -1,
                "crossing": a.get("crossing", np.nan),
                "crossing_point": a.get("crossing_point", np.nan),
                "decision_point": a.get("decision_point", np.nan),
                "age": dec("age", a["age"]) if "age" in a else None,
                "gender": dec("gender", a["gender"]) if "gender" in a else None,
                "designated": dec("designated", a["designated"]) if "designated" in a else None,
                "signalized": dec("signalized", a["signalized"]) if "signalized" in a else None,
                "intersection": dec("intersection", a["intersection"]) if "intersection" in a else None,
                "group_size": a.get("group_size", np.nan),
                "motion_direction": (dec("motion_direction", a["motion_direction"])
                                     if "motion_direction" in a else None),
            })

    frames_df = (pd.concat(frame_parts, ignore_index=True) if frame_parts
                 else pd.DataFrame(columns=FRAME_COLUMNS))
    if not frames_df.empty:
        frames_df["ego_stopped"] = frames_df["vehicle_action"] == "stopped"
    frames_df = frames_df.reindex(columns=FRAME_COLUMNS)
    boxes_df = (pd.concat(box_parts, ignore_index=True) if box_parts
                else pd.DataFrame(columns=BOX_COLUMNS)).reindex(columns=BOX_COLUMNS)
    return {
        "clips": pd.DataFrame(clip_rows, columns=CLIP_COLUMNS),
        "frames": frames_df,
        "boxes": boxes_df,
        "peds": pd.DataFrame(ped_rows, columns=PED_COLUMNS),
    }


def write_tables(tables, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(os.path.join(out_dir, f"jaad_gt_{name}.csv"), index=False, encoding="utf-8")


TABLE_FILES = {"clips": "jaad_gt_clips.csv", "frames": "jaad_gt_frames.csv",
               "peds": "jaad_gt_peds.csv", "boxes": "jaad_gt_boxes.csv"}


def load_tables(gt_dir):
    """Read the tables written by `write_tables` back into DataFrames.

    Only the EMPTY string is missing. JAAD's own label for "no traffic light" is the text
    'n/a'; pandas' default NA list contains it, so a plain read_csv silently turns 94% of
    the traffic_light column into NaN -- which then compares as "!= 'n/a'" and would score
    every unlabelled frame as a frame WITH a light.
    """
    out = {}
    for name, fname in TABLE_FILES.items():
        out[name] = pd.read_csv(os.path.join(gt_dir, fname), keep_default_na=False,
                                na_values=[""])
    return out


def main():
    ap = argparse.ArgumentParser(description="Build tidy JAAD ground-truth tables.")
    ap.add_argument("--jaad_root", default=DEFAULT_JAAD_ROOT)
    ap.add_argument("--out", default=None,
                    help="Directory for jaad_gt_*.csv (default: <jaad_root>/../jaad_gt)")
    args = ap.parse_args()
    tables = build_tables(args.jaad_root)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.jaad_root)), "jaad_gt")
    write_tables(tables, out)
    for k, v in tables.items():
        print(f"{k}: {len(v)} rows")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
