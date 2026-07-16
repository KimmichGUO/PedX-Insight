"""Unit tests for modules/age_gender/age_gender_detect.py ([P6], --mode ag).

Plain asserts, no pytest needed. Run from the repo root:

    python tests/test_age_gender.py

Covers:
  * the module now imports cleanly (paddlex ImportError stub is gone)
  * legacy entry-point signature run_age_gender(video_path) still works
  * missing pedestrian_img directory -> header-only CSV (default paths,
    exercised inside a temp cwd so nothing real is touched)
  * empty / non-png track folders -> no rows, but track present in the return
  * face-alignment geometry produces the 96x96 patch insightface expects
  * REAL archived crops (skipped if analysis_results is absent): output CSV
    has the legacy columns ['id','age','gender'], ids join as integers the
    way crossed_info.py does, ages are plausible integer years, genders are
    'male'/'female', and video_info.py's mode-of-age aggregation works
"""

import csv
import inspect
import os
import sys
import tempfile

import numpy as np

# Make the repo root importable no matter where the test is launched from.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# Test 1: the module must import cleanly (main.py's 'ag' branch depends on it).
from modules.age_gender import age_gender_detect as ag  # noqa: E402
from modules.age_gender.age_gender_detect import (  # noqa: E402
    _align_face_96,
    run_age_gender,
)


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def test_legacy_signature():
    params = inspect.signature(run_age_gender).parameters
    names = list(params)
    assert names[0] == "video_path", names
    # every added parameter must be optional, so run_age_gender(video_path)
    # keeps working for main.py
    for name in names[1:]:
        assert params[name].default is not inspect.Parameter.empty, name
    print("PASS test_legacy_signature")


def test_missing_pedestrian_img_header_only():
    with tempfile.TemporaryDirectory() as tmp:
        old_cwd = os.getcwd()
        os.chdir(tmp)  # default paths are relative; keep them inside tmp
        try:
            res = run_age_gender("no_such_video.mp4")
        finally:
            os.chdir(old_cwd)
        assert res == {}, res
        out = os.path.join(tmp, "analysis_results", "no_such_video",
                           "[P6]age_gender.csv")
        assert os.path.isfile(out), out
        rows = _read_csv(out)
        assert rows == [["id", "age", "gender"]], rows
    print("PASS test_missing_pedestrian_img_header_only")


def test_empty_and_non_png_folders():
    with tempfile.TemporaryDirectory() as tmp:
        img_root = os.path.join(tmp, "pedestrian_img")
        os.makedirs(os.path.join(img_root, "id_7"))          # empty track
        os.makedirs(os.path.join(img_root, "id_9"))
        with open(os.path.join(img_root, "id_9", "notes.txt"), "w") as f:
            f.write("not an image")
        out_csv = os.path.join(tmp, "[P6]age_gender.csv")
        res = run_age_gender("v.mp4", img_root=img_root, output_csv_path=out_csv)
        # tracks are visited but yield nothing -> in return dict as None,
        # absent from the CSV (consumers treat missing ids as 'None')
        assert res == {"id_7": {"age": None, "gender": None},
                       "id_9": {"age": None, "gender": None}}, res
        rows = _read_csv(out_csv)
        assert rows == [["id", "age", "gender"]], rows
    print("PASS test_empty_and_non_png_folders")


def test_unreadable_png_is_skipped():
    with tempfile.TemporaryDirectory() as tmp:
        img_root = os.path.join(tmp, "pedestrian_img")
        os.makedirs(os.path.join(img_root, "id_3"))
        with open(os.path.join(img_root, "id_3", "frame_30.png"), "wb") as f:
            f.write(b"\x89PNG garbage that cv2.imread cannot decode")
        out_csv = os.path.join(tmp, "[P6]age_gender.csv")
        res = run_age_gender("v.mp4", img_root=img_root, output_csv_path=out_csv)
        assert res["id_3"] == {"age": None, "gender": None}, res
        assert _read_csv(out_csv) == [["id", "age", "gender"]]
    print("PASS test_unreadable_png_is_skipped")


def test_align_face_geometry():
    # A synthetic image with a single white pixel at the face-box center must
    # land at the center of the 96x96 aligned patch.
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[100, 100] = 255
    bbox = (80.0, 80.0, 40.0, 40.0)  # center (100, 100)
    aligned = _align_face_96(img, bbox)
    assert aligned.shape == (96, 96, 3), aligned.shape
    ys, xs = np.where(aligned[:, :, 0] > 0)[:2]
    assert len(ys) > 0, "center pixel lost by the transform"
    cy, cx = ys.mean(), xs.mean()
    assert abs(cy - 48) <= 1.5 and abs(cx - 48) <= 1.5, (cy, cx)
    # scale = 96 / (1.5 * 40) = 1.6 -> the 40 px box spans 64 px in the patch
    print("PASS test_align_face_geometry")


def test_real_archived_crops():
    candidates = ["Paris1_UElofxU8Nvo", "Tokyo1_pnVAm2YhJLg",
                  "Berlin1_ORPGr4m2-Sw"]
    video = None
    for name in candidates:
        root = os.path.join(REPO_ROOT, "analysis_results", name,
                            "pedestrian_img")
        if os.path.isdir(root):
            video, img_root = name, root
            break
    if video is None:
        print("SKIP test_real_archived_crops (no archived crops on disk)")
        return

    with tempfile.TemporaryDirectory() as tmp:
        out_csv = os.path.join(tmp, "[P6]age_gender.csv")
        res = run_age_gender(video + ".mp4", img_root=img_root,
                             output_csv_path=out_csv)
        assert len(res) > 0
        rows = _read_csv(out_csv)
        assert rows[0] == ["id", "age", "gender"], rows[0]
        body = rows[1:]
        assert len(body) > 0, "expected at least one classified track"

        import pandas as pd
        df = pd.read_csv(out_csv)
        # crossed_info.py joins gender_df['id'] == track_id (int) — the id
        # column must therefore parse as integers
        assert df["id"].dtype.kind in "iu", df["id"].dtype
        assert set(df.columns) >= {"id", "age", "gender"}
        assert df["id"].is_unique
        assert set(df["gender"].unique()) <= {"male", "female"}, \
            df["gender"].unique()
        ages = df["age"]
        assert (ages == ages.astype(int)).all(), "ages must be integer years"
        assert ages.between(0, 100).all(), (ages.min(), ages.max())
        # video_info.py: mode of the age column must be computable
        assert not ages.mode().empty

        # CSV rows must be a subset of the return dict, minus the no-face ones
        classified = {int(k.replace("id_", "")) for k, v in res.items()
                      if v["age"] is not None}
        assert set(df["id"]) == classified

        n_total = len(res)
        print(f"PASS test_real_archived_crops ({video}: "
              f"{len(body)}/{n_total} tracks classified)")


if __name__ == "__main__":
    test_legacy_signature()
    test_missing_pedestrian_img_header_only()
    test_empty_and_non_png_folders()
    test_unreadable_png_is_skipped()
    test_align_face_geometry()
    test_real_archived_crops()
    print("ALL TESTS PASSED")
