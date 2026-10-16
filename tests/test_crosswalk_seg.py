"""Tests for modules/crosswalk/crosswalk_seg.py (the segmentation-based [E7] replacement).

The Mask2Former model is never loaded: `scene_seg._load_model` is monkeypatched with a fake
whose "segmentation" is decided by the grey level of the frame (>128 -> a crosswalk rectangle).

Covers: mask -> box geometry (padding, stripe merging, size gate, ordering), the [E7] output
contract the five downstream consumers rely on (one row per NATIVE frame, 0-indexed, Yes/No,
python-literal box lists), the 5-sample majority vote, the threshold seam, the [E10] by-product,
header-only outputs, and that the real consumer (crosswalk_usage) can parse the file.
"""

import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.crosswalk import crosswalk_seg as cs  # noqa: E402
from modules.scene_seg import scene_seg as ss  # noqa: E402

W, H, FPS = 640, 360, 10

# fake id2label: every class name scene_seg needs, plus "Void"
_NAMES = ["Void"] + sorted({n for names in ss.CLASS_GROUPS.values() for n in names})
FAKE_ID2LABEL = {i: n for i, n in enumerate(_NAMES)}
NAME2ID = {n: i for i, n in FAKE_ID2LABEL.items()}
RECT = (100, 50, 200, 60)            # x, y, w, h of the crosswalk the fake model "sees"


def _rect_mask(rects, shape=(H, W)):
    m = np.zeros(shape, bool)
    for x, y, w, h in rects:
        m[y:y + h, x:x + w] = True
    return m


class FakeProcessor:
    def __init__(self):
        self._grey = 0.0

    def __call__(self, images, return_tensors="pt", size=None, **kwargs):
        self._grey = float(np.asarray(images).mean())
        return {"pixel_values": torch.zeros(1, 3, 8, 8)}

    def post_process_semantic_segmentation(self, outputs, target_sizes=None):
        assert target_sizes == [(H, W)]
        seg = np.full((H, W), NAME2ID["Road"], dtype=np.int64)
        if self._grey > 128:                                   # a frame "with a crosswalk"
            x, y, w, h = RECT
            seg[y:y + h, x:x + w] = NAME2ID["Lane Marking - Crosswalk"]
        return [torch.from_numpy(seg)]


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(id2label=FAKE_ID2LABEL)
        self.dtype = torch.float32
        self.n_forward = 0

    def __call__(self, pixel_values=None, **kwargs):
        self.n_forward += 1
        return SimpleNamespace()


@pytest.fixture
def fake_loader(monkeypatch):
    state = {"loads": 0, "model": None}

    def _load(model_id, device):
        state["loads"] += 1
        state["model"] = FakeModel()
        return state["model"], FakeProcessor()

    monkeypatch.setattr(ss, "_load_model", _load)
    return state


@pytest.fixture
def results_root(tmp_path, monkeypatch):
    root = tmp_path / "results"
    monkeypatch.setattr(cs, "RESULTS_ROOT", str(root))
    return root


def make_video(path, greys):
    """One constant-grey frame per entry (MJPG keeps a flat frame within a grey level or two)."""
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    assert vw.isOpened()
    for g in greys:
        vw.write(np.full((H, W, 3), g, dtype=np.uint8))
    vw.release()
    return str(path)


def sampled_greys(pattern, n_native=None):
    """Grey levels for a clip whose analysed frames (every 10th) follow `pattern` (True = a
    crosswalk); the frames in between are never decoded, so their content is irrelevant."""
    n_native = n_native or (len(pattern) * FPS)
    greys = [50] * n_native
    for i, has in enumerate(pattern):
        greys[i * FPS] = 220 if has else 40
    return greys


# --------------------------------------------------------------------------- box geometry

def test_single_rectangle_box_is_padded_in_x_only():
    boxes = cs.crosswalk_boxes_from_mask(_rect_mask([RECT]))
    # width 200 -> pad 0.15 * 200 = 30 per side; y untouched
    assert boxes == [[70.0, 50.0, 330.0, 110.0]]


def test_zebra_stripes_merge_into_one_box():
    # five stacked stripes (8 px tall, 6 px gaps): the closing kernel (13 px at 640 wide) must
    # bridge the gaps, otherwise each stripe fails the 24 px height gate and nothing is found
    stripes = [(100, 20 + i * 14, 200, 8) for i in range(5)]
    boxes = cs.crosswalk_boxes_from_mask(_rect_mask(stripes))
    assert len(boxes) == 1
    x1, y1, x2, y2 = boxes[0]
    assert (y1, y2) == (20.0, 20.0 + 4 * 14 + 8) and x1 == 70.0 and x2 == 330.0


def test_size_gate_rejects_small_regions():
    assert cs.crosswalk_boxes_from_mask(_rect_mask([(10, 10, 30, 10)])) == []      # too small
    assert cs.crosswalk_boxes_from_mask(_rect_mask([(10, 10, 59, 40)])) == []      # 1 px too narrow
    assert cs.crosswalk_boxes_from_mask(_rect_mask([(10, 10, 80, 23)])) == []      # 1 px too short
    assert len(cs.crosswalk_boxes_from_mask(_rect_mask([(10, 10, 60, 24)]))) == 1  # exactly at the gate


def test_separate_regions_give_separate_boxes_largest_first():
    boxes = cs.crosswalk_boxes_from_mask(_rect_mask([(20, 20, 100, 40), (300, 200, 250, 80)]))
    assert len(boxes) == 2
    assert (boxes[0][2] - boxes[0][0]) > (boxes[1][2] - boxes[1][0])               # bigger area first


def test_empty_and_degenerate_masks():
    assert cs.crosswalk_boxes_from_mask(np.zeros((H, W), bool)) == []
    assert cs.crosswalk_boxes_from_mask(np.zeros((0, 0), bool)) == []


def test_size_gates_are_shared_with_the_legacy_module():
    import ast as _ast
    src = open(os.path.join(os.path.dirname(cs.__file__), "crosswalk.py"), encoding="utf-8").read()
    tree = _ast.parse(src)
    # the legacy module must IMPORT the gates (single source of truth), not redefine them
    assert not any(isinstance(n, _ast.Assign) and any(getattr(t, "id", "") == "MIN_CROSSWALK_BOX_W_PX"
                                                    for t in n.targets) for n in tree.body)
    assert "from modules.crosswalk.crosswalk_seg import" in src


# --------------------------------------------------------------------------- output contract

def test_contract_one_row_per_native_frame(tmp_path, fake_loader, results_root):
    video = make_video(tmp_path / "clipA.avi", sampled_greys([True, True, True], n_native=25))
    cs.run_crosswalk_detection(video, analyze_interval_sec=1.0, device="cpu")
    e7 = results_root / "clipA" / "[E7]crosswalk_detection.csv"
    df = pd.read_csv(e7)
    assert list(df.columns) == ["frame_id", "crosswalk_detected", "crosswalk_boxes"]
    assert df["frame_id"].tolist() == list(range(25))                    # native frames, 0-indexed
    assert set(df["crosswalk_detected"]) == {"Yes"}
    import ast
    boxes = [ast.literal_eval(b) for b in df["crosswalk_boxes"]]
    assert all(b == [[70.0, 50.0, 330.0, 110.0]] for b in boxes)         # python-literal box lists
    # model loaded once; only the 3 analysed frames (0, 10, 20) were run
    assert fake_loader["loads"] == 1 and fake_loader["model"].n_forward == 3
    # the by-product: raw per-analysed-frame fractions, the only record once the video is deleted
    e10 = pd.read_csv(results_root / "clipA" / "[E10]scene_segmentation.csv")
    assert e10["frame_id"].tolist() == [0, 10, 20]
    assert e10["crosswalk_frac"].tolist() == pytest.approx([round(200 * 60 / (W * H), 5)] * 3)
    assert (e10["crosswalk_present"] == 1).all()


def test_real_consumer_parses_the_file(tmp_path, fake_loader, results_root):
    from modules.crosswalk_usage.crosswalk_usage import load_crosswalk_boxes
    video = make_video(tmp_path / "clipB.avi", sampled_greys([True, False, True]))
    cs.run_crosswalk_detection(video, device="cpu")
    boxes = load_crosswalk_boxes(str(results_root / "clipB" / "[E7]crosswalk_detection.csv"))
    assert isinstance(boxes, dict) and len(boxes) > 0
    first = next(iter(boxes.values()))
    assert first and len(first[0]) == 4


def test_no_crosswalk_gives_all_no_and_empty_boxes(tmp_path, fake_loader, results_root):
    video = make_video(tmp_path / "clipC.avi", sampled_greys([False, False, False]))
    cs.run_crosswalk_detection(video, device="cpu")
    df = pd.read_csv(results_root / "clipC" / "[E7]crosswalk_detection.csv")
    assert set(df["crosswalk_detected"]) == {"No"} and set(df["crosswalk_boxes"]) == {"[]"}


# --------------------------------------------------------------------------- smoothing

def test_majority_vote_removes_flicker_and_borrows_boxes(tmp_path, fake_loader, results_root):
    # one missed sample inside a run of detections: the centred 5-sample vote restores it
    video = make_video(tmp_path / "clipD.avi", sampled_greys([True, True, False, True, True]))
    cs.run_crosswalk_detection(video, device="cpu")
    df = pd.read_csv(results_root / "clipD" / "[E7]crosswalk_detection.csv")
    assert set(df["crosswalk_detected"]) == {"Yes"}
    middle = df[(df["frame_id"] >= 20) & (df["frame_id"] < 30)]
    assert (middle["crosswalk_boxes"] != "[]").all()                     # boxes borrowed from a neighbour
    # an isolated detection among misses is suppressed
    video = make_video(tmp_path / "clipE.avi", sampled_greys([False, False, True, False, False]))
    cs.run_crosswalk_detection(video, device="cpu")
    df = pd.read_csv(results_root / "clipE" / "[E7]crosswalk_detection.csv")
    assert set(df["crosswalk_detected"]) == {"No"}


def test_threshold_is_the_scene_seg_constant(tmp_path, fake_loader, results_root, monkeypatch):
    video = make_video(tmp_path / "clipF.avi", sampled_greys([True, True, True]))
    # the fake crosswalk covers 0.052 of the frame: a threshold above that must turn it off
    monkeypatch.setitem(ss.PRESENCE_THRESHOLDS, "crosswalk", 0.5)
    cs.run_crosswalk_detection(video, device="cpu")
    df = pd.read_csv(results_root / "clipF" / "[E7]crosswalk_detection.csv")
    assert set(df["crosswalk_detected"]) == {"No"} and set(df["crosswalk_boxes"]) == {"[]"}


# --------------------------------------------------------------------------- robustness

def test_unreadable_video_header_only_and_no_model_load(tmp_path, fake_loader, results_root, capsys):
    cs.run_crosswalk_detection(str(tmp_path / "missing.avi"), device="cpu")
    e7 = pd.read_csv(results_root / "missing" / "[E7]crosswalk_detection.csv")
    e10 = pd.read_csv(results_root / "missing" / "[E10]scene_segmentation.csv")
    assert list(e7.columns) == cs.COLUMNS and e7.empty
    assert list(e10.columns) == ss.OUTPUT_COLUMNS and e10.empty
    assert fake_loader["loads"] == 0
    assert "Could not open video" in capsys.readouterr().out


def test_explicit_output_path_and_legacy_kwargs_are_accepted(tmp_path, fake_loader, results_root):
    video = make_video(tmp_path / "clipG.avi", sampled_greys([True]))
    out = tmp_path / "custom" / "e7.csv"
    out.parent.mkdir()
    # conf/show belong to the retired YOLO module's signature and must not break old call sites
    cs.run_crosswalk_detection(video, 1.0, str(out), conf=0.35, show=False, device="cpu")
    assert out.exists() and (out.parent / "[E10]scene_segmentation.csv").exists()


def test_main_dispatches_crosswalk_to_the_segmentation_module():
    main_src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "main.py"), encoding="utf-8").read()
    assert "from modules.crosswalk.crosswalk_seg import run_crosswalk_detection" in main_src
    assert "from modules.crosswalk.crosswalk import run_crosswalk_detection" not in main_src


def test_import_is_cheap():
    import subprocess
    code = ("import sys; import modules.crosswalk.crosswalk_seg; "
            "print('torch' in sys.modules, 'transformers' in sys.modules, 'ultralytics' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["False", "False", "False"], out.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
