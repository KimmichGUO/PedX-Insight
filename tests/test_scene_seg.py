"""Tests for modules/scene_seg/scene_seg.py ([E10], --mode scene_seg).

Run from the repo root:
    python -m pytest -q tests/test_scene_seg.py

The Mask2Former model is never loaded: `_load_model` is monkeypatched with a fake
(model, processor) pair whose semantic map is known in advance. Each synthetic frame is a
flat grey whose level encodes its frame index, and the fake map depends on that index, so
the CSV proves WHICH frames were analysed, not only how many.

Covers: exact output columns, [E2]-mirrored 0-indexed sampling and stride (including the
int() truncation for fractional intervals), frac arithmetic incl. the crosswalk union and
the 5-decimal rounding, presence thresholds (>= at the boundary), header-only output on
unreadable / missing / empty videos without loading the model, PEDX_RESULTS_ROOT
redirection, and class lookup by name (missing names fail loudly).
"""

import os
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from modules.scene_seg import scene_seg as ss  # noqa: E402

EXPECTED_COLUMNS = [
    "frame_id", "crosswalk_frac", "sidewalk_frac", "curb_frac", "road_frac",
    "traffic_light_frac", "traffic_sign_frac", "pothole_frac", "manhole_frac",
    "person_frac", "crosswalk_present", "sidewalk_present", "traffic_light_present",
    "model_id",
]

# A shuffled, Vistas-like id2label: the module must find classes by NAME, so the ids
# here deliberately differ from the real checkpoint's.
FAKE_ID2LABEL = {
    0: "Sky", 1: "Lane Marking - Crosswalk", 2: "Road", 3: "Traffic Light",
    4: "Crosswalk - Plain", 5: "Sidewalk", 6: "Person", 7: "Curb",
    8: "Traffic Sign (Front)", 9: "Traffic Sign (Back)", 10: "Manhole", 11: "Pothole",
    12: "Pedestrian Area", 13: "Curb Cut", 14: "Lane Marking - General", 15: "Building",
}
NAME2ID = {v: k for k, v in FAKE_ID2LABEL.items()}

W, H = 100, 50          # 5000 px per frame: 1 px = 0.0002 of the image
N_PIX = W * H
FPS = 10


def _grey(index):
    return 10 + 8 * index           # frame index -> flat grey level (<= 250 for 30 frames)


def _index_from_rgb(rgb):
    return int(round((float(rgb.mean()) - 10) / 8))


def known_map(index):
    """Semantic map (H, W) for the frame with this index. Pixel budget (of 5000):
    plain crosswalk 10 + zebra 15 + index -> union 25 + index; sidewalk 40, pedestrian
    area 10 -> 50 = exactly the 0.01 threshold; traffic light 1 px = exactly 0.0002 on
    even frames, 0 on odd frames; curb 3 + curb cut 2; road 1000 + general marking 500;
    sign front 7 (back 9 must be ignored); manhole 4; pothole 2; person 33; sky the rest.
    """
    counts = [
        ("Crosswalk - Plain", 10), ("Lane Marking - Crosswalk", 15 + index),
        ("Sidewalk", 40), ("Pedestrian Area", 10),
        ("Traffic Light", 1 if index % 2 == 0 else 0),
        ("Curb", 3), ("Curb Cut", 2), ("Road", 1000), ("Lane Marking - General", 500),
        ("Traffic Sign (Front)", 7), ("Traffic Sign (Back)", 9),
        ("Manhole", 4), ("Pothole", 2), ("Person", 33),
    ]
    flat = np.full(N_PIX, NAME2ID["Sky"], dtype=np.int64)
    pos = 0
    for name, n in counts:
        flat[pos:pos + n] = NAME2ID[name]
        pos += n
    np.random.RandomState(index).shuffle(flat)      # location must not matter
    return flat.reshape(H, W)


class FakeProcessor:
    def __init__(self):
        self.calls = []
        self._index = None

    def __call__(self, images, return_tensors="pt", size=None, **kwargs):
        assert return_tensors == "pt"
        self.calls.append(size)
        self._index = _index_from_rgb(np.asarray(images))
        return {"pixel_values": torch.zeros(1, 3, 8, 8)}

    def post_process_semantic_segmentation(self, outputs, target_sizes=None):
        assert target_sizes == [(H, W)], target_sizes     # original frame size
        return [torch.from_numpy(known_map(self._index))]


class FakeModel:
    def __init__(self, id2label=None):
        self.config = SimpleNamespace(id2label=id2label or FAKE_ID2LABEL)
        self.dtype = torch.float32
        self.n_forward = 0

    def __call__(self, pixel_values=None, **kwargs):
        assert torch.is_inference_mode_enabled()
        self.n_forward += 1
        return SimpleNamespace()


@pytest.fixture
def fake_loader(monkeypatch):
    state = {"loads": 0, "model": None, "processor": None}

    def _fake_load_model(model_id, device):
        state["loads"] += 1
        state["model"], state["processor"] = FakeModel(), FakeProcessor()
        return state["model"], state["processor"]

    monkeypatch.setattr(ss, "_load_model", _fake_load_model)
    return state


@pytest.fixture
def results_root(tmp_path, monkeypatch):
    root = tmp_path / "results"
    monkeypatch.setattr(ss, "RESULTS_ROOT", str(root))
    return root


def make_video(path, n_frames, fps=FPS):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    vw = cv2.VideoWriter(str(path), fourcc, fps, (W, H))
    assert vw.isOpened(), "cv2.VideoWriter could not open an MJPG writer"
    for i in range(n_frames):
        vw.write(np.full((H, W, 3), _grey(i), dtype=np.uint8))
    vw.release()
    return str(path)


def expected_row(index):
    return {
        "crosswalk_frac": round((25 + index) / N_PIX, 5),
        "sidewalk_frac": round(50 / N_PIX, 5),
        "curb_frac": round(5 / N_PIX, 5),
        "road_frac": round(1500 / N_PIX, 5),
        "traffic_light_frac": round((1 if index % 2 == 0 else 0) / N_PIX, 5),
        "traffic_sign_frac": round(7 / N_PIX, 5),
        "pothole_frac": round(2 / N_PIX, 5),
        "manhole_frac": round(4 / N_PIX, 5),
        "person_frac": round(33 / N_PIX, 5),
    }


# --------------------------------------------------------------------------- contract

def test_output_columns_constant():
    assert ss.OUTPUT_COLUMNS == EXPECTED_COLUMNS
    assert set(ss.PRESENCE_THRESHOLDS) == {"crosswalk", "sidewalk", "traffic_light"}
    # crosswalk is the JAAD-calibrated value (see the comment on PRESENCE_THRESHOLDS); the
    # other two are still the provisional ones.
    assert ss.PRESENCE_THRESHOLDS == {"crosswalk": 0.0005, "sidewalk": 0.01,
                                      "traffic_light": 0.0002}
    assert set(ss.CLASS_GROUPS) == set(ss.FRAC_GROUPS)
    assert set(ss.CLASS_GROUPS["crosswalk"]) == {"Crosswalk - Plain",
                                                 "Lane Marking - Crosswalk"}
    assert ss.CLASS_GROUPS["traffic_sign"] == ["Traffic Sign (Front)"]
    # groups are disjoint: no pixel can be counted in two *_frac columns
    names = [n for group in ss.CLASS_GROUPS.values() for n in group]
    assert len(names) == len(set(names))


def test_class_names_exist_in_real_checkpoint_config():
    """CLASS_GROUPS names must match the downloaded checkpoint's id2label (skips if the
    cache is absent on this machine)."""
    import glob
    import json
    cache = ss._resolve_cache_dir() or ""
    cfgs = glob.glob(os.path.join(
        cache, "models--facebook--mask2former-swin-large-mapillary-vistas-semantic",
        "snapshots", "*", "config.json"))
    if not cfgs:
        pytest.skip("Mask2Former Vistas config.json not in the local HF cache")
    with open(cfgs[0], encoding="utf-8") as f:
        id2label = json.load(f)["id2label"]
    ids = ss._group_class_ids(id2label)          # raises on any missing name
    assert len(ids["crosswalk"]) == 2


def test_missing_class_name_fails_loudly():
    broken = {k: v for k, v in FAKE_ID2LABEL.items() if v != "Lane Marking - Crosswalk"}
    with pytest.raises(KeyError, match="Lane Marking - Crosswalk"):
        ss._group_class_ids(broken)


def test_import_is_cheap():
    code = ("import sys; sys.path.insert(0, %r); "
            "import modules.scene_seg.scene_seg; "
            "print('torch' in sys.modules, 'transformers' in sys.modules)" % REPO_ROOT)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=REPO_ROOT, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["False", "False"], out.stdout


# --------------------------------------------------------------------------- sampling

def test_columns_sampling_fracs_and_presence(tmp_path, fake_loader, results_root):
    video = make_video(tmp_path / "clipA.avi", n_frames=25)
    out = ss.run_scene_segmentation(video, analyze_interval_sec=1.0, device="cpu")

    assert out == os.path.join(str(results_root), "clipA", "[E10]scene_segmentation.csv")
    df = pd.read_csv(out)
    assert list(df.columns) == EXPECTED_COLUMNS
    # fps 10, interval 1 s -> N = 10, 0-indexed like [E2]: frames 0, 10, 20 of 0..24
    assert df["frame_id"].tolist() == [0, 10, 20]
    assert fake_loader["loads"] == 1
    assert fake_loader["model"].n_forward == 3
    # native resolution by default: short side = frame height
    assert fake_loader["processor"].calls[0]["shortest_edge"] == H

    for _, row in df.iterrows():
        idx = int(row["frame_id"])
        for col, val in expected_row(idx).items():
            assert row[col] == pytest.approx(val, abs=1e-12), (idx, col, row[col], val)
        # crosswalk = union of plain + zebra (25 + idx px >= 10 px threshold)
        assert row["crosswalk_present"] == 1
        # sidewalk = Sidewalk + Pedestrian Area = 50 px = exactly the 0.01 threshold
        assert row["sidewalk_present"] == 1
        # 1 px = 0.0002 exactly at threshold on even frames; 0 px on odd frames
        assert row["traffic_light_present"] == (1 if idx % 2 == 0 else 0)
        assert row["model_id"] == ss.DEFAULT_MODEL_ID

    for col in ("frame_id", "crosswalk_present", "sidewalk_present",
                "traffic_light_present"):
        assert df[col].dtype.kind == "i", (col, df[col].dtype)


@pytest.mark.parametrize("interval, expected", [
    (0.5, [0, 5, 10, 15, 20]),              # N = int(10 * 0.5) = 5
    (0.25, list(range(0, 25, 2))),          # N = int(2.5) = 2 ([E2] truncation, not ceil)
    (0.0, list(range(25))),                 # N = max(1, 0) = 1
    (3.0, [0]),                             # N = 30 > video length
])
def test_stride_mirrors_traffic_light(tmp_path, fake_loader, results_root,
                                      interval, expected):
    video = make_video(tmp_path / "clipB.avi", n_frames=25)
    out = ss.run_scene_segmentation(video, analyze_interval_sec=interval, device="cpu")
    df = pd.read_csv(out)
    assert df["frame_id"].tolist() == expected
    # the analysed content really is frame `frame_id` (grey level -> zebra pixel count)
    expect_cw = [round((25 + i) / N_PIX, 5) for i in expected]
    assert df["crosswalk_frac"].tolist() == pytest.approx(expect_cw, abs=1e-12)


def test_presence_thresholds_are_used_and_tunable(tmp_path, fake_loader, results_root,
                                                  monkeypatch):
    video = make_video(tmp_path / "clipC.avi", n_frames=11)
    # crosswalk frame 0 = 25 px = 0.005; frame 10 = 35 px = 0.007
    monkeypatch.setitem(ss.PRESENCE_THRESHOLDS, "crosswalk", 0.006)
    monkeypatch.setitem(ss.PRESENCE_THRESHOLDS, "sidewalk", 0.01001)
    df = pd.read_csv(ss.run_scene_segmentation(video, device="cpu"))
    assert df["frame_id"].tolist() == [0, 10]
    assert df["crosswalk_present"].tolist() == [0, 1]
    assert df["sidewalk_present"].tolist() == [0, 0]


def test_short_side_and_explicit_output_path(tmp_path, fake_loader, results_root):
    video = make_video(tmp_path / "clipD.avi", n_frames=3)
    target = tmp_path / "elsewhere" / "custom.csv"
    out = ss.run_scene_segmentation(video, output_csv_path=str(target), device="cpu",
                                    short_side=24, model_id="my/model")
    assert out == str(target) and target.is_file()
    assert not results_root.exists()           # explicit path wins over RESULTS_ROOT
    df = pd.read_csv(target)
    assert df["frame_id"].tolist() == [0]
    assert df["model_id"].tolist() == ["my/model"]
    assert fake_loader["processor"].calls[0]["shortest_edge"] == 24


# --------------------------------------------------------------------------- empty input

def _assert_header_only(path):
    df = pd.read_csv(path)
    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) == 0


def test_unreadable_video_header_only(tmp_path, fake_loader, results_root, capsys):
    bad = tmp_path / "garbage.mp4"
    bad.write_bytes(b"this is not a video" * 100)
    out = ss.run_scene_segmentation(str(bad), device="cpu")
    assert out == os.path.join(str(results_root), "garbage", "[E10]scene_segmentation.csv")
    _assert_header_only(out)
    assert fake_loader["loads"] == 0           # model never loaded for nothing
    assert "0 frames analysed" in capsys.readouterr().out


def test_missing_video_header_only(tmp_path, fake_loader, results_root):
    out = ss.run_scene_segmentation(str(tmp_path / "nope.mp4"), device="cpu")
    _assert_header_only(out)
    assert fake_loader["loads"] == 0


def test_zero_frame_video_header_only(tmp_path, fake_loader, results_root):
    video = make_video(tmp_path / "empty.avi", n_frames=0)
    out = ss.run_scene_segmentation(video, device="cpu")
    _assert_header_only(out)
    assert fake_loader["loads"] == 0


# --------------------------------------------------------------------------- results root

def test_results_root_env_var_redirects_output(tmp_path):
    """PEDX_RESULTS_ROOT set BEFORE import must redirect the default output path."""
    root = tmp_path / "redirected"
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from modules.scene_seg import scene_seg as ss\n"
        "print(ss.RESULTS_ROOT)\n"
        "print(ss._default_output_path('/some/dir/Paris1_abc.mp4'))\n" % REPO_ROOT
    )
    env = dict(os.environ, PEDX_RESULTS_ROOT=str(root))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(tmp_path), env=env, timeout=120)
    assert out.returncode == 0, out.stderr
    lines = out.stdout.strip().splitlines()
    assert lines[0] == str(root)
    assert lines[1] == os.path.join(str(root), "Paris1_abc", "[E10]scene_segmentation.csv")


def test_results_root_monkeypatch_writes_there(tmp_path, fake_loader, results_root):
    video = make_video(tmp_path / "London1_xyz.avi", n_frames=1)
    ss.run_scene_segmentation(video, device="cpu")
    produced = results_root / "London1_xyz" / "[E10]scene_segmentation.csv"
    assert produced.is_file()
    assert pd.read_csv(produced)["frame_id"].tolist() == [0]
