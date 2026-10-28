"""[E10] Scene segmentation with Mask2Former (Swin-L) trained on Mapillary Vistas.

Runs `facebook/mask2former-swin-large-mapillary-vistas-semantic` (Hugging Face
`transformers`: Mask2FormerForUniversalSegmentation + AutoImageProcessor, then
`post_process_semantic_segmentation`) on sampled video frames and writes, per analysed
frame, the share of image pixels assigned to a few street-scene classes:

    <RESULTS_ROOT>/<video_name>/[E10]scene_segmentation.csv
    frame_id, crosswalk_frac, sidewalk_frac, curb_frac, road_frac, traffic_light_frac,
    traffic_sign_frac, pothole_frac, manhole_frac, person_frac,
    crosswalk_present, sidewalk_present, traffic_light_present, model_id

It is a candidate replacement for the YOLO crosswalk detector ([E7], fires on 0% of
frames in most videos) and the SGDepth sidewalk subprocess ([E9]); the JAAD benchmark
harness calibrates PRESENCE_THRESHOLDS. Not part of single_all.

Frame sampling -- copied verbatim from modules/traffic_light/traffic_light.py ([E2]) so the
two files' rows line up one-to-one:
  * frame_id is 0-INDEXED: the first decoded frame is frame_id 0 (like [E2]; tracking [B1]
    is 1-indexed, see AGENTS.md).
  * fps = ceil(CAP_PROP_FPS) (30 when unknown), stride N = max(1, int(fps * interval)).
    For the production 1 s interval this is ceil(fps * interval); for fractional
    intervals [E2]'s int() truncation is kept on purpose (row alignment beats elegance).
  * one row per ANALYSED frame (frame_id % N == 0). Non-sampled frames are skipped with
    cap.grab() (no decode); only sampled frames are decoded with cap.retrieve(). A
    sampled frame that fails to decode is skipped (no row): it was not analysed, so
    consumers that divide by len(df) keep an honest denominator.

Fractions:
  * `<group>_frac` = (pixels whose arg-max class is in CLASS_GROUPS[group]) / (H * W),
    measured on a semantic map resized back to the ORIGINAL frame size, rounded to 5
    decimals. Groups are disjoint, so fractions of different groups never double count.
  * `<group>_present` = int(rounded frac >= PRESENCE_THRESHOLDS[group]); using the rounded
    value keeps every *_present column exactly reproducible from the CSV.
  * Classes are mapped by NAME through the checkpoint's own config (`id2label`); class ids
    are never hard-coded. A missing name is a hard error, never a silent zero column.

Resolution (`short_side`): None (default) keeps the frame's NATIVE resolution, aspect
preserved (the processor only rounds each side up to a multiple of 32). An int rescales
the frame so its short side equals that value (normally a downscale, for speed). The hub's
preprocessor_config.json for this checkpoint asks for a fixed 384x384 SQUARE resize, which
distorts 16:9 dashcam frames and shrinks small objects; on JAAD video_0039 (1080p) it
lost most of the crosswalk on 4 of 12 frames and labelled scaffolding as sky, while being
only ~1.6x faster, so it is deliberately not used. (The original Mask2Former Vistas test
protocol is long side 2048, i.e. ~native for 1080p.) Native 1080p runs at ~0.15 s/frame
in FP16 on an RTX 4090 (~1.7 GB VRAM peak).

Weights: loaded with cache_dir = $PEDX_HF_CACHE, else E:/PedX-validation-data/hf_cache if
it exists, else the default HF cache; local files first, then a normal load. FP16 on CUDA
under torch.inference_mode(); the model and VRAM are released when the run ends.
"""

import importlib.util
import math
import os
import sys
import time

import cv2
import pandas as pd

from modules.paths import RESULTS_ROOT

DEFAULT_MODEL_ID = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
DEFAULT_MODEL_REVISION = "4772b6bf101d91f2534c106dc524d906aeb3c68a"
DEFAULT_HF_CACHE = "E:/PedX-validation-data/hf_cache"
OUTPUT_FILENAME = "[E10]scene_segmentation.csv"

# Mapillary Vistas v1.2 class names (exactly as in the checkpoint's id2label) per output
# group. Groups are disjoint.
#  * crosswalk: union of the plain (lines-only) crossing area and the zebra marking.
#  * sidewalk: "Pedestrian Area" (plazas, pedestrian streets) is walkable pedestrian space;
#    common Vistas->Cityscapes mappings fold it into Cityscapes 'sidewalk', the class the
#    SGDepth [E9] module detects. Curbs are reported separately (curb_frac).
#  * curb: "Curb Cut" is the lowered section of the same curb.
#  * road: "Lane Marking - General" is paint on the carriageway; leaving it out would make
#    road_frac drop with marking density. Crosswalk, manhole and pothole pixels have their
#    own columns, so the visible carriageway ~= road + crosswalk + manhole + pothole.
#  * traffic_sign: front-facing faces only (backs and sign frames carry no readable sign).
CLASS_GROUPS = {
    "crosswalk": ["Crosswalk - Plain", "Lane Marking - Crosswalk"],
    "sidewalk": ["Sidewalk", "Pedestrian Area"],
    "curb": ["Curb", "Curb Cut"],
    "road": ["Road", "Lane Marking - General"],
    "traffic_light": ["Traffic Light"],
    "traffic_sign": ["Traffic Sign (Front)"],
    "pothole": ["Pothole"],
    "manhole": ["Manhole"],
    "person": ["Person"],
}

# Presence thresholds on the pixel fraction.
#  * crosswalk: calibrated on the 346-clip JAAD benchmark (validation/results/JAAD_REPORT.md).
#    The F1-optimal threshold is ~0.0002 (stable across all five folds, 0.00017-0.00024); from
#    0.0002 to 0.0005 F1 is statistically indistinguishable (0.913 vs 0.906, 95% CIs overlap), so
#    0.0005 is used: no measurable loss, 2.5x the single-fragment noise floor, higher precision
#    (0.912) -- a spurious crosswalk feeds the usage metrics. The earlier provisional 0.002 gave
#    F1 0.873 (recall 0.82). Calibrated on JAAD (Canada/Ukraine): transfer to other cities is
#    unverified.
#  * sidewalk, traffic_light: still provisional (not calibrated).
PRESENCE_THRESHOLDS = {"crosswalk": 0.0005, "sidewalk": 0.01, "traffic_light": 0.0002}

FRAC_GROUPS = ["crosswalk", "sidewalk", "curb", "road", "traffic_light", "traffic_sign",
               "pothole", "manhole", "person"]
PRESENT_GROUPS = ["crosswalk", "sidewalk", "traffic_light"]
OUTPUT_COLUMNS = (["frame_id"]
                  + [f"{g}_frac" for g in FRAC_GROUPS]
                  + [f"{g}_present" for g in PRESENT_GROUPS]
                  + ["model_id"])

FRAC_DECIMALS = 5
# Upper bound handed to the processor's aspect-preserving resize so it never caps the
# long side (the short side alone sets the scale).
_NO_LONG_SIDE_CAP = 100_000


def _resolve_cache_dir():
    env = os.environ.get("PEDX_HF_CACHE")
    if env:
        return env
    if os.path.isdir(DEFAULT_HF_CACHE):
        return DEFAULT_HF_CACHE
    return None  # default HF cache


def _hide_broken_torchaudio():
    """transformers imports torchaudio at import time whenever it is installed. A torchaudio
    built for another CUDA than torch raises RuntimeError there and takes transformers down
    with it (seen here: torch cu126 vs torchaudio cu128). Audio is not needed, so a broken
    torchaudio is hidden (sys.modules entry None -> find_spec None -> 'not available')."""
    if "torchaudio" in sys.modules or importlib.util.find_spec("torchaudio") is None:
        return False
    try:
        import torchaudio  # noqa: F401
        return False
    except Exception as exc:  # RuntimeError (CUDA mismatch), OSError (DLL), ...
        sys.modules["torchaudio"] = None
        print(f"[scene_seg] torchaudio is installed but fails to import "
              f"({type(exc).__name__}); hiding it from transformers (audio is unused).")
        return True


def _load_model(model_id, device):
    """Return (model, processor) ready for inference on `device`.

    The single seam the tests replace with a fake. FP16 on CUDA, FP32 otherwise.
    """
    import torch
    _hide_broken_torchaudio()
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    cache_dir = _resolve_cache_dir()
    revision = os.environ.get("PEDX_SCENE_MODEL_REVISION") or (DEFAULT_MODEL_REVISION if model_id == DEFAULT_MODEL_ID else "main")
    try:
        processor = AutoImageProcessor.from_pretrained(
            model_id, revision=revision, cache_dir=cache_dir, local_files_only=True)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_id, revision=revision, cache_dir=cache_dir, local_files_only=True)
    except Exception as exc:
        print(f"[scene_seg] {model_id} not loadable from local cache "
              f"({type(exc).__name__}); trying a normal load.")
        processor = AutoImageProcessor.from_pretrained(model_id, revision=revision, cache_dir=cache_dir)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id, revision=revision, cache_dir=cache_dir)
    # A "swin.layernorm MISSING" line in transformers' load report is expected: SwinBackbone
    # declares that key ignorable and uses its per-stage hidden_states_norms instead.
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, processor


def _record_scene_assets(model, model_id, output_path, device, short_side):
    """Resolve local snapshot files without another hub request."""
    from pathlib import Path
    from modules.provenance import record_model_reference
    revision = getattr(model.config, "_commit_hash", None)
    if not revision:
        return
    roots = [_resolve_cache_dir(), os.environ.get("HF_HUB_CACHE"),
             str(Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")]
    files = []
    for root in roots:
        if not root:
            continue
        snapshot = Path(root) / ("models--" + model_id.replace("/", "--")) / "snapshots" / revision
        if snapshot.is_dir():
            files = [p for p in snapshot.iterdir() if p.is_file() and p.suffix in {".json", ".bin", ".safetensors"}]
            break
    record_model_reference(Path(output_path).parent, "scene_segmentation", model_id,
                           revision=revision, files=files,
                           parameters={"device": device, "short_side": short_side,
                                       "presence_thresholds": dict(PRESENCE_THRESHOLDS)})


def _group_class_ids(id2label):
    """{group: [class ids]} by NAME lookup in the model's id2label. Missing names raise."""
    label2id = {str(name): int(idx) for idx, name in id2label.items()}
    missing = sorted({n for names in CLASS_GROUPS.values() for n in names} - set(label2id))
    if missing:
        raise KeyError(f"[scene_seg] class names not in the model's id2label: {missing}. "
                       f"Check CLASS_GROUPS against the checkpoint's config.json.")
    return {g: [label2id[n] for n in names] for g, names in CLASS_GROUPS.items()}


def _row_from_counts(frame_id, counts, n_pixels, group_ids, model_id):
    """One CSV row from per-class pixel counts (indexable by class id)."""
    row = {"frame_id": int(frame_id)}
    for g in FRAC_GROUPS:
        px = sum(int(counts[i]) for i in group_ids[g] if i < len(counts))
        row[f"{g}_frac"] = round(px / n_pixels, FRAC_DECIMALS) if n_pixels else 0.0
    for g in PRESENT_GROUPS:
        row[f"{g}_present"] = int(row[f"{g}_frac"] >= PRESENCE_THRESHOLDS[g])
    row["model_id"] = model_id
    return row


def _segment_map(frame_bgr, model, processor, device, short_side):
    """The (h, w) semantic class map of one frame at the ORIGINAL frame size (torch long
    tensor, on whatever device the model ran on). Call under torch.inference_mode()."""
    import torch
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    target_short = int(short_side) if short_side else min(h, w)
    inputs = processor(images=rgb, return_tensors="pt",
                       size={"shortest_edge": target_short, "longest_edge": _NO_LONG_SIDE_CAP})
    pixel_values = inputs["pixel_values"].to(device=device, dtype=model.dtype)
    outputs = model(pixel_values=pixel_values)
    seg = processor.post_process_semantic_segmentation(outputs, target_sizes=[(h, w)])[0]
    return torch.as_tensor(seg).long()


def _segment_counts(frame_bgr, model, processor, device, short_side, num_labels):
    """Per-class pixel counts (numpy, length >= num_labels) of one frame's semantic map,
    measured at the original frame size."""
    import torch
    seg = _segment_map(frame_bgr, model, processor, device, short_side)
    return torch.bincount(seg.reshape(-1), minlength=num_labels).cpu().numpy()


def _default_output_path(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(RESULTS_ROOT, video_name, OUTPUT_FILENAME)


def _write(rows, output_csv_path):
    out_dir = os.path.dirname(output_csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(output_csv_path, index=False)


def run_scene_segmentation(video_path, analyze_interval_sec=1.0, output_csv_path=None,
                           model_id=DEFAULT_MODEL_ID, device=None, short_side=None):
    """Segment sampled frames of `video_path` and write [E10]scene_segmentation.csv.

    Returns the output CSV path. Empty/unreadable videos yield a header-only CSV.
    """
    if output_csv_path is None:
        output_csv_path = _default_output_path(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        cap.release()
        _write([], output_csv_path)
        print(f"Scene segmentation: 0 frames analysed, crosswalk_present in 0/0 frames. "
              f"Header-only results saved to {output_csv_path}")
        return output_csv_path

    # Mirrors [E2] traffic_light.py exactly (see module docstring).
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = math.ceil(fps) if fps > 0 else 30
    analyze_every_n_frames = max(1, int(fps * analyze_interval_sec))

    rows = []
    frame_id = -1
    decode_failures = 0
    model = processor = None
    group_ids = None
    num_labels = 0
    used_device = device
    seconds = 0.0
    torch = None
    try:
        while cap.isOpened():
            ok = cap.grab()
            if not ok:
                break
            frame_id += 1
            if frame_id % analyze_every_n_frames != 0:
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None or frame.size == 0:
                decode_failures += 1
                continue

            if model is None:  # lazy: empty videos never load the model
                import torch
                if used_device is None:
                    used_device = "cuda" if torch.cuda.is_available() else "cpu"
                model, processor = _load_model(model_id, used_device)
                _record_scene_assets(model, model_id, output_csv_path, used_device, short_side)
                id2label = model.config.id2label
                group_ids = _group_class_ids(id2label)
                num_labels = max(int(i) for i in id2label) + 1

            t0 = time.perf_counter()
            with torch.inference_mode():
                counts = _segment_counts(frame, model, processor, used_device, short_side,
                                         num_labels)
            seconds += time.perf_counter() - t0
            n_pixels = frame.shape[0] * frame.shape[1]
            rows.append(_row_from_counts(frame_id, counts, n_pixels, group_ids, model_id))
    finally:
        cap.release()
        if model is not None:
            del model, processor
            model = processor = None
            import gc
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    _write(rows, output_csv_path)

    n = len(rows)
    n_cw = sum(r["crosswalk_present"] for r in rows)
    share = (100.0 * n_cw / n) if n else 0.0
    speed = f", {seconds / n:.3f} s/frame on {used_device}" if n else ""
    failed = f", {decode_failures} sampled frames failed to decode" if decode_failures else ""
    print(f"Scene segmentation: {n} frames analysed, crosswalk_present in {n_cw}/{n} "
          f"({share:.1f}%) frames{speed}{failed}. Results saved to {output_csv_path}")
    return output_csv_path
