"""Age/gender estimation from archived per-track pedestrian crops ([P6]).

Replaces the removed paddlex pedestrian-attribute pipeline (paddlex capped
numpy<2.4 / pinned opencv 4.10 and was dropped in the numpy 2.5 / OpenCV 5
upgrade) with a dependency-light two-stage approach:

  1. Face detection: OpenCV's built-in YuNet (cv2.FaceDetectorYN) with
     face_detection_yunet_2023mar.onnx (~230 KB, github.com/opencv/opencv_zoo),
     run on the upsampled upper body of each pedestrian crop.
  2. Age + gender: InsightFace's genderage.onnx (~1.3 MB, extracted from
     buffalo_l.zip on github.com/deepinsight/insightface releases). Input is a
     96x96 aligned face crop; output is [female_logit, male_logit, age/100].
     Loaded via cv2.dnn.readNetFromONNX, with an onnxruntime fallback ONLY if
     cv2.dnn cannot load the model.

Input path convention (works WITHOUT the video, which is deleted after
analysis): analysis_results/<video_name>/pedestrian_img/id_<track_id>/*.png

Output: [P6]age_gender.csv with the legacy columns ['id', 'age', 'gender'].
'age' is integer years (the legacy paddlex buckets are gone; consumers —
crossed_info.py joins on 'id' and reads 'gender'/'age', video_info.py takes
the mode of 'age' — only need the column to exist). Tracks with no detected
face are skipped entirely; consumers already treat missing ids as 'None'.

Weights are cached under modules/age_gender/weights/ and auto-downloaded on
first use if absent.
"""

import os
import csv
import statistics
import urllib.request
import zipfile
from collections import Counter

import numpy as np
import cv2

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
YUNET_PATH = os.path.join(WEIGHTS_DIR, "face_detection_yunet_2023mar.onnx")
GENDERAGE_PATH = os.path.join(WEIGHTS_DIR, "genderage.onnx")

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
BUFFALO_L_URL = ("https://github.com/deepinsight/insightface/releases/"
                 "download/v0.7/buffalo_l.zip")

# YuNet score threshold. The default 0.9 is tuned for webcam-quality faces and
# misses small blurry dashcam/street-cam faces; 0.6 keeps precision acceptable
# while roughly doubling recall on the archived crops.
FACE_SCORE_THRESHOLD = 0.6
# Fraction of the pedestrian crop (from the top) treated as "upper body" for
# face detection. Generous on purpose: crops sometimes contain only part of
# the body, and detecting on a smaller region is cheap anyway.
UPPER_BODY_FRACTION = 0.5
# Upsample the upper-body region so its width is at least this many pixels
# (crops can be ~50 px wide; a face in them is ~15 px, below YuNet's sweet
# spot). Capped to avoid blowing up already-large crops.
UPSCALE_TARGET_WIDTH = 192.0
UPSCALE_MAX_FACTOR = 4.0

GENDERAGE_INPUT_SIZE = 96  # genderage.onnx expects a 96x96 aligned face crop


def ensure_weights(weights_dir=None):
    """Download missing model weights into `weights_dir` (idempotent).

    YuNet is fetched directly from the opencv_zoo repo. genderage.onnx ships
    only inside InsightFace's buffalo_l.zip release asset (~275 MB), so that
    zip is streamed to memory-backed storage and only the ~1.3 MB member is
    extracted.
    Returns (yunet_path, genderage_path).
    """
    weights_dir = weights_dir or WEIGHTS_DIR
    os.makedirs(weights_dir, exist_ok=True)
    yunet_path = os.path.join(weights_dir, os.path.basename(YUNET_PATH))
    genderage_path = os.path.join(weights_dir, os.path.basename(GENDERAGE_PATH))

    if not os.path.isfile(yunet_path):
        print(f"[age_gender] downloading YuNet face detector -> {yunet_path}")
        tmp = yunet_path + ".part"
        urllib.request.urlretrieve(YUNET_URL, tmp)
        os.replace(tmp, yunet_path)

    if not os.path.isfile(genderage_path):
        print("[age_gender] downloading buffalo_l.zip (~275 MB) to extract "
              f"genderage.onnx -> {genderage_path}")
        tmp_zip = genderage_path + ".zip.part"
        urllib.request.urlretrieve(BUFFALO_L_URL, tmp_zip)
        try:
            with zipfile.ZipFile(tmp_zip) as zf:
                # Released zips have stored the member either at the root or
                # under a buffalo_l/ prefix; accept both.
                member = next(n for n in zf.namelist()
                              if n.endswith("genderage.onnx"))
                with zf.open(member) as src, open(genderage_path, "wb") as dst:
                    dst.write(src.read())
        finally:
            if os.path.isfile(tmp_zip):
                os.remove(tmp_zip)

    return yunet_path, genderage_path


class _GenderAgeNet:
    """genderage.onnx via cv2.dnn, falling back to onnxruntime only if
    cv2.dnn cannot load the model (per the module contract)."""

    def __init__(self, model_path):
        self._session = None
        self._net = None
        try:
            self._net = cv2.dnn.readNetFromONNX(model_path)
        except cv2.error as cv_err:
            try:
                import onnxruntime  # noqa: F401  (fallback only)
            except ImportError as e:
                raise RuntimeError(
                    f"cv2.dnn could not load {model_path} ({cv_err}) and "
                    "onnxruntime is not installed as a fallback."
                ) from e
            self._session = onnxruntime.InferenceSession(
                model_path, providers=["CPUExecutionProvider"])
            self._input_name = self._session.get_inputs()[0].name

    def predict(self, blob):
        """blob: NCHW float32 (1,3,96,96). Returns 1-D array
        [female_logit, male_logit, age/100]."""
        if self._net is not None:
            self._net.setInput(blob)
            out = self._net.forward()
        else:
            out = self._session.run(None, {self._input_name: blob})[0]
        return np.asarray(out).reshape(-1)


def _make_face_detector(yunet_path):
    return cv2.FaceDetectorYN.create(
        yunet_path, "", (320, 320),
        score_threshold=FACE_SCORE_THRESHOLD,
        nms_threshold=0.3,
        top_k=50,
    )


def detect_best_face(detector, crop):
    """Detect the highest-confidence face in the upsampled upper body of a
    pedestrian crop. Returns (x, y, w, h) in ORIGINAL crop coordinates, or
    None if no face is found."""
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    if h < 16 or w < 16:
        return None

    upper = crop[: max(16, int(round(h * UPPER_BODY_FRACTION)))]
    uh, uw = upper.shape[:2]

    scale = min(UPSCALE_MAX_FACTOR, max(1.0, UPSCALE_TARGET_WIDTH / uw))
    if scale > 1.0:
        upper = cv2.resize(upper, (int(round(uw * scale)), int(round(uh * scale))),
                           interpolation=cv2.INTER_CUBIC)

    detector.setInputSize((upper.shape[1], upper.shape[0]))
    _, faces = detector.detect(upper)
    if faces is None or len(faces) == 0:
        return None

    best = faces[int(np.argmax(faces[:, 14]))]
    x, y, fw, fh = (best[:4] / scale).tolist()
    return float(x), float(y), float(fw), float(fh)


def _align_face_96(crop, bbox):
    """InsightFace Attribute-model preprocessing: similarity transform that
    maps the face-box center to the center of a 96x96 patch with
    scale = 96 / (1.5 * max(w, h)). Mirrors insightface's
    face_align.transform(img, center, 96, scale, rotate=0)."""
    x, y, w, h = bbox
    cx, cy = x + w / 2.0, y + h / 2.0
    s = GENDERAGE_INPUT_SIZE / (max(w, h) * 1.5)
    half = GENDERAGE_INPUT_SIZE / 2.0
    m = np.array([[s, 0.0, half - cx * s],
                  [0.0, s, half - cy * s]], dtype=np.float32)
    return cv2.warpAffine(crop, m,
                          (GENDERAGE_INPUT_SIZE, GENDERAGE_INPUT_SIZE),
                          borderValue=0.0)


def predict_age_gender(image_path, detector, net):
    """Classify a single pedestrian crop image.

    Returns (age_years:int, gender:str) or (None, None) when the image is
    unreadable or contains no detectable face.
    """
    crop = cv2.imread(image_path)
    if crop is None:
        return None, None
    bbox = detect_best_face(detector, crop)
    if bbox is None:
        return None, None

    aligned = _align_face_96(crop, bbox)
    # insightface Attribute: input_mean=0, input_std=1, RGB order.
    blob = cv2.dnn.blobFromImage(
        aligned, 1.0, (GENDERAGE_INPUT_SIZE, GENDERAGE_INPUT_SIZE),
        (0.0, 0.0, 0.0), swapRB=True)
    pred = net.predict(blob.astype(np.float32))
    if pred.shape[0] < 3:
        return None, None

    gender = "male" if pred[1] > pred[0] else "female"
    age = int(round(float(pred[2]) * 100.0))
    age = max(0, min(100, age))
    return age, gender


def run_age_gender(video_path, img_root=None, output_csv_path=None):
    """Entry point (legacy signature preserved; new args are optional
    overrides used by tests/validation so archived folders are never touched).

    Reads per-track crops from
    analysis_results/<video_name>/pedestrian_img/id_<track_id>/*.png and
    writes [P6]age_gender.csv with columns ['id', 'age', 'gender'].
    Per track: majority-vote gender, median age (integer years) across all
    crops with a detected face. Tracks with no face are not written.
    Missing pedestrian_img directory -> header-only CSV.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if img_root is None:
        img_root = os.path.join("./analysis_results", video_name, "pedestrian_img")
    if output_csv_path is None:
        output_csv_path = os.path.join("./analysis_results", video_name,
                                       "[P6]age_gender.csv")

    final_results = {}

    if os.path.isdir(img_root):
        yunet_path, genderage_path = ensure_weights()
        detector = _make_face_detector(yunet_path)
        net = _GenderAgeNet(genderage_path)

        for person_id in sorted(os.listdir(img_root)):
            person_folder = os.path.join(img_root, person_id)
            if not os.path.isdir(person_folder):
                continue

            ages, genders = [], []
            for img_file in sorted(os.listdir(person_folder)):
                if not img_file.lower().endswith(".png"):
                    continue
                age, gender = predict_age_gender(
                    os.path.join(person_folder, img_file), detector, net)
                if age is not None and gender is not None:
                    ages.append(age)
                    genders.append(gender)

            if ages:
                final_age = int(round(statistics.median(ages)))
                final_gender = Counter(genders).most_common(1)[0][0]
            else:
                final_age = None
                final_gender = None

            final_results[person_id] = {"age": final_age, "gender": final_gender}
            print(f"Finished analyzing {person_id}: "
                  f"Age={final_age}, Gender={final_gender}")
    else:
        print(f"[age_gender] no pedestrian_img directory at {img_root}; "
              "writing header-only CSV")

    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "age", "gender"])
        for person_id, res in final_results.items():
            if res["age"] is None or res["gender"] is None:
                continue  # no face found on any crop: leave the track absent
            clean_id = person_id.replace("id_", "")
            writer.writerow([clean_id, res["age"], res["gender"]])

    print(f"Age and gender results saved to {output_csv_path}")
    return final_results
